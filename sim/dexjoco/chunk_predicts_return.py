#!/usr/bin/env python3
"""고정 장면에서 **액션 청크가 리턴을 예측하는가** — critic 없이 지도학습으로 상한을 잰다.

test-time Q selection/guidance 는 전부 Q(s,A) 가 A 에 따라 달라진다는 전제 위에 있다.
그런데 그 전제가 성립하는지는 critic 을 학습하기 전에 데이터만으로 답할 수 있다:
같은 상태에서 서로 다른 액션 청크가 서로 다른 리턴으로 이어지는가.

고정 장면(--fixed-scene)이라 t=0 에서 모든 에피소드의 상태가 **완전히 동일**하다.
따라서 t=0 의 회귀는 상태 교란이 전혀 없는 깨끗한 측정이다. t>0 은 궤적이 갈라져
상태도 달라지므로 예측력의 일부가 상태에서 올 수 있다 — 그래서 프레임별로 따로 낸다.

  target : G_t = γ^(T-t) if 성공 else 0        (프레임 t 에서 본 실제 감가 리턴)
  feature: 실행 구간 [latency, latency+replan) 의 액션 청크 (raw parquet, 표준화)

  CV R² <= 0  ->  청크가 리턴을 예측하지 못한다. Q(s,A) 가 A 에 평평한 것이 **정답**이고,
                  guidance 로 얻을 수 있는 이득의 상한이 0 이다. critic 을 고쳐도 소용없다.
  CV R² >  0  ->  신호가 있다. critic 이 그것을 못 뽑았다면 학습 쪽 문제다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAT, R, A_DIM, G = 5, 20, 25, 0.995


def ridge_cv_r2(X, y, lams=(300., 100., 30., 10., 3., 1.), folds=5, seed=0):
    """쌍이 아니라 에피소드 단위 K-fold. 절편 포함(중심화로 처리)."""
    n = len(y)
    r = np.random.default_rng(seed)
    fold = r.permutation(n) % folds
    best = (-9, None)
    for lam in lams:
        pred = np.zeros(n)
        for k in range(folds):
            tr, te = fold != k, fold == k
            if tr.sum() < 3 or te.sum() == 0:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd < 1e-8] = 1.0
            Xt = (X[tr] - mu) / sd
            ym = y[tr].mean()
            w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
            pred[te] = ((X[te] - mu) / sd) @ w + ym
        ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if ss > best[0]:
            best = (ss, lam, pred.copy())
    return best


def main(sess: Path) -> None:
    eps = [json.loads(l) for l in (sess / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    ok = np.array([bool(e.get("success")) for e in eps])
    T = np.array([e["length"] for e in eps])
    print(f"[데이터] {len(eps)} 에피소드  성공 {ok.sum()} ({100*ok.mean():.1f}%)  "
          f"길이 {T.min()}~{T.max()}")

    acts = {}
    for e in eps:
        i = e["episode_index"]
        df = pd.read_parquet(sess / f"data/chunk-000/episode_{i:06d}.parquet", columns=["action"])
        acts[i] = np.vstack([np.asarray(v, dtype=np.float32) for v in df["action"]])

    print(f"\n{'프레임':>6} {'표본':>5} {'리턴 std':>9} {'CV R²':>8} {'λ':>6} {'순열 p':>8}")
    for t in (0, 20, 40, 60, 90, 120, 150):
        X, y = [], []
        for e, o, tl in zip(eps, ok, T):
            i = e["episode_index"]
            if tl <= t + LAT + R:
                continue
            a = acts[i]
            # 프레임 t 에서 커밋되는 청크의 실행 구간. 로그에는 실행된 액션만 있으므로
            # 프레임 t+LAT .. t+LAT+R-1 의 액션이 그 구간에 해당한다.
            X.append(a[t + LAT:t + LAT + R].reshape(-1))
            y.append(G ** (tl - t) if o else 0.0)
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        if len(y) < 25 or y.std() < 1e-9:
            print(f"{t:>6} {len(y):>5}  표본/분산 부족")
            continue
        r2, lam, _ = ridge_cv_r2(X, y)
        null = []
        for s in range(200):
            rr = np.random.default_rng(500 + s)
            null.append(ridge_cv_r2(X, y[rr.permutation(len(y))], lams=(lam,), seed=s)[0])
        pv = float((np.asarray(null) >= r2).mean())
        print(f"{t:>6} {len(y):>5} {y.std():9.4f} {r2:8.3f} {lam:6.0f} {pv:8.3f}"
              f"{'  ** 신호' if pv < 0.05 and r2 > 0 else ''}")

    print("\n해석: t=0 은 고정 장면이라 상태가 완전히 동일하다 — 여기서 R²<=0 이면\n"
          "      '같은 상태에서 액션이 결과를 가르지 못한다'는 뜻이고, test-time\n"
          "      selection/guidance 의 이득 상한이 0 이다 (Q-VGM 도 같은 ∇_A Q 를 쓴다).\n"
          "      t>0 은 궤적이 갈라져 상태 정보가 섞이므로 상한으로만 읽을 것.")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1
              else "/workspace/junmo_cho/dexjoco/rollout/fixed/s0_n100"))
