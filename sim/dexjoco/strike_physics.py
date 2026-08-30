#!/usr/bin/env python3
"""타격 구간의 **물리 요약값**으로 결과를 예측한다 — 표본 100개에 맞는 저차원 검정.

앞선 두 검정은 결론을 낼 수 없었다:
  · chunk_predicts_return.py : 절대 프레임 정렬 → 타격 시점이 60~358 로 흩어져 국면이 섞였다
  · strike_signal.py         : 사건에 정렬했지만 특징이 500차원인데 표본이 43~80 이라
                               ridge 로도 검정력이 없다. 게다가 창이 에피소드 밖으로 나가는
                               성공 에피소드가 제외되어 표본이 실패 쪽으로 치우쳤다

여기서는 환경의 물리를 그대로 쓴다. 못 삽입량이
    delta = 0.008 * min(3, |vz| / 0.02)        (panda_hammer_nail_env.py:642)
이므로 **타격 순간의 수직 속도**가 직접적인 결정 요인이고, 실측 ep1 처럼 세게 휘둘러도
빗나가면 깊이가 0 이므로 **타격 위치**도 필요하다. 고정 장면이라 못 위치가 상수여서
절대 x,y 를 그대로 쓸 수 있다.

특징 (액션 청크에서 뽑는다 = 정책이 실제로 고를 수 있는 것):
    z_min      명령된 최저 높이        (얼마나 깊이 내려찍나)
    vz_peak    명령된 최대 하강 속도    (삽입량에 직접 비례)
    x_at, y_at 최저점에서의 수평 위치   (못을 맞히나)
    grip       손가락 관절 평균         (망치를 쥐고 있나)

6차원이면 표본 100 개로 충분한 검정력이 나온다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = 20


def feats(chunk: np.ndarray) -> np.ndarray:
    """(R,25) 명령 청크 → 물리 요약 6차원."""
    z = chunk[:, 2]
    k = int(np.argmin(z))
    vz = np.diff(z)
    return np.array([z.min(), -vz.min(), chunk[k, 0], chunk[k, 1],
                     chunk[:, 9:].mean(), chunk[:, 9:].std()])


def ridge_cv(X, y, lams=(30., 10., 3., 1., .3, .1), folds=5, seed=0):
    n = len(y)
    fold = np.random.default_rng(seed).permutation(n) % folds
    best = -9.0
    for lam in lams:
        pred = np.zeros(n)
        for k in range(folds):
            tr, te = fold != k, fold == k
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd < 1e-8] = 1.0
            Xt = (X[tr] - mu) / sd
            ym = y[tr].mean()
            w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
            pred[te] = ((X[te] - mu) / sd) @ w + ym
        best = max(best, 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    return best


def main(sess: Path) -> None:
    eps = [json.loads(l) for l in (sess / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    rows, y, ok = [], [], []
    for e in eps:
        i = e["episode_index"]
        df = pd.read_parquet(sess / f"data/chunk-000/episode_{i:06d}.parquet",
                             columns=["observation.state", "action"])
        S = np.vstack([np.asarray(v, np.float32) for v in df["observation.state"]])
        A = np.vstack([np.asarray(v, np.float32) for v in df["action"]])
        s = int(np.argmin(np.diff(S[:, 2])))          # 타격 = 최대 하강 프레임
        if s < R:
            continue
        rows.append(feats(A[s - R:s]))                # 타격으로 **들어가는** 명령 청크
        y.append(e.get("final_nail_depth", 0.0))
        ok.append(bool(e.get("success")))
    X, y, ok = np.asarray(rows), np.asarray(y), np.asarray(ok)
    print(f"[표본] {len(y)} 에피소드 (성공 {ok.sum()}) — 창이 항상 에피소드 안이라 편향 없음")
    print(f"[타깃] final_nail_depth  평균 {y.mean():.4f} std {y.std():.4f}")

    names = ["z_min", "vz_peak", "x_at", "y_at", "grip_mean", "grip_std"]
    print(f"\n=== 개별 상관 (Pearson r, 타깃 = 깊이) ===")
    for j, nm in enumerate(names):
        r = np.corrcoef(X[:, j], y)[0, 1]
        # 순열 p
        null = [abs(np.corrcoef(X[:, j], np.random.default_rng(j * 999 + k).permutation(y))[0, 1])
                for k in range(2000)]
        pv = float(np.mean(np.asarray(null) >= abs(r)))
        print(f"  {nm:10} r = {r:+.3f}   p = {pv:.4f}{'  **' if pv < 0.05 else ''}")

    r2 = ridge_cv(X, y)
    null = [ridge_cv(X, np.random.default_rng(k).permutation(y), seed=k) for k in range(300)]
    pv = float(np.mean(np.asarray(null) >= r2))
    print(f"\n=== 6차원 전체 (5-fold CV) ===")
    print(f"  R² = {r2:+.3f}   순열 p = {pv:.4f}"
          f"{'  ** 액션이 결과를 예측한다' if pv < 0.05 and r2 > 0 else '  (신호 없음)'}")

    # 성공/실패로도 본다
    print(f"\n=== 성공 vs 실패 평균 차이 ===")
    for j, nm in enumerate(names):
        a, b = X[ok, j], X[~ok, j]
        d = (a.mean() - b.mean()) / max(X[:, j].std(), 1e-9)
        print(f"  {nm:10} 성공 {a.mean():+.4f}  실패 {b.mean():+.4f}   차이 {d:+.2f}σ")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1
              else "/workspace/junmo_cho/dexjoco/rollout/fixed/s0_n100"))
