#!/usr/bin/env python3
"""**타격 사건에 정렬해서** 액션이 상태 너머로 결과를 예측하는지 잰다.

앞선 chunk_predicts_return.py 는 절대 프레임(t=0,20,...)에 정렬했는데 그것이 틀렸다:
내려찍는 시점이 에피소드마다 164/178/312/167 프레임으로 제각각이라, 한 회귀 안에
파지 중인 에피소드와 이미 타격 중인 에피소드가 섞여 신호가 상쇄된다. hammer_nail 의
승패는 "망치를 잘 잡았나 / 못을 잘 내려찍었나" 에서 갈리므로 그 **사건**에 맞춰야 한다.

타격 검출: eef_z (state[2]) 의 하강 속도가 최대인 프레임. 환경이 못 삽입량을
    delta = 0.008 * min(3, |vz|/0.02)   (panda_hammer_nail_env.py:642)
로 주므로 수직 속도가 결과를 직접 좌우한다. 다만 속도만으로는 부족하다 — 실측 ep1 은
하강이 가장 빨랐는데 깊이 0 이었다 (못을 빗나감). 타격 위치도 액션 안에 있다.

핵심 비교 (이것이 critic 의 상한이다):
    R²(state 만)        상태로 설명되는 몫
    R²(state + action)  액션을 더했을 때
  둘의 차이가 0 이면 critic 이 액션을 봐도 얻을 것이 없다 — Q(s,A)=V(s) 가 정답이고
  test-time selection/guidance 의 이득 상한이 0 이다 (Q-VGM 도 같은 ∇_A Q 를 쓴다).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAT, R, A_DIM, G = 5, 20, 25, 0.995


def ridge_cv(X, y, lams=(1000., 300., 100., 30., 10., 3.), folds=5, seed=0):
    n = len(y)
    fold = np.random.default_rng(seed).permutation(n) % folds
    best = -9.0
    for lam in lams:
        pred = np.zeros(n)
        for k in range(folds):
            tr, te = fold != k, fold == k
            if tr.sum() < 5 or te.sum() == 0:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd < 1e-8] = 1.0
            Xt = (X[tr] - mu) / sd
            ym = y[tr].mean()
            w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
            pred[te] = ((X[te] - mu) / sd) @ w + ym
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        best = max(best, r2)
    return best


def main(sess: Path) -> None:
    eps = [json.loads(l) for l in (sess / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    S, A, meta = {}, {}, {}
    for e in eps:
        i = e["episode_index"]
        df = pd.read_parquet(sess / f"data/chunk-000/episode_{i:06d}.parquet",
                             columns=["observation.state", "action"])
        S[i] = np.vstack([np.asarray(v, np.float32) for v in df["observation.state"]])
        A[i] = np.vstack([np.asarray(v, np.float32) for v in df["action"]])
        meta[i] = e

    # 타격 프레임 = eef_z 하강 속도 최대 지점
    strike = {}
    for i, s in S.items():
        vz = np.diff(s[:, 2])
        strike[i] = int(np.argmin(vz))
    st = np.array([strike[i] for i in S])
    print(f"[데이터] {len(eps)} 에피소드  성공 {sum(bool(meta[i]['success']) for i in S)}")
    print(f"[타격 프레임] 중앙 {int(np.median(st))}  범위 {st.min()}~{st.max()}  "
          f"-> 절대 프레임 정렬이 왜 틀렸는지 보여주는 산포다")

    depth = np.array([meta[i].get("final_nail_depth", 0.0) for i in S])
    print(f"[타깃] final_nail_depth  평균 {depth.mean():.4f} std {depth.std():.4f} "
          f"(성공 임계 0.04)")

    print(f"\n{'정렬':>14} {'표본':>5} {'R²(state)':>10} {'R²(s+act)':>10} {'차이':>8} {'순열 p':>8}")
    for off in (0, 5, 10, 20, 30, 45):
        Xs, Xa, y = [], [], []
        for i in S:
            t = strike[i] - off                      # 타격 off 프레임 전에 커밋된 청크
            if t < 0 or t + LAT + R > len(A[i]):
                continue
            Xs.append(S[i][t])
            Xa.append(A[i][t + LAT:t + LAT + R].reshape(-1))
            y.append(meta[i].get("final_nail_depth", 0.0))
        if len(y) < 30:
            print(f"{f'타격-{off}':>14} {len(y):>5}  표본 부족")
            continue
        Xs, Xa, y = np.asarray(Xs, float), np.asarray(Xa, float), np.asarray(y)
        Xsa = np.hstack([Xs, Xa])
        r2s, r2sa = ridge_cv(Xs, y), ridge_cv(Xsa, y)
        # 순열: 액션 블록만 에피소드 간에 섞어 "액션이 더한 몫" 의 귀무분포를 만든다
        null = []
        for k in range(150):
            p = np.random.default_rng(700 + k).permutation(len(y))
            null.append(ridge_cv(np.hstack([Xs, Xa[p]]), y, seed=k) - ridge_cv(Xs, y, seed=k))
        pv = float((np.asarray(null) >= (r2sa - r2s)).mean())
        print(f"{f'타격-{off}':>14} {len(y):>5} {r2s:10.3f} {r2sa:10.3f} "
              f"{r2sa - r2s:+8.3f} {pv:8.3f}{'  ** 액션이 정보를 더한다' if pv < 0.05 else ''}")

    print("\n해석: '차이' 가 액션이 상태 너머로 더하는 예측력이다. 이것이 critic 이\n"
          "      액션 선택으로 얻을 수 있는 몫의 상한이고, 0 이면 어떤 아키텍처·데이터로도\n"
          "      test-time selection/guidance 는 이득이 없다.")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1
              else "/workspace/junmo_cho/dexjoco/rollout/fixed/s0_n100"))
