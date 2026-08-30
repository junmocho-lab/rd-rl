#!/usr/bin/env python3
"""데이터에 '같은 상태, 다른 액션' 의 반사실 신호가 있는가 — 쌍둥이 에피소드로 검정.

왜 이 질문인가: Q(s,a) 의 액션 항은 데이터가 **같은 s 에서 다른 a** 를 보여줘야 학습된다.
우리 롤아웃은 전부 같은 BC 정책 하나에서 나왔으므로 그 조건이 성립하는지 불확실하다.
probe_actsens 는 "학습된 critic 이 액션에 둔하다" 를 보였지만, 그것이 (1) 데이터에
신호가 없어서인지 (2) critic 이 못 뽑아낸 것인지는 가르지 못한다. 이 스크립트가 그것을 가른다.

자연 실험: 1000 에피소드 수집이 ep500 에서 재시작되어 env 전역 RNG 가 되감겼고,
그래서 ep k 와 ep k+500 은 **초기 조건이 동일**하다 (첫 프레임 픽셀차 0.6~0.8 = 코덱 노이즈,
다른 장면은 3~7). 정책 샘플만 다르다. 그중 148쌍은 결과까지 갈렸다.

검정: 초기 프레임(상태가 아직 거의 같은 구간)에서 **액션 청크의 차이만으로** 어느 쪽이
이길지 예측되는가. 쌍 단위 교차검증 + 순열검정으로 우연을 배제한다.

  예측됨   → 신호는 데이터에 있다. critic 이 못 뽑아낸 것 (아키텍처/목적함수 문제)
  안 됨    → 데이터에 반사실이 없다 (데모 혼합 / 노이즈 롤아웃이 필요)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/rlwrld2/home/junmo_cho/ws/rd-rl")
DS = ROOT / "rl-dataset/dexjoco/hammer_nail_d5r20"
ACTNORM = ROOT / "checkpoints/dexjoco_hammer_nail-critic/actnorm.npy"
LAT, R, A_DIM = 5, 20, 25          # exp yaml
# 구간을 훑는다. 초기만 보면 안 된다 — hammer_nail 의 승패는 망치를 내려치는 시점
# (중후반)에 갈리므로, 초기 구간만 재면 "결정적 액션이 있는 창" 을 통째로 놓친다.
# 대신 후반은 상태도 갈라져 있으므로 구간마다 |Δstate| 를 같이 찍어 해석을 가능하게 한다.
WINDOWS = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 360), (0, 360)]
STRIDE = 5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_ridge_logistic(X, y, lam, iters=300):
    """절편 없는 ridge 로지스틱. 쌍 차분 특징이라 절편이 있으면 안 된다 (부호 대칭)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) / len(y) + lam * w
        # 뉴턴 대신 고정 스텝 — 특징을 표준화해 두었으므로 안정적이다
        w -= 1.0 * g
    return w


def auc(score, y):
    pos, neg = score[y > 0.5], score[y < 0.5]
    if not len(pos) or not len(neg):
        return 0.5
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    eps = {e["episode_index"]: e for e in
           (json.loads(l) for l in (DS / "meta/episodes.jsonl").read_text().splitlines() if l.strip())}
    order = sorted(eps)
    off, c = {}, 0
    for e in order:                       # flat 프레임 순서 = 에피소드 인덱스 순 concat
        off[e], c = c, c + eps[e]["length"]
    act = np.load(ACTNORM, mmap_mode="r")
    assert act.shape[0] == c, f"actnorm {act.shape[0]} != 총 프레임 {c}"
    print(f"[데이터] {len(order)} 에피소드 / {c} 프레임 / actnorm {act.shape}")

    pairs = [(k, k + 500) for k in range(500) if k in eps and k + 500 in eps]
    disc = [(i, j) for i, j in pairs if bool(eps[i]["success"]) != bool(eps[j]["success"])]
    print(f"[쌍] 같은 장면 {len(pairs)}쌍 중 결과가 갈린 쌍 {len(disc)}")

    def state_of(e):
        df = pd.read_parquet(DS / f"data/chunk-000/episode_{e:06d}.parquet",
                             columns=["observation.state"])
        return np.vstack([np.asarray(v, dtype=np.float32) for v in df["observation.state"]])

    ST = {e: state_of(e) for ij in disc for e in ij}
    idx = np.array([t * A_DIM + jj for t in range(LAT, LAT + R) for jj in range(A_DIM)])

    def cv_auc(Xa, Ya, Ga, lam, folds=5, seed=0):
        r = np.random.default_rng(seed)
        pid = np.unique(Ga)
        fold = {p: k for p, k in zip(r.permutation(pid), np.arange(len(pid)) % folds)}
        fa = np.array([fold[g] for g in Ga])
        sc = np.zeros(len(Ya))
        for k in range(folds):
            tr, te = fa != k, fa == k
            if not te.any() or not tr.any():
                continue
            sc[te] = Xa[te] @ fit_ridge_logistic(Xa[tr], Ya[tr], lam)
        return auc(sc, Ya)

    print(f"\n{'구간':>12} {'표본':>6} {'|Δstate|':>9} {'|Δaction|':>10} "
          f"{'최고 CV AUC':>11} {'λ':>7} {'순열 p':>8}")
    for lo, hi in WINDOWS:
        X, Y, G, dsm = [], [], [], []
        for pi, (i, j) in enumerate(disc):
            T = min(eps[i]["length"], eps[j]["length"], hi)
            wins_i = bool(eps[i]["success"])
            si, sj = ST[i], ST[j]
            for t in range(lo, T, STRIDE):
                d = (np.asarray(act[off[i] + t]).reshape(-1)[idx]
                     - np.asarray(act[off[j] + t]).reshape(-1)[idx])
                X.append(d); Y.append(1.0 if wins_i else 0.0); G.append(pi)
                X.append(-d); Y.append(0.0 if wins_i else 1.0); G.append(pi)
                dsm.append(np.abs(si[t] - sj[t]).mean())
        if len(X) < 200:
            print(f"{f'{lo}-{hi}':>12} {len(X):>6}  표본 부족")
            continue
        X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y); G = np.asarray(G)
        raw = np.abs(X).mean()
        sd = X.std(0); sd[sd < 1e-8] = 1.0; X = X / sd
        bA, bL = 0.0, None
        for lam in (0.3, 0.1, 0.03, 0.01, 0.003):
            v = cv_auc(X, Y, G, lam)
            if v > bA:
                bA, bL = v, lam
        null = []
        for sd_ in range(120):
            r = np.random.default_rng(1000 + sd_)
            flip = {p: r.integers(2) for p in np.unique(G)}
            Yp = np.where([flip[g] for g in G], 1 - Y, Y)
            null.append(cv_auc(X, Yp, G, bL, seed=sd_))
        pv = float((np.asarray(null) >= bA).mean())
        print(f"{f'{lo}-{hi}':>12} {len(X):>6} {np.mean(dsm):9.5f} {raw:10.5f} "
              f"{bA:11.3f} {bL:7} {pv:8.3f}{'  **' if pv < 0.05 else ''}")

    print("\n판정:\n"
          "  AUC 가 유의하게 0.5 를 넘으면 -> 데이터에 반사실 신호가 있다.\n"
          "     critic 이 그것을 못 뽑아낸 것이므로 아키텍처/목적함수를 고친다.\n"
          "  넘지 못하면 -> 같은 상태에서 다른 액션의 결과가 데이터에 없다.\n"
          "     데모 혼합(Q-VGM 레시피) 또는 노이즈 롤아웃으로 데이터를 늘려야 한다.")


if __name__ == "__main__":
    main()
