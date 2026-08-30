#!/usr/bin/env python3
"""서버가 덤프한 **후보 청크들이 서로 얼마나 다른지** 잰다.

왜: test-time selection 은 후보들이 서로 달라야 의미가 있다. 지금까지 우리는 후보 산포를
코드 주석의 openarm 실측값(차원당 0.018)으로 가정했을 뿐 dexjoco 에서 잰 적이 없다.
후보들이 사실상 같은 액션이면 Q 가 평평한 것이 **정답**이고, 후보를 128개로 늘려도
전부 복제본이라 소용이 없다.

판정 기준은 오픈루프 리플레이 실측이다 (sim/dexjoco/replay_actions.py):
    노이즈 0.005 -> 성공 40%,  0.01 -> 10%
즉 액션이 0.005 만 흔들려도 성공 궤적이 깨진다. 따라서

    후보 산포 << 0.005  ->  후보들이 전부 같은 궤적. 고를 것이 없다 (샘플 수를 늘려도 무의미)
    후보 산포 >~ 0.005  ->  후보들 사이에 성공/실패가 실제로 갈린다. critic 이 못 고르는 것

  python sim/dexjoco/cand_spread.py <dump.npz> [<dump2.npz> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

A_DIM, LAT, R = 25, 5, 20


def report(path: Path) -> None:
    d = np.load(path)
    acts = d["acts"]                        # (frames, n_cand, window*A_DIM) 정규화 액션
    F, N, FULL = acts.shape
    W = FULL // A_DIM
    a = acts.reshape(F, N, W, A_DIM)
    exe = a[:, :, LAT:LAT + R, :]           # 실행 구간만 (prefix 는 RTC 가 고정한다)
    pre = a[:, :, :LAT, :]

    print(f"\n=== {path} ===")
    print(f"프레임 {F}  후보 {N}  창 {W}스텝 x {A_DIM}관절")
    # 후보 축의 표준편차 = "같은 상태에서 정책이 내놓는 액션의 산포"
    print(f"  prefix [0,{LAT})   차원당 std 중앙 {np.median(exe.std(1)) * 0 + np.median(pre.std(1)):.5f}"
          f"   <- RTC 가 고정하면 0 에 가깝다")
    s_exe = exe.std(1)                      # (F, R, A_DIM)
    print(f"  실행   [{LAT},{LAT + R}) 차원당 std 중앙 {np.median(s_exe):.5f}  "
          f"평균 {s_exe.mean():.5f}  90분위 {np.quantile(s_exe, .9):.5f}")
    for nm, lo, hi in (("eef_position", 0, 3), ("eef_rotation", 3, 9), ("hand_joints", 9, 25)):
        print(f"     {nm:14} std 중앙 {np.median(s_exe[:, :, lo:hi]):.5f}")
    # 후보 쌍 간 최대 거리 (차원당)
    dmax = []
    for f in range(F):
        x = exe[f].reshape(N, -1)
        dd = np.abs(x[:, None, :] - x[None, :, :]).mean(-1)
        dmax.append(dd.max())
    print(f"  후보 쌍 최대 |Δ| (차원당) 중앙 {np.median(dmax):.5f}  최대 {np.max(dmax):.5f}")
    # 물리량: 명령된 최저 z (성공/실패를 -0.61σ 로 가른 유일한 특징)
    zmin = a[:, :, LAT:LAT + R, 2].min(-1)  # (F, N)
    print(f"  명령 최저 z: 후보 간 std 중앙 {np.median(zmin.std(1)):.5f}  "
          f"(에피소드 간 성공/실패 차이는 0.61σ 였다)")
    m = np.median(s_exe)
    print(f"  → 판정: 실행 구간 산포 {m:.5f} vs 리플레이 붕괴 임계 0.005~0.01  "
          + ("**후보들이 전부 허용 오차 안 — 고를 것이 없다**" if m < 0.005 else
             "후보들이 허용 오차 밖으로 갈린다 — 고를 거리가 있다"))


if __name__ == "__main__":
    for p in sys.argv[1:] or ["/workspace/junmo_cho/dexjoco/dump/cand.npz"]:
        pp = Path(p)
        if pp.is_file():
            report(pp)
        else:
            print(f"[없음] {pp}")
