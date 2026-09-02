"""씬 초기조건별 성공률 — 어느 구간에서 실패하는가.

왜 재현이 가능한가
------------------
롤아웃은 에피소드마다 `np.random.seed(seed*1_000_003 + ep)` 를 걸고 env.reset() 을
부른다 (rollout_dexjoco.py:521-529). reset() 은 randomize=False 일 때 np.random 을
정확히 이 순서로만 소비한다 (panda_hammer_nail_env.py:396-441):

    delta_h    = uniform(0, 0.05)                       테이블 높이. 망치/못 z 를 같이 올린다
    hammer_xy  = uniform([-0.25,-0.35], [-0.40,-0.50])
    hammer_yaw = uniform(-10, 10)   [deg]
    nail_xy    = uniform([-0.10, 0.00], [0.00, 0.10])

따라서 시뮬레이터를 띄우지 않고 씬을 그대로 복원할 수 있다. --verify 로 실제 env 와
대조할 수 있다 (기본은 안 한다 — mujoco 가 필요하다).

**z 축에 대한 주의**: 망치와 못의 z 는 각자의 기준 높이 + delta_h 로, delta_h 하나가
정한다. 그래서 "yz 평면" 은 사실상 **y 대 테이블 높이**다. 축 이름을 그렇게 적는다.

성공 라벨은 데이터셋 파케이의 next.success 에서 온다 (한 번이라도 True 면 성공).

사용
----
  python sim/dexjoco/scene_heatmap.py --data rl-dataset/dexjoco/hammer_nail_d5r20
  python sim/dexjoco/scene_heatmap.py --data ... --bins 5 --out out/scene_C
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

H_LO, H_HI = np.array([-0.25, -0.35]), np.array([-0.40, -0.50])
N_LO, N_HI = np.array([-0.10, 0.00]), np.array([0.00, 0.10])


def scene_params(n_ep: int, seed: int = 0) -> dict[str, np.ndarray]:
    """에피소드 0..n_ep-1 의 씬 파라미터를 재현한다."""
    dh, hxy, hyaw, nxy = [], [], [], []
    for ep in range(n_ep):
        np.random.seed((seed * 1_000_003 + ep) % (2 ** 31 - 1))
        dh.append(np.random.uniform(0.0, 0.05))
        hxy.append(np.random.uniform(H_LO, H_HI))
        hyaw.append(np.random.uniform(-10, 10))
        nxy.append(np.random.uniform(N_LO, N_HI))
    hxy, nxy = np.array(hxy), np.array(nxy)
    return {"delta_h": np.array(dh), "hammer_x": hxy[:, 0], "hammer_y": hxy[:, 1],
            "hammer_yaw": np.array(hyaw), "nail_x": nxy[:, 0], "nail_y": nxy[:, 1]}


def load_success(root: Path) -> np.ndarray:
    import pandas as pd
    meta = json.loads((root / "meta" / "info.json").read_text())
    ch, n = meta.get("chunks_size", 1000), meta["total_episodes"]
    out = np.zeros(n, bool)
    for e in range(n):
        f = root / meta["data_path"].format(episode_chunk=e // ch, episode_index=e)
        out[e] = bool(pd.read_parquet(f, columns=["next.success"])["next.success"]
                      .to_numpy().any())
    return out


def grid(x, y, ok, bins, xr=None, yr=None):
    """(성공률, 표본수, x경계, y경계). 표본 0 인 칸은 nan."""
    xe = np.linspace(*(xr or (x.min(), x.max())), bins + 1)
    ye = np.linspace(*(yr or (y.min(), y.max())), bins + 1)
    tot, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    hit, _, _ = np.histogram2d(x[ok], y[ok], bins=[xe, ye])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(tot > 0, hit / tot, np.nan)
    return rate, tot, xe, ye


def smooth2d(x, y, ok, res=40, frac=0.12):
    """가우시안 커널 국소 성공률. 칸을 쪼개는 대신 **모든 점을 가중치로 쓴다** —
    표본을 잃지 않고 해상도를 올리는 유일한 방법이다. 반환은 (성공률, 유효표본, xg, yg).

    frac 은 커널 폭을 각 축 범위의 비율로 준다 (0.12 → 범위의 12%).
    유효표본 n_eff = (Σw)^2 / Σw^2 로, 그 지점 추정의 표준오차는 sqrt(p(1-p)/n_eff) 다.
    """
    xg = np.linspace(x.min(), x.max(), res)
    yg = np.linspace(y.min(), y.max(), res)
    hx, hy = frac * np.ptp(x), frac * np.ptp(y)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    w = np.exp(-0.5 * (((X[..., None] - x) / hx) ** 2 + ((Y[..., None] - y) / hy) ** 2))
    sw = w.sum(-1)
    rate = (w * ok).sum(-1) / np.maximum(sw, 1e-12)
    neff = sw ** 2 / np.maximum((w ** 2).sum(-1), 1e-12)
    return np.where(sw > 1e-9, rate, np.nan), neff, xg, yg


def profile1d(name, v, ok, bins=8):
    """1차원 주변 프로파일 + 95% 이항 신뢰구간. 칸당 표본이 커서 읽을 수 있다."""
    e = np.quantile(v, np.linspace(0, 1, bins + 1))
    print(f"\n  ── {name} 1차원 ({bins}분할, 95% CI)")
    print(f"    {'구간중앙':>10}{'n':>6}{'성공률':>9}{'95% CI':>16}")
    for i in range(bins):
        m = (v >= e[i]) & ((v <= e[i + 1]) if i == bins - 1 else (v < e[i + 1]))
        n_, p_ = int(m.sum()), float(ok[m].mean())
        se = 1.96 * np.sqrt(p_ * (1 - p_) / max(n_, 1))
        bar = "#" * int(round(40 * p_))
        print(f"    {(e[i] + e[i + 1]) / 2:>10.4f}{n_:>6}{100 * p_:>8.1f}%"
              f"  ±{100 * se:>4.1f}pp  {bar}")


def show(name, xn, yn, x, y, ok, bins):
    rate, tot, xe, ye = grid(x, y, ok, bins)
    print(f"\n  ── {name}   (행 {xn} ↓ , 열 {yn} →)")
    hdr = "".join(f"{(ye[j] + ye[j + 1]) / 2:>9.3f}" for j in range(bins))
    print(f"    {'':>9}{hdr}")
    for i in range(bins):
        cells = "".join(
            "        ·" if not np.isfinite(rate[i, j])
            else f"{100 * rate[i, j]:>7.0f}%" + ("!" if tot[i, j] < 10 else " ")
            for j in range(bins))
        print(f"    {(xe[i] + xe[i + 1]) / 2:>9.3f}{cells}")
    n = "".join(f"{int(tot[i].sum()):>9}" for i in range(bins))
    print(f"    {'행 표본':>9}{n}   (! = 표본 10 미만)")
    return rate, tot, xe, ye


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--prof-bins", type=int, default=8, help="1차원 프로파일 분할 수")
    p.add_argument("--smooth", type=float, default=0.12,
                   help="커널 폭 (각 축 범위의 비율). 0 이면 스무딩 그림을 안 그린다")
    p.add_argument("--out", type=Path, default=None, help="png 저장 경로 접두사")
    p.add_argument("--verify", type=int, default=0,
                   help="실제 env 를 띄워 앞 K 에피소드의 씬을 대조한다 (mujoco 필요)")
    a = p.parse_args()

    ok = load_success(a.data)
    n = len(ok)
    sc = scene_params(n, a.seed)
    print(f"에피소드 {n}   성공 {ok.sum()} = {100 * ok.mean():.1f}%   bins {a.bins}")

    if a.verify:
        import gymnasium  # noqa: F401
        import random as _r
        from dexjoco.sim.envs.panda_hammer_nail_env import PandaHammerNailEnv
        env = PandaHammerNailEnv(randomize=False)
        bad = 0
        for ep in range(a.verify):
            sd = (a.seed * 1_000_003 + ep) % (2 ** 31 - 1)
            _r.seed(sd); np.random.seed(sd)
            env.reset()
            got = np.array([env.hammer_ori_pose[0], env.hammer_ori_pose[1],
                            env.nail_ori_pose[0], env.nail_ori_pose[1],
                            float(env.delta_h)])
            exp = np.array([sc["hammer_x"][ep], sc["hammer_y"][ep],
                            sc["nail_x"][ep], sc["nail_y"][ep], sc["delta_h"][ep]])
            if not np.allclose(got, exp, atol=1e-9):
                bad += 1
                print(f"  [불일치] ep{ep}\n    env {got}\n    재현 {exp}")
        print(f"  [검증] {a.verify} 에피소드 중 불일치 {bad}개"
              f"{'  → 재현 정확' if bad == 0 else '  ** 재현 실패'}")
        if bad:
            raise SystemExit(1)

    panels = [
        ("hammer xy", "hammer_x", "hammer_y"),
        ("nail xy", "nail_x", "nail_y"),
        ("hammer y vs table height (yz)", "hammer_y", "delta_h"),
        ("nail y vs table height (yz)", "nail_y", "delta_h"),
        ("hammer x vs nail x", "hammer_x", "nail_x"),
        ("hammer y vs nail y", "hammer_y", "nail_y"),
        ("hammer yaw vs table height", "hammer_yaw", "delta_h"),
    ]
    res = []
    for nm, xn, yn in panels:
        res.append((nm, xn, yn) + show(nm, xn, yn, sc[xn], sc[yn], ok, a.bins))

    # 1차원 요약 — 어느 축이 실제로 성공률을 가르나
    print("\n  ── 축별 1차원 (사분위)")
    print(f"    {'축':<12}{'Q1':>9}{'Q2':>9}{'Q3':>9}{'Q4':>9}{'최대-최소':>10}")
    for k, v in sc.items():
        q = np.quantile(v, [0, .25, .5, .75, 1.0])
        rs = []
        for i in range(4):
            m = (v >= q[i]) & (v <= q[i + 1] if i == 3 else v < q[i + 1])
            rs.append(100 * ok[m].mean() if m.any() else np.nan)
        print(f"    {k:<12}" + "".join(f"{r:>8.1f}%" for r in rs)
              + f"{max(rs) - min(rs):>9.1f}")

    print("\n" + "=" * 70)
    for k in ("nail_x", "nail_y", "hammer_y", "delta_h", "hammer_yaw", "hammer_x"):
        profile1d(k, sc[k], ok, a.prof_bins)

    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(21, 9))
        for ax, (nm, xn, yn, rate, tot, xe, ye) in zip(axes.ravel(), res):
            im = ax.imshow(rate.T * 100, origin="lower", aspect="auto", vmin=0, vmax=100,
                           cmap="RdYlGn", extent=[xe[0], xe[-1], ye[0], ye[-1]])
            for i in range(a.bins):
                for j in range(a.bins):
                    if tot[i, j] > 0:
                        ax.text((xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2,
                                f"{100 * rate[i, j]:.0f}\nn={int(tot[i, j])}",
                                ha="center", va="center", fontsize=7)
            ax.set_title(nm, fontsize=9); ax.set_xlabel(xn); ax.set_ylabel(yn)
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes.ravel()[-1].axis("off")
        fig.suptitle(f"{a.data.name}   success rate % "
                     f"(overall {100 * ok.mean():.1f}%, {n} episodes)", fontsize=13)
        fig.tight_layout()
        a.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{a.out}.png", dpi=110)
        print(f"\n  저장: {a.out}.png")

        if a.smooth > 0:
            pairs = [("nail x", "nail y", "nail_x", "nail_y"),
                     ("hammer x", "hammer y", "hammer_x", "hammer_y"),
                     ("nail x", "table height", "nail_x", "delta_h"),
                     ("nail y", "table height", "nail_y", "delta_h")]
            fig2, ax2 = plt.subplots(2, 4, figsize=(20, 8))
            for c, (xl, yl, xn, yn) in enumerate(pairs):
                r, ne, xg, yg = smooth2d(sc[xn], sc[yn], ok.astype(float),
                                         frac=a.smooth)
                im = ax2[0, c].imshow(r.T * 100, origin="lower", aspect="auto",
                                      vmin=25, vmax=75, cmap="RdYlGn",
                                      extent=[xg[0], xg[-1], yg[0], yg[-1]])
                ax2[0, c].set_title(f"{xl} vs {yl}  (smoothed)", fontsize=9)
                ax2[0, c].set_xlabel(xl); ax2[0, c].set_ylabel(yl)
                fig2.colorbar(im, ax=ax2[0, c], fraction=0.046)
                # 유효표본 대비 몇 시그마 벗어났나 — 노이즈와 구조를 가른다
                se = np.sqrt(ok.mean() * (1 - ok.mean()) / np.maximum(ne, 1e-9))
                z = (r - ok.mean()) / np.maximum(se, 1e-12)
                im2 = ax2[1, c].imshow(z.T, origin="lower", aspect="auto",
                                       vmin=-4, vmax=4, cmap="coolwarm",
                                       extent=[xg[0], xg[-1], yg[0], yg[-1]])
                ax2[1, c].set_title(f"z vs overall {100 * ok.mean():.1f}%", fontsize=9)
                ax2[1, c].set_xlabel(xl); ax2[1, c].set_ylabel(yl)
                fig2.colorbar(im2, ax=ax2[1, c], fraction=0.046)
            fig2.suptitle(f"{a.data.name}  kernel-smoothed success rate "
                          f"(frac={a.smooth}, {n} episodes)", fontsize=13)
            fig2.tight_layout()
            fig2.savefig(f"{a.out}_smooth.png", dpi=110)
            print(f"  저장: {a.out}_smooth.png")


if __name__ == "__main__":
    main()
