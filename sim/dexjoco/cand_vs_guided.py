"""guidance 로 옮겨진 액션이 BC 후보 구름에서 얼마나 벗어났나 — 직접 측정.

배경 (왜 이게 필요한가)
----------------------
"parl 이 sel32 보다 좋은 이유는 sel32 가 절대 못 뽑는 액션으로 갔기 때문" 이라는
주장을 하려면, **같은 상태에서** BC 후보 구름과 상승된 액션을 나란히 놓고 재야 한다.
그전까지는 "이동거리 / 후보산포" 를 따로 잰 두 스칼라의 비로 갈음했는데, 그 후보산포
값(0.0047)은 다른 세팅(A)에서 잰 것을 yaml 주석으로 복사해 둔 것이었다.

여기서는 서버가 `--dump-obs` 로 남긴 npz — 결정 프레임마다 후보 32개(acts),
상승 시작점(parent), 실제로 내보낸 액션(chosen) — 를 읽어 프레임별로 잰다.

무엇을 재나
----------
guidance 가 실제로 미는 차원(cog_mask)만 남기고:

  δ  = chosen - parent          guidance 변위. u = δ/|δ| 가 "옮겨간 방향" 이다.
  σ_u = std_i (a_i - μ)·u       그 방향으로 후보 구름이 가진 산포
  z_u = (chosen - μ)·u / σ_u    **핵심 숫자.** 그 방향으로 몇 σ 나가 있나.

z_u 가 답을 준다. 후보를 아무리 많이 뽑아도 그 방향의 분포는 σ_u 를 못 넘으므로,
N 개를 뽑아 chosen 만큼 나간 것을 볼 확률은 대략 N·Φ(-z_u) 다. z_u 가 3 이면
N=1024 로 닿고, 16 이면 어떤 N 으로도 못 닿는다.

같이 재는 것:
  · 후보 구름의 차원당 산포 (그 세팅에서 실제로 측정된 값 — yaml 주석 대체)
  · chosen 에서 가장 가까운 후보까지 거리 / 후보끼리의 최근접 거리
  · 대각 마할라노비스 (후보는 정의상 ~1)
  · q_pre 최고 vs q_post — 상승이 실제로 Q 를 얼마나 올렸나

성공/실패와 에피소드 구간
------------------------
덤프에 ep(에피소드)·dec(그 안에서 몇 번째 결정인가) 가 들어 있으면, 롤아웃 출력
디렉토리의 next.success 를 붙여 **성공한 에피소드만** 볼 수 있고 결정 인덱스별로도
볼 수 있다. 결정은 replan(20) 스텝마다 한 번이므로 dec=k 는 env_step ≈ 5 + 20k 다.

물어보는 것: 실패한 에피소드는 더 멀리 갔나. 그리고 개입 여지가 에피소드 구간마다
다른가 (fuji 에서 "피더를 잡을 때는 BC 로 충분하고 옮길 때만 guidance 가 필요하다"
는 관찰의 sim 판 검증이다 — 맞으면 상태별 온도 tau(s) 의 근거가 된다).

사용
----
  python sim/dexjoco/cand_vs_guided.py <dump.npz> [<dump2.npz> ...]

덤프 옆의 롤아웃 디렉토리(<dump>_ 를 뗀 이름)를 자동으로 찾아 성공 라벨을 붙인다.

gm=0 (sel32) 덤프를 대조군으로 같이 주면 좋다 — 그쪽은 chosen 이 후보 중 하나이므로
z_u 가 32개 중 최댓값 수준(≈2)에서 나와야 한다. 안 나오면 측정이 틀린 것이다.
"""
import sys
from pathlib import Path

import numpy as np


def phi_sf(z):
    """표준정규 생존함수. scipy 없이 erfc 로."""
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2.0))


def load_success(npz_path: Path):
    """덤프 옆 롤아웃 디렉토리에서 에피소드별 성공 여부를 읽는다. (없으면 None)

    덤프는 <eval>/<NAME>__dump.npz 이고 롤아웃은 <eval>/<NAME>/ 이다.
    LeRobot 파케이의 next.success 가 한 번이라도 True 면 성공 (make_subset.py 와 같은 규약).
    """
    root = npz_path.parent / npz_path.name.replace("__dump.npz", "")
    info = root / "meta" / "info.json"
    if not info.exists():
        return None
    import json

    import pandas as pd
    meta = json.loads(info.read_text())
    ch = meta.get("chunks_size", 1000)
    out = {}
    for e in range(meta.get("total_episodes", 0)):
        f = root / meta["data_path"].format(episode_chunk=e // ch, episode_index=e)
        if not f.exists():
            continue
        out[e] = bool(pd.read_parquet(f, columns=["next.success"])["next.success"]
                      .to_numpy().any())
    return out or None


def analyze(path: Path):
    d = np.load(path, allow_pickle=False)
    acts = d["acts"].astype(np.float64)          # (F, n, D)
    chosen = d["chosen"].astype(np.float64)      # (F, D)
    parent = d["parent"].astype(np.float64) if "parent" in d else None
    mask = d["cog_mask"].astype(bool) if "cog_mask" in d else np.ones(acts.shape[-1], bool)
    gm = float(d["guide_move"]) if "guide_move" in d else float("nan")
    gs = int(d["guide_steps"]) if "guide_steps" in d else -1
    pk = int(d["parl_keep"]) if "parl_keep" in d else -1
    q_pre = d["q_pre"].astype(np.float64) if "q_pre" in d else None
    q_post = d["q_post"].astype(np.float64) if "q_post" in d else None
    ep = d["ep"].astype(int) if "ep" in d else None
    dec = d["dec"].astype(int) if "dec" in d else None

    F, n, D = acts.shape
    A = acts[:, :, mask]                          # (F, n, m)
    C = chosen[:, mask]                           # (F, m)
    P = parent[:, mask] if parent is not None else None
    m_dim = A.shape[-1]
    m = m_dim

    print(f"\n{'=' * 78}\n{path}")
    print(f"  프레임 {F} · 후보 {n} · 전체 {D}차원 중 guidance 차원 {m}개 "
          f"| guide_move={gm:g} steps={gs} parl_keep={pk}")

    mu = A.mean(1)                                # (F, m)
    dev = A - mu[:, None, :]                      # (F, n, m)

    # --- 후보 구름 자체의 크기 (그 세팅에서 실측) --------------------------------
    per_dim = A.std(1, ddof=1)                    # (F, m)
    spread = float(np.sqrt((per_dim ** 2).mean()))
    print(f"\n  [후보 구름]  차원당 산포(RMS std)      {spread:.5f}")
    print(f"               구름 반지름 |a-mu| 중앙값   "
          f"{np.median(np.linalg.norm(dev, axis=-1)):.4f}   (= 산포x√{m} ≈ "
          f"{spread * np.sqrt(m):.4f})")

    if P is None:
        print("  [!] parent 가 없는 구판 덤프 — 변위를 최근접 후보 기준으로 잡는다")
        P = A[np.arange(F), np.linalg.norm(A - C[:, None, :], axis=-1).argmin(1)]

    delta = C - P
    dn = np.linalg.norm(delta, axis=-1)           # (F,)
    moved = dn > 1e-9
    print(f"\n  [변위]       상승이 일어난 프레임        {moved.sum()}/{F}")
    if not moved.any():
        print("               (전부 0 — guidance 없음. 대조군이면 정상)")
    print(f"               |δ| 중앙값                  {np.median(dn):.4f}"
          f"   차원당 {np.median(dn) / np.sqrt(m):.5f}")

    # --- 핵심: 이동 방향으로 몇 σ 나갔나 ----------------------------------------
    # 방향이 정의되는 프레임만. 대조군(gm=0)은 chosen==parent 라 δ=0 이므로
    # 대신 "구름 중심 -> chosen" 방향을 쓴다 (선택이 끌어낸 방향).
    dir_src = np.where(moved[:, None], delta, C - mu)
    dl = np.linalg.norm(dir_src, axis=-1)
    ok = dl > 1e-12
    u = np.zeros_like(dir_src)
    u[ok] = dir_src[ok] / dl[ok][:, None]

    proj_cand = np.einsum("fnm,fm->fn", dev, u)   # (F, n) 후보의 u 방향 좌표
    sig_u = proj_cand.std(1, ddof=1)              # (F,)
    proj_c = np.einsum("fm,fm->f", C - mu, u)     # chosen 의 u 방향 좌표
    proj_p = np.einsum("fm,fm->f", P - mu, u)     # parent 의 u 방향 좌표

    good = ok & (sig_u > 1e-12)
    z_c = proj_c[good] / sig_u[good]
    z_p = proj_p[good] / sig_u[good]

    print(f"\n  [★ 이동 방향 u 로 본 위치]  (후보 구름 산포 = 1σ 기준, {good.sum()} 프레임)")
    print(f"               부모 후보 z          중앙값 {np.median(z_p):+7.2f}"
          f"   (32개 중 argmax 라 1~2 가 정상)")
    print(f"               chosen   z          중앙값 {np.median(z_c):+7.2f}"
          f"   p10 {np.percentile(z_c, 10):+.2f}  p90 {np.percentile(z_c, 90):+.2f}")
    zz = float(np.median(z_c))
    p = phi_sf(zz)
    if p <= 0:
        need = float("inf")
    else:
        need = 1.0 / p
    print(f"\n               → 그 방향으로 chosen 만큼 나간 후보를 한 번 보려면 "
          f"N ≈ {need:.3g}")
    print(f"                 (n={n} 로 실제 나오는 최댓값 z ≈ "
          f"{np.median(proj_cand.max(1)[good] / sig_u[good]):+.2f})")

    # --- 최근접 후보까지의 거리 -------------------------------------------------
    dist_c = np.linalg.norm(A - C[:, None, :], axis=-1)        # (F, n)
    nn_c = dist_c.min(1)
    pw = np.linalg.norm(A[:, :, None, :] - A[:, None, :, :], axis=-1)
    iu = np.arange(n)
    pw[:, iu, iu] = np.inf
    nn_a = pw.min(2).mean(1)
    print(f"\n  [최근접]     chosen→가장 가까운 후보    {np.median(nn_c):.4f}")
    print(f"               후보→후보 최근접 평균       {np.median(nn_a):.4f}"
          f"   비 {np.median(nn_c / np.maximum(nn_a, 1e-12)):.1f}배")

    # --- 대각 마할라노비스 -------------------------------------------------------
    sd = np.maximum(per_dim, 1e-9)
    mah_c = np.sqrt((((C - mu) / sd) ** 2).mean(-1))
    mah_a = np.sqrt((((A - mu[:, None, :]) / sd[:, None, :]) ** 2).mean(-1)).mean(1)
    print(f"\n  [마할라노비스] chosen {np.median(mah_c):7.2f}   후보 평균 "
          f"{np.median(mah_a):.2f} (정의상 ≈1)")

    if ep is not None and dec is not None:
        succ = load_success(path)
        z_all = np.full(F, np.nan)
        z_all[good] = z_c
        if succ is None:
            print("\n  [!] 롤아웃 디렉토리를 못 찾아 성공 라벨을 못 붙였다")
        else:
            lab = np.array([succ.get(int(e), None) for e in ep], dtype=object)
            for nm, m in (("성공", lab == True), ("실패", lab == False)):   # noqa: E712
                m = m & np.isfinite(z_all)
                if not m.any():
                    continue
                nep = len(set(ep[m].tolist()))
                print(f"\n  [{nm} 에피소드]  {nep}개 · 결정 {int(m.sum())}회"
                      f"   z 중앙값 {np.median(z_all[m]):+.2f}"
                      f"  |δ|/dim {np.median(dn[m]) / np.sqrt(m_dim):.5f}")
        # 결정 인덱스별. dec=k 는 env_step ≈ latency + replan*k 다.
        print(f"\n  [결정 스텝별]  (dec=k → env_step ≈ 5 + 20k)")
        print(f"     {'dec':>4} {'env_step':>9} {'n':>5} {'z 중앙값':>10} "
              f"{'후보산포':>9} {'Q상승':>9}")
        edges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 99)]
        for lo, hi in edges:
            m = (dec >= lo) & (dec < hi) & np.isfinite(z_all)
            if not m.any():
                continue
            rng = f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"
            es = f"{5 + 20 * lo}~{5 + 20 * (hi - 1)}"
            gq = (np.median(q_post[m] - q_pre[m].max(1))
                  if q_pre is not None else float("nan"))
            print(f"     {rng:>4} {es:>9} {int(m.sum()):>5} {np.median(z_all[m]):>10.2f} "
                  f"{np.median(per_dim[m].mean(-1)):>9.5f} {gq:>+9.4f}")

    if q_pre is not None and q_post is not None:
        print(f"\n  [Q]          선택만 했을 때 max Q      {np.median(q_pre.max(1)):.4f}")
        print(f"               상승 후 Q                  {np.median(q_post):.4f}"
              f"   상승분 {np.median(q_post - q_pre.max(1)):+.4f}")

    return dict(path=str(path), gm=gm, spread=spread, z=zz, need=need,
                nn_ratio=float(np.median(nn_c / np.maximum(nn_a, 1e-12))))


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    rows = [analyze(p) for p in paths]
    if len(rows) > 1:
        print(f"\n{'=' * 78}\n요약")
        print(f"  {'덤프':<34} {'gm':>6} {'후보산포':>9} {'z_u':>7} {'필요N':>10} {'최근접비':>8}")
        for r in rows:
            print(f"  {Path(r['path']).stem:<34} {r['gm']:>6g} {r['spread']:>9.5f} "
                  f"{r['z']:>7.2f} {r['need']:>10.3g} {r['nn_ratio']:>8.1f}")


if __name__ == "__main__":
    main()
