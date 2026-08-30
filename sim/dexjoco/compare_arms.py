#!/usr/bin/env python3
"""guidance 팔들의 성공률을 **같은 장면 기준**으로 비교한다.

    python sim/dexjoco/compare_arms.py bc sel g020

rollout_dexjoco.py --seed-per-episode 덕분에 팔마다 episode_index i 는 같은 장면
(테이블 높이·망치 xy/yaw·못 xy)이다. 따라서 두 비율 검정이 아니라 McNemar 로 읽는다 —
장면 난이도라는 가장 큰 분산원이 상쇄되므로 같은 표본에서 훨씬 작은 차이를 본다.

McNemar: 기준 팔이 실패하고 비교 팔이 성공한 장면 b, 그 반대 c 만 쓴다.
정확 이항검정 (b | b+c ~ Binom(0.5)) 이라 표본이 작아도 유효하다.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/junmo_cho/dexjoco/rollout/guidance")


def load(arm: str) -> dict[int, dict]:
    f = ROOT / arm / "meta/episodes.jsonl"
    if not f.is_file():
        raise SystemExit(f"{arm}: {f} 가 없다")
    return {e["episode_index"]: e for e in (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def exact_mcnemar(b: int, c: int) -> float:
    """양측 정확 검정. b, c 는 불일치 쌍의 개수."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main(arms: list[str]) -> None:
    data = {a: load(a) for a in arms}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"공통 에피소드(=같은 장면) {len(common)}개\n")

    print(f"{'arm':>6}  {'성공률':>16}  {'평균 depth':>10}  {'근접실패':>8}  {'평균길이':>8}")
    for a in arms:
        ok = np.array([bool(data[a][i]["success"]) for i in common])
        d = np.array([data[a][i].get("final_nail_depth", np.nan) for i in common])
        L = np.array([data[a][i]["length"] for i in common])
        p = ok.mean(); se = (p * (1 - p) / len(ok)) ** 0.5
        near = int(((~ok) & (d >= 0.02)).sum())
        print(f"{a:>6}  {ok.sum():3d}/{len(ok):3d} = {100*p:4.1f}% ±{100*1.96*se:3.1f}"
              f"  {d.mean():10.4f}  {near:8d}  {L.mean():8.0f}")

    base = arms[0]
    print(f"\n=== McNemar (기준 {base}, 같은 장면 짝지어 비교) ===")
    ok0 = np.array([bool(data[base][i]["success"]) for i in common])
    for a in arms[1:]:
        ok1 = np.array([bool(data[a][i]["success"]) for i in common])
        b = int((~ok0 & ok1).sum())      # base 실패 -> a 성공
        c = int((ok0 & ~ok1).sum())      # base 성공 -> a 실패
        p = exact_mcnemar(b, c)
        delta = 100 * (ok1.mean() - ok0.mean())
        print(f"  {base} -> {a}:  살린 장면 {b}, 죽인 장면 {c}, 불일치 {b+c}"
              f"  |  Δ성공률 {delta:+.1f}pp  p={p:.4f}"
              f"  {'**유의' if p < 0.05 else '(유의하지 않음)'}")

    print(f"\n=== 못 깊이 분포 (성공 임계 0.04) ===")
    edges = [0, 0.0005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08]
    print(f"{'구간':>16}" + "".join(f"{a:>8}" for a in arms))
    for lo, hi in zip(edges, edges[1:]):
        row = f"{lo:.4f}~{hi:.4f}".rjust(16)
        for a in arms:
            d = np.array([data[a][i].get("final_nail_depth", np.nan) for i in common])
            row += f"{int(((d > lo - 1e-9) & (d <= hi)).sum()):>8}"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1:] or ["bc", "sel", "g020"])
