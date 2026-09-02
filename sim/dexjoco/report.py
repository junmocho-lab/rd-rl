#!/usr/bin/env python3
"""모든 eval 결과를 하나의 표로 정리한다 (RESULTS.md).

씬이 고정(seed 0)이라 eval 의 k번째 에피소드와 수집 데이터의 k번째가 같은 장면이다.
그래서 성공/실패를 짝지어 McNemar 정확검정을 쓸 수 있다 — 둘 다 성공하거나 둘 다
실패한 장면은 차이에 대한 정보가 없으므로 버리고, 엇갈린 쌍만 센다.

  python3 sim/dexjoco/report.py           # 화면
  python3 sim/dexjoco/report.py --out RESULTS.md
"""
import argparse, glob, json, os, re
from math import comb

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS = [
    ("A", "d2r8_s0",  "(8,2) 고정씬",  "hammer_nail_d2r8_s0",  200, 10 * 9),
    ("B", "d5r20_s0", "(20,5) 고정씬", "hammer_nail_d5r20_s0", 200, 25 * 9),
    ("C", "d5r20",    "(20,5) 랜덤씬", "hammer_nail_d5r20",   1000, 25 * 9),
]


def mcnemar(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path):
    m = [json.loads(x) for x in open(path) if x.strip()]
    return (np.array([bool(e.get("success")) for e in m]),
            np.array([e["length"] for e in m]))


def parse(name):
    """디렉토리 이름 → (방법, guide_move, critic 필터, critic 스텝). 못 읽으면 None."""
    m = re.match(r"(sel32|parl|bc)(_gm[0-9]+)?__(success|all|success_ens|all_ens)@(\d+)k$", name)
    if m:
        meth, gm, ct, st = m.groups()
        gmv = {"_gm0": 0.0, "_gm0001": 0.001, "_gm0005": 0.005, "_gm001": 0.01,
               "_gm002": 0.02, "_gm01": 0.1, "_gm02": 0.2}.get(gm or "",
               0.05 if meth == "parl" else 0.0)
        return meth, gmv, ct, int(st) * 1000
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="")
    a = p.parse_args()
    L = []
    P = L.append
    P("# DexJoCo hammer_nail — 전체 결과\n")
    P("성공률은 `성공/전체`. **성공길이**는 성공한 에피소드만의 평균 프레임 수 "
      "(전체 평균은 실패가 대부분 360 타임아웃이라 성공률의 함수가 되므로 쓰지 않는다).\n")
    P("`승/패`와 `p`는 **같은 장면끼리 짝지은** McNemar 정확검정이다 — BC 가 실패한 장면에서 "
      "성공한 수 / BC 가 성공한 장면에서 실패한 수. 둘 다 성공·둘 다 실패한 장면은 버린다.\n")
    P("| 세팅 | (replan, latency) | critic 액션 | 수집 | BC |")
    P("|---|---|---|---|---|")
    base = {}
    for tag, exp, desc, ds, nep, adim in SETTINGS:
        ok, ln = load(f"{REPO}/rl-dataset/dexjoco/{ds}/meta/episodes.jsonl")
        base[tag] = (ok, ln)
        P(f"| **{tag}** | {desc} | {adim}차원 | {nep}ep | "
          f"{ok.sum()}/{len(ok)} = {100*ok.mean():.1f}% (길이 {ln[ok].mean():.0f}) |")
    P("")

    for tag, exp, desc, ds, nep, adim in SETTINGS:
        bok, bln = base[tag]
        P(f"\n## {tag}  {desc}\n")
        P(f"BC 기준선 **{bok.sum()}/{len(bok)} = {100*bok.mean():.1f}%**, "
          f"성공길이 {bln[bok].mean():.1f}, 최단 {int(bln[bok].min())}\n")
        P("| arm | critic | step | gm | n | 성공 | 성공률 | 성공길이 | 최단 | 승/패 | p |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for d in glob.glob(f"{REPO}/rl-dataset/dexjoco/eval_dexjoco_hammer_nail_{exp}/*"):
            f = f"{d}/meta/episodes.jsonl"
            if not os.path.isfile(f):
                continue
            nm = os.path.basename(d)
            ok, ln = load(f)
            n = len(ok)
            if n < 20:
                continue
            pr = parse(nm)
            meth, gmv, ct, st = pr if pr else ("?", None, "?", None)
            bn = bok[:n]
            w = int((ok & ~bn).sum())
            l = int((~ok & bn).sum())
            pv = mcnemar(w, l)
            sl = ln[ok] if ok.any() else np.array([0])
            rows.append((meth, ct, st if st else 0, gmv if gmv is not None else 0,
                         nm, n, int(ok.sum()), 100 * ok.mean(), sl.mean(),
                         int(sl.min()), w, l, pv))
        for r in sorted(rows, key=lambda r: -r[7]):
            meth, ct, st, gmv, nm, n, s, rate, sln, smin, w, l, pv = r
            sig = "\\*\\*\\*" if pv < 0.001 else "\\*\\*" if pv < 0.01 else "\\*" if pv < 0.05 else ""
            worse = " ⚠" if rate < 100 * bok[:n].mean() - 8 and pv < 0.05 else ""
            stx = f"{st//1000}K" if st else "—"
            gmx = f"{gmv:g}" if meth == "parl" else "—"
            P(f"| `{nm}` | {ct} | {stx} | {gmx} | {n} | {s} | **{rate:.1f}%** | "
              f"{sln:.0f} | {smin} | {w}/{l} | {pv:.4f} {sig}{worse} |")
        P("")
    # ── guide_move 곡선 ────────────────────────────────────────────────────────
    P("\n---\n\n## guide_move 곡선\n")
    P("`parl` 의 상승 폭. **총 이동거리 = guide_move x sqrt(편집차원)** 이고 스텝 수와 "
      "무관하다 (스텝 수는 경로 해상도다). gm=0 이면 후보가 안 바뀌고 top-10 의 argmax = "
      "전체 argmax 이므로 **parl 은 수학적으로 sel32 와 같아진다** — 정합성 검사로 쓴다.\n")
    for tag, exp, desc, ds, nep, adim in SETTINGS:
        bok, _ = base[tag]
        cells = {}
        for d in glob.glob(f"{REPO}/rl-dataset/dexjoco/eval_dexjoco_hammer_nail_{exp}/*"):
            f = f"{d}/meta/episodes.jsonl"
            if not os.path.isfile(f):
                continue
            pr = parse(os.path.basename(d))
            if not pr or pr[0] != "parl":
                continue
            meth, gmv, ct, st = pr
            ok, _ = load(f)
            if len(ok) < 95:
                continue
            n = len(ok); bn = bok[:n]
            w = int((ok & ~bn).sum()); l = int((~ok & bn).sum())
            cells[(ct, st, gmv)] = (100 * ok.mean(), n, w, l, mcnemar(w, l))
        if not cells:
            continue
        gms = sorted({k[2] for k in cells})
        combos = sorted({(k[0], k[1]) for k in cells})
        P(f"### {tag}  {desc}   (BC {100*bok.mean():.1f}%)\n")
        P("| critic | step | " + " | ".join(f"gm={g:g}" for g in gms) + " |")
        P("|---|---|" + "---|" * len(gms))
        for ct, st in combos:
            row = []
            for g in gms:
                v = cells.get((ct, st, g))
                if v is None:
                    row.append("—")
                else:
                    r, n, w, l, pv = v
                    mark = "**" if pv < 0.05 and r > 100 * bok[:n].mean() else ""
                    warn = " ⚠" if pv < 0.05 and r < 100 * bok[:n].mean() else ""
                    row.append(f"{mark}{r:.1f}%{mark} ({w}/{l}){warn}")
            P(f"| {ct} | {st//1000}K | " + " | ".join(row) + " |")
        P("")

    out = "\n".join(L)
    if a.out:
        open(f"{REPO}/{a.out}", "w").write(out)
        print(f"작성: {a.out}  ({len(L)}줄)")
    else:
        print(out)


main()
