"""eval 롤아웃을 모아 BC 기준선과 페어 비교한다.

씬이 고정(seed 0)이라 eval 의 k번째 에피소드와 수집 데이터의 k번째 에피소드가 같은 장면이다.
그래서 성공률 차이를 McNemar 정확검정으로 볼 수 있다 — 200개 표본에서 ±3.5pp 인 비페어
비교보다 훨씬 민감하다. 진행 중인 런도 앞 n개만 잘라서 비교하므로 중간 집계가 유효하다.

  python3 sim/dexjoco/summarize_evals.py [--exp d2r8_s0]
"""
import argparse, glob, json, os
from math import comb

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = argparse.ArgumentParser()
p.add_argument("--exp", default="", help="비우면 전부")
a = p.parse_args()

BC = {"d2r8_s0": "hammer_nail_d2r8_s0", "d5r20_s0": "hammer_nail_d5r20_s0",
      "d5r20": "hammer_nail_d5r20"}


def mcnemar(w, l):
    """이항 정확검정 양측. w/l = 상대는 실패했는데 내가 성공/그 반대."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


for exp, bcname in BC.items():
    if a.exp and a.exp != exp:
        continue
    root = f"{REPO}/rl-dataset/dexjoco/eval_dexjoco_hammer_nail_{exp}"
    if not os.path.isdir(root):
        continue
    b = load(f"{REPO}/rl-dataset/dexjoco/{bcname}/meta/episodes.jsonl")
    bok = np.array([bool(x.get("success")) for x in b])
    bl = np.array([x["length"] for x in b])
    print(f"\n══ {exp}   BC {bok.sum()}/{len(b)} = {100*bok.mean():.1f}%  "
          f"성공길이 {bl[bok].mean():.1f}")
    print(f"{'arm':>26} {'성공':>9} {'성공률':>7} {'성공길이':>8} {'승/패':>9} {'McNemar':>8}")
    rows = []
    for d in glob.glob(f"{root}/*"):
        f = f"{d}/meta/episodes.jsonl"
        if not os.path.isfile(f):
            continue
        m = load(f)
        n = len(m)
        ok = np.array([bool(x.get("success")) for x in m])
        L = np.array([x["length"] for x in m])
        sl = L[ok] if ok.any() else np.array([0])
        bn = bok[:n]
        w, l = int((ok & ~bn).sum()), int((~ok & bn).sum())
        rows.append((os.path.basename(d), n, int(ok.sum()), 100 * ok.mean(),
                     float(sl.mean()), w, l, mcnemar(w, l)))
    for r in sorted(rows, key=lambda r: -r[3]):
        sig = "**" if r[7] < 0.01 else "*" if r[7] < 0.05 else ""
        print(f"{r[0]:>26} {str(r[2])+'/'+str(r[1]):>9} {r[3]:6.1f}% {r[4]:8.1f} "
              f"{r[5]:4d}/{r[6]:<4d} {r[7]:8.4f} {sig}")
