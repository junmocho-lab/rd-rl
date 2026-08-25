#!/usr/bin/env python3
"""실패 시점 라벨 → 성공 에피소드에서 같은 장면 찾기 (critic 액션 서열 검증용).

두 단계로 쓴다:

  1) 라벨 템플릿 만들기 — **실패 에피소드만** CSV 로 뽑는다. 사람이 fail_sec / mode 를 채운다.
        python -m rl.probe_pairs --data <데이터셋> --template anno.csv [--camera wrist_left]
  2) 라벨을 채운 뒤 — 같은 장면 찾기 + critic 검증 (2x2 액션 스왑)
        python -m rl.probe_pairs --data <데이터셋> --exp fuji --checkpoints <ckpt> --anno anno.csv

     실패 프레임 t 에서 픽셀이 가장 비슷한 **성공** 에피소드 프레임을 찾고, 상태를 고정한 채
     액션만 갈아끼워 Q 를 비교한다:

        Q(s_fail, a_succ) > Q(s_fail, a_fail) ?    ← 올라가야 한다
        Q(s_succ, a_succ) > Q(s_succ, a_fail) ?    ← 이것도 성립해야 "상태 조건부" 서열이다

     첫 줄만 성립하면 critic 이 액션의 전역 특징을 본 것이고 롤아웃 argmax 에는 쓸모가 없다.
     t 뿐 아니라 t-Δ (Δ = 0, replan, 2·replan, ...) 를 함께 본다 — critic 은 실패가 보이는
     순간이 아니라 **결정 시점에** 걸러내야 쓸모가 있고, 그 리드타임이 여기서 나온다.

매칭을 인코더 latent 가 아니라 **픽셀**로 하는 이유: latent 는 학습이 진행되면 바뀌어서
SARSA 와 IQL 을 같은 잣대로 비교할 수 없다. 픽셀 매칭은 고정된 평가셋이 된다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rl.data import episode_files, find_sessions


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--template", type=Path, help="라벨 템플릿 CSV 를 여기 쓴다 (1단계)")
    p.add_argument("--anno", type=Path, help="채운 CSV 를 읽어 매칭+검증 (2단계)")
    p.add_argument("--exp", default="fuji")
    p.add_argument("--checkpoints", type=Path, help="2단계 필수 — critic.pt 가 있는 루트")
    p.add_argument("--deltas", default="0,8,16,32,64", help="t-Δ 프레임 목록")
    p.add_argument("--topk", type=int, default=3, help="검증 PNG 에 넣을 매칭 후보 수")
    p.add_argument("--stride", type=int, default=2, help="후보 프레임 간격 (인접 프레임은 거의 같다)")
    p.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    p.add_argument("--chunk", type=int, default=8, help="Q 계산 배치 (CPU 면 작게)")
    p.add_argument("--exclude", default="teleop",
                   help="세션 이름에 이 문자열이 있으면 건너뛴다 (기본: teleop — 사람 시연은 "
                        "정책의 실패 모드가 아니다). 빈 문자열이면 전부 포함")
    p.add_argument("--sessions", default="",
                   help="세션 이름에 이 문자열이 있는 것만 (기본: 전부)")
    p.add_argument("--camera", default="wrist_left",
                   help="video 칸에 적을 카메라 (부분 문자열 매칭, depth 는 제외한다)")
    p.add_argument("--include-success", action="store_true",
                   help="성공 에피소드도 넣는다 (기본은 실패만 — 마킹할 대상이 실패뿐이라서)")
    a = p.parse_args()
    if not a.template and not a.anno:
        raise SystemExit("--template (1단계) 또는 --anno (2단계) 중 하나를 줄 것")
    if a.anno:
        return probe(a)

    rows, n_skip, used, dropped = [], 0, [], []
    for s in find_sessions(a.data):
        if (a.exclude and a.exclude in s.name) or (a.sessions and a.sessions not in s.name):
            dropped.append(s.name)
            continue
        used.append(s.name)
        info = json.loads((s / "meta/info.json").read_text())
        fps = float(info["fps"])
        cams = [k for k in info["features"] if "image" in k and "depth" not in k]
        pick = [k for k in cams if a.camera in k]
        if not pick:
            raise SystemExit(f"{s.name}: '{a.camera}' 카메라가 없다 (있는 것: {cams})")
        cam = pick[0]
        for ep, f in sorted(episode_files(s).items()):
            df = pd.read_parquet(f, columns=["next.success"])
            ok = bool(df["next.success"].to_numpy(bool).any())
            if ok and not a.include_success:
                n_skip += 1
                continue
            n = len(df)
            rows.append({"video": str(s / "videos" / f"chunk-{ep // 1000:03d}" / cam
                                      / f"episode_{ep:06d}.mp4"),
                         "fail_sec": "", "mode": "",
                         "episode": ep, "seconds": round(n / fps, 2), "frames": n, "fps": fps,
                         "success": int(ok), "session": s.name})
    print(f"[세션] 씀 {len(used)}개: {used}")
    if dropped:
        print(f"[세션] 건너뜀 {len(dropped)}개: {dropped}")
    if not rows:
        raise SystemExit("실패 에피소드가 없다 (--include-success / --exclude '' 를 보라)")
    a.template.parent.mkdir(parents=True, exist_ok=True)
    with a.template.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[템플릿] {a.template}  실패 {len(rows)}행"
          + (f" (성공 {n_skip}개는 제외 — --include-success 로 넣을 수 있다)" if n_skip else ""))
    print(f"  video 칸 = {a.camera} 카메라의 에피소드 mp4 (h264 320x192, 바로 재생된다)")
    print("  채울 칸 2개:  fail_sec = 실패가 보이는 시점(초, mp4 플레이어 표시 그대로)")
    print("                mode     = 실패 종류 문자열 (예: insert_miss, drop_before_shelf)")
    print("  프레임 변환은 도구가 fps 로 한다 — 초만 적으면 된다.")
    return 0




def _feat(imgs, idx, cam: int, hw=(48, 80)) -> "np.ndarray":
    """매칭용 특징 = 한 카메라의 그레이스케일 축소. 학습과 무관하게 고정된 잣대."""
    out = np.empty((len(idx), hw[0] * hw[1]), np.float32)
    for k, i in enumerate(idx):                      # 기본 인덱싱이라 그 카메라 면만 읽는다
        g = cv2.cvtColor(np.asarray(imgs[int(i), cam]), cv2.COLOR_RGB2GRAY)
        out[k] = cv2.resize(g, (hw[1], hw[0]), interpolation=cv2.INTER_AREA).ravel()
    return out / 255.0


def probe(a) -> int:
    import torch
    import yaml

    from rl.data import build_flat, open_images, resolve_modality, video_keys
    from rl.nets import BatchEncoder, CriticEnsemble
    from rl.expo import ExpoConfig
    from rl.vla_rldx import load_state_action_processor, normalize_states

    repo = Path(__file__).resolve().parent.parent
    exp = yaml.safe_load((repo / "configs/exp" / f"{a.exp}.yaml").read_text())
    cfg = ExpoConfig.from_dict(exp.get("expo"))
    R, LAT, H = exp["replan_steps"], exp["inference_latency"], exp["action_horizon"]
    base = a.checkpoints / exp["base_policy"]
    work = a.checkpoints / f"{a.exp}-critic"
    rldx = repo / "third_party/RLDX-1"

    mod, _ = resolve_modality(a.data, None, rldx, exp["rldx_data_config"], base)
    sessions = find_sessions(a.data)                 # 학습과 **같은 목록/순서** 여야 한다
    flat = build_flat(sessions, mod)
    imgs, meta = open_images(work / "images.mm")               # 절대 다시 만들지 않는다
    if meta["sessions"] != [x.name for x in sessions] or meta["shape"][0] != len(flat):
        raise SystemExit(f"images.mm 이 이 --data 로 만들어진 것이 아니다 — 인덱스가 어긋난다\n"
                         f"  images.mm: {meta['sessions']} ({meta['shape'][0]} 프레임)\n"
                         f"  --data   : {[x.name for x in sessions]} ({len(flat)} 프레임)")
    norm = np.load(work / "actnorm.npy", mmap_mode="r")
    proc = load_state_action_processor(base, rldx, exp["rldx_data_config"])
    snorm = normalize_states(proc, mod.embodiment_tag, mod, flat.state)
    cam = video_keys(sessions[0], mod).index(
        next(k for k in video_keys(sessions[0], mod) if a.camera in k))
    print(f"[매칭 카메라] {video_keys(sessions[0], mod)[cam]} (채널 {cam})")

    # --- critic 로드 -------------------------------------------------------
    dev = a.device
    sd = torch.load(work / "critic.pt", map_location=dev)
    print(f"[critic] step {sd.get('step')}  latency {sd.get('latency')} replan {sd.get('replan')} "
          f"action_dim {sd.get('action_dim')} state_dim {sd.get('state_dim')}")
    enc = BatchEncoder(3 * mod.n_cams, cfg.latent_dim_image, cfg.encoder_stage_sizes,
                       cfg.encoder_num_filters).to(dev).eval()
    critic = CriticEnsemble(cfg.latent_dim_image, snorm.shape[1], (LAT + R) * mod.action_dim,
                            cfg.num_qs, cfg.latent_dim_state, cfg.include_state, cfg.hidden_dims,
                            cfg.critic_layer_norm).to(dev).eval()
    enc.load_state_dict(sd["enc"]); critic.load_state_dict(sd["critic"])

    # --- 인덱스 사전 -------------------------------------------------------
    si_of = {p.name: i for i, p in enumerate(sessions)}
    starts = {int(e): int(np.flatnonzero(flat.episode == e)[0]) for e in np.unique(flat.episode)}
    lens = {int(e): int((flat.episode == e).sum()) for e in np.unique(flat.episode)}

    anchors = []
    with a.anno.open() as fh:
        for r in csv.DictReader(fh):
            if not (r.get("fail_sec") or "").strip():
                continue
            si = si_of[r["session"]]
            gep = int(r["episode"]) + flat.ep_offset[si]
            t = int(round(float(r["fail_sec"]) * float(r["fps"])))
            anchors.append({"gep": gep, "t": min(t, lens[gep] - 1), "mode": r["mode"].strip(),
                            "ep": int(r["episode"]), "session": r["session"]})
    if not anchors:
        raise SystemExit(f"{a.anno} 에 fail_sec 이 채워진 행이 없다")
    print(f"[앵커] {len(anchors)}개  모드 "
          f"{ {m: sum(x['mode'] == m for x in anchors) for m in sorted({x['mode'] for x in anchors})} }")

    # --- 후보: 성공 에피소드 프레임 (같은 세션 필터 적용) -------------------
    keep_si = {i for i, p in enumerate(sessions)
               if not (a.exclude and a.exclude in p.name)
               and (not a.sessions or a.sessions in p.name)}
    cand = []
    for e in np.unique(flat.episode):
        fr = np.flatnonzero(flat.episode == e)
        if flat.session[fr[0]] not in keep_si or not flat.is_success[fr[-1]]:
            continue
        cand.append(fr[::a.stride])
    cand = np.concatenate(cand)
    print(f"[후보] 성공 에피소드 프레임 {len(cand)}개 (stride {a.stride})")
    F_cand = _feat(imgs, cand, cam)

    @torch.no_grad()
    def q_of(i_state, i_act):
        """Q(상태 = i_state 의 관측/상태, 액션 = i_act 의 로그된 청크). chunk 씩 나눠 돈다."""
        i_state, i_act = np.asarray(i_state), np.asarray(i_act)
        out = []
        for c in range(0, len(i_state), a.chunk):
            si, ai = i_state[c:c + a.chunk], i_act[c:c + a.chunk]
            x = np.asarray(imgs[si])
            o = torch.from_numpy(np.concatenate([x[:, k] for k in range(x.shape[1])], -1)).to(dev)
            st = torch.from_numpy(snorm[si]).to(dev)
            ac = torch.from_numpy(np.ascontiguousarray(
                np.asarray(norm[ai])[:, :LAT + R].reshape(len(ai), -1))).to(dev)
            out.append(critic(enc(o, stop_gradient=True), st, ac).min(0).values.float().cpu().numpy())
        return np.concatenate(out)

    deltas = [int(x) for x in a.deltas.split(",")]
    pairs, rows = [], []
    for d in deltas:
        qi = np.array([starts[x["gep"]] + max(0, x["t"] - d) for x in anchors])
        Fq = _feat(imgs, qi, cam)
        dist = ((Fq[:, None] - F_cand[None]) ** 2).sum(-1) / Fq.shape[1]
        order = np.argsort(dist, axis=1)
        mj = cand[order[:, 0]]
        qff, qfs = q_of(qi, qi), q_of(qi, mj)        # 실패 상태: 자기 액션 vs 성공 액션
        qss, qsf = q_of(mj, mj), q_of(mj, qi)        # 성공 상태: 자기 액션 vs 실패 액션
        w1, w2 = qfs > qff, qss > qsf
        rows.append((d, float(np.sqrt(dist[np.arange(len(qi)), order[:, 0]]).mean()),
                     float(w1.mean()), float(w2.mean()),
                     float((qfs - qff).mean()), float((qss - qsf).mean()),
                     [x["mode"] for x in anchors], w1, w2))
        if d == deltas[0]:
            pairs = [{"anchor": int(qi[k]), "match": [int(cand[j]) for j in order[k, :a.topk]],
                      "dist": [float(np.sqrt(dist[k, j])) for j in order[k, :a.topk]],
                      **{kk: anchors[k][kk] for kk in ("session", "ep", "t", "mode")}}
                     for k in range(len(qi))]

    print(f"\n{'Δ':>4} {'픽셀거리':>8} {'win1 Q(sf,as)>Q(sf,af)':>22} {'win2 Q(ss,as)>Q(ss,af)':>22}"
          f" {'ΔQ1':>8} {'ΔQ2':>8}")
    for d, pd_, w1m, w2m, dq1, dq2, *_ in rows:
        print(f"{d:>4} {pd_:>8.4f} {w1m:>21.0%} {w2m:>21.0%} {dq1:>+8.4f} {dq2:>+8.4f}")

    print("\n모드별 (Δ = %d)" % deltas[0])
    d, _, _, _, _, _, modes, w1, w2 = rows[0]
    for m in sorted(set(modes)):
        sel = np.array([mm == m for mm in modes])
        print(f"  {m:16s} n={int(sel.sum()):2d}  win1 {w1[sel].mean():.0%}  win2 {w2[sel].mean():.0%}")

    # --- 검증 시트: 앵커 | 매칭 top-k ---------------------------------------
    tiles = []
    for pr in pairs:
        row = [np.asarray(imgs[pr["anchor"], cam])]
        for j, dd in zip(pr["match"], pr["dist"]):
            im = np.asarray(imgs[j, cam]).copy()
            cv2.putText(im, f"{dd:.3f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            row.append(im)
        im0 = row[0].copy()
        cv2.putText(im0, f"ep{pr['ep']} t={pr['t']} {pr['mode'][:12]}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        row[0] = im0
        tiles.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(tiles, axis=0)
    out_png = work / "eval" / "probe_matches.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet[..., ::-1])
    (work / "eval" / "probe_pairs.json").write_text(json.dumps(pairs, indent=2) + "\n")
    print(f"\n[검증] {out_png}  (왼쪽=실패 앵커, 오른쪽={a.topk}개 매칭, 숫자=픽셀거리)")
    print(f"[쌍]   {work / 'eval' / 'probe_pairs.json'}")
    return 0



if __name__ == "__main__":
    import sys
    sys.exit(main())
