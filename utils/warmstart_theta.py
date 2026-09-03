#!/usr/bin/env python3
"""오프라인 qvgm critic 체크포인트 → 온라인 EXPO 의 warm-start θ₀.

무엇을 만드는가 (learner 가 뜨기 **전에** 제자리에 두면 learner 가 그대로 이어받는다):
    <out>/init/theta.pt      critic/target = 오프라인 가중치, residual/temp = seed 초기화
    <out>/init/meta.json     출처·sha·검증 결과
    <out>/init/DONE          learner/loop.py export_init 이 이걸 보고 θ₀ 재생성을 건너뛴다
    <out>/featstats.npz      오프라인 ckpt 의 feat_mu/feat_sd — $L_RUNS/<run id>/buffer/ 에
                             둘 것 (ingest 가 "없으면 계산" 이라 미리 두면 그걸 쓴다.
                             warm-start 가중치는 이 통계로 표준화된 입력을 가정한다)

구조가 exp yaml 의 critic 블록과 다르면 (num_qs/bins/hidden/latent/action_index)
로드 전에 죽는다 — 조용히 절반만 맞는 warm start 가 제일 위험하다.

usage (로컬에서 만들고 kubectl cp 로 올린다):
    third_party/RLDX-1/.pixi/envs/rldx/bin/python utils/warmstart_theta.py \\
        --critic checkpoints/<exp>-critic/<tag>/critic_001000.pt \\
        --exp-config configs/exp/fuji_online.yaml --out /tmp/warm_init --seed 0
    kubectl -n $L_NS cp /tmp/warm_init/init $L_POD:$L_CKPT/expo/<run id>/init
    kubectl -n $L_NS exec $L_POD -- mkdir -p $L_RUNS/<run id>/buffer
    kubectl -n $L_NS cp /tmp/warm_init/featstats.npz $L_POD:$L_RUNS/<run id>/buffer/featstats.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/RLDX-1"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--critic", required=True, type=Path, help="오프라인 qvgm critic .pt")
    p.add_argument("--exp-config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0, help="residual/temp 초기화 seed")
    a = p.parse_args()

    import numpy as np
    import torch
    import yaml

    from learner.loop import qvgm_spec, qvgm_theta, write_atomic
    from rl.data import resolve_modality
    from rl.expo import DummyVLA, EXPOLearner, ExpoConfig
    from rl.nets import explore_spec

    exp = yaml.safe_load(a.exp_config.read_text())
    mod, src = resolve_modality(REPO, REPO / exp["modality"], REPO / "third_party/RLDX-1",
                                exp["rldx_data_config"], None)
    qv = qvgm_spec(exp, mod)
    if qv is None:
        sys.exit("exp yaml 에 critic 블록이 없다 — qvgm 모드가 아니면 warm start 대상이 아니다")
    ecfg = ExpoConfig.from_dict(exp.get("expo"))
    lat, rep = int(exp["inference_latency"]), int(exp["replan_steps"])

    sd = torch.load(a.critic, map_location="cpu")
    if sd.get("kind") != "qvgm":
        sys.exit(f"{a.critic}: kind={sd.get('kind')} — qvgm critic 이 아니다")

    # ── 구조 검증: 조용히 절반만 맞는 warm start 를 막는다 ─────────────────────
    checks = {
        "latency": (sd.get("latency"), lat),
        "replan": (sd.get("replan"), rep),
        "action_dim": (sd.get("action_dim"), mod.action_dim),
        "state_dim": (sd.get("state_dim"), mod.state_dim),
        "latent": (sd.get("latent"), qv["latent"]),
        "state_latent": (sd.get("state_latent"), qv["state_latent"]),
        "hidden_dims": (list(sd.get("hidden_dims") or []), list(qv["hidden"])),
        "bins": (sd.get("bins"), qv["bins"]),
        "num_qs": (sd.get("num_qs"), ecfg.num_qs),
        "n_steps": (sd.get("n_steps"), qv["n_steps"]),
        "inject": (bool(sd.get("inject", True)), qv["inject"]),
        "action_index(len)": (len(sd.get("action_index") or []), len(qv["action_index"])),
        "dfeat": (int(sd["feat_mu"].reshape(-1).shape[0]), qv["dfeat"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        for k, (got, want) in bad.items():
            print(f"  불일치 {k}: ckpt={got} vs exp={want}")
        sys.exit("구조가 exp yaml 과 다르다 — 같은 설정으로 다시 구운 critic 을 쓸 것")
    if list(sd.get("action_index") or []) != list(qv["action_index"]):
        sys.exit("action_index 내용이 다르다 (explore_groups 나 modality 가 학습 때와 다름)")

    # ── θ₀: critic/target = 오프라인, residual/temp = seed 초기화 ───────────────
    spec = explore_spec(mod.offsets("action"), exp.get("explore_groups") or [],
                        mod.action_dim, rep, lat)
    L = EXPOLearner(DummyVLA(mod.action_dim, int(exp["action_horizon"])), spec, mod.state_dim,
                    mod.n_cams, rep, ecfg, device="cpu", seed=a.seed, latency=lat, qvgm=qv)
    L.critic.enc.load_state_dict(sd["enc"])
    L.critic.q.load_state_dict(sd["critic"])
    L.target_critic.enc.load_state_dict(sd.get("tenc", sd["enc"]))
    L.target_critic.q.load_state_dict(sd.get("target", sd["critic"]))

    mu = sd["feat_mu"].reshape(-1).float().numpy()
    sdv = sd["feat_sd"].reshape(-1).float().numpy()
    out_sd = qvgm_theta(L, mu, sdv, exp, mod, qv)
    if sd.get("ens_std_ref") is not None:
        out_sd["ens_std_ref"] = float(sd["ens_std_ref"])  # 서빙 OOD 진단 기준선 승계

    init = a.out / "init"
    init.mkdir(parents=True, exist_ok=True)
    theta = init / "theta.pt"
    torch.save(out_sd, theta)
    np.savez(a.out / "featstats.npz", mu=mu.astype(np.float32),
             sd=sdv.astype(np.float32))
    sha = hashlib.sha256(theta.read_bytes()).hexdigest()
    meta = {
        "kind": "init", "warm_start": str(a.critic), "warm_step": sd.get("step"),
        "seed": a.seed, "theta_sha256": sha, "artifacts": ["theta.pt"],
        "critic_backend": "qvgm",
        "keys": ["enc", "critic", "tenc", "target", "residual", "temp"],
        "featstats": "오프라인 ckpt 의 feat_mu/sd — buffer/featstats.npz 로 배치할 것",
        "modality_source": src, "torch": torch.__version__,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_atomic(init / "meta.json", meta)
    write_atomic(init / "DONE", {"kind": "init", "theta_sha256": sha,
                                 "finished_at": meta["finished_at"]})
    print(f"[warm] θ₀ {theta}  {theta.stat().st_size/1e6:.0f} MB  sha256 {sha[:16]}")
    print(f"[warm] critic/target ← {a.critic.name} (step {sd.get('step')}), "
          f"residual/temp ← seed {a.seed} 초기화")
    print(f"[warm] featstats.npz ← ckpt 의 feat_mu/sd (버퍼에 미리 둘 것 — 위 usage 참고)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
