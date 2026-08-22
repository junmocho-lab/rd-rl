"""torchrun 아래에서 여러 GPU 를 쓰기 위한 최소 도구.

**왜 DistributedDataParallel 을 안 쓰나.** DDP 는 forward 에 훅을 걸어 backward 와
all-reduce 를 겹친다. 그러려면 "모듈마다 forward 한 번 → backward 한 번" 이 지켜져야
하는데 EXPO 의 한 update 는 그 모양이 아니다:

  update_critic     encoder 를 두 번 forward 한다 (next_obs 는 stop_gradient=True,
                    obs 는 grad 필요). target_critic·residual 은 no_grad 로 부른다.
  update_residual   critic 을 forward 하지만 grad 는 residual 것만 쓴다 (critic 파라미터에
                    .grad 가 생기지만 opt_critic 이 다음 스텝에서 zero_grad 로 지운다).
  candidate_actions no_grad 안에서 residual·target_critic 을 또 부른다.

이 구조에 DDP 를 씌우면 "Expected to mark a variable ready only once" / unused parameter
오류를 피하려고 find_unused_parameters=True 를 켜야 하고, 그러면 매 스텝 파라미터 전수
검사가 붙는다. 대신 backward 뒤에 **직접 all-reduce** 한다. 동기화할 파라미터가
critic 앙상블 + 인코더 + residual + LoRA 로 다 합쳐 50M 미만이고, 한 update 의 지배적
비용은 VLA forward(수 초)라 통신을 겹치지 않아도 손해가 없다.

**파라미터가 rank 사이에서 갈라지지 않는 근거.** 시작할 때 rank 0 의 값을 전부
broadcast 하고(`broadcast_params`), 이후 모든 rank 가 (a) 같은 평균 gradient 로 (b) 같은
옵티마이저 상태를 갱신한다. target critic 의 polyak 도 같은 입력에 같은 연산이다.
`EXPOLearner.gen`(REDQ 부분집합 뽑기) 도 같은 seed 라 rank 마다 같은 멤버를 고른다 —
여기가 갈라지면 rank 마다 다른 critic 부분집합으로 타깃을 만들게 된다.

배치만 rank 마다 달라야 한다 (`Trainer.rng` 를 seed+rank 로 만든다). 그래서 실효 배치는
batch_size × world_size 다.
"""

from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def enabled() -> bool:
    """torchrun 아래에서 2개 이상의 프로세스로 도는가."""
    return world_size() > 1


def is_main() -> bool:
    """산출물·센티넬·wandb 를 쓰는 것은 이 rank 뿐이다."""
    return rank() == 0


def init() -> tuple[int, int, torch.device]:
    """(rank, world_size, device). 단일 프로세스면 process group 을 만들지 않는다."""
    r, w, lr = rank(), world_size(), local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
        device = torch.device(f"cuda:{lr}")
    else:
        device = torch.device("cpu")
    if w > 1 and not dist.is_initialized():
        # timeout 을 길게 잡는다 — 첫 라운드에서 VLA(13.8GB) 로드에 40초, 이미지 memmap
        # 디코딩에 분 단위가 걸리고 그동안 다른 rank 는 barrier 에서 기다린다.
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    return r, w, device


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_grads(params) -> None:
    """gradient 를 rank 평균으로 맞춘다. backward 직후, opt.step() 전에 부른다."""
    if not dist.is_initialized():
        return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    w = float(dist.get_world_size())
    # 텐서 하나씩 all_reduce 하면 4.78M LoRA 가 수천 번의 작은 통신이 된다. 하나로 합쳐
    # 한 번에 보낸다 (_flatten_dense_tensors 는 torch 내부에서 DDP 도 쓰는 것).
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= w
    for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


@torch.no_grad()
def broadcast_params(modules, src: int = 0) -> None:
    """rank 0 의 파라미터·버퍼를 모든 rank 에 복사한다.

    같은 seed 로 만들면 이론상 같지만, θ 를 파일에서 이어받는 경로(`_load_theta`)와
    LoRA 주입처럼 순서에 민감한 초기화가 섞여 있어 명시적으로 맞춘다. 여기서 어긋난 채
    시작하면 gradient 만 평균되고 파라미터는 영원히 갈라진 상태로 학습된다.
    """
    if not dist.is_initialized():
        return
    for m in modules:
        if m is None:
            continue
        for t in list(m.parameters()) + list(m.buffers()):
            dist.broadcast(t.data, src=src)


def gather_object(obj) -> list:
    """모든 rank 의 값을 모든 rank 가 받는다. 실패 여부를 합의할 때 쓴다."""
    if not dist.is_initialized():
        return [obj]
    box = [None] * dist.get_world_size()
    dist.all_gather_object(box, obj)
    return box


def broadcast_object(obj, src: int = 0):
    """rank 0 이 정한 것을 모두에게 알린다 (어느 라운드를 처리할지 등).

    메일박스를 rank 마다 각자 폴링하면 READY 를 보는 시점이 갈려서 어떤 rank 는 라운드 N,
    어떤 rank 는 대기 상태가 된다 — 그 상태로 all_reduce 에 들어가면 그대로 멈춘다.
    """
    if not dist.is_initialized():
        return obj
    box = [obj if rank() == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]
