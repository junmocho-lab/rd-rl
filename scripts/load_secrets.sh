# 토큰 로더 — sbatch 스크립트가 `source scripts/load_secrets.sh` 로 부른다.
#
# 우선순위 (뒤가 앞을 덮는다):
#   1. 이미 환경에 있는 값        (셸에서 export 했거나 --export=ALL 로 넘어온 것)
#   2. ~/.rldx_secrets.sh         (예전 방식 — 다른 레포와 공유할 때)
#   3. <repo>/.secrets.sh         (이 레포 전용, .gitignore 됨)  <- 가장 강함
#
# 왜 레포 안에 두나: 잡마다 클러스터/계정이 달라도 레포만 따라오면 토큰이 따라온다.
# 왜 커밋하면 안 되나: 이 레포의 sbatch/offline_rl/offline_critic_fuji.sbatch:14 에
# wandb 키가 평문으로 들어가 이미 origin/main 에 올라가 있다 — 같은 일을 반복하지 않는다.
#
# 만들기:
#   cp .secrets.sh.example .secrets.sh && chmod 600 .secrets.sh && $EDITOR .secrets.sh

_ls_repo="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

_LS_VARS="HF_TOKEN HUGGINGFACE_HUB_TOKEN WANDB_API_KEY"

for _f in "$HOME/.rldx_secrets.sh" "$_ls_repo/.secrets.sh"; do
    [ -f "$_f" ] || continue
    # 남이 읽을 수 있으면 경고한다 (조용히 넘어가면 토큰이 새는 걸 모른다).
    _perm=$(stat -c %a "$_f" 2>/dev/null || echo "")
    case "$_perm" in
        *[1-7][1-7]|*[1-7]) echo "  [경고] $_f 권한이 $_perm 이다 — chmod 600 할 것" ;;
    esac
    # ★ **빈 값은 덮어쓰지 않는다.** 템플릿을 그대로 복사하면 `export WANDB_API_KEY=`
    #   가 들어 있는데, 그냥 source 하면 그 빈 값이 홈 파일에서 이미 잡은 진짜 키를
    #   지워 버린다 (실측: 홈에 86자 키가 있는데 로더가 "없음" 을 찍었다).
    #   그래서 source 전후를 비교해 빈 값만 되돌린다.
    for _v in $_LS_VARS; do eval "_prev_$_v=\${$_v:-}"; done
    # shellcheck disable=SC1090
    . "$_f"
    for _v in $_LS_VARS; do
        eval "_cur=\${$_v:-}; _old=\${_prev_$_v:-}"
        [ -z "$_cur" ] && [ -n "$_old" ] && eval "$_v=\$_old"
    done
done
for _v in $_LS_VARS; do unset "_prev_$_v"; done
unset _f _perm _v _cur _old _LS_VARS _ls_repo

# HF 쪽은 두 이름을 다 보는 라이브러리가 있어 하나만 채워도 되게 맞춰 준다.
[ -n "${HF_TOKEN:-}" ] && : "${HUGGINGFACE_HUB_TOKEN:=$HF_TOKEN}"
[ -n "${HUGGINGFACE_HUB_TOKEN:-}" ] && : "${HF_TOKEN:=$HUGGINGFACE_HUB_TOKEN}"
export HF_TOKEN HUGGINGFACE_HUB_TOKEN WANDB_API_KEY 2>/dev/null || true

# 무엇이 잡혔는지만 알린다. **값은 절대 찍지 않는다** (로그가 out/ 에 남는다).
_mask() { [ -n "$1" ] && echo "설정됨(${#1}자)" || echo "없음"; }
echo "  [secrets] HF_TOKEN $(_mask "${HF_TOKEN:-}")  WANDB_API_KEY $(_mask "${WANDB_API_KEY:-}")"
unset -f _mask
