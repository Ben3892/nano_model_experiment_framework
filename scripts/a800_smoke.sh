#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/a800_smoke.sh PATCH_ROOT OUTPUT [all|single|dp2|prepare]" >&2
  echo "       bash scripts/a800_smoke.sh PATCH_ROOT OUTPUT rollout CHECKPOINT ITERATION" >&2
  exit 2
}

mode=${3:-all}
case "$mode" in
  all|single|dp2|prepare) (( $# == 2 || $# == 3 )) || usage ;;
  rollout)
    (( $# == 5 )) || usage
    checkpoint="$(realpath "$4")"
    ;;
  *) usage ;;
esac
patch_root="$(realpath "$1")"
output="$(realpath -m "$2")"
cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$output/hf-datasets}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD
if [[ "$mode" == rollout ]]; then
  device=${SMOKE_GPU_DEVICES-${CUDA_VISIBLE_DEVICES-0}}
  CUDA_VISIBLE_DEVICES="${device%%,*}" exec "$python_bin" -m src.rsi.rollout_smoke \
    --patch-root "$patch_root" --output "$output" --init-from "$checkpoint" --init-iteration "$5"
fi

mkdir -p "$(dirname "$output")"
mkdir "$output"
args=(--backend megatron-patch --patch-root "$patch_root" --seq-len 64
      --batch-size 4 --eval-batches 2 --seed 42 --lr 0.001)

prepare() {
  CUDA_VISIBLE_DEVICES= "$python_bin" -m src.train "${args[@]}" --prepare-only "$@"
}

if [[ "$mode" == prepare ]]; then
  for processes in 1 2; do
    for stage in pretrain sft; do
      prepare --stage "$stage" --nproc-per-node "$processes" --steps 2 \
        --output "$output/prepare-dp$processes-$stage"
    done
  done
  exit
fi

IFS=, read -r gpu0 gpu1 _ <<< "${SMOKE_GPU_DEVICES-${CUDA_VISIBLE_DEVICES-0,1}}"
if [[ -z "$gpu0" || ( "$mode" != single && -z "$gpu1" ) ]]; then
  echo "Select the required GPU(s) with SMOKE_GPU_DEVICES or CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

run_pair() {
  local devices="$1" processes="$2" steps="$3"
  local prefix="$output/dp$processes"
  CUDA_VISIBLE_DEVICES="$devices" "$python_bin" -m src.train "${args[@]}" \
    --stage pretrain --nproc-per-node "$processes" --steps "$steps" \
    --output "$prefix-pretrain" 2>&1 | tee "$prefix-pretrain.console.log"

  local parent="$prefix-pretrain/checkpoint"
  prepare --stage sft --nproc-per-node "$processes" --steps "$steps" \
    --init-from "$parent" --init-iteration "$steps" --output "$prefix-sft-prepare"

  CUDA_VISIBLE_DEVICES="$devices" "$python_bin" -m src.train "${args[@]}" \
    --stage sft --nproc-per-node "$processes" --steps "$steps" \
    --init-from "$parent" --init-iteration "$steps" \
    --output "$prefix-sft" 2>&1 | tee "$prefix-sft.console.log"
}

case "$mode" in
  all) run_pair "$gpu0" 1 4; run_pair "$gpu0,$gpu1" 2 2 ;;
  single) run_pair "$gpu0" 1 4 ;;
  dp2) run_pair "$gpu0,$gpu1" 2 2 ;;
esac
