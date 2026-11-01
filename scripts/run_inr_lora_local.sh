#!/usr/bin/env bash
set -euo pipefail

# 先在 tmux 中激活 Conda 环境，再传入两张或四张已安排使用的物理 GPU。
# 例：bash scripts/run_inr_lora_local.sh 4,5
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${1:?用法：bash scripts/run_inr_lora_local.sh 4,5 或 2,3,4,5}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf '请提供 GPU 编号，以逗号分隔，不含空格。\n' >&2
    exit 2
fi
IFS=, read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
export NUM_GPUS=${#gpu_ids[@]}
if [[ "$NUM_GPUS" -ne 2 && "$NUM_GPUS" -ne 4 ]]; then
    printf '本启动器支持两张或四张 GPU。\n' >&2
    exit 2
fi
if [[ $(printf '%s\n' "${gpu_ids[@]}" | sort -u | wc -l) -ne "$NUM_GPUS" ]]; then
    printf 'GPU 编号不能重复。\n' >&2
    exit 2
fi

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd -- "$(dirname -- "$SCRIPT_PATH")/.." && pwd)
export MODEL_PATH=${MODEL_PATH:-/home/zhenglifeng/models/Qwen2-VL-2B-Instruct-895c3a4}
export DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data/image_cls_cil_dataset/imagenet-r}
RUN_ID=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
export SAVE_ROOT=${SAVE_ROOT:-/mnt/conda/zhenglifeng/rapo-fork/inr-10task-lora-${RUN_ID}}

# 使用新实现的全覆盖 LoRA；首次完整运行只做一个类别顺序。
export LORA_FULL_COVERAGE=true
export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export CLASS_ORDER_SEEDS=${CLASS_ORDER_SEEDS:-1990}
export ACTOR_MICRO_BATCH_SIZE_UPDATE=${ACTOR_MICRO_BATCH_SIZE_UPDATE:-1}
export ACTOR_MICRO_BATCH_SIZE_EXPERIENCE=${ACTOR_MICRO_BATCH_SIZE_EXPERIENCE:-1}

# /home 已可写；仍将可再生缓存留在本节点 /tmp，结果留在 /mnt/conda。
RUNTIME_ROOT=/tmp/rapo-zhenglifeng
export XDG_CACHE_HOME="$RUNTIME_ROOT/cache"
export XDG_CONFIG_HOME="$RUNTIME_ROOT/config"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
for required_path in "$MODEL_PATH/config.json" "$DATA_ROOT/train_5shots" "$DATA_ROOT/test"; do
    if [[ ! -r "$required_path" ]]; then
        printf '模型或数据路径不可读：%s\n' "$required_path" >&2
        exit 2
    fi
done
mkdir -p "$SAVE_ROOT" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
    "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
    "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"
cp -- "$SCRIPT_PATH" "$SAVE_ROOT/launch.sh"
{
    printf 'START_TIME=%s\nHOST=%s\nPYTHON=%s\n' "$(date -Is)" "$(hostname)" "$(command -v python)"
    for key in CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES NUM_GPUS MODEL_PATH DATA_ROOT SAVE_ROOT \
        LORA_FULL_COVERAGE LORA_RANK LORA_ALPHA ACTOR_LR CLASS_ORDER_SEEDS \
        ACTOR_MICRO_BATCH_SIZE_UPDATE ACTOR_MICRO_BATCH_SIZE_EXPERIENCE; do
        printf '%s=%s\n' "$key" "${!key}"
    done
    printf 'GIT_COMMIT='
    git rev-parse HEAD
    git status --short
} > "$SAVE_ROOT/run-info.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
    --query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu \
    --format=csv > "$SAVE_ROOT/gpu-before.csv"
printf '本次结果目录：%s\n' "$SAVE_ROOT"
printf '实时监视命令：tail -n 40 -f %q\n' "$SAVE_ROOT/resources.log"

# 监视只读取选中的 GPU；RAM、Swap 和 user.slice 是节点/用户组口径。
monitor_resources() {
    while :; do
        date -Is
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
            --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
        free -h
        for group in /sys/fs/cgroup/user.slice "/sys/fs/cgroup/user.slice/user-$(id -u).slice"; do
            for field in memory.current memory.high memory.max; do
                if [[ -r "$group/$field" ]]; then
                    printf '%s=' "$group/$field"
                    cat "$group/$field"
                fi
            done
        done
        cat /proc/pressure/memory
        printf '\n'
        sleep 5
    done
}
monitor_resources > "$SAVE_ROOT/resources.log" 2>&1 &
monitor_pid=$!
stop_monitor() {
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap stop_monitor EXIT

# nvidia-smi 能读不代表 CUDA 能算；逐卡计算小张量，然后退出检查进程。
python -u - <<'PY' 2>&1 | tee "$SAVE_ROOT/cuda-check.log"
import os
import torch
expected = int(os.environ["NUM_GPUS"])
assert torch.cuda.device_count() == expected, f"Expected {expected} visible GPUs, got {torch.cuda.device_count()}"
for i in range(expected):
    result = torch.ones(1, device=f"cuda:{i}").sum().item()
    print(f"CUDA {i}: {torch.cuda.get_device_name(i)}; compute OK ({result})", flush=True)
PY

if bash scripts/img_cls/inr/train_10task.sh 2>&1 | tee "$SAVE_ROOT/console.log"; then
    run_status=0
    printf '训练脚本正常结束，退出码 0。\n' | tee -a "$SAVE_ROOT/console.log"
else
    run_status=$?
    printf '训练脚本异常退出，退出码 %s。\n' "$run_status" | tee -a "$SAVE_ROOT/console.log"
fi
printf 'END_TIME=%s\nEXIT_CODE=%s\n' "$(date -Is)" "$run_status" >> "$SAVE_ROOT/run-info.txt"
exit "$run_status"
