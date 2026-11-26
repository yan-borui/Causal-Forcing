#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
official="$root/vendor/Causal-Forcing/long_video"
pilot="$root/runs/segment-pilot-20261124"
export PATH="/usr/lib/wsl/lib:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
py="$root/.venv-gen/bin/python"
echo $$ > run.pid
echo waiting > run.status
trap 'echo "$?" > run.status' EXIT
for name in configs pipeline model trainer utils wan; do
    if [[ ! -e "$name" ]]; then ln -s "$official/$name" "$name"; fi
done
for name in assets wan_models quality-deps evaluate_segments.py; do
    if [[ ! -e "$name" ]]; then ln -s "$pilot/$name" "$name"; fi
done
mkdir -p output
while [[ "$(cat weights.status)" == running || "$(cat ../attention-audit-four/run.status)" == running ]]; do sleep 6; done
[[ "$(cat weights.status)" == 0 ]]
git -C "$official" rev-parse HEAD
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -l 5 > output/gpu.csv &
monitor=$!
trap 'rc=$?; kill "$monitor" 2>/dev/null || true; echo "$rc" > run.status' EXIT
echo generation > run.status
date -Iseconds
"$py" -c 'import torch,torchvision,flash_attn; print(torch.cuda.get_device_name(),torch.cuda.device_count(),torch.__version__,torchvision.__version__,flash_attn.__version__)'
"$py" generate_long.py
export VBENCH_CACHE_DIR="$root/vbench-cache"
export PYTHONPATH="$PWD/quality-deps:$root/vendor/VBench${PYTHONPATH:+:$PYTHONPATH}"
for metric in subject quality motion; do
    echo "$metric" > run.status
    date -Iseconds
    "$py" score.py "$metric"
done
date -Iseconds
