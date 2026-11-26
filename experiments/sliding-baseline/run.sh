#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
official="$root/vendor/Causal-Forcing"
archives="$root/runs"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PATH="/usr/lib/wsl/lib:$PATH"
py="$root/.venv-gen/bin/python"
echo $$ > run.pid
echo setup > run.status
trap 'echo "$?" > run.status' EXIT
for name in configs pipeline model trainer utils wan demo_utils wan_models; do
    if [[ ! -e "$name" ]]; then ln -s "$official/$name" "$name"; fi
done
for name in generate_long.py score.py; do
    if [[ ! -e "$name" ]]; then ln -s "$archives/causal-official-30/$name" "$name"; fi
done
for name in assets quality-deps evaluate_segments.py; do
    if [[ ! -e "$name" ]]; then ln -s "$archives/segment-pilot-20261124/$name" "$name"; fi
done
mkdir -p output
git -C "$official" rev-parse HEAD
date -Iseconds
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -l 5 > output/gpu.csv &
monitor=$!
trap 'rc=$?; kill "$monitor" 2>/dev/null || true; echo "$rc" > run.status' EXIT
"$py" -c 'import torch,torchvision,flash_attn,av;print(torch.cuda.get_device_name(),torch.cuda.device_count(),torch.__version__,torch.version.cuda,torchvision.__version__,flash_attn.__version__,av.__version__)'
echo generation > run.status
"$py" generate_sliding.py
export VBENCH_CACHE_DIR="$root/vbench-cache"
export PYTHONPATH="$PWD/quality-deps:$root/vendor/VBench${PYTHONPATH:+:$PYTHONPATH}"
for metric in subject quality motion; do
    echo "$metric" > run.status
    date -Iseconds
    "$py" score.py "$metric"
done
date -Iseconds
