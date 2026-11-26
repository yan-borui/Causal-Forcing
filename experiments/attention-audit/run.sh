#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
archive="$root/runs/segment-pilot-20261124"
for name in wan utils model configs demo_utils wan_models pipeline trainer generate.py cache_probe.py; do
    if [[ ! -e "$name" ]]; then ln -s "$archive/$name" "$name"; fi
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export PATH="/usr/lib/wsl/lib:$PATH"
python_bin="$root/.venv-gen/bin/python"
echo $$ > run.pid
echo running > run.status
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -l 3 > output/gpu.csv &
monitor=$!
trap 'rc=$?; kill "$monitor" 2>/dev/null || true; echo "$rc" > run.status' EXIT
date -Iseconds
"$python_bin" -c 'import torch,torchvision,flash_attn,av; print(torch.cuda.get_device_name(),torch.cuda.device_count(),torch.__version__,torch.version.cuda,torchvision.__version__,flash_attn.__version__,av.__version__)'
"$python_bin" replay.py
"$python_bin" compare.py
date -Iseconds
