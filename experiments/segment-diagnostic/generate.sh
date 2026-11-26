#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
if [[ ! -e pipeline/causal_inference.py ]]; then
    tar xzf "$root/runs/deep30-dspc/source.tar.gz"
fi
if ! grep -q 'getattr(self, "cache_probe"' wan/modules/causal_model.py; then
    git apply --check --ignore-space-change deepforcing.patch
    git apply --ignore-space-change deepforcing.patch
fi
mkdir -p wan_models
if [[ ! -e wan_models/Wan2.1-T2V-1.3B ]]; then
    ln -s "$root/weights/Wan2.1-T2V-1.3B" wan_models/Wan2.1-T2V-1.3B
fi
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export PATH="/usr/lib/wsl/lib:$PATH"
mkdir -p output
echo $$ > generation.pid
echo running > generation.status
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -l 2 > output/generation-gpu.csv &
monitor=$!
trap 'rc=$?; kill "$monitor" 2>/dev/null || true; echo "$rc" > generation.status' EXIT
python_bin="$root/.venv-gen/bin/python"
"$python_bin" -c 'import torch,torchvision,flash_attn,importlib.metadata as m; print(torch.cuda.get_device_name(),torch.__version__,torchvision.__version__,torch.version.cuda,flash_attn.__version__); print({k:m.version(k) for k in ["diffusers","transformers","av"]})'
"$python_bin" generate.py
