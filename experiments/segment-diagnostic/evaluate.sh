#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
export VBENCH_CACHE_DIR="$root/vbench-cache"
export PYTHONPATH="$PWD/quality-deps:$root/vendor/VBench${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/usr/lib/wsl/lib:$HOME/.local/bin:$PATH"
echo $$ > assessment.pid
echo waiting > assessment.status
trap 'echo "$?" > assessment.status' EXIT
while [[ ! -f generation.status || "$(cat generation.status)" == running ]]; do sleep 10; done
[[ "$(cat generation.status)" == 0 ]]
py="$root/.venv-gen/bin/python"
if [[ ! -d quality-deps/pyiqa ]]; then
    uv pip install --python "$py" --target ./quality-deps --no-deps --index-url https://pypi.org/simple pyiqa==0.1.13
fi
"$py" -c 'import torch, torchvision, importlib.metadata as m; print(torch.cuda.get_device_name(), torch.__version__, torchvision.__version__, {k:m.version(k) for k in ["pyiqa", "decord", "numpy"]})'
for metric in motion subject quality; do
    echo "$metric" > assessment.status
    "$py" evaluate_segments.py "$metric"
done
