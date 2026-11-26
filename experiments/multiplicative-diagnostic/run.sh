#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
pilot="$root/runs/segment-pilot-20261124"
quota="$root/runs/quota-study-20261124"
py="$root/.venv-gen/bin/python"
export PATH="/usr/lib/wsl/lib:$PATH"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
mkdir -p output
echo $$ > run.pid
echo preparing > run.status
monitor=""
finish() {
    rc=$?
    if [[ -n $monitor ]]; then kill "$monitor" 2>/dev/null || true; fi
    echo "$rc" > run.status
}
trap finish EXIT
stamp() {
    echo "$(date +%s.%N) $1" >> phases.tsv
    echo "$1" > run.status
    echo "$1 $(date -Iseconds)"
}
if [[ ! -e pipeline/causal_inference.py ]]; then tar xzf "$root/runs/deep30-dspc/source.tar.gz"; fi
if ! grep -q 'getattr(self, "cache_probe"' wan/modules/causal_model.py; then
    git apply --check --ignore-space-change "$pilot/deepforcing.patch"
    git apply --ignore-space-change "$pilot/deepforcing.patch"
fi
for file in cache_probe.py generate.py evaluate_segments.py prompts.json; do
    if [[ ! -e $file ]]; then cp "$pilot/$file" "$file"; fi
done
if [[ ! -e paired_run.py ]]; then cp "$quota/paired_run.py" paired_run.py; fi
if ! grep -q 'from multiplicative import' generate.py; then
    git apply --check --recount --ignore-space-change trial.patch
    git apply --recount --ignore-space-change trial.patch
fi
for name in assets quality-deps plot-deps; do
    if [[ ! -e $name ]]; then ln -s "$pilot/$name" "$name"; fi
done
mkdir -p wan_models
if [[ ! -e wan_models/Wan2.1-T2V-1.3B ]]; then ln -s "$root/weights/Wan2.1-T2V-1.3B" wan_models/Wan2.1-T2V-1.3B; fi
echo '["multiplicative"]' > methods.json
stamp generating
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -l 2 > output/gpu.csv &
monitor=$!
"$py" -c 'import torch,torchvision,flash_attn;print(torch.cuda.get_device_name(),torch.__version__,torchvision.__version__,flash_attn.__version__)'
"$py" paired_run.py
stamp generated
export VBENCH_CACHE_DIR="$root/vbench-cache"
export PYTHONPATH="$PWD/quality-deps:$root/vendor/VBench${PYTHONPATH:+:$PYTHONPATH}"
for metric in motion subject quality; do
    stamp "$metric"
    "$py" evaluate_segments.py "$metric"
done
stamp evaluated
