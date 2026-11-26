#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
previous="$root/runs/quota-study-20261124"
pilot="$root/runs/segment-pilot-20261124"
py="$root/.venv-gen/bin/python"
export PATH="/usr/lib/wsl/lib:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
mkdir -p output
echo $$ > run.pid
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
stamp preparing
for name in configs demo_utils model pipeline trainer utils wan wan_models assets quality-deps; do
    if [[ ! -e $name ]]; then ln -s "$previous/$name" "$name"; fi
done
for name in cache_probe.py quota_selection.py generate.py paired_run.py evaluate_segments.py prompts.json; do
    if [[ ! -e $name ]]; then cp "$previous/$name" "$name"; fi
done
cmp prompts.json "$pilot/prompts.json"
if ! git apply --reverse --check --recount weight.patch 2>/dev/null; then
    git apply --check --recount weight.patch
    git apply --recount weight.patch
fi
printf '["capped_015"]\n' > methods.json
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$py" -c 'import av,flash_attn,torch,torchvision; print(torch.cuda.get_device_name(),torch.cuda.device_count(),torch.__version__,torch.version.cuda,torchvision.__version__,flash_attn.__version__,av.__version__)'
rg 'self.diversity_weight =|weight = self.diversity_weight|scores\[0, old\] =' cache_probe.py
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -l 2 > output/gpu.csv &
monitor=$!
stamp generating
"$py" paired_run.py
stamp generated
export VBENCH_CACHE_DIR="$root/vbench-cache"
export PYTHONPATH="$PWD/quality-deps:$root/vendor/VBench${PYTHONPATH:+:$PYTHONPATH}"
for metric in motion subject quality; do
    stamp "$metric"
    "$py" evaluate_segments.py "$metric"
done
stamp evaluated
kill "$monitor" 2>/dev/null || true
wait "$monitor" 2>/dev/null || true
monitor=""
tar czf results.partial.tar.gz run.sh weight.patch methods.json prompts.json phases.tsv run.log \
    generate.py cache_probe.py quota_selection.py paired_run.py evaluate_segments.py \
    output/gpu.csv output/meta output/traces output/videos output/scores output/prefix
mv -- results.partial.tar.gz results.tar.gz
