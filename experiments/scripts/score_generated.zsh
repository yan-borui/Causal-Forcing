#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h:h}
method=${1:?Specify the method}
run_name=${2:?Specify the generation run}
run_dir=$root/runs/generation/$run_name/$method
grep -q '^generation_end ' "$run_dir/timing.txt"
print -r -- running > "$run_dir/exit_status"
trap 'print -r -- "$?" > "$run_dir/exit_status"' EXIT
generated=$root/data/generated/$run_name/$method
score_input=$root/data/score_input/$run_name/$method
out=$root/results/generated/$run_name/$method
export VBENCH_CACHE_DIR="$root/vbench-cache" PYTHONUNBUFFERED=1
"$root/.venv-gen/bin/python" "$root/scripts/link_generated.py" "$root/data/text_align.txt" "$method" "$generated" "$score_input"
cd "$root/vendor/VBench"
print -r -- "evaluation_start $(date -Is)" | tee -a "$run_dir/timing.txt"
"$root/.venv-eval/bin/python" evaluate.py --dimension dynamic_degree --mode custom_input --videos_path "$score_input" --output_path "$out" 2>&1 | tee "$run_dir/evaluation.log"
print -r -- "evaluation_end $(date -Is)" | tee -a "$run_dir/timing.txt"
