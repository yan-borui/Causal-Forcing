#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h:h}
run_name=${1:?Provide a new run name}
cd "$root"
mkdir -p runs/generation
pair_log=$root/runs/generation/$run_name-pair-status
print -r -- waiting > "$pair_log"
trap 'print -r -- "$?" > "$pair_log"' EXIT
while [[ $(cat runs/weights/exit_status) == running ]]; do sleep 15; done
[[ $(cat runs/weights/exit_status) == 0 ]]
[[ $(cat runs/generation/setup_status) == 0 && $(cat runs/reference/exit_status) == 0 ]]
print -r -- running > "$pair_log"
for method in causal-forcing self-forcing; do
    method_dir=$root/runs/generation/$run_name/$method
    if [[ -f "$method_dir/exit_status" ]]; then
        method_status=$(cat "$method_dir/exit_status")
        if [[ "$method_status" == 0 ]]; then continue; fi
        if [[ "$method_status" != running ]] && grep -q '^generation_end ' "$method_dir/timing.txt"; then
            zsh "$root/scripts/score_generated.zsh" "$method" "$run_name"
            continue
        fi
        print -u2 -- "Inspect the existing inference run before reusing $method_dir"
        exit 1
    fi
    zsh "$root/scripts/generate_and_score.zsh" "$method" "$run_name"
done
