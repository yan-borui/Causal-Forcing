#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h:h}
method=${1:?Specify causal-forcing or self-forcing}
run_name=${2:?Use a fresh run name to keep the seed sequence reproducible}
case "$method" in
    causal-forcing)
        repo=Causal-Forcing
        config=configs/causal_forcing_dmd_chunkwise.yaml
        checkpoint=$root/weights/chunkwise/causal_forcing.pt
        extra=()
        ;;
    self-forcing)
        repo=Self-Forcing
        config=configs/self_forcing_dmd.yaml
        checkpoint=$root/weights/checkpoints/self_forcing_dmd.pt
        extra=(--use_ema)
        ;;
    *) exit 2 ;;
esac

run_dir=$root/runs/generation/$run_name/$method
mkdir -p "${run_dir:h}"
mkdir "$run_dir"
export PATH="/usr/lib/wsl/lib:$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
export VBENCH_CACHE_DIR="$root/vbench-cache" TOKENIZERS_PARALLELISM=false
prompts=$run_dir/prompts.txt
sed '/^[[:space:]]*$/d' "$root/data/text_align.txt" > "$prompts"
generated=$root/data/generated/$run_name/$method
out=$root/results/generated/$run_name/$method
mkdir -p "$generated" "$out"
print -r -- $$ > "$run_dir/pid"
print -r -- running > "$run_dir/exit_status"
monitor_pid=''
finish() {
    local code=$?
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
    print -r -- "$code" > "$run_dir/exit_status"
    print -r -- "exit $code $(date -Is)" >> "$run_dir/timing.txt"
}
trap finish EXIT
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -l 5 > "$run_dir/gpu.csv" &
monitor_pid=$!
py=$root/.venv-gen/bin/python
cd "$root/vendor/$repo"
git rev-parse HEAD > "$run_dir/source_revision.txt"
cp -- "$config" "$run_dir/config.yaml"
"$py" -c 'import sys, torch, importlib.metadata as m; print(sys.version, torch.version.cuda, torch.cuda.get_device_name(0)); print("\n".join(f"{k}=={m.version(k)}" for k in ("torch", "torchvision", "flash-attn", "diffusers", "transformers", "av", "numpy")))' > "$run_dir/environment.txt"
args=(inference.py --config_path "$config" --checkpoint_path "$checkpoint" --data_path "$prompts" --output_folder "$generated" --num_output_frames 21 --seed 0 "${extra[@]}")
print -r -- "${(j: :)args}" > "$run_dir/command.txt"
print -r -- "generation_start $(date -Is)" | tee -a "$run_dir/timing.txt"
"$py" "${args[@]}" 2>&1 | tee "$run_dir/inference.log"
print -r -- "generation_end $(date -Is)" | tee -a "$run_dir/timing.txt"
zsh "$root/scripts/score_generated.zsh" "$method" "$run_name"
