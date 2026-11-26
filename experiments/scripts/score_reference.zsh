#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h:h}
run_dir=$root/runs/reference
mkdir -p "$run_dir" "$root/data" "$root/vendor" "$root/results/reference"
trap 'print -r -- "$?" > "$run_dir/exit_status"' EXIT
print -r -- $$ > "$run_dir/pid"
cd "$root"
export PATH="/usr/lib/wsl/lib:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 VBENCH_CACHE_DIR="$root/vbench-cache"
export GIT_LFS_SKIP_SMUDGE=1

fetch() {
    local url=$1 dest=$2
    if [[ -s "$dest" ]]; then return; fi
    curl -fsSL --retry 3 --retry-all-errors --connect-timeout 12 --max-time 180 "$url" -o "$dest.partial"
    mv -- "$dest.partial" "$dest"
    print -r -- "fetched ${dest:t}"
}

source_setup() {
    if [[ ! -d vendor/VBench/.git ]]; then
        git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.1.5 --filter=blob:none --sparse https://github.com/Vchitect/VBench.git vendor/VBench
    fi
    git -C vendor/VBench sparse-checkout set vbench
    git -C vendor/VBench rev-parse HEAD > "$run_dir/vbench_revision.txt"
}

env_setup() {
    local base_python=${VBENCH_BASE_PYTHON:?Set VBENCH_BASE_PYTHON to the existing PyTorch environment}
    if [[ ! -x .venv-eval/bin/python ]]; then
        uv venv --python "$base_python" .venv-eval
    fi
    local site_dir=$(.venv-eval/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    "$base_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' > "$site_dir/torch_base.pth"
    uv pip install --python .venv-eval/bin/python --no-deps --index-url https://pypi.org/simple decord==0.6.0 easydict==1.13 numpy==1.26.4 opencv-python-headless==4.11.0.86
}

print -r -- "prepare $(date -Is)"
fetch 'https://github.com/user-attachments/files/26291952/text_align.txt' data/text_align.txt &
pid_prompts=$!
fetch 'https://drive.usercontent.google.com/download?id=1MFiCNHgjLcQsOFiL2njBeWHrP2TSd70L&export=download&confirm=t' data/causal-forcing.zip &
pid_cf=$!
fetch 'https://drive.usercontent.google.com/download?id=1FTKRuvmTZt64TDTgT70Gow1FTIZ7U5M_&export=download&confirm=t' data/self-forcing.zip &
pid_sf=$!
fetch 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip' data/raft-models.zip &
pid_raft=$!
source_setup &
pid_src=$!
env_setup &
pid_env=$!
for task_pid in $pid_prompts $pid_cf $pid_sf $pid_raft $pid_src $pid_env; do wait "$task_pid"; done

py=$root/.venv-eval/bin/python
"$py" "$root/scripts/prepare_reference.py" "$root"
"$py" -c 'import sys, torch, torchvision, cv2, decord, numpy; print(sys.version, torch.__version__, torch.version.cuda, torchvision.__version__, cv2.__version__, decord.__version__, numpy.__version__, torch.cuda.get_device_name(0))' | tee "$run_dir/environment.txt"
cd "$root/vendor/VBench"
for model in causal-forcing self-forcing; do
    print -r -- "$model start $(date -Is)"
    "$py" evaluate.py --dimension dynamic_degree --mode custom_input --videos_path "$root/data/reference/$model" --output_path "$root/results/reference/$model" 2>&1 | tee "$run_dir/$model.log"
    print -r -- "$model end $(date -Is)"
done
