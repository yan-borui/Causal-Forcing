#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h:h}
cd "$root"
run_dir=$root/runs/generation
mkdir -p "$run_dir" "$root/vendor"
trap 'print -r -- "$?" > "$run_dir/setup_status"' EXIT
print -r -- running > "$run_dir/setup_status"
export PATH="$HOME/.local/bin:$PATH" GIT_LFS_SKIP_SMUDGE=1

checkout() {
    local name=$1 url=$2 revision=$3 dest="$root/vendor/$1"
    if [[ ! -d "$dest/.git" ]]; then git init -q "$dest"; fi
    if ! git -C "$dest" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git -c http.version=HTTP/1.1 -C "$dest" fetch --depth=1 "$url" "$revision"
    fi
    git -C "$dest" checkout -q --detach "$revision"
    git -C "$dest" rev-parse HEAD > "$run_dir/$name-revision.txt"
    mkdir -p "$dest/wan_models"
    if [[ ! -e "$dest/wan_models/Wan2.1-T2V-1.3B" ]]; then
        ln -s "$root/weights/Wan2.1-T2V-1.3B" "$dest/wan_models/Wan2.1-T2V-1.3B"
    fi
}

environment() {
    local base_python=${GEN_BASE_PYTHON:?Set GEN_BASE_PYTHON to a PyTorch 2.8 and FlashAttention 2 environment}
    if [[ ! -x .venv-gen/bin/python ]]; then uv venv --python "$base_python" .venv-gen; fi
    local site_dir=$(.venv-gen/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
    "$base_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))' > "$site_dir/torch_base.pth"
    uv pip install --python .venv-gen/bin/python --index-url https://pypi.org/simple --no-deps diffusers==0.31.0 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3 easydict==1.13 lmdb==1.7.5 av==13.1.0 numpy==1.24.4 opencv-python-headless==4.11.0.86 imageio==2.37.0 imageio-ffmpeg==0.6.0
}

checkout Causal-Forcing https://github.com/thu-ml/Causal-Forcing.git 52c39015bcb307da785b476822aefce4488ea342 &
cf_pid=$!
checkout Self-Forcing https://github.com/guandeh17/Self-Forcing.git 33593df3e81fa3ec10239271dd2c100facac6de1 &
sf_pid=$!
environment &
env_pid=$!
for task_pid in $cf_pid $sf_pid $env_pid; do wait "$task_pid"; done
print -r -- "ready $(date -Is)"
