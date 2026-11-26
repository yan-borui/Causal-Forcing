#!/usr/bin/env zsh
set -euo pipefail
root=${0:A:h:h}
cd "$root"
log_dir=$root/runs/weights
mkdir -p "$log_dir" "$root/weights/Wan2.1-T2V-1.3B/google/umt5-xxl" "$root/weights/chunkwise" "$root/weights/checkpoints"
trap 'print -r -- "$?" > "$log_dir/exit_status"' EXIT
print -r -- running > "$log_dir/exit_status"
print -r -- $$ > "$log_dir/pid"
print -r -- "start $(date -Is)"

if (( ! $+commands[aria2c] )); then
    pkg_dir=$root/download-tools
    mkdir -p "$pkg_dir"
    cd "$pkg_dir"
    if [[ ! -x usr/bin/aria2c || ! -e usr/lib/x86_64-linux-gnu/libcares.so.2 ]]; then
        apt-get download aria2 libaria2-0 libcares2
        for pkg in ./*.deb; do dpkg-deb -x "$pkg" .; done
    fi
    export PATH="$pkg_dir/usr/bin:$PATH" LD_LIBRARY_PATH="$pkg_dir/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    cd "$root"
fi

wan_url='https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B/resolve/master'
wan_dir=$root/weights/Wan2.1-T2V-1.3B
job() {
    print -r -- "$1"
    print -r -- "  dir=$2"
    print -r -- "  out=$3"
}
{
    for item in config.json google/umt5-xxl/{spiece.model,tokenizer.json,special_tokens_map.json,tokenizer_config.json} Wan2.1_VAE.pth diffusion_pytorch_model.safetensors models_t5_umt5-xxl-enc-bf16.pth; do
        job "$wan_url/$item" "${wan_dir}/${item:h}" "${item:t}"
    done
    job 'https://hf-mirror.com/zhuhz22/Causal-Forcing/resolve/main/chunkwise/causal_forcing.pt' "$root/weights/chunkwise" causal_forcing.pt
    job 'https://hf-mirror.com/gdhe17/Self-Forcing/resolve/main/checkpoints/self_forcing_dmd.pt' "$root/weights/checkpoints" self_forcing_dmd.pt
} > "$log_dir/urls.txt"

aria2c --input-file="$log_dir/urls.txt" --continue=true --auto-file-renaming=false --allow-overwrite=false --file-allocation=none --max-concurrent-downloads=4 --max-connection-per-server=12 --split=12 --min-split-size=8M --retry-wait=3 --max-tries=12 --connect-timeout=15 --timeout=45 --console-log-level=warn --download-result=full --summary-interval=30 --show-console-readout=false
[[ $(stat -c %s "$wan_dir/diffusion_pytorch_model.safetensors") == 5676070424 ]]
[[ $(stat -c %s "$wan_dir/models_t5_umt5-xxl-enc-bf16.pth") == 11361920418 ]]
[[ $(stat -c %s "$wan_dir/Wan2.1_VAE.pth") == 507609880 ]]
[[ $(stat -c %s weights/chunkwise/causal_forcing.pt) == 5676282643 ]]
[[ $(stat -c %s weights/checkpoints/self_forcing_dmd.pt) == 5676252553 ]]
print -r -- "end $(date -Is)"
