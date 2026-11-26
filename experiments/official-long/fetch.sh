#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
dest="$root/weights/chunkwise"
size=17028883814
echo running > weights.status
trap 'echo "$?" > weights.status' EXIT
if [[ -s "$dest/longvideo.pt" ]]; then
    [[ "$(stat -c %s "$dest/longvideo.pt")" == "$size" ]]
    exit 0
fi
tools_dir="$root/download-tools"
export PATH="$tools_dir/usr/bin:$PATH"
export LD_LIBRARY_PATH="$tools_dir/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
url="https://hf-mirror.com/zhuhz22/Causal-Forcing/resolve/2f8eb8bb6eeb1238da9d13e5420d342a74d634a6/chunkwise/longvideo.pt"
aria2c --dir="$dest" --out=longvideo.pt.part --continue=true --auto-file-renaming=false --allow-overwrite=false --file-allocation=none --max-connection-per-server=12 --split=12 --min-split-size=16M --retry-wait=4 --max-tries=8 --connect-timeout=15 --timeout=45 --summary-interval=30 --show-console-readout=false "$url"
[[ "$(stat -c %s "$dest/longvideo.pt.part")" == "$size" ]]
mv -n -- "$dest/longvideo.pt.part" "$dest/longvideo.pt"
