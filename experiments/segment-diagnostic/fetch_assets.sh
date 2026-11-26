#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets
fetch() {
    local name=$1 url=$2
    if [[ ! -s "assets/$name" ]]; then
        curl -fLsS --retry 3 --connect-timeout 15 "$url" -o "assets/$name.partial"
        mv -- "assets/$name.partial" "assets/$name"
    fi
}
fetch_dino_source() {
    local revision=7c446df5b9f45747937fb0d72314eb9f7b66930a
    mkdir -p assets/dino
    for file in hubconf.py vision_transformer.py utils.py LICENSE; do
        fetch "dino/$file" "https://raw.githubusercontent.com/facebookresearch/dino/$revision/$file"
    done
    echo "$revision" > assets/dino_revision.txt
}
if [[ ${1:-all} == dino-source ]]; then
    fetch_dino_source
    exit 0
fi
fetch dino_vitbase16_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth &
dino_pid=$!
fetch musiq_spaq_ckpt-358bb6af.pth https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth &
musiq_pid=$!
fetch_dino_source
wait "$dino_pid"
wait "$musiq_pid"
ls -lh assets/*.pth
