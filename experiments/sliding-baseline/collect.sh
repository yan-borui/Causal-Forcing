#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo waiting > bundle.status
trap 'echo "$?" > bundle.status' EXIT
while true; do
    status=$(cat run.status)
    case "$status" in
        0) break ;;
        setup|generation|subject|quality|motion) sleep 15 ;;
        *) echo "Run exited with status $status"; exit 1 ;;
    esac
done
echo collecting > bundle.status
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 ../../.venv-gen/bin/python workshop.py --with-sliding
tar czf results-with-videos.partial.tar.gz run.status run.log output/config.yaml output/gpu.csv output/meta output/videos output/scores output/workshop
mv -- results-with-videos.partial.tar.gz results-with-videos.tar.gz
