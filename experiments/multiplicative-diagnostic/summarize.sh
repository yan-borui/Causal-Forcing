#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo waiting > score.status
trap 'echo "$?" > score.status' EXIT
while [[ ! -f run.status || ! $(cat run.status) =~ ^[0-9]+$ ]]; do sleep 8; done
[[ $(cat run.status) == 0 ]]
echo summarizing > score.status
CUDA_VISIBLE_DEVICES= ../../.venv-gen/bin/python score_trial.py
