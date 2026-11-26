#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
echo waiting > summary.status
trap 'echo "$?" > summary.status' EXIT
while [[ ! -f run.status || ! $(cat run.status) =~ ^[0-9]+$ ]]; do sleep 8; done
[[ $(cat run.status) == 0 ]]
export MPLBACKEND=Agg
export PYTHONPATH="$PWD/plot-deps${PYTHONPATH:+:$PYTHONPATH}"
echo comparing > summary.status
"$root/.venv-gen/bin/python" compare.py
echo rendering > summary.status
"$root/.venv-gen/bin/python" visualize.py
