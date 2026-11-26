#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
export MPLBACKEND=Agg
export PATH="$HOME/.local/bin:$PATH"
echo $$ > analysis.pid
echo waiting > analysis.status
trap 'echo "$?" > analysis.status' EXIT
while [[ ! -f assessment.status || "$(cat assessment.status)" =~ ^(waiting|motion|subject|quality)$ ]]; do
    sleep 10
done
[[ "$(cat assessment.status)" == 0 ]]
if [[ ! -d plot-deps/matplotlib ]]; then
    uv pip install --python "$root/.venv-gen/bin/python" --target ./plot-deps --index-url https://pypi.org/simple numpy==1.26.4 pillow==10.4.0 matplotlib==3.8.4
fi
export PYTHONPATH="$PWD/plot-deps${PYTHONPATH:+:$PYTHONPATH}"
echo summarizing > analysis.status
"$root/.venv-gen/bin/python" analyze.py
echo rendering > analysis.status
"$root/.venv-gen/bin/python" visualize.py
