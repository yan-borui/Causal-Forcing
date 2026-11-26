#!/usr/bin/env bash
# One downstream experiment, conditional on the frozen same-seed selection rule.
set -euo pipefail
cd "$(dirname "$0")"
study=$PWD
root=$(cd ../.. && pwd)
replica="$root/runs/quota-replication-20261124"
python_bin="$root/.venv-gen/bin/python"
echo $$ > replication.pid
echo queued > replication.status
finish() {
    rc=$?
    if ((rc != 0)); then echo "failed $rc" > "$study/replication.status"; fi
}
trap finish EXIT
while [[ ! -f summary.status || ! $(cat summary.status) =~ ^[0-9]+$ ]]; do sleep 12; done
[[ $(cat summary.status) == 0 ]]
choice=$("$python_bin" -c 'import json; print(json.load(open("output/summary.json"))["replicate_choice"] or "none")')
if [[ $choice == none ]]; then
    echo no_qualifying_method > replication.status
    exit 0
fi
case "$choice" in capped|two_stage) ;; *) exit 2 ;; esac
mkdir "$replica"
for file in configure.py run.sh paired_run.py quota_selection.py selection.patch compare.py visualize.py summarize.sh; do
    cp "$study/$file" "$replica/$file"
done
echo "selected $choice" > replication.status
cd "$replica"
export PATH="/usr/lib/wsl/lib:$PATH"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name --format=csv
while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
    sleep 20
done
echo "running $choice" > "$study/replication.status"
bash run.sh replicate "$choice" > run.log 2>&1
bash summarize.sh > summary.log 2>&1
echo "completed $choice" > "$study/replication.status"
