#!/usr/bin/env bash
set -eu
run=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$run/../.." && pwd)
cd "$run"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 PATH="/usr/lib/wsl/lib:$PATH"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,nounits -lms 200 > gpu.csv &
monitor=$!
trap 'rc=$?; kill "$monitor" 2>/dev/null || true; echo "$rc" > exit_status' EXIT
echo $$ > pid
mkdir -p wan_models
ln -sfn "$root/weights/Wan2.1-T2V-1.3B" wan_models/Wan2.1-T2V-1.3B
head -n 1 "$root/data/text_align.txt" > prompt.txt
"$root/.venv-gen/bin/python" -c 'import torch,flash_attn;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),flash_attn.__version__)'
"$root/.venv-gen/bin/python" measure.py --config_path configs/self_forcing_dmd/self_forcing_dmd_sink10.yaml --checkpoint_path "$root/weights/chunkwise/causal_forcing.pt" --data_path prompt.txt --num_output_frames 123 --seed 0 --save_with_index --output_folder video --Budget 16 --Recent 4
