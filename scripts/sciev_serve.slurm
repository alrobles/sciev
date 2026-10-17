#!/bin/bash
#SBATCH --job-name=sciev_serve
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/logs/serve_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/serve_%j.err

set -euo pipefail

RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}

export PYTHONPATH=/beegfs/a474r867/pylibs/serve:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$CODE
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export SCIEV_MANIFEST=$RJ/serve/manifest.json
export SCIEV_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false

PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
[ -x "$PY" ] || PY=/kuhpc/scratch/kbs/a474r867/sciev/venv/bin/python

PORT=$($PY - <<'EOF'
import socket
s = socket.socket()
s.bind(("0.0.0.0", 0))
print(s.getsockname()[1])
s.close()
EOF
)

HOST=$(hostname -s)
echo "=================================================================="
echo "sciev serve: $HOST:$PORT  (job $SLURM_JOB_ID)"
echo "manifest:  $SCIEV_MANIFEST"
echo "python:    $PY"
echo "tunnel:    ssh -N -L $PORT:$HOST:$PORT $USER@hpc.crc.ku.edu"
echo "=================================================================="

cd "$CODE"
$PY -m uvicorn sciev.serve:app --host 0.0.0.0 --port "$PORT" &
UVPID=$!

# warmup: force engine load so first client request is not cold
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:$PORT/v1/models > /tmp/sciev_models_$SLURM_JOB_ID.json 2>/dev/null; then
    echo "engine ready after ${i}0s:"; cat /tmp/sciev_models_$SLURM_JOB_ID.json; break
  fi
  sleep 10
done

wait $UVPID
