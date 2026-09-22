#!/bin/bash
#SBATCH --job-name=tc_teacher
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/toolcall/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/toolcall/%x_%j.err

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
ECO=/beegfs/a474r867/ecoreasoner
BAT=$RJ/data/toolcall_battery
RUNS=$RJ/runs/toolcall
mkdir -p "$RUNS"

OLLAMA_DIR="${HOME}/work/ollama/.ollama"
CONTAINER="${HOME}/work/ollama/ollama-v03215.sif"
MODEL="olmo2:32b"

workdir=$(mktemp -d -p /tmp)
mkdir -p -m 700 ${workdir}/run ${workdir}/tmp ${workdir}/cuda_cache
mkdir -p ${OLLAMA_DIR}/models ${OLLAMA_DIR}/cache
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

export APPTAINER_BIND="${workdir}/run:/run,${workdir}/tmp:/tmp,${OLLAMA_DIR}:/root/.ollama,${workdir}/cuda_cache:/root/.cuda_cache,/dev/shm:/dev/shm"
export APPTAINERENV_OLLAMA_HOST="0.0.0.0:${PORT}"
export APPTAINERENV_OLLAMA_MODELS="/root/.ollama/models"
export APPTAINERENV_OLLAMA_KEEP_ALIVE="5h"
export APPTAINERENV_OLLAMA_NUM_PARALLEL="8"
export APPTAINERENV_OLLAMA_MAX_LOADED_MODELS="1"
export APPTAINERENV_OLLAMA_FLASH_ATTENTION="1"
export APPTAINERENV_OLLAMA_LOAD_TIMEOUT="30m"

echo "node=$HOSTNAME port=$PORT model=$MODEL"
apptainer exec --nv ${CONTAINER} ollama serve &
SERVE_PID=$!

# wait for server
for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${PORT}/api/version" >/dev/null 2>&1 && break
    sleep 5
done
echo "server up"

APPTAINERENV_OLLAMA_HOST="127.0.0.1:${PORT}" \
    apptainer exec ${CONTAINER} ollama pull "$MODEL" \
    || { echo "32b pull failed, falling back to 13b"; MODEL="olmo2:13b";
         APPTAINERENV_OLLAMA_HOST="127.0.0.1:${PORT}" \
         apptainer exec ${CONTAINER} ollama pull "$MODEL"; }
echo "model ready: $MODEL"

PY() { apptainer exec --overlay "$ECO/blackwell-torch.overlay":ro "$ECO/pytorch-cuda.sif" /bb/bwvenv/bin/python "$@"; }
cd "$RJ"

for TAG in train dev eval; do
    echo "=== TEACHER $TAG ==="
    PY data/teacher_label_ollama.py \
        --text-jsonl "$BAT/toolcall_decisions_${TAG}_text.jsonl" \
        --url "http://127.0.0.1:${PORT}" --model "$MODEL" \
        --workers 8 \
        --out "$RUNS/soft_${TAG}.jsonl"
done

kill $SERVE_PID 2>/dev/null || true
echo "ALL_DONE"
