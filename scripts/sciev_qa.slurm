#!/usr/bin/env bash
#SBATCH --job-name=sciev_qa
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|pro6000"
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/sciev/logs/sciev_qa_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/sciev_qa_%j.err

# sciev-QA v1: legitimate synthetic MC-QA from papers_db fulltext.
#   passages (CPU) → generate (gemma-4-31B-qat teacher)
#   → certify (OLMo-2-32B validator: sufficiency + necessity gates) → emit.
# Constraint a100|pro6000: OLMo-2-32B bf16 ~65GB needs >=80GB.
#   sbatch scripts/sciev_qa.slurm   (resume-safe: gen skips done pids)

set -euo pipefail
BASE=/beegfs/a474r867
RJ=${RJ:-$BASE/sciev}
CODE=${CODE:-$RJ}
SIF=$BASE/ecoreasoner/pytorch-cuda.sif
PY=$BASE/bioai/venv/bin/python
QADIR=${QADIR:-$RJ/data/sciev_qa}
N_PASSAGES=${N_PASSAGES:-3000}
GEN_MODEL=${GEN_MODEL:-allenai/OLMo-2-1124-13B-Instruct}
VAL_MODEL=${VAL_MODEL:-allenai/OLMo-2-0325-32B-Instruct}
GEN_HF=${GEN_HF:-$BASE/hf_cache}
VAL_HF=${VAL_HF:-$BASE/hf-cache}
LIMIT=${LIMIT:-0}
export HF_HOME=$BASE/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_V1=0
export TRITON_CACHE_DIR=$BASE/triton_cache
mkdir -p "$QADIR" "$RJ/logs"
echo "[$(hostname)] job=$SLURM_JOB_ID gen=$GEN_MODEL val=$VAL_MODEL n=$N_PASSAGES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$CODE"
if [ ! -f "$QADIR/passages.jsonl" ]; then
    echo "== stage passages =="
    "$PY" data/build_sciev_qa.py --stage passages \
        --papers-db "$BASE/ecoreasoner/data/papersdb/papers_db.jsonl" \
        --out "$QADIR/passages.jsonl" --n-passages "$N_PASSAGES"
fi
if [ ! -f "$QADIR/qa_cert.jsonl" ]; then
    echo "== stage generate ($GEN_MODEL) =="
    apptainer exec --nv "$SIF" env HF_HOME="$GEN_HF" PYTHONPATH="$BASE/pylibs_vllm" python3 \
        "$CODE/data/build_sciev_qa.py" --stage generate \
        --passages "$QADIR/passages.jsonl" --gen "$QADIR/qa_gen.jsonl" \
        --model "$GEN_MODEL" ${LIMIT:+--limit "$LIMIT"}
    echo "== stage certify ($VAL_MODEL) =="
    apptainer exec --nv "$SIF" env HF_HOME="$VAL_HF" PYTHONPATH="$BASE/pylibs_vllm" python3 \
        "$CODE/data/build_sciev_qa.py" --stage certify \
        --gen "$QADIR/qa_gen.jsonl" --cert "$QADIR/qa_cert.jsonl" \
        --model "$VAL_MODEL"
fi
echo "== stage emit =="
PYTHONPATH="$CODE:$BASE/pylibs/tf446" "$PY" data/build_sciev_qa.py --stage emit \
    --cert "$QADIR/qa_cert.jsonl" --out "$QADIR/sciev_qa_v1.jsonl"
echo "ALL_DONE"
