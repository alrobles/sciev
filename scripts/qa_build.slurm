#!/bin/bash
#SBATCH --job-name=sciev_qa_build_v2
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"
printf 'TRACE_SCIEV_QA_BUILD_V2 job=%s\n' "${SLURM_JOB_ID:-local}"
"$PY" "${BUILDER:-$RJ/data/build_qa_choice.py}" --rj "$RJ" --out-dir "${OUT_DIR:-$RJ/data/qa_battery_v3}"
"$PY" "$RJ/scripts/audit_qa_dataset.py" --qa-dir "${OUT_DIR:-$RJ/data/qa_battery_v3}/systemone-v2" \
    --gpqa-dir "$RJ/data/bench_external/gpqa/systemone-v2" --out "$RJ/runs/sci/gpqa_v3_data_audit.json"
