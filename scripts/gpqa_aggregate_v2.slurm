#!/bin/bash
#SBATCH --job-name=sciev_aggregate_v2
#SBATCH --partition=kbs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
OUT=${OUT:-$RJ/runs/sci/phase_a_v2}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
for CFG in main diamond; do
    EVALS=()
    DEVS=()
    for VI in 0 1 2 3 4; do
        EVALS+=("$OUT/ensemble/v${VI}/gpqa_${CFG}.json")
        DEVS+=("$OUT/ensemble/v${VI}/policy_predictions.json")
    done
    for METHOD in probability_mean log_probability_mean; do
        "$PY" -m scripts.ensemble_instr_variants --evals "${EVALS[@]}" \
            --dev-evals "${DEVS[@]}" --method "$METHOD" \
            --out "$OUT/ensemble_${METHOD}_${CFG}.json"
    done
done
printf 'ALL_DONE\n'
