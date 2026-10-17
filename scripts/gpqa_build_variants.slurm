#!/bin/bash
#SBATCH --job-name=sciev_variants_v2
#SBATCH --partition=kbs
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
OUT=${OUT:-$RJ/runs/sci/phase_a_v2}
QA=$RJ/data/qa_battery_v2/systemone-v2
GPQA=$RJ/data/bench_external/gpqa/systemone-v2
VAR=$OUT/instructions/systemone-v2
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
printf 'TRACE_SCIEV_VARIANTS_V2 job=%s\n' "$SLURM_JOB_ID"
for NAME in calibration policy main diamond; do
    SOURCE=$QA/qa_choice_${NAME}_text.jsonl
    if [ "$NAME" = main ] || [ "$NAME" = diamond ]; then
        SOURCE=$GPQA/gpqa_${NAME}_choice_eval_clean_text.jsonl
    fi
    "$PY" "$CODE/data/build_instr_variants.py" --text-file "$SOURCE" \
        --out-prefix "$VAR/$NAME" --include-original --max-ctx 672 \
        --instructions 'Select the best answer.|Choose the correct option.|Identify the correct answer.|Which of the following options is correct?'
done
printf 'ALL_DONE\n'
