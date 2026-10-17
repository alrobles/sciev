#!/bin/bash
#SBATCH --job-name=sciev_phase_a_v2
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --array=0-7%2
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%A_%a.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%A_%a.err

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
OUT=${OUT:-$RJ/runs/sci/phase_a_v2}
GPQA=$RJ/data/bench_external/gpqa/systemone-v2
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=$CODE:/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446
cd "$CODE"
mkdir -p "$OUT"
printf 'TRACE_SCIEV_PHASE_A_V2 job=%s task=%s code=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$CODE"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

case "$SLURM_ARRAY_TASK_ID" in
    0|1)
        MODE=full_mask
        if [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then MODE=label; fi
        for CFG in main diamond; do
            "$PY" -m scripts.eval_llada_pll --scoring "$MODE" \
                --text-file "$GPQA/gpqa_${CFG}_choice_eval_clean_text.jsonl" \
                --out "$OUT/${MODE}_${CFG}.json"
        done
        ;;
    2|3)
        CFG=main
        if [ "$SLURM_ARRAY_TASK_ID" = 3 ]; then CFG=diamond; fi
        "$PY" -m scripts.eval_llada_pll --scoring leave_one_out --batch-size 4 \
            --text-file "$GPQA/gpqa_${CFG}_choice_eval_clean_text.jsonl" \
            --out "$OUT/leave_one_out_${CFG}.json"
        ;;
    4|5|6)
        CFG=main
        OFFSET=0
        LIMIT=220
        PART=0
        if [ "$SLURM_ARRAY_TASK_ID" = 5 ]; then OFFSET=220; PART=1; fi
        if [ "$SLURM_ARRAY_TASK_ID" = 6 ]; then CFG=diamond; LIMIT=193; fi
        "$PY" -m scripts.eval_llada_prompted --gen-len 64 --steps 64 --prompt reasoning \
            --offset "$OFFSET" --limit "$LIMIT" \
            --text-file "$GPQA/gpqa_${CFG}_choice_eval_clean_text.jsonl" \
            --out "$OUT/prompted64_${CFG}_part${PART}.json"
        ;;
    7)
        for CFG in main diamond; do
            "$PY" -m scripts.eval_llada_prompted --gen-len 6 --steps 6 --prompt direct \
                --text-file "$GPQA/gpqa_${CFG}_choice_eval_clean_text.jsonl" \
                --out "$OUT/prompted6_${CFG}.json"
        done
        ;;
esac
printf 'ALL_DONE\n'
