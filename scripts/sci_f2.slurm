#!/bin/bash
#SBATCH --job-name=sci_f2
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err

set -euo pipefail
RJ=/beegfs/a474r867/sciev
ECO=/beegfs/a474r867/ecoreasoner
BAT=$RJ/data/sci_battery
RUNS=$RJ/runs/sci
SIF=$ECO/pytorch-cuda.sif
OVL=$ECO/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
mkdir -p "$RUNS"

CKPT=$ECO/runs/f2-spanes/checkpoint-g50000/model.pt
CFG=$ECO/harness/configs/f0-span-esqueleto.yaml
export HF_HOME=/beegfs/a474r867/hf-cache
cd "$RJ"

cat "$BAT"/sci_choice_train.jsonl \
    "$BAT"/sci_noul_train.jsonl \
    "$BAT"/sci_score_train.jsonl > "$BAT"/sci_all_train.jsonl
cat "$BAT"/sci_choice_dev.jsonl \
    "$BAT"/sci_noul_dev.jsonl \
    "$BAT"/sci_score_dev.jsonl > "$BAT"/sci_all_dev.jsonl

run() { apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" "$@"; }

echo "=== TRAIN sci spanpool ft (f2-50k init) ==="
run -m sciev.train \
    --decisions-train "$BAT/sci_all_train.jsonl" \
    --ckpt "$CKPT" --config "$CFG" \
    --r2-mode spanpool --steps 8000 --accum 1 \
    --lr 2e-5 --head-lr 3e-4 --warmup 200 --device cuda \
    --out "$RUNS/sci_sp_ft"

for KIND in choice noul score; do
    echo "=== EVAL $KIND (sci_sp_ft) ==="
    run -m sciev.eval \
        --ckpt "$RUNS/sci_sp_ft/decision.pt" --config "$CFG" \
        --r2-mode spanpool \
        --r2-temp-fit-decisions "$BAT/sci_all_dev.jsonl" \
        --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_sci_f2_${KIND}.json"
done
echo "ALL_DONE"
