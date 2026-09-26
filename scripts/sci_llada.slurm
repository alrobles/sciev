#!/bin/bash
#SBATCH --job-name=sci_llada
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# The capacity test: LLaDA-8B-Instruct (pretrained dLLM, Apache-2.0) as
# backbone via HFBackbone — freeze (head only) + full-FT spanpool on the
# scientific decision battery.
#
# Env: bioai venv (torch 2.14 cu130) + PYTHONPATH tf446 pin — LLaDA remote
# code requires transformers 4.x (5.x renames _tied_weights_keys internals).

set -euo pipefail
RJ=/beegfs/a474r867/sciev
ECO=/beegfs/a474r867/ecoreasoner
BAT=$RJ/data/sci_battery
RUNS=$RJ/runs/sci
PY=/beegfs/a474r867/bioai/venv/bin/python
LLADA=GSAI-ML/LLaDA-8B-Instruct
mkdir -p "$RUNS"
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

cat "$BAT"/sci_choice_train.jsonl \
    "$BAT"/sci_noul_train.jsonl \
    "$BAT"/sci_score_train.jsonl > "$BAT"/sci_all_train.jsonl
cat "$BAT"/sci_choice_dev.jsonl \
    "$BAT"/sci_noul_dev.jsonl \
    "$BAT"/sci_score_dev.jsonl > "$BAT"/sci_all_dev.jsonl

echo "=== TRAIN sci spanpool FREEZE (LLaDA-8B, head only) ==="
$PY -m sciev.train \
    --decisions-train "$BAT/sci_all_train.jsonl" \
    --hf-backbone "$LLADA" \
    --r2-mode spanpool --steps 6000 --accum 1 --freeze \
    --head-lr 3e-4 --warmup 200 --device cuda \
    --out "$RUNS/sci_llada_freeze"

echo "=== TRAIN sci spanpool FT (LLaDA-8B) ==="
$PY -m sciev.train \
    --decisions-train "$BAT/sci_all_train.jsonl" \
    --hf-backbone "$LLADA" \
    --r2-mode spanpool --steps 4000 --accum 1 \
    --lr 1e-5 --head-lr 3e-4 --warmup 200 --device cuda \
    --out "$RUNS/sci_llada_ft"

for TAG in sci_llada_freeze sci_llada_ft; do
    for KIND in choice noul score; do
        echo "=== EVAL $KIND ($TAG) ==="
        $PY -m sciev.eval \
            --ckpt "$RUNS/$TAG/decision.pt" \
            --r2-mode spanpool \
            --r2-temp-fit-decisions "$BAT/sci_all_dev.jsonl" \
            --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
            --device cuda --out "$RUNS/eval_${TAG}_${KIND}.json"
    done
done
echo "ALL_DONE"
