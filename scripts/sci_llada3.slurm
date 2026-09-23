#!/bin/bash
#SBATCH --job-name=sci_llada3
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.err
#
# Per-type heads on frozen LLaDA-8B — best recipe per kind from sci_llada2:
#   pt_choice : AttnPoolHead L{8,16,24,32}, orders=1, choice train only
#   pt_noul   : DecisionHead (mlp), orders=4, noul train only
#   pt_score  : DecisionHead (mlp), orders=4, score train only
# Each evals on its own kind, temp-fit on its own kind's dev split.

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
BAT=$RJ/data/sci_battery
RUNS=$RJ/runs/sci
PY=/beegfs/a474r867/bioai/venv/bin/python
LLADA=GSAI-ML/LLaDA-8B-Instruct
LAYERS="-1,-9,-17,-25"
mkdir -p "$RUNS"
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

train() {  # tag head_kind orders steps kind
    [ -f "$RUNS/$1/decision.pt" ] && { echo "=== SKIP $1 (done) ==="; return; }
    echo "=== TRAIN $1 ==="
    $PY -m reverse_jev.train \
        --decisions-train "$BAT/sci_$5_train.jsonl" \
        --hf-backbone "$LLADA" \
        --r2-mode spanpool --head-kind "$2" --r2-layers="$LAYERS" \
        --orders "$3" --steps "$4" --accum 1 --freeze \
        --head-lr 3e-4 --warmup 200 --device cuda \
        --out "$RUNS/$1"
}

train pt_choice attnpool 1 2000 choice
train pt_noul   mlp      4 3000 noul
train pt_score  mlp      4 3000 score

for KIND in choice noul score; do
    echo "=== EVAL $KIND (pt_$KIND) ==="
    $PY -m reverse_jev.eval \
        --ckpt "$RUNS/pt_$KIND/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" \
        --r2-temp-fit-decisions "$BAT/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
        --device cuda --out "$RUNS/eval_pt_${KIND}.json"
done
echo "ALL_DONE"
