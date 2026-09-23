#!/bin/bash
#SBATCH --job-name=sci_llada2
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.err
#
# Ablation on LLaDA-8B (frozen backbone, head only):
#   mlp_o4 : DecisionHead + --orders 4        (order-robustness alone)
#   ap_o1  : AttnPoolHead L{8,16,24,32} + o1  (readout upgrade alone)
#   ap_o4  : AttnPoolHead + --orders 4        (both)
# NB: --r2-layers uses =form (values starting with - are parsed as flags).

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

cat "$BAT"/sci_choice_train.jsonl \
    "$BAT"/sci_noul_train.jsonl \
    "$BAT"/sci_score_train.jsonl > "$BAT"/sci_all_train.jsonl
cat "$BAT"/sci_choice_dev.jsonl \
    "$BAT"/sci_noul_dev.jsonl \
    "$BAT"/sci_score_dev.jsonl > "$BAT"/sci_all_dev.jsonl

train() {  # tag head_kind layers orders steps
    [ -f "$RUNS/$1/decision.pt" ] && { echo "=== SKIP $1 (done) ==="; return; }
    echo "=== TRAIN $1 ==="
    $PY -m reverse_jev.train \
        --decisions-train "$BAT/sci_all_train.jsonl" \
        --hf-backbone "$LLADA" \
        --r2-mode spanpool --head-kind "$2" --r2-layers="$3" \
        --orders "$4" --steps "$5" --accum 1 --freeze \
        --head-lr 3e-4 --warmup 200 --device cuda \
        --out "$RUNS/$1"
}

train mlp_o4 mlp      -1        4 3000
train ap_o1  attnpool "$LAYERS" 1 4000
train ap_o4  attnpool "$LAYERS" 4 3000

for TAG in mlp_o4 ap_o1 ap_o4; do
    for KIND in choice noul score; do
        echo "=== EVAL $KIND ($TAG) ==="
        $PY -m reverse_jev.eval \
            --ckpt "$RUNS/$TAG/decision.pt" \
            --r2-mode spanpool --r2-layers="$LAYERS" \
            --r2-temp-fit-decisions "$BAT/sci_all_dev.jsonl" \
            --decisions-eval "$BAT/sci_${KIND}_eval.jsonl" \
            --device cuda --out "$RUNS/eval_${TAG}_${KIND}.json"
    done
done
echo "ALL_DONE"
