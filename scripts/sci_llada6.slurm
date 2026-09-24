#!/bin/bash
#SBATCH --job-name=sci_llada6
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/reverse-jev/runs/sci/%x_%j.err
#
# c_* recipe retrained on the DAPT-adapted backbone (LLaDA-8B + papers LoRA).
# Requires runs/dapt/lora-final (from dapt_llada). Heads: da_*.
# Evals: GPQA main/diamond + SciFact noul/score + elite regression.

set -euo pipefail
RJ=/beegfs/a474r867/reverse-jev
BAT=$RJ/data/sci_battery
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
DAPT=$RJ/runs/dapt/lora-final
PY=/beegfs/a474r867/bioai/venv/bin/python
LAYERS="-1,-9,-17,-25"
mkdir -p "$RUNS"
LOCK=$RUNS/.sci_llada6.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another sci_llada6 holds the lock; exiting"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

if [ ! -f "$DAPT/adapter_config.json" ]; then
    echo "missing $DAPT — run dapt_llada first"; exit 1
fi

train_head () {  # name kind data extra...
    local name=$1 kind=$2 data=$3; shift 3
    if [ -f "$RUNS/$name/decision.pt" ]; then
        echo "=== SKIP $name (done) ==="; return
    fi
    echo "=== TRAIN $name ==="
    $PY -m reverse_jev.train \
        --hf-backbone GSAI-ML/LLaDA-8B-Instruct --lora-adapter "$DAPT" \
        --decisions-train "$data" \
        --freeze --r2-mode spanpool --r2-layers="$LAYERS" \
        --canonical-order --head-kind "$kind" \
        --steps 3000 --head-lr 1e-3 --accum 8 "$@" \
        --out "$RUNS/$name"
}

eval_head () {  # name evalfile out
    local name=$1 ev=$2 out=$3
    echo "=== EVAL $out ($name) ==="
    $PY -m reverse_jev.eval \
        --ckpt "$RUNS/$name/decision.pt" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${name#da_}_dev.jsonl" \
        --decisions-eval "$ev" \
        --device cuda --out "$RUNS/eval_${out}.json"
}

train_head da_choice attnpool "$BAT/sci_choice_train.jsonl"
train_head da_noul   mlp      "$BAT/sci_noul_train.jsonl"
train_head da_score  mlp      "$BAT/sci_score_train.jsonl" --ordinal 1.0

# elite regression (adapted backbone, same battery)
for K in choice noul score; do
    eval_head da_$K "$BAT/sci_${K}_eval.jsonl" "da_${K}_elite"
done
# external scientific benchmarks
eval_head da_choice "$EXT/gpqa/gpqa_main_choice_eval.jsonl"    da_choice_gpqa_main
eval_head da_choice "$EXT/gpqa/gpqa_diamond_choice_eval.jsonl" da_choice_gpqa_diamond
eval_head da_noul   "$EXT/scifact/scifact_dev_noul_eval.jsonl" da_noul_scifact
eval_head da_score  "$EXT/scifact/scifact_dev_score_eval.jsonl" da_score_scifact

echo "ALL_DONE"
