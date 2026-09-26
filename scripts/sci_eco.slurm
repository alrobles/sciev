#!/bin/bash
#SBATCH --job-name=sci_eco
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Baseline: c_* recipe on the ecoreasoner own-pretrained backbone bw1_sr
# (MdLMMoE 1.4B, 16 layers, hidden 1024, heads 16, 8 experts top-2,
# seq_len 768). This is the "trajectory zero" point: same heads/data,
# smallest backbone. NOTE: sft_moe_v2 lora targets HF LLaDA-MoE-7B
# (model.layers.*.self_attn.*), NOT bw1_sr (blocks.*) — incompatible.
# R2 truncates ctx to seq_len//2=384 and clamps OOB ids, so it runs.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
CKPT=/beegfs/a474r867/ecoreasoner/outputs/bw1_sr/checkpoint-g20/model.pt
CFG=$RJ/scripts/bw1sr.yaml
PY=/beegfs/a474r867/bioai/venv/bin/python
TAG=eb
LAYERS="-1,-5,-9,-13"   # 16-layer model: same quarter spacing as LLaDA {-1,-9,-17,-25}/32
mkdir -p "$RUNS"
LOCK=$RUNS/.sci_eco.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another sci_eco holds the lock; exiting"; exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

train_head () {  # name kind data extra...
    local name=$1 kind=$2 data=$3; shift 3
    if [ -f "$RUNS/$name/decision.pt" ]; then
        echo "=== SKIP $name (done) ==="; return
    fi
    echo "=== TRAIN $name ==="
    $PY -m sciev.train \
        --decisions-train "$data" \
        --ckpt "$CKPT" --config "$CFG" \
        --freeze --r2-mode spanpool --r2-layers="$LAYERS" \
        --canonical-order --head-kind "$kind" \
        --steps 3000 --head-lr 1e-3 --accum 8 "$@" \
        --out "$RUNS/$name"
}

eval_head () {  # name evalfile out
    local name=$1 ev=$2 out=$3
    echo "=== EVAL $out ($name) ==="
    $PY -m sciev.eval \
        --ckpt "$RUNS/$name/decision.pt" --config "$CFG" \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_${name#*_}_dev.jsonl" \
        --decisions-eval "$ev" \
        --device cuda --out "$RUNS/eval_${out}.json"
}

train_head ${TAG}_choice attnpool "$BAT/sci_choice_train.jsonl"
train_head ${TAG}_noul   mlp      "$BAT/sci_noul_train.jsonl"
train_head ${TAG}_score  mlp      "$BAT/sci_score_train.jsonl" --ordinal 1.0

for K in choice noul score; do
    eval_head ${TAG}_$K "$BAT/sci_${K}_eval.jsonl" "${TAG}_${K}_elite"
done
eval_head ${TAG}_choice "$EXT/gpqa/gpqa_main_choice_eval.jsonl"    ${TAG}_choice_gpqa_main
eval_head ${TAG}_choice "$EXT/gpqa/gpqa_diamond_choice_eval.jsonl" ${TAG}_choice_gpqa_diamond
eval_head ${TAG}_noul   "$EXT/scifact/scifact_dev_noul_eval.jsonl" ${TAG}_noul_scifact
eval_head ${TAG}_score  "$EXT/scifact/scifact_dev_score_eval.jsonl" ${TAG}_score_scifact

echo "ALL_DONE"
