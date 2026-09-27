#!/bin/bash
#SBATCH --job-name=r2_matched_KIND
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:q6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Round 2 (M9): rerun the matched fr_/da_ heads with per-item predictions
# so paired per-seed deltas/CIs can be computed on identical item sets.
# Submit three times with KIND=choice|noul|score via --export.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LAYERS=${LAYERS:--1,-9,-17,-25}
KIND=${KIND:?submit with --export=KIND=choice|noul|score}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

# clean eval files were pre-generated into data/*/systemone-v2/
for CFG in main diamond; do
    [ -e "$EXT/gpqa/systemone-v2/gpqa_${CFG}_choice_eval_clean.jsonl" ] || [ "$KIND" != "choice" ] || { echo "missing clean gpqa_$CFG"; exit 1; }
done

# elite eval target: the systemone-v2 battery (group-disjoint, complete
# evidence, zero question-truncated rows) — same files as the archived
# matched evals, under the renamed repo root.
ELITE_EVAL="$BAT/systemone-v2/sci_${KIND}_eval.jsonl"

run_eval () {  # ckpt evalfile tag
    local ck=$1 ev=$2 tag=$3
    local dst="$RUNS/eval_r2_${tag}.json"
    if [ -e "$dst" ]; then echo "skip existing $dst"; return; fi
    echo "=== EVAL $tag ==="
    "$PY" -m sciev.eval \
        --ckpt "$ck" \
        --decision-type "$KIND" --seed 7331 \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/systemone-v2/sci_${KIND}_dev.jsonl" \
        --decisions-eval "$ev" --predictions \
        --device cuda --out "$dst"
}

for BB in fr da; do
    for S in 7331 7332 7333; do
        CK="$RUNS/${BB}_matched_s${S}_${KIND}/decision.pt"
        [ -f "$CK" ] || { echo "missing $CK"; continue; }
        run_eval "$CK" "$ELITE_EVAL" "${BB}_s${S}_${KIND}_elite"
        case "$KIND" in
            choice)
                run_eval "$CK" "$EXT/gpqa/systemone-v2/gpqa_main_choice_eval_clean.jsonl"    "${BB}_s${S}_${KIND}_gpqa_main"
                run_eval "$CK" "$EXT/gpqa/systemone-v2/gpqa_diamond_choice_eval_clean.jsonl" "${BB}_s${S}_${KIND}_gpqa_diamond"
                ;;
            noul)
                run_eval "$CK" "$EXT/scifact/systemone-v2/clean_dev_noul_eval.jsonl"  "${BB}_s${S}_${KIND}_scifact"
                ;;
            score)
                run_eval "$CK" "$EXT/scifact/systemone-v2/clean_dev_score_legacy_proxy_eval.jsonl" "${BB}_s${S}_${KIND}_scifact"
                ;;
        esac
    done
done
echo "ALL_DONE"
