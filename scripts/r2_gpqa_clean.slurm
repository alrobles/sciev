#!/bin/bash
#SBATCH --job-name=r2_gpqa_clean
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/sciev/runs/sci/%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/runs/sci/%x_%j.err
#
# Round 2 (M3 clean + M6 options-only):
#   1. Regenerate GPQA eval files with the current converter (duplicate-option
#      items are excluded and logged -> gpqa_*_clean.jsonl).
#   2. Build options-only variants (question stripped; ctx keeps only the
#      instruction line) to measure option-side/parametric cues.
#   3. Eval c_choice with per-item predictions on:
#      clean main, clean diamond, options-only main, options-only diamond.

set -euo pipefail
RJ=/beegfs/a474r867/sciev
BAT=$RJ/data/sci_battery
EXT=$RJ/data/bench_external
RUNS=$RJ/runs/sci
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
LAYERS=${LAYERS:--1,-9,-17,-25}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=/beegfs/a474r867/pylibs/peft013:/beegfs/a474r867/pylibs/tf446:$RJ
cd "$RJ"

echo "=== STEP 1: regenerate clean GPQA eval files (dup options excluded) ==="
for CFG in main diamond; do
    SRC="$EXT/gpqa/dataset/gpqa_${CFG}.csv"
    DST="$EXT/gpqa/gpqa_${CFG}_choice_eval_clean.jsonl"
    if [ ! -e "$DST" ]; then
        "$PY" data/convert_benchmarks.py --bench gpqa --src "$SRC" --out "$DST"
    fi
done

echo "=== STEP 2: options-only variants (ctx = instruction only) ==="
"$PY" - <<'PYEOF'
import json, os
os.environ.setdefault("HF_HOME", "/beegfs/a474r867/hf-cache")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct",
                                    local_files_only=True)
INSTR = "Which answer is correct?"
for cfg in ("main", "diamond"):
    src = f"/beegfs/a474r867/sciev/data/bench_external/gpqa/gpqa_{cfg}_choice_eval_clean.jsonl"
    dst = f"/beegfs/a474r867/sciev/data/bench_external/gpqa/gpqa_{cfg}_choice_eval_optonly.jsonl"
    if os.path.exists(dst):
        continue
    ctx = tok.encode(f"Question: {INSTR}", add_special_tokens=False)
    n = 0
    with open(dst, "w") as fo:
        for line in open(src):
            r = json.loads(line)
            fo.write(json.dumps({**r, "ctx": ctx,
                                 "qid": r.get("qid", "") + "__optonly"}) + "\n")
            n += 1
    print(f"wrote {dst} n={n}")
PYEOF

echo "=== STEP 3: evals ==="
run_eval () {  # evalfile tag
    local ev=$1 tag=$2
    local dst="$RUNS/eval_r2_${tag}.json"
    if [ -e "$dst" ]; then echo "skip existing $dst"; return; fi
    echo "=== EVAL $tag ==="
    "$PY" -m sciev.eval \
        --ckpt "$RUNS/c_choice/decision.pt" \
        --decision-type choice --seed 7331 \
        --r2-mode spanpool --r2-layers="$LAYERS" --canonical-order \
        --r2-temp-fit-decisions "$BAT/sci_choice_dev.jsonl" \
        --decisions-eval "$ev" --predictions \
        --device cuda --out "$dst"
}

run_eval "$EXT/gpqa/gpqa_main_choice_eval_clean.jsonl"       gpqa_main_clean
run_eval "$EXT/gpqa/gpqa_diamond_choice_eval_clean.jsonl"    gpqa_diamond_clean
run_eval "$EXT/gpqa/gpqa_main_choice_eval_optonly.jsonl"     gpqa_main_optonly
run_eval "$EXT/gpqa/gpqa_diamond_choice_eval_optonly.jsonl"  gpqa_diamond_optonly

echo "ALL_DONE"
