#!/usr/bin/env bash
#SBATCH --job-name=e0-readout
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/sciev/logs/e0_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/e0_%j.err
# =============================================================================
# E0 — readout swap sin entrenar (gate test-and-drop de la linea sciev):
# para cada par (ctx/ok/bad) compara dos scorers sobre el MISMO checkpoint:
#   r1_first_token : ctx + [MASK] -> logit[ok[0]] vs logit[bad[0]]  (Jev-style)
#   legacy_denoise : denoise_loss(ctx+ok) < denoise_loss(ctx+bad)   (Fase-3)
# Si R1 > legacy y > azar: el substrato discrimina, fallaba el readout -> GO E1.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
RJ=/beegfs/a474r867/sciev
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
PY=/bb/bwvenv/bin/python
RUNS=$BASE/runs
OUT=$RJ/runs/e0
mkdir -p "$OUT" "$RJ/logs"

export PYTHONPATH=$RJ

# (tag, checkpoint, config yaml)
read -r -d '' SPECS <<'EOF' || true
f0-span-esqueleto|/beegfs/a474r867/ecoreasoner/runs/f0-span-esqueleto/checkpoint-g10000/model.pt|/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto.yaml
f0-span-v2-weight-tying|/beegfs/a474r867/ecoreasoner/runs/f0-span-v2-weight-tying/checkpoint-g10000/model.pt|/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-v2-weight-tying.yaml
f0-span-esqueleto-v2-piloto|/beegfs/a474r867/ecoreasoner/runs/f0-span-esqueleto-v2-piloto/checkpoint-g10000/model.pt|/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto-v2.yaml
f2-spanes-50k|/beegfs/a474r867/ecoreasoner/runs/f2-spanes/checkpoint-g50000/model.pt|/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-esqueleto.yaml
EOF

for PAIRS in pairs_hard_v4_holdout_clean pairs_hard_v3_eval; do
    echo "=== BATTERY $PAIRS ==="
    while IFS='|' read -r tag ckpt cfg; do
        [ -z "$tag" ] && continue
        echo "--- $tag vs $PAIRS ---"
        apptainer exec --nv --overlay "$OVL":ro "$SIF" \
            "$PY" -m sciev.eval \
            --ckpt "$ckpt" --config "$cfg" \
            --pairs-dir "$RUNS/$PAIRS" \
            --device cuda \
            --out "$OUT/${tag}__${PAIRS}.json"
    done <<< "$SPECS"
done

echo "E0_COMPLETE"
