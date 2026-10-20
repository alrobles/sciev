#!/bin/bash
#SBATCH --job-name=sciev_litdump_index
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:30:00
#SBATCH --output=/beegfs/a474r867/sciev/logs/litdump_%x_%j.out
#SBATCH --error=/beegfs/a474r867/sciev/logs/litdump_%x_%j.err

# Chunked FTS5 litdump index for grounded-retrieval corpora.
#   sbatch --export=ALL,HF_SPEC='wikimedia/wikipedia|20231101.en|train',DB=/beegfs/a474r867/litdump/wiki/wiki_full.db,SOURCE='wiki|en-20231101' scripts/litdump_index.slurm
#   sbatch --export=ALL,IN=/path/records.jsonl,DB=/path/out.db,SOURCE='papers|papersdb',EXTRA='--latex' scripts/litdump_index.slurm
# Writes $DB.partial then atomically renames to $DB on success.

set -euo pipefail
RJ=${RJ:-/beegfs/a474r867/sciev}
CODE=${CODE:-$RJ}
PY=${PY:-/beegfs/a474r867/bioai/venv/bin/python}
export HF_HOME=/beegfs/a474r867/hf-cache
export PYTHONPATH=$CODE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

: "${DB:?set DB=/path/to/out.db}"
: "${SOURCE:?set SOURCE=corpus-tag}"
EXTRA=${EXTRA:-}
INPUT_ARGS=()
if [ -n "${HF_SPEC:-}" ]; then
    INPUT_ARGS=(--hf "$HF_SPEC")
elif [ -n "${IN:-}" ]; then
    INPUT_ARGS=(--in $IN)
else
    echo "set HF_SPEC=NAME|CONFIG|SPLIT or IN=/path/records.jsonl" >&2
    exit 1
fi

mkdir -p "$(dirname "$DB")" "$RJ/logs"
[ -f "$DB" ] && { echo "refusing overwrite: $DB"; exit 0; }
rm -f "$DB.partial" "$DB.partial-wal" "$DB.partial-shm"
printf 'TRACE_LITDUMP job=%s input=%s db=%s source=%s extra=%s\n' \
    "$SLURM_JOB_ID" "${HF_SPEC:-$IN}" "$DB" "$SOURCE" "$EXTRA"
cd "$CODE"
"$PY" data/build_litdump_index.py "${INPUT_ARGS[@]}" --db "$DB.partial" \
    --source "$SOURCE" --progress-every 100000 $EXTRA
mv -n "$DB.partial" "$DB"
echo "INDEX_DONE $DB"
