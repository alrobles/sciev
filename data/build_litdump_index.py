"""Build a chunked SQLite FTS5 index from a JSONL/parquet record stream.

Generic version of the litdump pubmed_full.db builder: every input record
(id, title, text, tags) is split into ~CHUNK_WORDS-word overlapping
passages and indexed under the same schema the retrieval builders
(`data/build_gpqa_retrieved.py --db`) already consume:

    articles_fts(id UNINDEXED, title, abstract, mesh)

`abstract` holds the chunk text; `mesh` holds "<source>|<tags>" so a
single db can mix corpora (wiki|en-20231101, arxiv|quant-ph, ...).

Usage:
    python data/build_litdump_index.py --in records.jsonl --db wiki_full.db \
        --source wiki --progress-every 200000
    python data/build_litdump_index.py --hf "wikimedia/wikipedia|20231101.en|train" \
        --db wiki_full.db --source wiki

Input JSONL fields: id|doc_id, title, text|abstract, tags (optional).
--hf streams NAME|CONFIG|SPLIT via `datasets` straight into the index.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

CHUNK_WORDS, CHUNK_OVERLAP = 350, 60
SQL_META = ("CREATE TABLE IF NOT EXISTS articles (rowid INTEGER PRIMARY KEY,"
            " doc_id TEXT, source TEXT, title TEXT, chunk_id INTEGER, mesh TEXT)")
SQL_FTS = ("CREATE VIRTUAL TABLE articles_fts USING fts5("
           "id UNINDEXED, title, abstract, mesh)")

LATEX_COMMENT = re.compile(r"(?<!\\)%[^\n]*")
LATEX_ENV = re.compile(r"\\(?:begin|end)\s*\{[^}]*\}(?:\[[^\]]*\])?")
LATEX_DISPLAY = re.compile(r"\$\$[^$]*\$\$|\\\[.*?\\\]", re.S)
LATEX_INLINE = re.compile(r"\$[^$]*\$")
LATEX_CMD = re.compile(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?")
LATEX_JUNK = re.compile(r"[{}^_~&]")
WHITESPACE = re.compile(r"\s+")


def scrub_latex(text):
    """Light LaTeX → prose scrub: keeps natural-language words, drops
    commands/preamble/math so FTS5 token stats reflect real text."""
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    text = LATEX_COMMENT.sub(" ", text)
    text = LATEX_ENV.sub(" ", text)
    text = LATEX_DISPLAY.sub(" ", text)
    text = LATEX_INLINE.sub(" ", text)
    text = LATEX_CMD.sub(" ", text)
    text = LATEX_JUNK.sub(" ", text)
    return WHITESPACE.sub(" ", text)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def iter_parquet(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["id", "title", "text", "tags"])
    for record in table.to_pylist():
        yield record


def iter_hf(spec):
    from datasets import load_dataset

    name, config, split = spec.split("|", 2)
    ds = load_dataset(name, config, split=split, streaming=True)
    yield from ds


def norm_record(record):
    doc_id = str(record.get("id") or record.get("doc_id")
                 or record.get("key") or "")
    title = str(record.get("title") or "")
    text = str(record.get("text") or record.get("abstract") or "")
    tags = record.get("tags") or record.get("categories") or record.get("domain")
    mesh = "|".join(tags) if isinstance(tags, (list, tuple)) else str(tags or "")
    return doc_id, title.strip(), text.strip(), mesh


def chunks(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        yield " ".join(words), 0
        return
    step = size - overlap
    for index, start in enumerate(range(0, len(words) - overlap, step)):
        yield " ".join(words[start:start + size]), index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", default=[],
                    help="input .jsonl / .parquet files")
    ap.add_argument("--hf", dest="hf_specs", nargs="+", default=[],
                    help="HF dataset specs NAME|CONFIG|SPLIT (streamed)")
    ap.add_argument("--db", required=True)
    ap.add_argument("--source", required=True, help="source tag, e.g. wiki|en-20231101")
    ap.add_argument("--min-words", type=int, default=80,
                    help="drop documents/chunks shorter than this")
    ap.add_argument("--latex", action="store_true",
                    help="scrub LaTeX commands/preamble/math before chunking")
    ap.add_argument("--progress-every", type=int, default=50000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        raise SystemExit(f"refusing to overwrite existing index: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(SQL_META)
    conn.execute(SQL_FTS)

    n_docs = n_chunks = 0
    started = time.time()
    sources = [(p, None) for p in args.inputs] + [(None, spec) for spec in args.hf_specs]
    if not sources:
        raise SystemExit("no inputs: pass --in or --hf")
    for input_path, hf_spec in sources:
        if hf_spec is not None:
            records = iter_hf(hf_spec)
        else:
            records = (iter_parquet(input_path) if str(input_path).endswith(".parquet")
                       else iter_jsonl(input_path))
        batch_meta, batch_fts = [], []
        for record in records:
            doc_id, title, text, mesh = norm_record(record)
            if args.latex:
                text = scrub_latex(text)
            if not doc_id or len(text.split()) < args.min_words:
                continue
            full_mesh = f"{args.source}|{mesh}" if mesh else args.source
            seen, chunk_rows = set(), 0
            for text_chunk, chunk_id in chunks(text):
                if text_chunk in seen or len(text_chunk.split()) < args.min_words:
                    continue
                seen.add(text_chunk)
                fts_id = f"{doc_id}:{chunk_id}"
                batch_meta.append((doc_id, args.source, title, chunk_id, full_mesh))
                batch_fts.append((fts_id, title, text_chunk, full_mesh))
                chunk_rows += 1
            n_docs += 1
            n_chunks += chunk_rows
            if len(batch_fts) >= 2000:
                with conn:
                    conn.executemany("INSERT INTO articles VALUES (NULL,?,?,?,?,?)", batch_meta)
                    conn.executemany("INSERT INTO articles_fts VALUES (?,?,?,?)", batch_fts)
                batch_meta, batch_fts = [], []
            if n_docs % args.progress_every == 0:
                rate = n_docs / max(1e-9, time.time() - started)
                print(f"{n_docs} docs, {n_chunks} chunks, {rate:.0f} docs/s", flush=True)
            if args.limit and n_docs >= args.limit:
                break
        if batch_fts:
            with conn:
                conn.executemany("INSERT INTO articles VALUES (NULL,?,?,?,?,?)", batch_meta)
                conn.executemany("INSERT INTO articles_fts VALUES (?,?,?,?)", batch_fts)
        if args.limit and n_docs >= args.limit:
            break

    conn.execute("PRAGMA optimize")
    conn.close()
    print(json.dumps({"db": str(db_path), "docs": n_docs, "chunks": n_chunks,
                      "elapsed_s": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
