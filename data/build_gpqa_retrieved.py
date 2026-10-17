"""Build retrieved-evidence GPQA eval variants from a PubMed FTS5 index.

Same pairing contract as build_gpqa_grounded.py — identical options and
ordering, only the state gains an evidence block — but the evidence is
real retrieval (BM25 over litdump/pubmed_full.db) instead of the paired
oracle explanation or nev trace. This prices practical retrieval on the
oracle--nev axis of the grounded-GPQA experiment.

Usage:
    python data/build_gpqa_retrieved.py \
        --text-file .../gpqa_main_choice_eval_clean_text.jsonl \
        --db /beegfs/a474r867/litdump/pubmed/pubmed_full.db \
        --out .../gpqa_main_choice_eval_retrieved.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from sciev.data import input_fingerprints, stable_json, write_dataset
from sciev.decisions import InputOverflow, encode_question, validate_decision_row

MAX_CTX, MAX_OPT = 896, 120  # wider than clean: evidence must survive
MAX_TERMS, MAX_DOC_CHARS, EVIDENCE_CHARS = 24, 600, 2400
META_KEYS = ("split_group", "group_id", "pid", "source", "source_file",
             "source_line", "source_row", "group_scope", "content_hash",
             "sample_id", "reasoning_type", "meta", "label_status",
             "evidence_regime", "task", "split")
STOPWORDS = frozenset(
    "a an the and or of to in is are was were be been for on at by with "
    "that this these those which what when where who whom how why not no "
    "it its he she they we you i do does did have has had can could will "
    "would should may might must shall from as if then than so such into "
    "about between through during before after above below up down out "
    "over under again further once each other some any all both few more "
    "most only own same very just also".split())


def question_text(state):
    text = state.strip()
    if text.lower().startswith("question:"):
        text = text.split(":", 1)[1]
    return text.strip()


def query_terms(text, max_terms=MAX_TERMS):
    terms, seen = [], set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text.lower()):
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:max_terms]


def find_fts(conn):
    """Locate the FTS5 table and its (title, body) column pair."""
    candidates = []
    for name, sql in conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE sql LIKE '%fts5%'").fetchall():
        cols = [row[1] for row in conn.execute(
            f"PRAGMA table_info({name})").fetchall()]
        if not cols:
            continue
        low = [c.lower() for c in cols]
        title = cols[low.index("title")] if "title" in low else cols[0]
        body = cols[low.index("abstract")] if "abstract" in low else cols[-1]
        score = ("pubmed" in name.lower()) + ("article" in name.lower())
        candidates.append((score, name, title, body, cols))
    if not candidates:
        raise ValueError("no FTS5 table found in db")
    _, name, title, body, cols = sorted(candidates, reverse=True)[0]
    return name, title, body, cols


def retrieve(conn, table, title_col, body_col, terms, top_k):
    if not terms:
        return []
    sql = (f"SELECT rowid, {title_col}, {body_col}, bm25({table}) "
           f"FROM {table} WHERE {table} MATCH ? "
           f"ORDER BY bm25({table}) LIMIT ?")
    # Rare-first two-stage retrieval. A flat OR scans every posting
    # list and never finishes on a 30M-doc index; intersections are
    # nearly free. Stage 1 ANDs the 4 longest terms; stage 2 ORs
    # ANDed pairs ("t1" "t2") OR ("t3" "t4") ... so every posting-list
    # merge stays tiny. Questions still empty after pairs are excluded.
    rare = sorted(terms, key=len, reverse=True)
    and_terms = rare[:4]
    match = " ".join(f'"{t}"' for t in and_terms)
    rows = conn.execute(sql, (match, top_k)).fetchall()
    stage = "and"
    if len(rows) < top_k and len(rare) >= 2:
        stage = "pairs"
        pair_terms = rare[:8] if len(rare) >= 8 else rare[:]
        pairs = [pair_terms[i:i + 2] for i in range(0, len(pair_terms), 2)]
        match = " OR ".join(
            "(" + " ".join(f'"{t}"' for t in p) + ")" for p in pairs)
        seen = {r[0] for r in rows}
        rows = rows + [r for r in conn.execute(sql, (match, top_k * 2))
                       if r[0] not in seen][:top_k]
        rows = rows[:top_k]
    docs = []
    for rowid, title, body, score in rows:
        docs.append({"rowid": rowid, "bm25": score,
                     "title": (title or "").strip(),
                     "text": (body or "").strip()[:MAX_DOC_CHARS]})
    return docs, stage, and_terms


def build_retrieved(records, conn, fts, tok, *, top_k, max_ctx, max_opt):
    table, title_col, body_col, _ = fts
    ids, texts, exclusions = [], [], []
    for i, rec in enumerate(records):
        qid, question = next(iter(rec["questions"].items()))
        if len(rec["questions"]) != 1 or question.get("type") != "choice":
            raise ValueError("retrieved variants require one choice question per state")
        terms = query_terms(question_text(rec["state"]))
        t0 = time.time()
        docs, stage, and_terms = retrieve(
            conn, table, title_col, body_col, terms, top_k)
        print(f"[retrieved] {i + 1}/{len(records)} {qid} "
              f"docs={len(docs)} stage={stage} "
              f"dt={time.time() - t0:.1f}s", flush=True)
        if not docs:
            exclusions.append({"qid": qid, "sample_id": rec.get("sample_id"),
                               "reason": "no_retrieved_evidence",
                               "terms": terms})
            continue
        evidence = "\n\n".join(
            f"Title: {d['title']}\n{d['text']}" for d in docs)[:EVIDENCE_CHARS]
        state = f"{rec['state']}\n\nEvidence: {evidence}"
        try:
            encoded = encode_question(tok, state, question,
                                      max_ctx=max_ctx, max_opt=max_opt,
                                      overflow="truncate")
        except InputOverflow:
            raise AssertionError("overflow=truncate must not raise")
        metadata = {key: question.get(key, rec.get(key)) for key in META_KEYS
                    if key in question or key in rec}
        metadata.update(evidence_regime="retrieved_bm25",
                        task="gpqa_choice_grounded",
                        evidence_arm="retrieved",
                        evidence_field="pubmed_bm25",
                        retrieval={"db": table, "terms": terms,
                                   "stage": stage, "and_terms": and_terms,
                                   "docs": [{"rowid": d["rowid"],
                                             "bm25": d["bm25"],
                                             "title": d["title"][:120]}
                                            for d in docs]})
        identity = hashlib.sha256(stable_json(encoded).encode()).hexdigest()
        grounded_qid = f"{qid}_gr-retrieved"
        row = validate_decision_row({**metadata, **encoded,
                                     "gold": encoded["option_keys"].index(question["label"]),
                                     "qid": grounded_qid, "decision_id": identity})
        ids.append(row)
        text_meta = {**metadata, "encoding": row["encoding"],
                     "schema_version": row["schema_version"]}
        texts.append({**text_meta, "state": state, "questions": {
            grounded_qid: {**text_meta, **question, "kind": row["kind"],
                           "option_keys": row["option_keys"],
                           "decision_id": identity}}})
    return ids, texts, exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True, help="clean *_text.jsonl")
    ap.add_argument("--db", required=True, help="pubmed_full.db (FTS5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    records = [json.loads(line) for line in Path(args.text_file).open()
               if line.strip()]
    if not records:
        ap.error("text-file must be nonempty")
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    fts = find_fts(conn)
    ids, texts, exclusions = build_retrieved(
        records, conn, fts, tok, top_k=args.top_k,
        max_ctx=args.max_ctx, max_opt=args.max_opt)
    conn.close()
    truncated = sum(1 for row in ids
                    if (row.get("truncation") or {}).get("context_tokens"))
    stat = Path(args.db).stat()
    manifest = {"builder": "build_gpqa_retrieved", "builder_version": 3,
                "evidence_arm": "retrieved",
                "retrieval": {"db": args.db, "db_bytes": stat.st_size,
                              "db_mtime": stat.st_mtime, "fts_table": fts[0],
                              "title_col": fts[1], "body_col": fts[2],
                              "columns": fts[3], "top_k": args.top_k},
                "inputs": input_fingerprints([args.text_file]),
                "tokenizer": args.tokenizer,
                "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                "exclusions": exclusions,
                "counts": {"input_records": len(records), "rows": len(ids),
                           "ctx_truncated": truncated}}
    out = Path(args.out)
    write_dataset({out: ids,
                   out.with_name(f"{out.stem}_text.jsonl"): texts},
                  out.with_suffix(".manifest.json"), manifest)
    print(f"[wrote] {out}: {len(ids)} rows ({truncated} ctx-truncated, "
          f"{len(exclusions)} excluded)", flush=True)


if __name__ == "__main__":
    main()
