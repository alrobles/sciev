"""Build sciev-QA: a legitimate synthetic MC-QA benchmark from papers_db fulltext.

Legitimacy is engineered, not claimed — every emitted item carries a
machine-checkable certificate:

  * sufficiency: an independent validator answers correctly WITH the passage
    (greedy) → the evidence is sufficient.
  * necessity: the same validator answers blind (greedy + 2 sampled) → keep
    only items where blind hits ≤1/3 → the evidence is necessary.
  * provenance: pid → papers_db document → FTS chunk recall@k measurable.
  * generator and validator are different model families.

Stages (one slurm job runs them in order):
  passages  CPU: sample papers_db → passages.jsonl {pid,title,domain,passage}
  generate  vLLM: teacher MC gen → qa_gen.jsonl
  certify   vLLM: validator gates → qa_cert.jsonl
  emit      CPU: → sciev_qa_v1_{clean,evidence}.jsonl + _text + manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_litdump_index import scrub_latex  # noqa: E402

PASSAGE_WORDS, PASSAGE_MIN = 800, 300
GEN_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
VAL_MODEL = "allenai/OLMo-2-0325-32B-Instruct"
LETTERS = "ABCD"

GEN_PROMPT = """You are given a passage from a scientific paper. Write ONE graduate-level
multiple-choice question that can be answered ONLY using this passage.

Requirements:
- A strong scientist WITHOUT this passage should NOT be able to answer:
  the question must depend on specific methods, values, or findings in it.
- Exactly 4 options, exactly one correct. The three distractors must be
  plausible to an expert (same domain vocabulary, plausible mechanism,
  value or conclusion) but contradicted or unsupported by the passage.
- The question must read like a standalone exam question — do not mention
  "the passage", "the paper" or "the authors".
- Output STRICT JSON only:
{{"question": "...", "options": ["...", "...", "...", "..."],
 "answer": 0, "type": "definitional|relational|numerical|negation"}}

Passage:
\"\"\"
{passage}
\"\"\"
JSON:"""

BLIND_PROMPT = """Answer this graduate-level multiple-choice question.
Reply with the single letter of the correct option only.

{question}

{options}
Answer:"""

EV_PROMPT = """Use the passage to answer this multiple-choice question.
Reply with the single letter of the correct option only.

Passage:
\"\"\"
{passage}
\"\"\"

{question}

{options}
Answer:"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_LETTER = re.compile(r"^\s*([A-D])\b")


def iter_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def extract_passage(text, words=PASSAGE_WORDS, rng=None):
    """Random window of `words` words from the middle of a scrubbed doc."""
    ws = text.split()
    if len(ws) < PASSAGE_MIN:
        return None
    lo, hi = int(len(ws) * 0.15), max(int(len(ws) * 0.15), len(ws) - words)
    start = (rng or random).randint(lo, hi) if hi > lo else 0
    return " ".join(ws[start:start + words])


def stage_passages(args):
    rng = random.Random(args.seed)
    reservoir, seen = [], 0
    for record in iter_jsonl(args.papers_db):
        if record.get("kind") != "fulltext":
            continue
        seen += 1
        if len(reservoir) < args.n_passages:
            reservoir.append(record)
        else:
            j = rng.randrange(seen)
            if j < args.n_passages:
                reservoir[j] = record
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        n = 0
        for record in reservoir:
            passage = extract_passage(scrub_latex(str(record.get("text") or "")), rng=rng)
            if not passage:
                continue
            row = {"pid": record["key"], "title": record.get("title") or "",
                   "domain": record.get("domain") or "",
                   "passage": passage,
                   "passage_sha256": hashlib.sha256(passage.encode()).hexdigest()}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(json.dumps({"stage": "passages", "docs_seen": seen, "emitted": n,
                      "out": str(out)}))


def parse_gen(text):
    block = _JSON_BLOCK.search(text)
    if not block:
        return None
    try:
        item = json.loads(block.group(0))
    except json.JSONDecodeError:
        return None
    options = item.get("options")
    if (not isinstance(options, list) or len(options) != 4
            or len(set(map(str.strip, map(str, options)))) != 4
            or not isinstance(item.get("answer"), int)
            or not 0 <= item["answer"] < 4
            or not str(item.get("question") or "").strip()):
        return None
    return {"question": item["question"].strip(),
            "options": [str(o).strip() for o in options],
            "gold": item["answer"], "qtype": str(item.get("type") or "")}


def llm(args, model):
    from vllm import LLM
    kw = {"model": model, "dtype": args.dtype, "max_model_len": args.maxlen,
          "gpu_memory_utilization": 0.90}
    if args.quantization:
        kw["quantization"] = args.quantization
    return LLM(**kw)


def gen_prompts(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    def wrap(prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            return prompt
    return wrap


def stage_generate(args):
    from vllm import SamplingParams
    rows = list(iter_jsonl(args.passages))
    rows = rows[args.shard::args.nshards]
    wrap = gen_prompts(args.model)
    model = llm(args, args.model)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=700)
    out = Path(args.out)
    done_pids = {r["pid"] for r in iter_jsonl(out)} if out.exists() else set()
    rows = [r for r in rows if r["pid"] not in done_pids]
    if args.limit:
        rows = rows[:args.limit]
    n_ok = 0
    with out.open("a", encoding="utf-8") as fh:
        for i in range(0, len(rows), args.maxdocs):
            wave = rows[i:i + args.maxdocs]
            outs = model.generate([wrap(GEN_PROMPT.format(passage=r["passage"]))
                                   for r in wave], sp)
            for row, res in zip(wave, outs):
                item = parse_gen(res.outputs[0].text)
                if item is None:
                    continue
                fh.write(json.dumps({**row, **item, "teacher": args.model},
                                    ensure_ascii=False) + "\n")
                n_ok += 1
            fh.flush()
            print(f"[gen] {i + len(wave)}/{len(rows)} kept={n_ok}", flush=True)
    print(json.dumps({"stage": "generate", "emitted": n_ok, "out": str(out)}))


def letter_of(text):
    m = _LETTER.match(text.strip()) or _LETTER.search(text.strip())
    return LETTERS.find(m.group(1)) if m else -1


def stage_certify(args):
    from vllm import SamplingParams
    rows = list(iter_jsonl(args.gen))
    wrap = gen_prompts(args.model)
    model = llm(args, args.model)
    greedy = SamplingParams(temperature=0, max_tokens=8)
    sampled = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=8, n=2)
    def fmt_opts(item):
        return "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(item["options"]))
    prompts_ev, prompts_blind, prompts_blind2 = [], [], []
    for r in rows:
        opts = fmt_opts(r)
        prompts_ev.append(wrap(EV_PROMPT.format(passage=r["passage"],
                                              question=r["question"], options=opts)))
        prompts_blind.append(wrap(BLIND_PROMPT.format(question=r["question"],
                                                     options=opts)))
        prompts_blind2.append(wrap(BLIND_PROMPT.format(question=r["question"],
                                                      options=opts)))
    outs_ev = model.generate(prompts_ev, greedy)
    outs_b1 = model.generate(prompts_blind, greedy)
    outs_b2 = model.generate(prompts_blind2, sampled)
    out = Path(args.out)
    n_cert = 0
    with out.open("w", encoding="utf-8") as fh:
        for row, o_ev, o_b1, o_b2 in zip(rows, outs_ev, outs_b1, outs_b2):
            pred_ev = letter_of(o_ev.outputs[0].text)
            blinds = [letter_of(o_b1.outputs[0].text)]
            blinds += [letter_of(o.text) for o in o_b2.outputs]
            blind_hits = sum(1 for p in blinds if p == row["gold"])
            gates = {"validator": args.model, "pred_evidence": pred_ev,
                     "pred_blind": blinds, "evidence_sufficient": pred_ev == row["gold"],
                     "evidence_necessary": blind_hits <= 1}
            if gates["evidence_sufficient"] and gates["evidence_necessary"]:
                fh.write(json.dumps({**row, "gates": gates}, ensure_ascii=False) + "\n")
                n_cert += 1
    print(json.dumps({"stage": "certify", "input": len(rows),
                      "certified": n_cert, "out": str(out)}))


def stage_emit(args):
    from sciev.data import (add_decision, content_fingerprint, input_fingerprints,
                            record_metadata, write_dataset)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    items = list(iter_jsonl(args.cert))
    out = Path(args.out)
    for arm in ("clean", "evidence"):
        id_rows, text_by_state, exclusions = [], {}, []
        arm_out = out.with_name(f"{out.stem}_{arm}.jsonl")
        for row in items:
            options = list(row["options"])
            rng.shuffle(options)
            gold_text = row["options"][row["gold"]]
            split = ("eval" if int(content_fingerprint([args.seed, row["pid"]])[:8], 16) % 10 < 4
                     else "dev")
            metadata = dict(record_metadata(row, "sciev_qa", "passage_mcq"),
                            meta=f"sciev-qa-v1|{row['domain']}", label_status="certified_synthetic",
                            evidence_regime=("gold_passage" if arm == "evidence"
                                             else "no_passage_parametric"),
                            task="sciev_qa", split=split, pid=row["pid"],
                            teacher=row.get("teacher"), gates=row.get("gates"),
                            passage_sha256=row.get("passage_sha256"))
            state = (f"Question: {row['question']}" if arm == "clean"
                     else f"{row['passage']}\n\nQuestion: {row['question']}")
            add_decision(id_rows, text_by_state, tok, state,
                         f"scievqa_{row['pid']}",
                         {"type": "choice", "instructions": "Which answer is correct?",
                          "criteria": {o: None for o in options}, "label": gold_text},
                         options.index(gold_text), metadata,
                         args.max_ctx, args.max_opt, "exclude", exclusions)
        manifest = {"builder": "build_sciev_qa", "builder_version": 1, "arm": arm,
                    "inputs": input_fingerprints([str(args.cert)]),
                    "tokenizer": args.tokenizer, "max_ctx": args.max_ctx,
                    "max_opt": args.max_opt, "exclusions": exclusions,
                    "counts": {"certified_items": len(items), "rows": len(id_rows)}}
        report = write_dataset(
            {arm_out: id_rows,
             arm_out.with_name(f"{arm_out.stem}_text.jsonl"): list(text_by_state.values())},
            arm_out.with_suffix(".manifest.json"), manifest)
        print(f"[wrote] {arm_out}: {len(id_rows)} rows")
    print(json.dumps({"stage": "emit", "out": str(out)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("passages", "generate", "certify", "emit"))
    ap.add_argument("--papers-db", default="/beegfs/a474r867/ecoreasoner/data/papersdb/papers_db.jsonl")
    ap.add_argument("--passages", default="data/sciev_qa/passages.jsonl")
    ap.add_argument("--gen", default="data/sciev_qa/qa_gen.jsonl")
    ap.add_argument("--cert", default="data/sciev_qa/qa_cert.jsonl")
    ap.add_argument("--out", default="data/sciev_qa/sciev_qa_v1.jsonl")
    ap.add_argument("--n-passages", type=int, default=3000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--maxlen", type=int, default=8192)
    ap.add_argument("--maxdocs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0,
                    help="generate stage: cap number of passages processed")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max-ctx", type=int, default=896)
    ap.add_argument("--max-opt", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7331)
    args = ap.parse_args()
    if args.stage == "generate":
        args.model = args.model or GEN_MODEL
        args.out = args.gen
        stage_generate(args)
    elif args.stage == "certify":
        args.model = args.model or VAL_MODEL
        args.out = args.cert
        stage_certify(args)
    elif args.stage == "passages":
        stage_passages(args)
    else:
        stage_emit(args)


if __name__ == "__main__":
    main()
