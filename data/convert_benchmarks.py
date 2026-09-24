#!/usr/bin/env python3
"""Convert public benchmarks into reverse-jev decision rows.

Outputs the same {ctx:[ids], opts:[[ids]...], gold, kind, qid} format as
build_sci_decisions.py, plus a text row file for --remote eval.

Benchmarks:
  gpqa    csv (Question, Correct Answer, Incorrect Answer 1-3) -> choice K=4
          ctx = "Question: ..." (no passage: parametric reasoning)
  scifact parquet (claim, title, abstract[list], verdict) -> noul K=2,
          score K=3. ctx = title + abstract + claim.
          noul: SUPPORT->yes, CONTRADICT/NEI->no
          score: SUPPORT->2, CONTRADICT->1, NEI->0

Usage:
  python convert_benchmarks.py --bench gpqa \
      --src bench_external/gpqa/dataset/gpqa_main.csv \
      --out bench_external/gpqa/gpqa_main_choice_eval.jsonl
  python convert_benchmarks.py --bench scifact \
      --src bench_external/scifact/validation.parquet \
      --out-prefix bench_external/scifact/scifact_dev
"""
import argparse
import csv
import json
import random
from pathlib import Path

INSTR_CHOICE = "Which answer is correct?"
INSTR_NOUL = "Is the claim supported by the passage?"
SCORE_LEGEND = ["unrelated answer", "related but wrong answer",
                "correct answer"]
MAX_CTX = 640
MAX_OPT = 120


def enc(tok, text, cap):
    return tok.encode(text, add_special_tokens=False)[:cap]


def write_rows(path, id_rows, text_rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r) for r in id_rows) + "\n")
    tp = Path(str(path).replace(".jsonl", "_text.jsonl"))
    tp.write_text("\n".join(json.dumps(r) for r in text_rows) + "\n")
    print(f"[wrote] {path}: {len(id_rows)}  (+text {tp.name})")


def conv_gpqa(src, out, tok, rng):
    id_rows, text_by_state = [], {}
    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        q, gold = r["Question"].strip(), r["Correct Answer"].strip()
        distr = [r[f"Incorrect Answer {j}"].strip() for j in (1, 2, 3)]
        if not q or not gold or any(not d for d in distr):
            continue
        opts = distr + [gold]
        rng.shuffle(opts)
        gold_i = opts.index(gold)
        state = f"Question: {q}"
        ctx = enc(tok, f"{state}\nQuestion: {INSTR_CHOICE}", MAX_CTX)
        id_rows.append({"ctx": ctx,
                        "opts": [enc(tok, o, MAX_OPT) for o in opts],
                        "gold": gold_i, "kind": "choice",
                        "qid": f"gpqa_{i}",
                        "meta": r.get("Subdomain", "")})
        ts = text_by_state.setdefault(state, {"state": state,
                                              "questions": {}})
        ts["questions"][f"gpqa_{i}"] = {
            "type": "choice", "instructions": INSTR_CHOICE,
            "criteria": {o: None for o in opts}, "label": gold}
    write_rows(out, id_rows, list(text_by_state.values()))


def conv_scifact(src, out_prefix, tok, rng):
    import pandas as pd
    df = pd.read_parquet(src)
    noul_rows, score_rows = [], []
    text_by_state = {}
    split = Path(src).stem
    for i, r in df.iterrows():
        claim = str(r["claim"]).strip()
        title = str(r["title"]).strip()
        abstract = " ".join(str(s) for s in r["abstract"]).strip()
        verdict = str(r["verdict"])           # SUPPORT / CONTRADICT / NEI
        if not claim or not abstract:
            continue
        state = (f"Title: {title}\nAbstract: {abstract}\n"
                 f"Claim: {claim}")
        ctx = enc(tok, f"{state}\nQuestion: {INSTR_NOUL}", MAX_CTX)
        yes = enc(tok, "yes", MAX_OPT)
        no = enc(tok, "no", MAX_OPT)
        supported = verdict == "SUPPORT"
        noul_rows.append({"ctx": ctx, "opts": [yes, no],
                          "gold": 0 if supported else 1, "kind": "noul",
                          "qid": f"sf_{split}_{i}_n"})
        score_gold = {"SUPPORT": 2, "CONTRADICT": 1, "NEI": 0}[verdict]
        score_rows.append({"ctx": ctx, "opts": [
            enc(tok, x, MAX_OPT) for x in SCORE_LEGEND],
            "gold": score_gold, "kind": "score",
            "qid": f"sf_{split}_{i}_s"})
        ts = text_by_state.setdefault(state, {"state": state,
                                              "questions": {}})
        ts["questions"][f"sf_{split}_{i}_n"] = {
            "type": "noul", "instructions": INSTR_NOUL, "label": supported}
        ts["questions"][f"sf_{split}_{i}_s"] = {
            "type": "score", "instructions": "Rate the claim.",
            "criteria": SCORE_LEGEND, "label": score_gold}
    write_rows(f"{out_prefix}_noul_eval.jsonl", noul_rows,
               [{"state": t["state"],
                 "questions": {k: v for k, v in t["questions"].items()
                               if k.endswith("_n")}}
                for t in text_by_state.values()])
    write_rows(f"{out_prefix}_score_eval.jsonl", score_rows,
               [{"state": t["state"],
                 "questions": {k: v for k, v in t["questions"].items()
                               if k.endswith("_s")}}
                for t in text_by_state.values()])


def conv_classification(src, out, tok, rng, cfg):
    """Generic K-way classification -> choice rows.

    cfg: {text_field, label_names, ctx_fmt, question, label_field?,
          labels_file?}. Options = label_names in fixed order, gold = label.
    """
    labels = cfg.get("label_names")
    if cfg.get("labels_file"):
        labels = [x.replace("_", " ")
                  for x in json.loads(Path(cfg["labels_file"]).read_text())]
    if src.endswith(".parquet"):
        import pandas as pd
        rows = pd.read_parquet(src).to_dict("records")
    else:
        rows = [json.loads(l) for l in open(src, encoding="utf-8")
                if l.strip()]
    id_rows, text_by_state = [], {}
    skipped = 0
    for i, r in enumerate(rows):
        text = str(r[cfg["text_field"]]).strip()
        gold = int(r[cfg["label_field"]])
        if not text or not (0 <= gold < len(labels)):
            skipped += 1
            continue
        state = cfg["ctx_fmt"].format(t=text)
        ctx = enc(tok, f"{state}\nQuestion: {cfg['question']}", MAX_CTX)
        id_rows.append({"ctx": ctx,
                        "opts": [enc(tok, o, MAX_OPT) for o in labels],
                        "gold": gold, "kind": "choice",
                        "qid": f"{cfg['tag']}_{i}"})
        ts = text_by_state.setdefault(state, {"state": state,
                                              "questions": {}})
        ts["questions"][f"{cfg['tag']}_{i}"] = {
            "type": "choice", "instructions": cfg["question"],
            "criteria": {o: None for o in labels}, "label": labels[gold]}
    write_rows(out, id_rows, list(text_by_state.values()))
    if skipped:
        print(f"[skip] {skipped} rows")


CLASSIF_CFGS = {
    "sst2": {"text_field": "sentence", "label_field": "label",
             "label_names": ["negative", "positive"],
             "ctx_fmt": "Text: {t}",
             "question": "What is the sentiment of this text?"},
    "ag_news": {"text_field": "text", "label_field": "label",
                "label_names": ["World", "Sports", "Business",
                                "Science/Technology"],
                "ctx_fmt": "Article: {t}",
                "question": "Which category does this article belong to?"},
    "banking77": {"text_field": "text", "label_field": "label",
                  "ctx_fmt": "Customer message: {t}",
                  "question": "What is the customer's intent?"},
    "enron": {"text_field": "text", "label_field": "label",
              "label_names": ["not spam", "spam"],
              "ctx_fmt": "Email: {t}",
              "question": "Is this email spam?"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["gpqa", "scifact",
                                        *CLASSIF_CFGS], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None, help="gpqa: output jsonl")
    ap.add_argument("--out-prefix", default=None,
                    help="scifact: prefix for _noul/_score_eval.jsonl")
    ap.add_argument("--labels-file", default=None,
                    help="json list of label names (banking77)")
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.bench == "gpqa":
        conv_gpqa(args.src, args.out, tok, rng)
    elif args.bench == "scifact":
        conv_scifact(args.src, args.out_prefix, tok, rng)
    else:
        cfg = dict(CLASSIF_CFGS[args.bench], tag=args.bench)
        if args.labels_file:
            cfg["labels_file"] = args.labels_file
        conv_classification(args.src, args.out, tok, rng, cfg)


if __name__ == "__main__":
    main()
