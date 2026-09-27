"""Build the qa_choice training/dev/eval battery from public MCQA corpora.

Sources (all 4-option, official labeled splits):
  train: ARC-Easy train, ARC-Challenge train, OpenBookQA train, SciQ train,
         MMLU auxiliary_train (subsample), MedMCQA train (subsample)
  dev:   ARC-E/C validation, OpenBookQA validation, SciQ validation,
         MedMCQA validation (subsample)
  eval:  ARC-E/C test, OpenBookQA test, SciQ test

Every item is screened for near-duplicate questions against the clean GPQA
evaluation sets (token-Jaccard >= 0.6 -> excluded, counted) and exact
question+options duplicates are dropped. Emits decisions jsonl +
_text mirrors + manifest like convert_benchmarks.

Usage:
    python data/build_qa_choice.py --out-dir data/qa_battery
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from sciev.data import (add_decision, content_fingerprint, exclude_record,
                        group_records, normalized_content, record_metadata,
                        write_dataset)

INSTR = "Which answer is correct?"
MAX_CTX, MAX_OPT = 640, 120

MMLU_TRAIN_SUB = 20000
MEDMCQA_TRAIN_SUB = 20000
MEDMCQA_DEV_SUB = 1000
SEED = 7331


def _tokset(text):
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def norm_items():
    """Yield unified {question, options[4], gold, source, split} dicts."""
    from datasets import load_dataset

    def emit_opts(texts, gold_letter, labels="ABCD"):
        opts = [str(t).strip() for t in texts]
        gold = labels.index(gold_letter) if gold_letter in labels else int(gold_letter)
        return opts, gold

    for cfg, name in (("ARC-Easy", "arc_easy"), ("ARC-Challenge", "arc_chal")):
        ds = load_dataset("allenai/ai2_arc", cfg)
        for split in ("train", "validation", "test"):
            for r in ds[split]:
                texts = r["choices"]["text"]
                labels = r["choices"]["label"]
                gold = labels.index(r["answerKey"])
                yield {"question": r["question"].strip(), "options": [str(t).strip() for t in texts],
                       "gold": gold, "source": name, "split": split, "src_id": r["id"]}

    ds = load_dataset("allenai/openbookqa", "main")
    for split in ("train", "validation", "test"):
        for r in ds[split]:
            texts = r["question"]["choices"]["text"] if "question" in r and isinstance(r["question"], dict) else r["choices"]["text"]
            labels = r["question"]["choices"]["label"] if "question" in r and isinstance(r["question"], dict) else r["choices"]["label"]
            gold = labels.index(r["answerKey"])
            qtxt = r["question_stem"] if "question_stem" in r else r["question"]["stem"]
            yield {"question": qtxt.strip(), "options": [str(t).strip() for t in texts],
                   "gold": gold, "source": "openbookqa", "split": split, "src_id": r["id"]}

    ds = load_dataset("allenai/sciq")
    for split in ("train", "validation", "test"):
        for r in ds[split]:
            opts = [r["distractor1"], r["distractor2"], r["distractor3"],
                    r["correct_answer"]]
            opts = [str(o).strip() for o in opts]
            yield {"question": r["question"].strip(), "options": opts,
                   "gold": 3, "source": "sciq", "split": split,
                   "src_id": content_fingerprint(r["question"])[:16]}

    ds = load_dataset("cais/mmlu", "auxiliary_train")["train"]
    idxs = random.Random(SEED).sample(range(len(ds)), min(MMLU_TRAIN_SUB, len(ds)))
    for i in idxs:
        r = ds[int(i)]
        if "train" in r and isinstance(r["train"], dict):
            r = r["train"]
        yield {"question": str(r["question"]).strip(),
               "options": [str(o).strip() for o in r["choices"]],
               "gold": int(r["answer"]), "source": "mmlu_aux", "split": "train",
               "src_id": f"mmlu_{i}"}

    ds = load_dataset("openlifescienceai/medmcqa")
    idxs = random.Random(SEED).sample(range(len(ds["train"])),
                                      min(MEDMCQA_TRAIN_SUB, len(ds["train"])))
    for i in idxs:
        r = ds["train"][int(i)]
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        yield {"question": str(r["question"]).strip(),
               "options": [str(o).strip() for o in opts],
               "gold": int(r["cop"]), "source": "medmcqa", "split": "train",
               "src_id": f"med_{r['id']}"}
    idxs = random.Random(SEED).sample(range(len(ds["validation"])),
                                      min(MEDMCQA_DEV_SUB, len(ds["validation"])))
    for i in idxs:
        r = ds["validation"][int(i)]
        opts = [r["opa"], r["opb"], r["opc"], r["opd"]]
        yield {"question": str(r["question"]).strip(),
               "options": [str(o).strip() for o in opts],
               "gold": int(r["cop"]), "source": "medmcqa", "split": "validation",
               "src_id": f"med_{r['id']}"}


def gpqa_question_sets(rj):
    sets = []
    for cfg in ("main", "diamond"):
        p = Path(rj) / "data/bench_external/gpqa/systemone-v2" / \
            f"gpqa_{cfg}_choice_eval_clean_text.jsonl"
        for line in open(p):
            st = json.loads(line)["state"]
            sets.append(_tokset(st))
    return sets


def build(items, split_name, out, tok, rng, gpqa_sets):
    id_rows, text_by_state, exclusions = [], {}, []
    rows = []
    seen = {}
    for it in items:
        if len(it["options"]) != 4 or len({normalized_content(o) for o in it["options"]}) != 4:
            exclusions.append({"reason": "invalid_options", "src_id": it["src_id"]})
            continue
        key = content_fingerprint([it["question"], sorted(it["options"])])
        if key in seen:
            exclusions.append({"reason": "duplicate_record", "src_id": it["src_id"]})
            continue
        seen[key] = 1
        qset = _tokset(it["question"])
        if qset and max((len(qset & g) / len(qset | g) for g in gpqa_sets), default=0) >= 0.6:
            exclusions.append({"reason": "gpqa_near_duplicate", "src_id": it["src_id"],
                               "source": it["source"]})
            continue
        it["sample_id"] = key
        rows.append(it)
    rows = group_records(rows, "question", scope="question", source="qa_battery")
    for r in rows:
        opts = list(r["options"])
        rng.shuffle(opts)
        gold_text = r["options"][r["gold"]]
        metadata = dict(record_metadata(r, r["source"], "parametric_mcq"),
                        meta=r["source"], label_status="source_reference",
                        evidence_regime="no_passage_parametric", task="qa_choice",
                        split=split_name)
        add_decision(id_rows, text_by_state, tok, f"Question: {r['question']}",
                     f"qa_{r['src_id'][:24]}",
                     {"type": "choice", "instructions": INSTR,
                      "criteria": {o: None for o in opts}, "label": gold_text},
                     opts.index(gold_text), metadata, MAX_CTX, MAX_OPT,
                     "exclude", exclusions)
    manifest = {"builder": "build_qa_choice", "split": split_name,
                "tokenizer": getattr(tok, "name_or_path", type(tok).__name__),
                "exclusions": exclusions,
                "counts": {"input_records": len(items), "accepted": len(id_rows)}}
    return write_rows(out, id_rows, list(text_by_state.values()), manifest)


def write_rows(path, id_rows, text_rows, manifest):
    path = Path(path)
    tp = path.with_name(f"{path.stem}_text.jsonl")
    report = write_dataset({path: id_rows, tp: text_rows},
                           path.with_suffix(".manifest.json"), manifest)
    print(f"[wrote] {report['outputs'][0]['path']}: {len(id_rows)} (+text)")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rj", default=".")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    rng = random.Random(SEED)

    gpqa_sets = gpqa_question_sets(args.rj)
    print(f"[gpqa] {len(gpqa_sets)} reference question sets")

    items = list(norm_items())
    by_split = {"train": [], "dev": [], "eval": []}
    for it in items:
        tgt = ("dev" if it["split"] == "validation"
               else "eval" if it["split"] == "test" else "train")
        by_split[tgt].append(it)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, its in by_split.items():
        print(f"[{name}] {len(its)} source items")
        build(its, name, out / f"qa_choice_{name}.jsonl", tok, rng, gpqa_sets)


if __name__ == "__main__":
    main()
