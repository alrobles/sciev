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
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from sciev.data import (add_decision, content_fingerprint, group_records,
                        input_fingerprints, load_decisions_ids, normalized_content,
                        record_metadata, stable_json, write_dataset)
from sciev.protocol import assert_disjoint_splits, dataset_contract

INSTR = "Which answer is correct?"
MAX_CTX, MAX_OPT = 640, 120
MMLU_TRAIN_SUB = 20000
MEDMCQA_TRAIN_SUB = 20000
MEDMCQA_DEV_SUB = 1000
SEED = 7331


def _tokset(text):
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def source_identity(record, split, index):
    identifier = record.get("id")
    return str(identifier) if identifier is not None and str(identifier).strip() else f"{split}:{index}"


def norm_items(provenance=None, seed=SEED):
    """Yield unified {question, options[4], gold, source, split} dicts."""
    from datasets import load_dataset

    specs = [("allenai/ai2_arc", "ARC-Easy", "arc_easy"),
             ("allenai/ai2_arc", "ARC-Challenge", "arc_chal"),
             ("allenai/openbookqa", "main", "openbookqa"),
             ("allenai/sciq", None, "sciq"),
             ("cais/mmlu", "auxiliary_train", "mmlu_aux"),
             ("openlifescienceai/medmcqa", None, "medmcqa")]
    for repo, config, source in specs:
        data = load_dataset(repo, config) if config else load_dataset(repo)
        splits = ("train",) if source == "mmlu_aux" else (
            ("train", "validation") if source == "medmcqa" else ("train", "validation", "test"))
        for split in splits:
            ds = data[split]
            cap = (MMLU_TRAIN_SUB if source == "mmlu_aux" else
                   MEDMCQA_TRAIN_SUB if source == "medmcqa" and split == "train" else
                   MEDMCQA_DEV_SUB if source == "medmcqa" else len(ds))
            indices = (random.Random(seed).sample(range(len(ds)), min(cap, len(ds)))
                       if cap < len(ds) else range(len(ds)))
            digest = hashlib.sha256()
            if provenance is not None:
                provenance.append({"dataset": repo, "config": config, "split": split,
                                   "source": source, "available_records": len(ds),
                                   "selected_records": len(indices),
                                   "dataset_fingerprint": ds._fingerprint,
                                   "license": ds.info.license})
            for index in indices:
                raw = ds[int(index)]
                payload = stable_json(raw).encode("utf-8")
                digest.update(payload + b"\n")
                r = raw["train"] if source == "mmlu_aux" and isinstance(raw.get("train"), dict) else raw
                if source in ("arc_easy", "arc_chal", "openbookqa"):
                    question = r.get("question_stem", r.get("question"))
                    choices = r["choices"]
                    options = choices["text"]
                    gold = choices["label"].index(r["answerKey"])
                elif source == "sciq":
                    question = r["question"]
                    options = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
                    gold = 3
                elif source == "mmlu_aux":
                    question, options, gold = r["question"], r["choices"], int(r["answer"])
                else:
                    question = r["question"]
                    options = [r[key] for key in ("opa", "opb", "opc", "opd")]
                    gold = int(r["cop"])
                source_id = source_identity(r, split, index)
                yield {"question": question, "options": options, "gold": gold,
                       "source": source, "split": split, "source_split": split,
                       "src_id": f"{source}_{source_id}", "source_id": source_id,
                       "source_file": f"{repo}/{config or 'default'}/{split}",
                       "source_row": int(index) + 1,
                       "raw_record_sha256": hashlib.sha256(payload).hexdigest(),
                       "subject": r.get("subject", r.get("subject_name")),
                       "topic": r.get("topic_name"), "choice_type": r.get("choice_type")}
            if provenance is not None:
                provenance[-1]["selected_raw_records_sha256"] = digest.hexdigest()


def gpqa_question_sets(rj, source_paths=None):
    root = Path(rj) / "data/bench_external/gpqa"
    paths, questions = [], []
    for cfg in ("main", "diamond"):
        path = root / "systemone-v2" / f"gpqa_{cfg}_choice_eval_clean_text.jsonl"
        with path.open() as stream:
            questions.extend(json.loads(line)["state"].removeprefix("Question: ") for line in stream if line.strip())
        paths.append(path)
    for cfg in ("main", "diamond", "extended"):
        path = root / "dataset" / f"gpqa_{cfg}.csv"
        if path.exists():
            with path.open(newline="") as stream:
                questions.extend(row["Question"] for row in csv.DictReader(stream))
            paths.append(path)
    if source_paths is not None:
        source_paths.extend(paths)
    return [_tokset(question) for question in sorted(set(questions))]


class QuestionScreen:
    def __init__(self, references, threshold=0.6):
        self.references = [set(ref) for ref in references]
        self.threshold = threshold
        self.index = defaultdict(list)
        for index, tokens in enumerate(self.references):
            for token in tokens:
                self.index[token].append(index)

    def matches(self, question):
        tokens = _tokset(question)
        overlaps = Counter(index for token in tokens for index in self.index.get(token, ()))
        return any(overlap / (len(tokens) + len(self.references[index]) - overlap) >= self.threshold
                   for index, overlap in overlaps.items())


def partition_items(items, gpqa_sets):
    splits = {name: [] for name in ("train", "dev", "eval")}
    exclusions = {name: [] for name in splits}
    screen, candidates, labels = QuestionScreen(gpqa_sets), [], defaultdict(set)
    priority = {"train": 0, "dev": 1, "eval": 2}

    def exclude(row, reason):
        exclusions[row["split"]].append({key: row[key] for key in
                                        ("source", "src_id", "source_row", "sample_id", "split_group") if key in row} |
                                       {"reason": reason})

    for original in items:
        row = dict(original)
        row["split"] = {"validation": "dev", "test": "eval"}.get(row["split"], row["split"])
        if row["split"] not in splits:
            raise ValueError(f"unsupported source split {row['split']!r}")
        question, options, gold = row.get("question"), row.get("options"), row.get("gold")
        if (not isinstance(question, str) or not question.strip()
                or not isinstance(options, (list, tuple)) or len(options) != 4
                or any(not isinstance(option, str) or not option.strip() for option in options)
                or type(gold) is not int or not 0 <= gold < 4
                or row.get("choice_type") not in (None, "single")):
            exclude(row, "invalid_schema")
            continue
        row["question"] = question.strip()
        row["options"] = [option.strip() for option in options]
        normalized_options = [normalized_content(option) for option in options]
        if len(set(normalized_options)) != 4:
            exclude(row, "invalid_options")
            continue
        row["sample_id"] = content_fingerprint([normalized_content(question), sorted(normalized_options)])
        if screen.matches(row["question"]):
            exclude(row, "gpqa_near_duplicate")
            continue
        labels[row["sample_id"]].add(normalized_options[gold])
        if row.get("source_id"):
            row["pid"] = f"{row['source']}:{row['source_id']}"
        candidates.append(row)
    grouped = group_records(candidates, "question", scope="question", source="qa_battery")
    owners = {}
    for row in grouped:
        group, role = row["split_group"], row["split"]
        if group not in owners or priority[role] > priority[owners[group]]:
            owners[group] = role
    seen = set()
    for row in grouped:
        if len(labels[row["sample_id"]]) != 1:
            exclude(row, "conflicting_gold")
        elif row["split"] != owners[row["split_group"]]:
            exclude(row, "heldout_question_overlap")
        elif row["sample_id"] in seen:
            exclude(row, "duplicate_record")
        else:
            seen.add(row["sample_id"])
            splits[row["split"]].append(row)
    return splits, exclusions


def encode_rows(items, split_name, tok, rng, exclusions, max_ctx=MAX_CTX, max_opt=MAX_OPT):
    id_rows, text_by_state = [], {}
    for row in items:
        options = list(row["options"])
        rng.shuffle(options)
        gold_text = row["options"][row["gold"]]
        metadata = dict(record_metadata(row, row["source"], "parametric_mcq"),
                        meta=row["source"], label_status="source_reference",
                        evidence_regime="no_passage_parametric", task="qa_choice", split=split_name)
        metadata.update({key: row[key] for key in
                         ("source_split", "subject", "topic", "raw_record_sha256") if row.get(key) is not None})
        add_decision(id_rows, text_by_state, tok, f"Question: {row['question']}",
                     f"qa_{row['src_id']}",
                     {"type": "choice", "instructions": INSTR,
                      "criteria": {option: None for option in options}, "label": gold_text},
                     options.index(gold_text), metadata, max_ctx, max_opt, "exclude", exclusions)
    return id_rows, list(text_by_state.values())


def write_rows(path, id_rows, text_rows, manifest):
    path = Path(path)
    report = write_dataset({path: id_rows, path.with_name(f"{path.stem}_text.jsonl"): text_rows},
                           path.with_suffix(".manifest.json"), manifest)
    print(f"[wrote] {report['outputs'][0]['path']}: {len(id_rows)} (+text)", flush=True)
    return report


def dev_role(group, seed=SEED):
    bucket = int(content_fingerprint([seed, group])[:8], 16) % 4
    return "select" if bucket < 2 else "calibration" if bucket == 2 else "policy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rj", default=".")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX)
    ap.add_argument("--max-opt", type=int, default=MAX_OPT)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sources, gpqa_paths = [], []
    references = gpqa_question_sets(args.rj, gpqa_paths)
    items = list(norm_items(sources, args.seed))
    splits, exclusions = partition_items(items, references)
    encoded = {name: encode_rows(rows, name, tok, random.Random(args.seed), exclusions[name],
                                 args.max_ctx, args.max_opt) for name, rows in splits.items()}
    disjoint = assert_disjoint_splits({name: pair[0] for name, pair in encoded.items()})
    disjoint["splits"] = {name: {key: value for key, value in contract.items()
                                 if key not in ("identity_hashes", "input_hashes", "effective_input_hashes")}
                           for name, contract in disjoint["splits"].items()}
    for cfg in ("main", "diamond"):
        path = Path(args.rj) / f"data/bench_external/gpqa/systemone-v2/gpqa_{cfg}_choice_eval_clean.jsonl"
        assert_disjoint_splits({**{name: pair[0] for name, pair in encoded.items()},
                               "gpqa": load_decisions_ids(path)})
    manifest = {"builder": "build_qa_choice", "builder_version": 3, "seed": args.seed,
                "text_policy": "preserve internal whitespace; normalize identities only",
                "builder_code": input_fingerprints([__file__])[0],
                "tokenizer": args.tokenizer, "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                "source_snapshots": sources, "gpqa_reference_files": input_fingerprints(gpqa_paths),
                "input_records_sha256": content_fingerprint(items), "split_disjointness": disjoint,
                "split_policy": "global question groups; official eval > dev > train; discard lower-priority copies",
                "gpqa_screen": {"method": "token_jaccard", "threshold": 0.6,
                                "reference_questions": len(references), "paraphrase_isolation_proven": False}}
    out = Path(args.out_dir)
    for name, (ids, texts) in encoded.items():
        write_rows(out / f"qa_choice_{name}.jsonl", ids, texts,
                   {**manifest, "split": name, "exclusions": exclusions[name],
                    "counts": {"accepted": len(ids), "source_counts": dict(Counter(r['source'] for r in ids))}})
    dev_ids, dev_texts = encoded["dev"]
    for role in ("select", "calibration", "policy"):
        ids = [row for row in dev_ids if dev_role(row["split_group"], args.seed) == role]
        texts = [row for row in dev_texts if dev_role(row["split_group"], args.seed) == role]
        dataset_contract(ids)
        write_rows(out / f"qa_choice_{role}.jsonl", ids, texts,
                   {**manifest, "split": role, "parent_split": "dev",
                    "role_policy": "group hash buckets 2:1:1 for selection:temperature:acceptance",
                    "counts": {"accepted": len(ids)}})
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
