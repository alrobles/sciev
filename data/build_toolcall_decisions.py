#!/usr/bin/env python3
"""Build K-way decision batteries from ecoreasoner tool-call records.

Input records: {"prompt": str, "gold": [{"tool": str, "args": {...}}]}

Outputs (id-tokenized for the dLLM backbone + text JSONL for remote eval):
  toolcall_choice_{train,dev,eval}.jsonl   K=10 options, gold tool
  toolcall_noul_{train,dev,eval}.jsonl     K=2 (valid/invalid), corrupted calls
  toolcall_score_{train,dev,eval}.jsonl    K=3 rubric: wrong tool / wrong args / correct
  toolcall_decisions_eval_text.jsonl       text System-One rows for --remote (Jev)

Record format for the model path: {"ctx":[ids], "opts":[[ids]...], "gold":i}
Negative generation: wrong tool (other tool, own args) and wrong args
(right tool, one arg value swapped with a value from another record).
"""
import argparse
import json
import random
from pathlib import Path

from reverse_jev.data import (
    add_decision, content_fingerprint, exclude_record, group_records,
    input_fingerprints, normalized_content, partition_groups, read_jsonl,
    record_metadata, stable_json, validate_fractions, write_dataset,
)

TOOLS = [
    "bioclim_download", "gbif_occurrence", "inaturalist_occurrence",
    "iucn_status", "maxent_train", "ncbi_taxonomy", "opentree_phylogeny",
    "srtm_elevation", "timetree_divergence", "try_traits",
]

INSTR_CHOICE = "Which tool should handle this ecological data request?"
INSTR_NOUL = "Is the proposed tool call valid for the request?"
SCORE_LEGEND = ["wrong tool", "right tool but wrong arguments", "correct call"]


def _validate_record(record, location):
    if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        raise ValueError(f"{location}: prompt must be a nonempty string")
    if not isinstance(record.get("gold"), list) or not record["gold"]:
        raise ValueError(f"{location}: gold must be a nonempty array of calls")
    for call in record["gold"]:
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str) or not isinstance(call.get("args", {}), dict):
            raise ValueError(f"{location}: each gold call needs a tool and an arguments object")
        if any(not isinstance(key, str) for key in call.get("args", {})):
            raise ValueError(f"{location}: argument keys must be strings")
        stable_json(call)


def load_records(path):
    rows = []
    for line_number, row in read_jsonl(path):
        _validate_record(row, f"{path}:{line_number}")
        row = dict(row, source_file=str(path), source_line=line_number)
        row.setdefault("source", "ecoreasoner.toolcalls")
        rows.append(row)
    return rows


def prepare_records(records, exclusions=None):
    for index, row in enumerate(records):
        _validate_record(row, f"record {index}")
    rows = [dict(row, prompt=row["prompt"].strip()) for row in records]
    if not all("split_group" in row for row in rows):
        rows = group_records(rows, "prompt", scope="request", source="ecoreasoner.toolcalls")
    references = {}
    for row in rows:
        references.setdefault(normalized_content(row["prompt"]), set()).add(stable_json(row["gold"]))
    unique = {}
    for row in rows:
        row["sample_id"] = content_fingerprint([row["source"], row["prompt"], row["gold"]])
        if len(row["gold"]) != 1:
            exclude_record(exclusions, row, "multiple_gold_calls")
        elif row["gold"][0]["tool"] not in TOOLS:
            exclude_record(exclusions, row, "unknown_tool", tool=row["gold"][0]["tool"])
        elif len(references[normalized_content(row["prompt"])]) > 1:
            exclude_record(exclusions, row, "ambiguous_prompt_reference")
        else:
            key = (row["content_hash"], stable_json(row["gold"]))
            if key in unique:
                exclude_record(exclusions, row, "duplicate_request", retained_sample_id=unique[key]["sample_id"])
                unique[key].setdefault("duplicate_sources", []).append(record_metadata(row, row["source"], "tool_call"))
            else:
                unique[key] = row
    return list(unique.values())


def split_records(fase, lit, rng, dev_frac=0.15, exclusions=None):
    validate_fractions(dev_frac, 0.0)
    combined = group_records([*fase, *lit], "prompt", scope="request", source="ecoreasoner.toolcalls")
    evaluation = combined[len(fase):]
    eval_groups = {row["split_group"] for row in evaluation}
    training = []
    for row in combined[:len(fase)]:
        if row["split_group"] in eval_groups:
            exclude_record(exclusions, row, "eval_group_overlap")
        else:
            training.append(row)
    splits = partition_groups(prepare_records(training, exclusions), rng, dev_frac, 0.0)
    splits["eval"] = [dict(row, split="eval") for row in prepare_records(evaluation, exclusions)]
    return splits


def arg_pool(records):
    """arg_name -> list of observed values (for plausible corruptions)."""
    pool = {}
    for r in records:
        for g in r["gold"]:
            for k, v in g.get("args", {}).items():
                pool.setdefault(k, {})[stable_json(v)] = v
    return {k: [vs[key] for key in sorted(vs)] for k, vs in sorted(pool.items())}


def corrupt_args(args, pool, rng):
    """Right tool, one argument swapped for a value seen elsewhere.

    Returns None if no argument can be changed (degenerate pool)."""
    if not args:
        return None
    keys = sorted(args.keys())
    rng.shuffle(keys)
    for key in keys:
        cands = [v for v in pool.get(key, []) if stable_json(v) != stable_json(args[key])]
        if cands:
            out = dict(args)
            out[key] = cands[rng.randrange(len(cands))]
            return out
    return None


def call_text(tool, args):
    a = ", ".join(f"{k}={stable_json(v)}" for k, v in sorted(args.items()))
    return f"{tool}({a})"


def build(records, tok, rng, tag, *, exclusions=None, max_ctx=640, max_opt=120):
    """records -> (id_decisions, text_decisions)."""
    records = prepare_records(records, exclusions)
    if any(r.get("split", tag) != tag for r in records):
        raise ValueError("negative pools must contain only records from the requested split")
    pool = arg_pool(records)
    id_rows, text_by_state = [], {}
    for r in records:
        prompt = r["prompt"]
        gold = r["gold"][0]
        tool, args = gold["tool"], gold.get("args", {})
        state = f"Request: {prompt}"
        metadata = dict(record_metadata(r, "ecoreasoner.toolcalls", "tool_call"), split=tag)
        identity = f"{tag}_{r['sample_id'][:20]}"
        provenance = {"strategy": "source_reference", "source_pid": r["pid"], "source": r["source"],
                      "source_group": r["split_group"], "source_sample_id": r["sample_id"],
                      "split": tag, "verified": False}

        # ---- choice: which tool? (K=10) ----
        add_decision(id_rows, text_by_state, tok, state, f"choice_{identity}",
                     {"type": "choice", "instructions": INSTR_CHOICE,
                      "criteria": {t: None for t in TOOLS}, "label": tool}, TOOLS.index(tool),
                     dict(metadata, label_status="source_reference", label_space=TOOLS,
                          negative_provenance=dict(provenance, strategy="fixed_tool_vocabulary")),
                     max_ctx, max_opt)

        # ---- noul + score: gold call vs corrupted calls ----
        wrong_tool = rng.choice([t for t in TOOLS if t != tool])
        wrong_args = corrupt_args(args, pool, rng)
        variants = [
            (tool, args, True, 2),          # correct
            (wrong_tool, args, False, 0),   # wrong tool
        ]
        corrupt_provenance = None
        if wrong_args is not None:
            variants.append((tool, wrong_args, False, 1))  # wrong args
            key = next(k for k in args if stable_json(args[k]) != stable_json(wrong_args[k]))
            donor = next(o for o in records for call in o["gold"]
                         if key in call.get("args", {}) and stable_json(call["args"][key]) == stable_json(wrong_args[key]))
            corrupt_provenance = {"strategy": "argument_swap", "argument": key,
                                  "original_value": args[key], "replacement_value": wrong_args[key],
                                  "source_pid": donor["pid"], "source": donor["source"],
                                  "source_group": donor["split_group"], "source_sample_id": donor["sample_id"],
                                  "split": tag, "verified": False}
        else:
            exclude_record(exclusions, r, "insufficient_argument_pool", split=tag, kind="noul/score")
        for j, (t2, a2, valid, score) in enumerate(variants):
            st2 = f"{state}\nProposed call: {call_text(t2, a2)}"
            prov = (corrupt_provenance if score == 1 else dict(
                provenance, strategy="source_reference" if valid else "tool_swap"))
            meta = dict(metadata, negative_provenance=prov,
                        label_status="source_reference" if valid else "heuristic")
            add_decision(id_rows, text_by_state, tok, st2, f"noul_{identity}_{j}",
                         {"type": "noul", "instructions": INSTR_NOUL, "label": valid},
                         0 if valid else 1, meta, max_ctx, max_opt)
            add_decision(id_rows, text_by_state, tok, st2, f"score_{identity}_{j}",
                         {"type": "score", "instructions": "Rate the proposed call.",
                          "criteria": SCORE_LEGEND, "label": score}, score,
                         dict(meta, label_status="heuristic_proxy", rubric_provenance="synthetic_strategy"),
                         max_ctx, max_opt)
    text_rows = list(text_by_state.values())
    return id_rows, text_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-dir", required=True, help="dir with toolcalls_*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--max-ctx", type=int, default=640)
    ap.add_argument("--max-opt", type=int, default=120)
    args = ap.parse_args()

    out = Path(args.out)
    paths = [Path(args.l1_dir) / f"{name}.jsonl" for name in (
        "toolcalls_fase3_500", "toolcalls_lit_gold", "toolcalls_lit_evolucion", "toolcalls_lit_pilot4")]
    fase = load_records(paths[0])
    lit = [row for path in paths[1:] for row in load_records(path)]
    exclusions = []
    splits = split_records(fase, lit, random.Random(args.seed), args.dev_frac, exclusions)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    files = {}
    for tag, recs in splits.items():
        id_rows, text_rows = build(recs, tok, random.Random(f"{args.seed}:{tag}"), tag,
                                   exclusions=exclusions, max_ctx=args.max_ctx, max_opt=args.max_opt)
        for kind in ("choice", "noul", "score"):
            files[out / f"toolcall_{kind}_{tag}.jsonl"] = [r for r in id_rows if r["kind"] == kind]
        files[out / f"toolcall_decisions_{tag}_text.jsonl"] = text_rows
    manifest = {"builder": "build_toolcall_decisions", "seed": args.seed, "inputs": input_fingerprints(paths),
                "tokenizer": args.tokenizer, "max_ctx": args.max_ctx, "max_opt": args.max_opt,
                "split_unit": "connected source identifiers and normalized request content",
                "dev_frac": args.dev_frac, "eval_overlap_policy": "exclude overlapping training-source groups",
                "negative_policy": "split-local typed argument swaps and tool swaps; not human-verified",
                "score_policy": "synthetic strategy proxy, not independently verified call validity",
                "counts": {"input_records": len(fase) + len(lit),
                           "split_records": {tag: len(rs) for tag, rs in splits.items()},
                           "split_groups": {tag: len({r["split_group"] for r in rs}) for tag, rs in splits.items()}},
                "exclusions": exclusions}
    report = write_dataset(files, out / "toolcall_manifest.json", manifest)
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
