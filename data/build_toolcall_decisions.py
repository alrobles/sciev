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

TOOLS = [
    "bioclim_download", "gbif_occurrence", "inaturalist_occurrence",
    "iucn_status", "maxent_train", "ncbi_taxonomy", "opentree_phylogeny",
    "srtm_elevation", "timetree_divergence", "try_traits",
]

INSTR_CHOICE = "Which tool should handle this ecological data request?"
INSTR_NOUL = "Is the proposed tool call valid for the request?"
SCORE_LEGEND = ["wrong tool", "right tool but wrong arguments", "correct call"]


def load_records(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip()]


def arg_pool(records):
    """arg_name -> list of observed values (for plausible corruptions)."""
    pool = {}
    for r in records:
        for g in r["gold"]:
            for k, v in g.get("args", {}).items():
                pool.setdefault(k, set()).add(str(v))
    return {k: sorted(vs) for k, vs in pool.items()}


def corrupt_args(args, pool, rng):
    """Right tool, one argument swapped for a value seen elsewhere.

    Returns None if no argument can be changed (degenerate pool)."""
    if not args:
        return None
    keys = sorted(args.keys())
    rng.shuffle(keys)
    for key in keys:
        cands = [v for v in pool.get(key, []) if v != str(args[key])]
        if cands:
            out = dict(args)
            out[key] = cands[rng.randrange(len(cands))]
            return out
    return None


def call_text(tool, args):
    a = ", ".join(f'{k}="{v}"' for k, v in sorted(args.items()))
    return f"{tool}({a})"


def build(records, tok, rng, tag):
    """records -> (id_decisions, text_decisions)."""
    id_rows, text_by_state = [], {}
    n_ctx = 0
    for i, r in enumerate(records):
        prompt = r["prompt"]
        gold = r["gold"][0]
        tool, args = gold["tool"], gold.get("args", {})
        if tool not in TOOLS:
            continue
        n_ctx += 1
        state = f"Request: {prompt}"

        # ---- choice: which tool? (K=10) ----
        opts_ids = [tok.encode(t, add_special_tokens=False) for t in TOOLS]
        gold_i = TOOLS.index(tool)
        id_rows.append({"ctx": tok.encode(
            f"{state}\nQuestion: {INSTR_CHOICE}", add_special_tokens=False),
            "opts": opts_ids, "gold": gold_i, "kind": "choice"})
        text_by_state.setdefault(state, {"state": state, "questions": {}})
        text_by_state[state]["questions"][f"choice_{tag}_{i}"] = {
            "type": "choice", "instructions": INSTR_CHOICE,
            "criteria": {t: None for t in TOOLS}, "label": tool}

        # ---- noul + score: gold call vs corrupted calls ----
        wrong_tool = rng.choice([t for t in TOOLS if t != tool])
        wrong_args = corrupt_args(args, arg_pool_cache, rng)
        variants = [
            (tool, args, True, 2),          # correct
            (wrong_tool, args, False, 0),   # wrong tool
        ]
        if wrong_args is not None:
            variants.append((tool, wrong_args, False, 1))  # wrong args
        for j, (t2, a2, valid, score) in enumerate(variants):
            st2 = f"{state}\nProposed call: {call_text(t2, a2)}"
            id_rows.append({"ctx": tok.encode(
                f"{st2}\nQuestion: {INSTR_NOUL}", add_special_tokens=False),
                "opts": [tok.encode("yes", add_special_tokens=False),
                         tok.encode("no", add_special_tokens=False)],
                "gold": 0 if valid else 1, "kind": "noul"})
            id_rows.append({"ctx": tok.encode(
                f"{st2}\nQuestion: rate the proposed call: "
                + " / ".join(SCORE_LEGEND), add_special_tokens=False),
                "opts": [tok.encode(x, add_special_tokens=False)
                         for x in SCORE_LEGEND],
                "gold": score, "kind": "score"})
            text_by_state.setdefault(st2, {"state": st2, "questions": {}})
            text_by_state[st2]["questions"][f"noul_{tag}_{i}_{j}"] = {
                "type": "noul", "instructions": INSTR_NOUL, "label": valid}
            text_by_state[st2]["questions"][f"score_{tag}_{i}_{j}"] = {
                "type": "score", "instructions": "Rate the proposed call.",
                "criteria": SCORE_LEGEND, "label": score}
    text_rows = list(text_by_state.values())
    return id_rows, text_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-dir", required=True, help="dir with toolcalls_*.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    global arg_pool_cache
    fase = load_records(Path(args.l1_dir) / "toolcalls_fase3_500.jsonl")
    lit = []
    for f in ("toolcalls_lit_gold", "toolcalls_lit_evolucion",
              "toolcalls_lit_pilot4"):
        lit += load_records(Path(args.l1_dir) / f"{f}.jsonl")
    arg_pool_cache = arg_pool(fase + lit)

    rng.shuffle(fase)
    n_dev = int(len(fase) * args.dev_frac)
    dev, train = fase[:n_dev], fase[n_dev:]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for tag, recs in (("train", train), ("dev", dev), ("eval", lit)):
        id_rows, text_rows = build(recs, tok, rng, tag)
        by_kind = {}
        for r in id_rows:
            by_kind.setdefault(r.pop("kind"), []).append(r)
        for kind, rows in by_kind.items():
            fp = out / f"toolcall_{kind}_{tag}.jsonl"
            fp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"[wrote] {fp.name}: {len(rows)}")
        if tag == "eval":
            fp = out / "toolcall_decisions_eval_text.jsonl"
            fp.write_text("\n".join(json.dumps(r) for r in text_rows) + "\n")
            print(f"[wrote] {fp.name}: {len(text_rows)}")


if __name__ == "__main__":
    main()
