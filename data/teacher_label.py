#!/usr/bin/env python3
"""Teacher labeling for decision batteries via an open model (OLMo-2).

Runs on the cluster with vLLM offline batch inference. For each labeled
System-One question we prompt the teacher with lettered options and read
the first-token logprobs -> a distribution over options ("soft" labels).

Output jsonl: {"qid", "kind", "gold", "soft":[p...], "teacher_argmax",
               "letter_mass"}
joined to id-rows by qid via --soft-labels in reverse_jev.train.

Usage (in apptainer on a GPU node):
  python data/teacher_label.py \
      --text-jsonl data/toolcall_battery/toolcall_decisions_train_text.jsonl \
      --model allenai/OLMo-2-0325-32B-Instruct --out soft_train.jsonl
"""
import argparse
import json
import string
from pathlib import Path

LETTERS = string.ascii_uppercase  # A..Z


def render_prompt(state, q):
    """state + question -> lettered-option chat prompt."""
    from reverse_jev.decisions import render_value

    qt = q["type"]
    if qt == "choice":
        keys = list(q["criteria"])
        opts = [key if value is None else f"{key}: {render_value(value, 'choice description', allow_empty=True)}"
                for key, value in q["criteria"].items()]
        gold = keys.index(q["label"])
    elif qt == "noul":
        opts = ["yes", "no"]
        criteria = q.get("criteria")
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                raise ValueError("noul criteria must define true and false")
            opts = [f"{option}: {render_value(criteria[key], 'noul criterion')}"
                    for option, key in zip(opts, ("true", "false"))]
        label = q["label"]
        if label is True or (isinstance(label, str) and label.lower() in ("true", "yes")):
            gold = 0
        elif label is False or (isinstance(label, str) and label.lower() in ("false", "no")):
            gold = 1
        else:
            raise ValueError("noul label must be true or false")
    elif qt == "score":
        opts = list(q["criteria"]) if isinstance(q["criteria"], list) \
            else [q["criteria"][k] for k in sorted(q["criteria"], key=int)]
        opts = [render_value(option, "score criterion") for option in opts]
        label = q["label"]
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise ValueError("score label must be an integer level")
        gold = int(label)
    else:
        return None, None, None
    if not 2 <= len(opts) <= len(LETTERS) or not 0 <= gold < len(opts):
        raise ValueError("teacher questions require 2..26 options and an in-range label")
    labels = [f"{LETTERS[i]}) {option}" for i, option in enumerate(opts)]
    body = (f"{render_value(state, 'state', allow_empty=True)}\n\n"
            f"{render_value(q['instructions'], 'instructions')}\n\n" + "\n".join(labels))
    body += "\n\nAnswer with only the letter of the best option."
    return body, opts, gold


def letter_probs_from_pairs(pairs, K):
    """[(token_str, logprob)] -> per-letter probability mass + coverage."""
    import math
    mass = [0.0] * K
    for tok_str, lp in pairs:
        t = (tok_str or "").strip().upper()
        if len(t) == 1 and t in LETTERS[:K]:
            mass[LETTERS.index(t)] += math.exp(lp)
    total = sum(mass)
    if total <= 0:
        return [1.0 / K] * K, 0.0
    return [m / total for m in mass], total


def letter_probs(logprob_dict, K):
    """vLLM top-logprobs dict at position 0 -> per-letter mass."""
    return letter_probs_from_pairs(
        [(lp.decoded_token, lp.logprob) for lp in logprob_dict.values()], K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-jsonl", required=True)
    ap.add_argument("--model", default="allenai/OLMo-2-0325-32B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.text_jsonl).read_text().splitlines()
            if l.strip()]

    prompts, meta = [], []
    for rec in rows:
        for qid, q in rec["questions"].items():
            body, opts, gold = render_prompt(rec["state"], q)
            if body is None:
                continue
            prompts.append([{"role": "user", "content": body}])
            meta.append((qid, q["type"], gold, len(opts)))
    if args.limit:
        prompts, meta = prompts[:args.limit], meta[:args.limit]
    print(f"[data] {len(prompts)} teacher prompts", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len,
              enforce_eager=False)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.chat(prompts, sp)

    agree = {}
    n_written = 0
    with open(args.out, "w") as f:
        for (qid, kind, gold, K), out in zip(meta, outs):
            lps = out.outputs[0].logprobs
            if not lps:
                continue
            soft, mass = letter_probs(lps[0], K)
            t_arg = max(range(K), key=lambda i: soft[i])
            agree.setdefault(kind, []).append(int(t_arg == gold))
            f.write(json.dumps({"qid": qid, "kind": kind, "gold": gold,
                                "soft": [round(p, 6) for p in soft],
                                "teacher_argmax": t_arg,
                                "letter_mass": round(mass, 4)}) + "\n")
            n_written += 1
    rep = {k: {"n": len(v), "teacher_gold_agreement": round(sum(v) / len(v), 4)}
           for k, v in agree.items()}
    print(f"[wrote] {args.out}: {n_written}")
    print(json.dumps({"teacher_agreement": rep}, indent=2))


if __name__ == "__main__":
    main()
