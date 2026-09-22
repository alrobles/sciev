#!/usr/bin/env python3
"""Teacher labeling via a running Ollama server (KU HPC GPU node).

Same output schema as teacher_label.py: {"qid","kind","gold","soft",
"teacher_argmax","letter_mass"} joined to id-rows by qid.

Usage:
  python data/teacher_label_ollama.py \
      --text-jsonl toolcall_decisions_train_text.jsonl \
      --url http://localhost:11434 --model olmo2:13b --out soft_train.jsonl
"""
import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from teacher_label import render_prompt, letter_probs_from_pairs


def query(url, model, body, timeout=300):
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": body}],
               "options": {"temperature": 0.0, "num_predict": 1},
               "logprobs": True, "top_logprobs": 20}
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def first_token_toplogprobs(resp):
    """Ollama chat response -> [(token_str, logprob)] at position 0.

    Ollama returns logprobs as a flat list of per-position objects
    {"token", "logprob", "top_logprobs": [...]}; OpenAI-style nests it
    under {"content": [...]} — handle both.
    """
    lp = resp.get("logprobs")
    if isinstance(lp, list):
        content = lp
    elif isinstance(lp, dict):
        content = lp.get("content") or []
    else:
        content = []
    if not content:
        return []
    first = content[0]
    return [(t.get("token"), t.get("logprob"))
            for t in (first.get("top_logprobs") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-jsonl", required=True)
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.text_jsonl).read_text().splitlines()
            if l.strip()]
    tasks = []
    for rec in rows:
        for qid, q in rec["questions"].items():
            body, opts, gold = render_prompt(rec["state"], q)
            if body is None:
                continue
            tasks.append((qid, q["type"], gold, len(opts), body))
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[data] {len(tasks)} teacher prompts", flush=True)

    def work(t):
        qid, kind, gold, K, body = t
        try:
            resp = query(args.url, args.model, body)
            pairs = first_token_toplogprobs(resp)
            soft, mass = letter_probs_from_pairs(pairs, K)
            t_arg = max(range(K), key=lambda i: soft[i])
            return {"qid": qid, "kind": kind, "gold": gold,
                    "soft": [round(p, 6) for p in soft],
                    "teacher_argmax": t_arg, "letter_mass": round(mass, 4)}
        except Exception as e:
            return {"qid": qid, "kind": kind, "gold": gold, "error": str(e)}

    agree, errs, done = {}, [], 0
    with open(args.out, "w") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        for rec in ex.map(work, tasks):
            done += 1
            if done % 200 == 0:
                print(f"[{done}/{len(tasks)}]", flush=True)
            if "error" in rec:
                errs.append(rec)
                continue
            f.write(json.dumps(rec) + "\n")
            agree.setdefault(rec["kind"], []).append(
                int(rec["teacher_argmax"] == rec["gold"]))
    rep = {k: {"n": len(v), "teacher_gold_agreement": round(sum(v) / len(v), 4)}
           for k, v in agree.items()}
    print(f"[wrote] {args.out}; errors={len(errs)}")
    print(json.dumps({"teacher_agreement": rep,
                      "errors": errs[:5]}, indent=2))


if __name__ == "__main__":
    main()
