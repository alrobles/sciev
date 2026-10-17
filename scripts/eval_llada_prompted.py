"""Generative MCQA baseline: prompted LLaDA-8B-Instruct.

Contrasts the decision-head pipeline with ordinary generation on the same
GPQA items. Each item is presented as a chat prompt with lettered options;
the model fills a short masked suffix by masked-diffusion sampling (greedy
low-confidence unmasking), and the first A-D letter produced is scored.

Usage:
    python eval_llada_prompted.py --text-file gpqa_main_choice_eval_text.jsonl \
        --out eval_llada_prompted_gpqa_main.json [--limit N]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sciev.protocol import file_fingerprint

MODEL = "GSAI-ML/LLaDA-8B-Instruct"
LETTERS = "ABCDEFGH"


def diffuse_fill(model, ids, mask_id, gen_len, steps, device):
    """Greedy masked-diffusion fill of a gen_len-token suffix."""
    if not 1 <= steps <= gen_len:
        raise ValueError("require 1 <= steps <= gen_len")
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("diffusion baseline requires one unpadded prompt")
    x = torch.cat([ids, torch.full((1, gen_len), mask_id, dtype=torch.long,
                                   device=device)], dim=1)
    for step in range(steps):
        masked = (x == mask_id)
        masked[:, :ids.shape[1]] = False
        n_masked = int(masked.sum().item())
        logits = model(input_ids=x).logits[masked].float()
        logits[:, mask_id] = float("-inf")
        cand = logits.argmax(-1)
        conf = torch.softmax(logits, -1).gather(1, cand.unsqueeze(1)).squeeze(1)
        k = max(1, -(-n_masked // (steps - step)))  # ceil(remaining/steps)
        topk = conf.topk(k).indices
        positions = masked.view(-1).nonzero().squeeze(1)
        x.view(-1)[positions[topk]] = cand[topk]
    return x


def parse_answer(text, n_options):
    letters = LETTERS[:n_options]
    explicit = re.findall(
        rf"\b(?i:final\s+answer|correct\s+answer|answer)\s*(?i:is\s*)?:?\s*[\(\[]?([{letters}])\b",
        text)
    if explicit:
        return LETTERS.index(explicit[-1].upper())
    match = re.fullmatch(rf"\s*[\(\[]?([{letters}])[\)\].:]?\s*", text, flags=re.IGNORECASE)
    return LETTERS.index(match.group(1).upper()) if match else -1


def load_items(path, limit=None):
    items, seen = [], set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for qid, question in row["questions"].items():
                if question.get("type") != "choice":
                    raise ValueError("MCQA diagnostics require choice questions")
                options = list(question["criteria"])
                if not 2 <= len(options) <= len(LETTERS) or question["label"] not in options:
                    raise ValueError(f"invalid MCQA labels for {qid}")
                if qid in seen:
                    raise ValueError(f"duplicate qid {qid}")
                seen.add(qid)
                metadata = {key: question.get(key, row.get(key)) for key in
                            ("sample_id", "split_group", "content_hash", "decision_id")}
                items.append({**metadata, "qid": qid, "state": row["state"],
                              "opts": options, "gold": options.index(question["label"])})
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        items = items[:limit]
    if not items:
        raise ValueError("empty evaluation cohort")
    return items


def chat_prefix(tok, item, instruction, device):
    lines = [item["state"], "", *(f"{LETTERS[i]}) {option}" for i, option in enumerate(item["opts"])), instruction]
    messages = [{"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   return_tensors="pt").to(device)


def mask_token_id(model, tok):
    for value in (getattr(model.config, "mask_token_id", None),
                  getattr(model.config, "mask_id", None), tok.mask_token_id):
        if value is not None:
            return int(value)
    raise ValueError("model/tokenizer must declare the diffusion mask token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--gen-len", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--prompt", choices=("direct", "reasoning"), default="direct")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not 1 <= args.steps <= args.gen_len:
        ap.error("require 1 <= steps <= gen_len")
    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    all_items = load_items(args.text_file)
    if args.offset < 0 or args.limit is not None and args.limit < 1:
        ap.error("offset must be nonnegative and limit positive")
    items = all_items[args.offset:None if args.limit is None else args.offset + args.limit]
    if not items:
        ap.error("selected evaluation shard is empty")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(args.device).eval()
    mask_id = mask_token_id(model, tok)
    instruction = ("Reason through the problem, then finish with 'Final answer: X', where X is the answer letter."
                   if args.prompt == "reasoning" else "Answer with a single letter.")
    records, input_tokens = [], 0
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, item in enumerate(items):
            ids = chat_prefix(tok, item, instruction, args.device)
            if ids.shape[1] + args.gen_len > 2048:
                raise ValueError(f"prompt exceeds the declared context budget: {item['qid']}")
            x = diffuse_fill(model, ids, mask_id, args.gen_len, args.steps, args.device)
            generated = tok.decode(x[0, ids.shape[1]:], skip_special_tokens=True).strip()
            prediction = parse_answer(generated, len(item["opts"]))
            input_tokens += x.numel() * args.steps
            records.append({key: value for key, value in item.items() if key not in ("state", "opts")} |
                           {"option_keys": item["opts"], "prediction": prediction,
                            "correct": int(prediction == item["gold"]), "generated": generated,
                            "forward_passes": args.steps})
            if (index + 1) % 25 == 0:
                print(f"[prompted] {index + 1}/{len(items)}", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {"n": len(items), "acc": sum(row["correct"] for row in records) / len(items),
              "unparsed": sum(row["prediction"] < 0 for row in records), "model": args.model,
              "protocol_version": 2, "prompt": args.prompt, "gen_len": args.gen_len, "steps": args.steps,
              "offset": args.offset, "cohort_size": len(all_items),
              "forward_passes": len(items) * args.steps, "model_input_tokens": input_tokens,
              "elapsed_s": elapsed, "seconds_per_decision": elapsed / len(items),
              "input_file": file_fingerprint(args.text_file), "code": file_fingerprint(__file__),
              "model_revision": getattr(model.config, "_commit_hash", None),
              "probability_metrics": None, "probability_metrics_note": "generated answers do not define calibrated option probabilities",
              "records": records}
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
