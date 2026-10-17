"""Masked pseudo-loglikelihood (PLL) MCQA baseline on frozen LLaDA-8B.

Scores each option by appending it as masked tokens after an "Answer:"
prefix and taking the masked-LM log-probability of its true tokens —
i.e., how likely the backbone finds each completion *without
generation*. Ranks options by PLL; measures the raw backbone knowledge
the decision head is meant to extract.

Usage:
    python eval_llada_pll.py --text-file <eval_text.jsonl> --out <report.json>
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sciev.metrics import classification_metrics
from sciev.protocol import file_fingerprint
from scripts.eval_llada_prompted import chat_prefix, load_items, mask_token_id

MODEL = "GSAI-ML/LLaDA-8B-Instruct"
LETTERS = "ABCDEFGH"


def option_pll(model, prefix_ids, opt_ids, mask_id, mode="full_mask", batch_size=4):
    """Sum/mean log p(opt tokens | prefix) under independent masking."""
    if not opt_ids or batch_size < 1 or mode not in ("full_mask", "leave_one_out"):
        raise ValueError("require nonempty option, positive batch_size and a supported scoring mode")
    pos = prefix_ids.shape[1]
    target = torch.tensor(opt_ids, dtype=torch.long, device=prefix_ids.device)
    if mode == "full_mask":
        seq = torch.cat([prefix_ids, torch.full_like(target.unsqueeze(0), mask_id)], dim=1)
        logits = model(input_ids=seq).logits[0, pos:pos + len(opt_ids)].float()
        token_logp = F.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    else:
        completed = torch.cat([prefix_ids, target.unsqueeze(0)], dim=1)
        parts = []
        for start in range(0, len(opt_ids), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(opt_ids)), device=target.device)
            batch = torch.arange(len(indices), device=target.device)
            seq = completed.expand(len(indices), -1).clone()
            seq[batch, pos + indices] = mask_id
            logits = model(input_ids=seq).logits[batch, pos + indices].float()
            parts.append(F.log_softmax(logits, dim=-1).gather(-1, target[indices, None]).squeeze(-1))
        token_logp = torch.cat(parts)
    return float(token_logp.sum()), float(token_logp.mean())


def label_scores(model, prefix_ids, token_ids, mask_id):
    if any(len(ids) != 1 for ids in token_ids) or len({ids[0] for ids in token_ids}) != len(token_ids):
        raise ValueError("letter scoring requires distinct single-token answer labels")
    seq = torch.cat([prefix_ids, torch.tensor([[mask_id]], device=prefix_ids.device)], dim=1)
    logp = F.log_softmax(model(input_ids=seq).logits[0, -1].float(), dim=-1)
    return [(float(logp[ids[0]]), float(logp[ids[0]])) for ids in token_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scoring", choices=("full_mask", "leave_one_out", "label"), default="full_mask")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("batch-size must be positive")
    if Path(args.out).exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    items = load_items(args.text_file, args.limit)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(args.device).eval()
    mask_id = mask_token_id(model, tok)
    records, probabilities_sum, probabilities_mean, golds = [], [], [], []
    forwards = input_tokens = 0
    instruction = "Answer with a single letter." if args.scoring == "label" else "Answer with the exact text of the correct option."
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for index, item in enumerate(items):
            prefix = chat_prefix(tok, item, instruction, args.device)
            answers = list(LETTERS[:len(item["opts"])]) if args.scoring == "label" else item["opts"]
            tokens = [tok.encode(answer, add_special_tokens=False) for answer in answers]
            if prefix.shape[1] + max(map(len, tokens)) > 2048:
                raise ValueError(f"prompt exceeds the declared context budget: {item['qid']}")
            if args.scoring == "label":
                scores = label_scores(model, prefix, tokens, mask_id)
                forwards += 1
                input_tokens += prefix.shape[1] + 1
            else:
                scores = [option_pll(model, prefix, ids, mask_id, args.scoring, args.batch_size) for ids in tokens]
                forwards += sum(1 if args.scoring == "full_mask" else math.ceil(len(ids) / args.batch_size) for ids in tokens)
                input_tokens += sum((prefix.shape[1] + len(ids)) * (1 if args.scoring == "full_mask" else len(ids)) for ids in tokens)
            sum_probs, mean_probs = torch.softmax(torch.tensor(scores, dtype=torch.float64), dim=0).T.tolist()
            probabilities_sum.append(sum_probs)
            probabilities_mean.append(mean_probs)
            golds.append(item["gold"])
            pred_sum = max(range(len(scores)), key=lambda j: scores[j][0])
            pred_mean = max(range(len(scores)), key=lambda j: scores[j][1])
            records.append({key: value for key, value in item.items() if key not in ("state", "opts")} |
                           {"option_keys": item["opts"], "pred_sum": pred_sum, "prediction": pred_mean,
                            "correct": int(pred_mean == item["gold"]), "scores": scores,
                            "probabilities_sum": sum_probs, "probabilities": mean_probs,
                            "max_probability": max(mean_probs)})
            if (index + 1) % 25 == 0:
                print(f"[{args.scoring}] {index + 1}/{len(items)}", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    sum_metrics = classification_metrics(probabilities_sum, golds)
    mean_metrics = classification_metrics(probabilities_mean, golds)
    report = {"protocol_version": 2, "scoring": args.scoring,
              "same_position_logits": True, "true_leave_one_out_pll": args.scoring == "leave_one_out",
              "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
              "input_file": file_fingerprint(args.text_file), "code": file_fingerprint(__file__),
              "prompt_code": file_fingerprint(Path(__file__).with_name("eval_llada_prompted.py")),
              "batch_size": args.batch_size, "sum_metrics": sum_metrics,
              "decisions_eval": {**mean_metrics, "predictions": records, "file": args.text_file,
                                 "temperature": 1.0, "calibration": "none; raw normalized scores, not fitted probabilities",
                                 "forward_passes": forwards, "model_input_tokens": input_tokens,
                                 "elapsed_s": elapsed, "seconds_per_decision": elapsed / len(items)}}
    with Path(args.out).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"scoring": args.scoring, "sum": sum_metrics, "mean": mean_metrics,
                      "forward_passes": forwards, "elapsed_s": elapsed}, indent=2))


if __name__ == "__main__":
    main()
