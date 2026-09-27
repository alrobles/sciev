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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "GSAI-ML/LLaDA-8B-Instruct"
LETTERS = "ABCDEFGH"


def diffuse_fill(model, ids, mask_id, gen_len, steps, device):
    """Greedy masked-diffusion fill of a gen_len-token suffix."""
    x = torch.cat([ids, torch.full((1, gen_len), mask_id, dtype=torch.long,
                                   device=device)], dim=1)
    for _ in range(steps):
        masked = (x == mask_id)
        n_masked = int(masked.sum().item())
        if not n_masked:
            break
        logits = model(input_ids=x).logits.float()
        cand = logits.argmax(-1)
        conf = torch.softmax(logits, -1).max(-1).values
        conf = torch.where(masked, conf, torch.full_like(conf, -1.0))
        k = max(1, -(-n_masked // steps))  # ceil(remaining/steps)
        topk = conf.view(-1).topk(min(k, n_masked)).indices
        x.view(-1)[topk] = cand.view(-1)[topk]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen-len", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(args.device).eval()
    mask_id = getattr(model.config, "mask_token_id", None) or \
        getattr(model.config, "mask_id", tok.mask_token_id)

    items = []
    for line in open(args.text_file):
        row = json.loads(line)
        for qid, q in row["questions"].items():
            opts = list(q["criteria"].keys())
            gold = opts.index(q["label"]) if q["label"] in opts \
                else int(q["label"])
            items.append({"qid": qid, "state": row["state"],
                          "opts": opts, "gold": gold})
    if args.limit:
        items = items[: args.limit]

    t0 = time.time()
    records, correct, unparsed = [], 0, 0
    with torch.no_grad():
        for it in items:
            lines = [it["state"], ""]
            for j, opt in enumerate(it["opts"]):
                lines.append(f"{LETTERS[j]}) {opt}")
            lines.append("Answer with a single letter.")
            msgs = [{"role": "user", "content": "\n".join(lines)}]
            try:
                ids = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt"
                ).to(args.device)
            except Exception:
                ids = tok("\n".join(lines) + "\nAnswer:",
                          return_tensors="pt").input_ids.to(args.device)
            x = diffuse_fill(model, ids, mask_id, args.gen_len,
                             args.steps, args.device)
            gen = tok.decode(x[0, ids.shape[1]:],
                             skip_special_tokens=True)
            m = re.search(r"[A-H]", gen)
            pred = LETTERS.index(m.group(0)) if m else -1
            unparsed += int(pred < 0)
            ok = int(pred == it["gold"])
            correct += ok
            records.append({"qid": it["qid"], "gold": it["gold"],
                            "prediction": pred, "correct": ok,
                            "generated": gen.strip()})
    out = {"n": len(items), "acc": round(correct / max(len(items), 1), 4),
           "unparsed": unparsed, "model": args.model,
           "gen_len": args.gen_len, "steps": args.steps,
           "elapsed_s": round(time.time() - t0, 2),
           "records": records}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "records"},
                     indent=2))


if __name__ == "__main__":
    main()
