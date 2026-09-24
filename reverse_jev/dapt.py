"""Domain-adaptive pretraining (DAPT) for a masked-diffusion HF backbone.

Continues the backbone's native objective — masked token prediction with
t ~ U(0,1) and loss scaled by 1/t (LLaDA recipe) — on a domain corpus, via
LoRA so it fits on a single GPU. The resulting adapter can be loaded on top
of the base model by HFBackbone (--lora-adapter) before head training.

Corpus: jsonl files with a text field (default "text"; "passage" works too).
"""
import argparse
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F


def iter_texts(patterns, field, limit=None, seed=7331):
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(pat)))
    rng = random.Random(seed)
    rng.shuffle(files)
    n = 0
    for path in files:
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                txt = row.get(field) or row.get("text") or ""
                if len(txt) >= 200:
                    yield txt
                    n += 1
                    if limit and n >= limit:
                        return


def lora_wrap(model, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    # LLaDA blocks use OLMo-style fused/attn+ffn projections; collect actual
    # Linear leaf names and target the standard set.
    want = {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj",
            "down_proj", "att_proj", "attn_out", "ff_proj", "ff_out",
            "qkv_proj", "out_proj"}
    found = sorted({m.rsplit(".", 1)[-1] for m, mod in model.named_modules()
                    if isinstance(mod, torch.nn.Linear)})
    targets = [t for t in found if t in want]
    if not targets:  # fall back to all non-embed/head linears inside blocks
        targets = [t for t in found if t not in ("lm_head", "wte", "emb")]
    print("[dapt] LoRA targets:", targets)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=targets)
    return get_peft_model(model, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-backbone", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--docs", type=int, default=0, help="cap corpus docs")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7331)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_backbone,
                                        trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.hf_backbone, trust_remote_code=True,
        torch_dtype=torch.bfloat16).to(dev)
    mask_id = getattr(base.config, "mask_token_id",
                      base.config.vocab_size - 1)
    print("[dapt] mask_id:", mask_id)

    model = lora_wrap(base, args.lora_r, args.lora_alpha, args.lora_dropout)
    # LLaDA remote-code doesn't support HF gradient_checkpointing_enable;
    # it has its own block-level activation checkpointing instead.
    import sys
    inner = getattr(base, "model", base)
    mod = sys.modules.get(type(inner).__module__)
    acs = getattr(mod, "ActivationCheckpointingStrategy", None)
    if acs is not None and hasattr(inner, "set_activation_checkpointing"):
        inner.set_activation_checkpointing(acs.whole_layer)
        print("[dapt] activation checkpointing: whole_layer")
    model.print_trainable_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(s / 200, 0.5 * (1 + math.cos(
            math.pi * s / max(args.steps, 1)))))

    texts = iter_texts(args.corpus, args.field,
                       limit=args.docs or None, seed=args.seed)
    it = iter(texts)
    model.train()
    t0 = time.time()
    ema = None
    os.makedirs(args.out, exist_ok=True)

    def next_batch():
        nonlocal it
        docs = []
        while len(docs) < args.bs:
            try:
                docs.append(next(it))
            except StopIteration:
                it = iter(iter_texts(args.corpus, args.field,
                                     limit=args.docs or None,
                                     seed=args.seed + 1))
        enc = tok(docs, return_tensors="pt", truncation=True,
                  max_length=args.seq_len, padding="longest")
        return enc["input_ids"].to(dev)

    for step in range(1, args.steps + 1):
        ids = next_batch()
        t = torch.rand(ids.size(0), 1, device=ids.device)
        p_mask = t.expand_as(ids).float()
        noise = torch.rand_like(ids, dtype=torch.float)
        m = noise < p_mask
        corrupted = torch.where(m, torch.full_like(ids, mask_id), ids)
        out = model(input_ids=corrupted)
        logits = out.logits.float()
        ce = F.cross_entropy(
            logits[m].view(-1, logits.size(-1)),
            ids[m].view(-1), reduction="mean")
        loss = ce / t.mean().clamp(min=1e-3) / args.grad_accum
        loss.backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        raw = (ce / t.mean()).item()
        ema = raw if ema is None else 0.98 * ema + 0.02 * raw
        if step % 50 == 0:
            print(f"[dapt] step {step}/{args.steps} ce/t {raw:.3f} "
                  f"ema {ema:.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.save_every == 0:
            path = os.path.join(args.out, f"lora-g{step}")
            model.save_pretrained(path)
            print("[dapt] saved", path, flush=True)

    final = os.path.join(args.out, "lora-final")
    model.save_pretrained(final)
    json.dump(vars(args), open(os.path.join(final, "dapt_args.json"), "w"),
              indent=1)
    print("[dapt] DONE", final)


if __name__ == "__main__":
    main()
