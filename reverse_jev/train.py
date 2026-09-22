"""reverse_jev.train — decision fine-tune on the R1 readout (RCDL-lite).

Objective: state + instructions + [MASK] -> softmax over candidate token-id
sets -> cross-entropy toward the labeled option. Optionally adds a
REINFORCE/scoring-rule term (RLCD-style): G Gaussian-perturbed candidates,
reward = log p(gold) + 0.75 * spherical, advantage vs group mean.

Per-example forwards + grad accumulation (the backbone has no padding mask;
batching would leak pad tokens into attention). Slow but honest — micro
runs only, per the repo's <4h test-and-drop rule.

  python -m reverse_jev.train --data train.jsonl --dev dev.jsonl \
      --ckpt runs/f0/checkpoint-g10000/model.pt --out runs/dec-001 \
      --steps 2000 --lr 2e-5 --accum 8 --rl 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import iter_decisions
from .model import load_backbone, MdLMMoE, DEFAULT_CONFIG
from .readout import option_id_sets, noul_id_sets, build_sequence


def _example(model, tok, state, q, label, device, max_len):
    """Build one (ids, mask_pos, option_sets, gold_idx) training example."""
    qt = q["type"]
    ids, mpos = build_sequence(tok, state, q["instructions"],
                               model.mask_id, max_len)
    if qt == "choice":
        names = list(q["criteria"].keys())
        sets = [option_id_sets(tok, n) for n in names]
        gold = names.index(label)
    elif qt == "noul":
        yes, no = noul_id_sets(tok)
        sets = [yes, no]
        gold = 0 if label in (True, "true", "yes") else 1
    elif qt == "score":
        sets = [option_id_sets(tok, str(i)) for i in range(len(q["criteria"]))]
        gold = int(label)
    else:
        return None
    return (torch.tensor(ids, dtype=torch.long, device=device),
            mpos, sets, gold)


def _sliced_logits(row, sets):
    return torch.stack([torch.logsumexp(row[s], dim=-1) if s
                        else torch.tensor(float("-inf"), device=row.device)
                        for s in sets])


def _reward(probs, gold):
    """Proper scoring rule: log score + 0.75 * spherical (Laya recipe)."""
    p = probs.clamp_min(1e-9)
    log_score = torch.log(p[gold])
    onehot = torch.zeros_like(p)
    onehot[gold] = 1.0
    spherical = (onehot * p).sum() / p.norm()
    return log_score + 0.75 * spherical


# ---------------- R2: marker-head training on (ctx, ok, bad) id-pairs --------

def _r2_example(model, ctx, ok, bad, rng, device):
    """ctx + [M] opt_a + [M] opt_b with randomized order -> (ids, pos, gold)."""
    from .model import marker_layout
    max_ctx = model.seq_len // 2
    ctx = ctx[:max_ctx]
    cap = (model.seq_len - len(ctx) - 2) // 2
    ok, bad = ok[:cap], bad[:cap]
    if rng.random() < 0.5:
        opts, gold = (ok, bad), 0
    else:
        opts, gold = (bad, ok), 1
    ids, pos = marker_layout(ctx, opts, model.mask_id)
    ids = torch.tensor(ids, dtype=torch.long, device=device).clamp(0, model.mask_id)
    return ids, pos, gold


def train_r2(model, head, pairs, args, device):
    """CE on marker logits; randomized option order each epoch."""
    import random as _r
    rng = _r.Random(args.seed)
    params = [{"params": head.parameters(), "lr": args.head_lr}]
    if not args.freeze:
        params.append({"params": model.parameters(), "lr": args.lr})
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    model.train(not args.freeze)
    head.train()
    t0 = time.time()
    order = list(range(len(pairs)))
    acc_hist = []
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        ctx, ok, bad = pairs[order[step % len(order)]]
        ids, pos, gold = _r2_example(model, ctx, ok, bad, rng, device)
        h = model(ids.unsqueeze(0), skip_head=True).squeeze(0)
        logits = head(h[pos])
        loss = F.cross_entropy(logits.unsqueeze(0),
                               torch.tensor([gold], device=device))
        (loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            lr_scale = min(1.0, (step + 1) / max(1, args.warmup))
            for g, base in zip(opt.param_groups, [args.head_lr, args.lr][:len(params)]):
                g["lr"] = base * lr_scale
            opt.step()
            opt.zero_grad(set_to_none=True)
        acc_hist.append(int(logits.argmax().item() == gold))
        if step % 50 == 0:
            acc = sum(acc_hist[-200:]) / min(len(acc_hist), 200)
            print(f"[{time.strftime('%H:%M:%S')}] step {step} "
                  f"loss {loss.item():.4f} train_acc_200 {acc:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="decisions JSONL (R1 path)")
    ap.add_argument("--pairs-train", default=None,
                    help="dir with pairs_L*.jsonl (R2 marker-head path, ids)")
    ap.add_argument("--levels", default="L0,L1,L2,L3",
                    help="comma list of pair levels to train on (R2)")
    ap.add_argument("--freeze", action="store_true",
                    help="R2: freeze backbone, train DecisionHead only")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--dev", default=None, help="held-out for temperature fit")
    ap.add_argument("--ckpt", default=None, help="init backbone from checkpoint")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rl", type=float, default=0.0,
                    help="weight of the REINFORCE scoring-rule term (0=CE only)")
    ap.add_argument("--rl_samples", type=int, default=4)
    ap.add_argument("--rl_noise", type=float, default=0.4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- R2 path: marker head on id-pairs, no tokenizer ----------
    if args.pairs_train:
        from .data import load_pairs_dir
        from .model import DecisionHead
        cfg = None
        if args.config:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text()).get("model", {})
        model = load_backbone(args.ckpt, cfg, device=args.device)
        head = DecisionHead(model.tok_emb.embedding_dim).to(args.device)
        want = set(args.levels.split(","))
        pairs = [p for lvl, ps in load_pairs_dir(args.pairs_train).items()
                 if lvl in want for p in ps]
        if not pairs:
            raise SystemExit("no pairs loaded from --pairs-train")
        print(f"[data] {len(pairs)} pairs "
              f"(levels={sorted(want)}, freeze={args.freeze})")
        model, head = train_r2(model, head, pairs, args, args.device)
        torch.save({"model": model.state_dict(), "head": head.state_dict(),
                    "meta": {"mode": "r2_marker", "levels": sorted(want),
                             "pairs_train": args.pairs_train}},
                   out_dir / "decision.pt")
        (out_dir / "training_config.json").write_text(
            json.dumps(vars(args), indent=2))
        print("COMPLETE")
        return

    # ---------- R1 path: decisions JSONL ----------
    if not args.data:
        ap.error("--data or --pairs-train required")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.ckpt:
        cfg = None
        if args.config:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text()).get("model", {})
        model = load_backbone(args.ckpt, cfg, device=args.device)
    else:
        cfg = dict(DEFAULT_CONFIG)
        if args.config:
            import yaml
            cfg.update(yaml.safe_load(Path(args.config).read_text()).get("model", {}))
        cfg["vocab"] = tok.vocab_size
        model = MdLMMoE(**cfg).to(args.device)
    model.train()

    examples = []
    for state, qid, q, label in iter_decisions(args.data):
        ex = _example(model, tok, state, q, label, args.device, args.max_len)
        if ex:
            examples.append(ex)
    if not examples:
        raise SystemExit("no labeled decisions in --data")
    print(f"[data] {len(examples)} decisions")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    it = 0
    for step in range(args.steps):
        ex = examples[it % len(examples)]
        it += 1
        ids, mpos, sets, gold = ex
        logits = model(ids.unsqueeze(0)).squeeze(0)
        sliced = _sliced_logits(logits[mpos], sets)
        loss = F.cross_entropy(sliced.unsqueeze(0),
                               torch.tensor([gold], device=args.device))

        if args.rl > 0:
            # REINFORCE over Gaussian-perturbed candidate distributions
            G = args.rl_samples
            noise = torch.randn(G, sliced.numel(), device=args.device) * args.rl_noise
            noise = noise - noise.mean(dim=-1, keepdim=True)  # center: no shift
            cand = sliced.detach().unsqueeze(0) + noise
            probs = torch.softmax(cand, dim=-1)
            rewards = torch.stack([_reward(p, gold) for p in probs])
            adv = rewards - rewards.mean()
            # log-density of each candidate under N(sliced, sigma): const + (z-mu)
            pg = -(adv.detach() * (noise / (args.rl_noise ** 2)) * sliced.unsqueeze(0)).sum(-1).mean()
            loss = loss + args.rl * pg

        (loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] step {step} "
                  f"loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % 200 == 0:
            torch.save({"model": model.state_dict()}, out_dir / "model.pt")

    torch.save({"model": model.state_dict()}, out_dir / "model.pt")

    # post-hoc temperature on dev (never on train)
    if args.dev:
        from .eval import fit_temperature
        model.eval()
        dev_rows = list(iter_decisions(args.dev))
        t = fit_temperature(model, tok, dev_rows, args.device, args.max_len)
        (out_dir / "temperature.json").write_text(json.dumps({"temperature": t}))
        print(f"[temp] fitted T={t:.3f} on {len(dev_rows)} dev decisions")

    (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))
    print("COMPLETE")


if __name__ == "__main__":
    main()
