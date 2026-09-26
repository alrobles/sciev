"""sciev.train — decision fine-tune on the R1 readout (RCDL-lite).

Objective: state + instructions + [MASK] -> softmax over candidate token-id
sets -> cross-entropy toward the labeled option. Optionally adds a
REINFORCE/scoring-rule term (RLCD-style): G Gaussian-perturbed candidates,
reward = log p(gold) + 0.75 * spherical, advantage vs group mean.

Per-example forwards + grad accumulation (the backbone has no padding mask;
batching would leak pad tokens into attention). Slow but honest — micro
runs only, per the repo's <4h test-and-drop rule.

  python -m sciev.train --data train.jsonl --dev dev.jsonl \
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
from .model import (load_backbone, MdLMMoE, DEFAULT_CONFIG,
                    forward_feats)
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

def _canonical_perm(opts):
    """Deterministic option order: sort by token-id content.

    Makes the presented order a pure function of the option *set*, so any
    input permutation collapses to the same sequence (flip_rate = 0 by
    construction at eval)."""
    return sorted(range(len(opts)), key=lambda j: tuple(opts[j]))


def _r2_example(model, ctx, opts, gold, rng, device, mode="marker",
                canonical=False, strict=False):
    """Build one R2 training example with an option permutation.

    marker   : ctx + [M] opt_1 + ... + [M] opt_K -> head reads h[marker]
    spanpool : ctx + opt_1 + ... + opt_K         -> head reads mean h[span]
    canonical: deterministic content-sorted order instead of random
    """
    from .decisions import prepare_decision, validate_decision_row
    row = validate_decision_row({"ctx": ctx, "opts": opts, "gold": gold})
    order = None if canonical else rng.sample(range(len(opts)), len(opts))
    prepared = prepare_decision(model, row["ctx"], row["opts"], mode,
                                canonical=canonical, order=order, strict=strict)
    ids = torch.tensor(prepared.ids, dtype=torch.long, device=device)
    return ids, prepared.positions, prepared.order.index(gold), prepared.order


def apply_scientific_recipe(args):
    """Apply the shared scientific-v1 recipe to parsed CLI arguments."""
    if args.recipe is None:
        return
    if args.recipe != "scientific-v1":
        raise ValueError(f"unknown recipe {args.recipe!r}")
    if args.decision_type not in {"choice", "noul", "score"}:
        raise ValueError("--decision-type is required with --recipe scientific-v1")
    from .protocol import scientific_recipe
    recipe = scientific_recipe(args.decision_type)
    args.freeze = recipe["freeze"]
    args.r2_mode = recipe["r2_mode"]
    args.canonical_order = recipe["canonical_order"]
    args.orders = recipe["orders"]
    args.head_kind = recipe["head_kind"]
    args.steps = recipe["steps"]
    args.head_lr = recipe["head_lr"]
    args.warmup = recipe["warmup"]
    args.accum = recipe["accum"]
    args.ordinal = recipe["ordinal"]
    args.strict_inputs = True
    args.training_contract = "scientific-v1"


def r2_checkpoint(model, head, examples, args):
    """Self-describing R2 head checkpoint with inference + training metadata."""
    from . import protocol
    effective_layers = [-1] if args.head_kind == "mlp" else list(args.layers_list)
    encodings = {ex.get("encoding") for ex in examples if isinstance(ex, dict)}
    inference = {"head_kind": args.head_kind, "mode": args.r2_mode,
                 "layers": effective_layers,
                 "canonical_order": args.canonical_order,
                 "strict_inputs": bool(args.strict_inputs),
                 "decision_type": args.decision_type,
                 "encoding": next(iter(encodings)) if len(encodings) == 1 else None,
                 "seq_len": model.seq_len}
    ckpt = {"head": head.state_dict(),
            "head_kind": args.head_kind,
            "n_layers": len(effective_layers),
            "inference": inference,
            "meta": {"mode": f"r2_{args.r2_mode}",
                     "pairs_train": args.pairs_train,
                     "decisions_train": args.decisions_train,
                     "recipe": getattr(args, "training_contract", None),
                     "seed": args.seed}}
    if all(isinstance(ex, dict) for ex in examples):
        ckpt["training_data"] = protocol.dataset_contract(examples)
    inputs = [p for p in (args.pairs_train, args.decisions_train) if p]
    if inputs:
        ckpt["input_files"] = [protocol.file_fingerprint(p) for p in inputs]
    return ckpt


def train_r2(model, head, examples, args, device):
    """CE on option logits; randomized option order each step.

    examples: [(ctx, [opt_ids...], gold_idx)] for pairs, or row dicts
    {"ctx","opts","gold","qid"?,"soft"?} from load_decisions_ids.
    """
    import random as _r
    from .decisions import ENCODING_VERSION, validate_decision_row
    if not examples:
        raise ValueError("training requires at least one decision")
    if args.steps < 1 or args.accum < 1 or args.orders < 1:
        raise ValueError("steps, accum, and orders must be positive")
    for example in examples:
        if isinstance(example, dict):
            validate_decision_row(example)
            if args.ordinal > 0 and example.get("kind") not in (None, "score"):
                raise ValueError("ordinal loss requires score decisions, not nominal classes")
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
    order = list(range(len(examples)))
    acc_hist = []
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        ex = examples[order[step % len(order)]]
        strict = getattr(args, "strict_inputs", None)
        if isinstance(ex, dict):
            ctx, opts, gold = ex["ctx"], ex["opts"], ex["gold"]
            soft = ex.get("soft")
            if strict is None:
                strict = ex.get("encoding") == ENCODING_VERSION
        else:
            ctx, opts, gold = ex
            soft = None
            strict = bool(strict)
        # --orders N: average logits over N option orders (canonical space).
        # canonical order makes all N draws identical -> single forward.
        n_orders = 1 if args.canonical_order else args.orders
        lg_orders = []
        for _ in range(n_orders):
            ids, pos, gold_p, perm = _r2_example(
                model, ctx, opts, gold, rng, device, mode=args.r2_mode,
                canonical=args.canonical_order, strict=strict)
            lg = forward_feats(model, head, ids, args.r2_mode, pos,
                               layers=args.layers_list)
            lg_c = torch.empty_like(lg)
            lg_c[perm] = lg          # map back to canonical option order
            lg_orders.append(lg_c)
        logits = torch.stack(lg_orders).mean(0)
        loss = F.cross_entropy(logits.unsqueeze(0),
                               torch.tensor([gold], device=device))
        if soft is not None and args.soft_weight > 0:
            # distillation: KL(student_T || teacher_T), canonical order
            T = args.soft_temp
            t = torch.tensor(soft, dtype=torch.float,
                             device=device).clamp_min(1e-9)
            t = t / t.sum()
            kl = F.kl_div(F.log_softmax(logits / T, dim=-1),
                          t, reduction="batchmean") * T * T
            loss = (1 - args.soft_weight) * loss + args.soft_weight * kl
        if args.ordinal > 0 and logits.numel() > 2:
            # ordinal auxiliary loss (score levels are ordered 0<1<2):
            # BCE(P(y >= j), 1{gold >= j}) on cumulative softmax probs
            K_ = logits.numel()
            probs = torch.softmax(logits, dim=-1)
            p_ge = 1.0 - probs.cumsum(-1)[:-1]     # P(y >= j+1)
            tgt = torch.tensor(
                [float(gold >= j + 1) for j in range(K_ - 1)],
                device=device)
            loss = loss + args.ordinal * F.binary_cross_entropy(
                p_ge.clamp(1e-7, 1 - 1e-7), tgt)
        if args.rl > 0:
            # RCDL-lite: REINFORCE over Gaussian-perturbed distributions,
            # reward = proper scoring rule (log + 0.75*spherical)
            G = args.rl_samples
            noise = torch.randn(G, logits.numel(), device=device) * args.rl_noise
            noise = noise - noise.mean(dim=-1, keepdim=True)
            cand = logits.detach().unsqueeze(0) + noise
            probs = torch.softmax(cand, dim=-1)
            rewards = torch.stack([_reward(p, gold) for p in probs])
            adv = rewards - rewards.mean()
            pg = -(adv.detach().unsqueeze(-1) * (noise / (args.rl_noise ** 2))
                   * logits.unsqueeze(0)).sum(-1).mean()
            loss = loss + args.rl * pg
        window_size = min(args.accum, args.steps - (step // args.accum) * args.accum)
        (loss / window_size).backward()
        if (step + 1) % args.accum == 0 or step + 1 == args.steps:
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
    ap.add_argument("--decisions-train", default=None,
                    help="K-way decisions jsonl {ctx,opts,gold} (R2 path)")
    ap.add_argument("--levels", default="L0,L1,L2,L3",
                    help="comma list of pair levels to train on (R2)")
    ap.add_argument("--recipe", choices=["scientific-v1"], default=None,
                    help="matched scientific training recipe; requires "
                         "--decision-type (R2 path)")
    ap.add_argument("--decision-type", choices=["choice", "noul", "score"],
                    default=None, help="task type for the matched recipe")
    ap.add_argument("--strict-inputs", action="store_true", default=None,
                    help="reject any input truncation instead of silently "
                         "truncating (R2; default follows data encoding)")
    ap.add_argument("--allow-truncation", dest="strict_inputs",
                    action="store_false",
                    help="explicitly permit truncation with reporting")
    ap.add_argument("--freeze", action="store_true",
                    help="R2: freeze backbone, train DecisionHead only")
    ap.add_argument("--r2-mode", choices=["marker", "spanpool"],
                    default="marker")
    ap.add_argument("--head-kind", choices=["mlp", "attnpool"], default="mlp",
                    help="attnpool: attention pooling over option tokens + "
                         "multi-layer features (spanpool mode only)")
    ap.add_argument("--r2-layers", default="-1",
                    help="comma list of hidden-layer indices for attnpool "
                         "(negatives ok, e.g. '-1,-9,-17,-25')")
    ap.add_argument("--orders", type=int, default=1,
                    help="option orders averaged per training example "
                         "(order-robustness; >1 slows each step)")
    ap.add_argument("--canonical-order", action="store_true",
                    help="present options in deterministic content-sorted "
                         "order; flip_rate = 0 by construction")
    ap.add_argument("--ordinal", type=float, default=0.0,
                    help="weight of ordinal cumulative-BCE auxiliary loss "
                         "(ordered K-way decisions like score levels)")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--dev", default=None, help="held-out for temperature fit")
    ap.add_argument("--ckpt", default=None, help="init backbone from checkpoint")
    ap.add_argument("--hf-backbone", default=None,
                    help="HF model name/path (e.g. GSAI-ML/LLaDA-8B-Instruct) "
                         "instead of an ecoreasoner --ckpt")
    ap.add_argument("--lora-adapter", default=None,
                    help="LoRA adapter dir (e.g. DAPT output) merged into the "
                         "HF backbone before training")
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
    ap.add_argument("--soft-labels", default=None,
                    help="jsonl {qid, soft:[p...]} — KL distillation term")
    ap.add_argument("--soft-weight", type=float, default=0.5)
    ap.add_argument("--soft-temp", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- R2 path: option head on id-examples, no tokenizer ----------
    if args.pairs_train or args.decisions_train:
        apply_scientific_recipe(args)
        from . import protocol
        protocol.ensure_fresh_output(out_dir)
        from .data import load_pairs_dir, load_decisions_ids
        from .model import (DecisionHead, AttnPoolHead, HFBackbone,
                            forward_feats)
        cfg = None
        if args.config:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text()).get("model", {})
        if args.hf_backbone:
            from .model import HFBackbone
            model = HFBackbone(args.hf_backbone, device=args.device,
                               lora_adapter=args.lora_adapter)
        else:
            model = load_backbone(args.ckpt, cfg, device=args.device)
        args.layers_list = [int(x) for x in args.r2_layers.split(",")]
        if args.head_kind == "attnpool":
            if args.r2_mode != "spanpool":
                ap.error("--head-kind attnpool requires --r2-mode spanpool")
            head = AttnPoolHead(model.tok_emb.embedding_dim,
                                n_layers=len(args.layers_list)
                                ).to(args.device)
        else:
            head = DecisionHead(model.tok_emb.embedding_dim).to(args.device)
        examples = []
        if args.pairs_train:
            want = set(args.levels.split(","))
            examples += [(c, [o, b], 0)
                         for lvl, ps in load_pairs_dir(args.pairs_train).items()
                         if lvl in want for c, o, b in ps]
        if args.decisions_train:
            examples += load_decisions_ids(args.decisions_train)
        if args.soft_labels:
            from .data import load_soft_labels
            soft = load_soft_labels(args.soft_labels)
            n_soft = 0
            for ex in examples:
                if isinstance(ex, dict) and ex.get("qid") in soft:
                    ex["soft"] = soft[ex["qid"]]
                    n_soft += 1
            print(f"[data] soft labels attached: {n_soft}/{len(examples)}")
        if not examples:
            raise SystemExit("no examples loaded")
        print(f"[data] {len(examples)} examples "
              f"(mode={args.r2_mode}, freeze={args.freeze})")
        model, head = train_r2(model, head, examples, args, args.device)
        ckpt = r2_checkpoint(model, head, examples, args)
        if isinstance(model, HFBackbone):
            ckpt["hf_backbone"] = model.hf_name   # head only; 16GB not stored
            if args.lora_adapter:
                ckpt["lora_adapter"] = args.lora_adapter
        else:
            ckpt["model"] = model.state_dict()
        torch.save(ckpt, out_dir / "decision.pt")
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
            pg = -(adv.detach().unsqueeze(-1) * (noise / (args.rl_noise ** 2)) * sliced.unsqueeze(0)).sum(-1).mean()
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
