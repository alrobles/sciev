import argparse
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from .data import content_fingerprint, load_decisions_ids
from .eval import eval_decisions_ids
from .model import AttnPoolHead, HFBackbone
from .protocol import assert_disjoint_splits, dataset_contract, ensure_fresh_output, file_fingerprint
from .train import r2_checkpoint, train_r2


def selection_subset(rows, limit, seed):
    groups = sorted({row["split_group"] for row in rows}, key=lambda group: content_fingerprint([seed, group]))
    chosen, count = set(), 0
    sizes = Counter(row["split_group"] for row in rows)
    for group in groups:
        if count >= limit:
            break
        chosen.add(group)
        count += sizes[group]
    return [row for row in rows if row["split_group"] in chosen]


def require_training_data(rows):
    if any(str(row.get("source", "")).lower() == "gpqa" or
           str(row.get("task", "")).lower().startswith("gpqa") for row in rows):
        raise ValueError("GPQA is forbidden in training and model selection")
    if any(row.get("kind") != "choice" or row.get("encoding") != "systemone-v2" for row in rows):
        raise ValueError("QA experiments require systemone-v2 choice data")
    return dataset_contract(rows)


def trainable_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_trainable(model, state):
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("resume trainable backbone parameters do not match this adapter")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter))


def enable_lora(model, rank, alpha, dropout):
    from .dapt import lora_wrap

    base = model.hf
    inner = getattr(base, "model", base)
    module = sys.modules[type(inner).__module__]
    strategy = getattr(module, "ActivationCheckpointingStrategy", None)
    if strategy is None or not hasattr(inner, "set_activation_checkpointing"):
        raise ValueError("this LoRA recipe requires the LLaDA activation-checkpointing API")
    model.hf = lora_wrap(base, rank, alpha, dropout)
    inner.set_activation_checkpointing(strategy.whole_layer)
    base.enable_input_require_grads()
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names or any("lora_" not in name for name in names):
        raise ValueError("only LoRA adapter parameters may be trainable in the backbone")
    model.hf.print_trainable_parameters()


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions-train", required=True)
    ap.add_argument("--selection-dev", required=True)
    ap.add_argument("--heldout", action="append", default=[])
    ap.add_argument("--science-train")
    ap.add_argument("--science-repeats", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--head-init")
    ap.add_argument("--hf-backbone", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--r2-layers", default="-1,-9,-17,-25")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--select-limit", type=int, default=512)
    ap.add_argument("--max-seconds", type=float, default=16500)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if min(args.steps, args.accum, args.save_every, args.eval_every, args.select_limit, args.science_repeats) < 1:
        ap.error("budgets and sampling counts must be positive")
    if args.lora_rank < 0 or args.steps % args.accum or args.save_every % args.accum or args.eval_every % args.accum:
        ap.error("rank must be nonnegative; steps/save/eval intervals must be accumulation multiples")
    if args.smoke and args.steps > 16:
        ap.error("smoke runs must explicitly request at most 16 steps")
    if args.max_seconds <= 0 or args.grad_clip <= 0:
        ap.error("max-seconds and grad-clip must be positive")
    out = Path(args.out).resolve()
    if not args.resume:
        ensure_fresh_output(out)
        if (out / "last_training.pt").exists():
            raise FileExistsError("use --resume or a fresh output directory")
    elif Path(args.resume).resolve().parent != out:
        ap.error("resume must use the original run directory")
    out.mkdir(parents=True, exist_ok=True)
    args.freeze = args.lora_rank == 0
    args.layers_list = [int(value) for value in args.r2_layers.split(",")]
    args.head_kind, args.r2_mode = "attnpool", "spanpool"
    args.decision_type = "choice"
    args.canonical_order = args.strict_inputs = True
    args.orders, args.ordinal, args.rl, args.soft_weight, args.pairs_train = 1, 0.0, 0.0, 0.0, None
    args.training_contract = "qa-v2-frozen" if args.freeze else "qa-v2-lora"
    rows = load_decisions_ids(args.decisions_train)
    input_paths = [args.decisions_train, args.selection_dev, *args.heldout]
    if args.science_train:
        rows += load_decisions_ids(args.science_train) * args.science_repeats
        input_paths.append(args.science_train)
    training_contract = require_training_data(rows)
    initial_head = None
    if args.head_init:
        initial_head = torch.load(args.head_init, map_location="cpu", weights_only=True)
        inference = initial_head.get("inference", {})
        known = initial_head.get("training_data", {}).get("identity_hashes")
        if (not known or not set(known).issubset(training_contract["identity_hashes"])
                or initial_head.get("lora_adapter") or initial_head.get("hf_backbone") != args.hf_backbone
                or inference.get("layers") != args.layers_list or inference.get("mode") != "spanpool"
                or inference.get("canonical_order") is not True):
            raise ValueError("head initialization requires a compatible frozen head trained only on the allowed data")
        input_paths.append(args.head_init)
    selection = load_decisions_ids(args.selection_dev)
    require_training_data(selection)
    assert_disjoint_splits({"train": rows, "selection": selection})
    for path in args.heldout:
        assert_disjoint_splits({"train": rows, "selection": selection, "heldout": load_decisions_ids(path)})
    total_rows = len(rows)
    if args.smoke:
        rows = sorted(rows, key=lambda row: len(row["ctx"]) + sum(map(len, row["opts"])), reverse=True)[:8]
    selection = selection_subset(selection, args.select_limit, args.seed)
    fingerprints = [{**file_fingerprint(path),
                     "role": "initial_head" if path == args.head_init else
                             "training" if path in (args.decisions_train, args.science_train) else
                             "model_selection" if path == args.selection_dev else "heldout_overlap_guard"}
                    for path in input_paths]
    config = {key: value for key, value in vars(args).items() if key not in ("resume", "max_seconds", "device")}
    resume = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resume and (resume["configuration"] != config or resume["input_files"] != fingerprints):
        raise ValueError("resume configuration or input fingerprints changed")
    torch.manual_seed(args.seed)
    model = HFBackbone(args.hf_backbone, device=args.device)
    if args.lora_rank:
        enable_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    else:
        model.requires_grad_(False)
    torch.manual_seed(args.seed)
    head = AttnPoolHead(model.tok_emb.embedding_dim, len(args.layers_list)).to(args.device)
    if initial_head is not None:
        if initial_head.get("model_revision") != getattr(model.hf.config, "_commit_hash", None):
            raise ValueError("initial head backbone revision differs from the runtime model")
        head.load_state_dict(initial_head["head"])
    if resume:
        head.load_state_dict(resume["head"])
        restore_trainable(model, resume["backbone_trainable"])
    metadata = r2_checkpoint(model, head, rows, args)
    metadata.pop("head")
    if initial_head is not None:
        metadata["training_data"] = training_contract
        metadata["initial_head"] = file_fingerprint(args.head_init)
    metadata.update(hf_backbone=model.hf_name, input_files=fingerprints,
                    selection_data=dataset_contract(selection),
                    model_revision=getattr(model.hf.config, "_commit_hash", None),
                    code=[file_fingerprint(Path(__file__)), file_fingerprint(Path(__file__).with_name("train.py")),
                          file_fingerprint(Path(__file__).with_name("model.py"))])
    if resume and resume.get("model_revision") != metadata["model_revision"]:
        raise ValueError("resume backbone revision changed")
    progress = dict(resume["progress"]) if resume else {"completed_steps": 0, "best_step": None, "best_selection": None}
    atomic_json(out / "training_config.json", {**config, "input_files": fingerprints,
                "full_sampling_rows": total_rows, "actual_sampling_rows": len(rows),
                "source_sampling_counts": dict(Counter(row.get("source") for row in rows)),
                "selection_n": len(selection), "selection_identity_sha256": content_fingerprint([row["qid"] for row in selection]),
                "training_data_scope": "GPQA excluded; selection/temperature/policy/test roles disjoint",
                "code": metadata["code"]})
    stop = {"requested": False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(requested=True))
    started = time.perf_counter()
    initial_adapter = trainable_state(model) if args.smoke and args.lora_rank else None
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def checkpoint(step, model, head, optimizer, rng, order):
        progress["completed_steps"] = step
        stopping = stop["requested"] or time.perf_counter() - started >= args.max_seconds
        final = step == args.steps
        evaluate = step % args.eval_every == 0 or stopping or final
        if not (step % args.save_every == 0 or evaluate):
            return False
        if evaluate:
            model.eval()
            head.eval()
            metrics = eval_decisions_ids(model, head, selection, args.device, mode="spanpool",
                                          layers=args.layers_list, canonical=True, decision_type="choice", strict=True,
                                          return_predictions=args.smoke)
            previous = progress["best_selection"]
            score = (metrics["acc"], -(metrics["nll"] if metrics["nll"] is not None else float("inf")))
            old_score = (-1, -float("inf")) if previous is None else (
                previous["acc"], -(previous["nll"] if previous["nll"] is not None else float("inf")))
            if score > old_score:
                best = {**metadata, "head": head.state_dict(), "selected_step": step,
                        "selection": {key: metrics[key] for key in ("n", "acc", "nll", "brier", "ece")}}
                if args.lora_rank:
                    adapter = out / "best_adapter"
                    model.hf.save_pretrained(adapter)
                    best["lora_adapter"] = str(adapter)
                if args.smoke:
                    best["reload_reference"] = metrics["predictions"][0]
                atomic_torch(out / "decision.pt", best)
                progress.update(best_step=step, best_selection=best["selection"])
            atomic_json(out / f"selection_{step:06d}.json", metrics)
            model.train(not args.freeze)
            head.train()
        state = {"step": step, "head": head.state_dict(), "backbone_trainable": trainable_state(model),
                 "optimizer": optimizer.state_dict(), "rng_state": rng.getstate(), "order": list(order),
                 "torch_rng_state": torch.get_rng_state(), "configuration": config, "input_files": fingerprints,
                 "model_revision": metadata["model_revision"], "progress": dict(progress)}
        if args.device.startswith("cuda"):
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
            progress["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
            progress["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024 ** 3
            progress["gpu"] = torch.cuda.get_device_name()
        atomic_torch(out / "last_training.pt", state)
        atomic_json(out / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        return stopping and not final

    train_r2(model, head, rows, args, args.device, resume=resume, checkpoint_callback=checkpoint)
    progress["budget_complete"] = progress["completed_steps"] == args.steps
    progress["invocation_elapsed_s"] = time.perf_counter() - started
    if initial_adapter is not None:
        current = trainable_state(model)
        changed = sum(not torch.equal(value, current[name]) for name, value in initial_adapter.items())
        progress["changed_adapter_tensors"] = changed
        if not changed:
            raise RuntimeError("LoRA smoke failed: no adapter tensor changed")
    if args.smoke:
        import gc
        from .model import load_decision

        del model, head
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        saved = torch.load(out / "decision.pt", map_location="cpu", weights_only=True)
        model, head = load_decision(out / "decision.pt", device=args.device)
        model.eval()
        observed = eval_decisions_ids(model, head, selection[:1], args.device, mode="spanpool",
                                      layers=args.layers_list, canonical=True, decision_type="choice", strict=True,
                                      return_predictions=True)["predictions"][0]
        delta = max(abs(left - right) for left, right in zip(
            observed["probabilities"], saved["reload_reference"]["probabilities"]))
        progress["reload_max_probability_delta"] = delta
        progress["reload_verified"] = delta < 0.03
        if not progress["reload_verified"]:
            raise RuntimeError(f"saved checkpoint reload changed probabilities by {delta}")
    atomic_json(out / "progress.json", progress)
    print("COMPLETE" if progress["budget_complete"] else "RESUME_REQUIRED", flush=True)
    if not progress["budget_complete"]:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
