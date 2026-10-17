"""sciev.biencoder — CLM-style contrastive bi-encoder arm (matched backbone).

Reimplements the Contrastive-LM architecture (Kwok et al. 2026;
github.com/Contrastive-LM/CLM) on sciev's own frozen backbone: state and
action are embedded independently (mean-pooled hidden states), projected
by two MLP heads and scored by a scale-tempered cosine similarity. The
InfoNCE objective uses a question's distractor options as hard negatives
and the rest of the batch as in-batch negatives, in both directions.

The frozen backbone makes embeddings deterministic, so they are cached
by content hash and head training runs without further backbone
forwards. This arm is the matched-backbone comparison for the joint
bidirectional decision heads (sciev.train / sciev.qa_train): same
encoder, same data, dot-product instead of cross-attention. CLM-8B's
released weights require the Qwen3-8B encoder, which is out of scope
for this project; this module compares architectures, not artifacts.

  python -m sciev.biencoder embed --backbone GSAI-ML/LLaDA-8B-Instruct \
      --decisions train.jsonl dev.jsonl --cache embeds.pt --device cuda
  python -m sciev.biencoder train --decisions-train train.jsonl \
      --selection-dev select.jsonl --cache embeds.pt --out runs/bi
  python -m sciev.biencoder eval --heads runs/bi/decision.pt \
      --cache embeds.pt --calibration-dev cal.jsonl --policy-dev pol.jsonl \
      --evaluation gpqa_main=gpqa.jsonl --out-dir runs/bi_eval
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import content_fingerprint, load_decisions_ids
from .metrics import classification_metrics, evaluate_acceptance_policy, fit_acceptance_policy
from .protocol import (assert_checkpoint_disjoint, assert_disjoint_splits,
                       dataset_contract, ensure_fresh_output, file_fingerprint)

CACHE_FORMAT = "sciev-biencoder-embeds-v1"
CHECKPOINT_FORMAT = "sciev-biencoder-v1"
PREDICTION_KEYS = ("qid", "kind", "pid", "split_group", "group_id",
                   "option_keys", "sample_id", "content_hash", "decision_id", "source")


class ProjectionHead(nn.Module):
    """MLP projector: (dim -> width)^(depth-1) -> LayerNorm -> Linear -> L2-normalized."""

    def __init__(self, dim, proj_dim, width=None, depth=2):
        super().__init__()
        width = width or dim
        blocks, d = [], dim
        for _ in range(max(0, depth - 1)):
            blocks += [nn.Linear(d, width), nn.GELU()]
            d = width
        blocks += [nn.LayerNorm(d), nn.Linear(d, proj_dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return F.normalize(self.net(x.float()), dim=-1)


class BiEncoderHeads(nn.Module):
    """State/action projection pair with a learnable logit scale (CLIP-style)."""

    def __init__(self, dim, proj_dim=512, width=None, depth=2, logit_scale=None):
        super().__init__()
        self.cfg = {"dim": dim, "proj_dim": proj_dim, "width": width,
                    "depth": depth}
        self.state_head = ProjectionHead(dim, proj_dim, width, depth)
        self.action_head = ProjectionHead(dim, proj_dim, width, depth)
        self.logit_scale = nn.Parameter(torch.tensor(
            math.log(1 / 0.07) if logit_scale is None else float(logit_scale)))

    def scale(self):
        return self.logit_scale.exp().clamp(max=100.0)

    def decision_logits(self, state_vec, action_vecs):
        """(D,) state x (K,D) actions -> (K,) scaled cosine logits."""
        device = self.logit_scale.device
        s = self.state_head(state_vec.unsqueeze(0).to(device))[0]
        return (self.action_head(action_vecs.to(device)) @ s) * self.scale()


def _layer_output(model, ids, layer):
    if hasattr(model, "hidden_layers"):
        return model.hidden_layers(ids, [layer])[0, 0]
    return model(ids, skip_head=True)[0]


def embed_sequences(model, sequences, device, layer=-1):
    """One forward per sequence (these backbones have no padding mask);
    masked-free mean pool of the requested layer's hidden states."""
    vectors = []
    with torch.inference_mode():
        for sequence in sequences:
            ids = torch.tensor(list(sequence), dtype=torch.long,
                               device=device).unsqueeze(0)
            vectors.append(_layer_output(model, ids, layer).mean(0).float().cpu())
    return torch.stack(vectors) if vectors else torch.zeros(0, 1)


def cache_key(sequence):
    return content_fingerprint([int(token) for token in sequence])


def _row_sequences(rows):
    for row in rows:
        yield row["ctx"]
        yield from row["opts"]


def embed_dataset(model, rows, device, layer=-1, existing=None, batch=64,
                  on_batch=None):
    """Extend an embedding cache {key: (D,) float32 tensor} for all rows.

    on_batch(cache, done, total), if given, runs after every batch —
    use it for periodic atomic persistence so a timeout resumes.
    """
    cache = dict(existing or {})
    wanted, seen = [], set()
    for sequence in _row_sequences(rows):
        key = cache_key(sequence)
        if key not in cache and key not in seen:
            seen.add(key)
            wanted.append(list(sequence))
    for start in range(0, len(wanted), batch):
        vectors = embed_sequences(model, wanted[start:start + batch], device, layer)
        for sequence, vector in zip(wanted[start:start + batch], vectors):
            cache[cache_key(sequence)] = vector
        if on_batch is not None:
            on_batch(cache, min(start + batch, len(wanted)), len(wanted))
    return cache


def save_cache(path, cache, *, backbone, layer, source_files, complete=True):
    payload = {"format": CACHE_FORMAT, "backbone": backbone, "layer": layer,
               "source_files": source_files, "complete": complete,
               "vectors": {key: tensor for key, tensor in cache.items()}}
    _atomic_torch(Path(path), payload)


def load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path} is not a {CACHE_FORMAT} embedding cache")
    vectors = payload.get("vectors")
    if not isinstance(vectors, dict) or not vectors:
        raise ValueError("embedding cache contains no vectors")
    return payload


def merge_caches(paths):
    merged, provenance = {}, []
    for path in paths:
        payload = load_cache(path)
        for key, tensor in payload["vectors"].items():
            other = merged.get(key)
            if other is not None and not torch.equal(other, tensor):
                raise ValueError(f"conflicting cached embedding for {key}")
            merged[key] = tensor
        provenance.append(file_fingerprint(path))
    return merged, provenance


def _row_vectors(cache, rows):
    """Map rows to (state_vec, action_vecs, gold); fails loudly on cache misses."""
    tensors, missing = [], 0
    for row in rows:
        keys = [cache_key(row["ctx"]), *[cache_key(option) for option in row["opts"]]]
        if any(key not in cache for key in keys):
            missing += 1
            continue
        tensors.append((cache[keys[0]], torch.stack([cache[key] for key in keys[1:]]),
                        int(row["gold"]), row))
    if missing:
        raise ValueError(f"embedding cache is missing {missing} rows")
    return tensors


def infonce_loss(heads, state_vecs, action_vecs, golds):
    """Bidirectional InfoNCE: state->actions over own options + the whole
    batch (distractors are the hard negatives), action->state symmetric."""
    states = heads.state_head(state_vecs)
    actions = heads.action_head(action_vecs.reshape(-1, action_vecs.shape[-1]))
    scale = heads.scale()
    bsz, k = action_vecs.shape[:2]
    targets = torch.arange(bsz, device=state_vecs.device) * k + golds
    forward = F.cross_entropy((states @ actions.T) * scale, targets)
    gold_actions = actions.view(bsz, k, -1)[torch.arange(bsz, device=state_vecs.device), golds]
    reverse = F.cross_entropy((gold_actions @ states.T) * scale,
                              torch.arange(bsz, device=state_vecs.device))
    return (forward + reverse) / 2


def eval_vectors(heads, vectors, temperature=1.0, return_predictions=False):
    """Score every row from cached embeddings -> protocol-compatible metrics."""
    probs_rows, golds, predictions, records = [], [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for state_vec, action_vecs, gold, row in vectors:
            logits = heads.decision_logits(state_vec, action_vecs).double()
            probs = torch.softmax(logits / temperature, dim=0).tolist()
            prediction = min(range(len(probs)), key=lambda i: (-probs[i], i))
            probs_rows.append(probs)
            golds.append(gold)
            predictions.append(prediction)
            if return_predictions:
                records.append({
                    **{key: row[key] for key in PREDICTION_KEYS if key in row},
                    "gold": gold, "prediction": prediction,
                    "probabilities": probs, "max_probability": max(probs),
                    "correct": int(prediction == gold)})
    metrics = classification_metrics(probs_rows, golds, predictions=predictions)
    metrics.update(elapsed_s=time.perf_counter() - started,
                   forward_passes=0, backbone_forwards=0,
                   scoring="cached_embeddings; one head forward per decision")
    if return_predictions:
        metrics["predictions"] = records
    return metrics


def recall_at_k(heads, vectors, ks, pool="row"):
    """Recall@k / MRR of the dot-product readout — can it shortlist the gold?

    pool="row":   rank the gold action among the row's own options. This is
                  the K>4 prefilter question: banking77 rows carry all 77
                  labels, so recall@4 tells whether a joint head over the
                  top-4 shortlist could still fire.
    pool="suite": rank it among every unique action vector in the file —
                  global action retrieval, the harsher setting (deduped by
                  vector identity; ties never count against the gold).
    """
    if not vectors:
        raise ValueError("recall evaluation requires at least one row")
    ks = sorted({int(k) for k in ks})
    if not ks or ks[0] < 1:
        raise ValueError("ks must be positive integers")
    device = heads.logit_scale.device
    ranks = []
    pool_size = None
    with torch.inference_mode():
        if pool == "row":
            for state_vec, action_vecs, gold, _ in vectors:
                logits = heads.decision_logits(state_vec, action_vecs)
                ranks.append(1 + int((logits > logits[gold]).sum()))
        elif pool == "suite":
            index, pool_vecs, gold_index = {}, [], []
            for _, action_vecs, gold, _ in vectors:
                for j, vec in enumerate(action_vecs):
                    key = vec.numpy().tobytes()
                    if key not in index:
                        index[key] = len(pool_vecs)
                        pool_vecs.append(vec)
                    if j == gold:
                        gold_index.append(index[key])
            pool_size = len(pool_vecs)
            projected = heads.action_head(torch.stack(pool_vecs).to(device))
            for i, (state_vec, _, _, _) in enumerate(vectors):
                sims = projected @ heads.state_head(
                    state_vec.unsqueeze(0).to(device))[0]
                ranks.append(1 + int((sims > sims[gold_index[i]]).sum()))
        else:
            raise ValueError(f"unknown pool: {pool}")
    n = len(ranks)
    metrics = {"n": n, "pool": pool, "mrr": sum(1.0 / r for r in ranks) / n}
    if pool_size is not None:
        metrics["pool_size"] = pool_size
    for k in ks:
        metrics[f"recall@{k}"] = sum(r <= k for r in ranks) / n
    return metrics


def fit_bi_temperature(heads, vectors):
    """Scalar temperature minimizing NLL over cached-vector logits."""
    groups = defaultdict(list)
    for state_vec, action_vecs, gold, _ in vectors:
        groups[len(action_vecs)].append((state_vec, action_vecs, gold))
    if not groups:
        raise ValueError("temperature fitting requires at least one decision")

    def nll(temperature):
        total = 0.0
        for group in groups.values():
            for state_vec, action_vecs, gold in group:
                logits = heads.decision_logits(state_vec, action_vecs).double()
                total += float(F.cross_entropy(logits.div(temperature).unsqueeze(0),
                                               torch.tensor([gold])).item())
        return total

    grid = list(torch.linspace(0.2, 5.0, 97)) + list(torch.linspace(5.5, 20.0, 30))
    best = min(grid, key=lambda t: nll(float(t)))
    lo, hi = max(0.05, float(best) * 0.5), float(best) * 2.0
    for _ in range(40):
        a, b = math.log(lo), math.log(hi)
        c, d = b - (b - a) * 0.618, a + (b - a) * 0.618
        if nll(math.exp(c)) < nll(math.exp(d)):
            hi = math.exp(d)
        else:
            lo = math.exp(c)
    return math.exp((math.log(lo) + math.log(hi)) / 2)


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_torch(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_heads(path, heads, **metadata):
    payload = {"format": CHECKPOINT_FORMAT,
               "state_head": heads.state_head.state_dict(),
               "action_head": heads.action_head.state_dict(),
               "logit_scale": float(heads.logit_scale.detach()),
               "cfg": dict(heads.cfg), **metadata}
    _atomic_torch(Path(path), payload)


def load_heads(path, device="cpu", dim=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    cfg = checkpoint["cfg"]
    heads = BiEncoderHeads(dim or cfg["dim"], cfg["proj_dim"], cfg["width"],
                           cfg["depth"], logit_scale=checkpoint["logit_scale"])
    heads.state_head.load_state_dict(checkpoint["state_head"])
    heads.action_head.load_state_dict(checkpoint["action_head"])
    return heads.to(device).eval(), checkpoint


# ---------------------------------------------------------------- CLI: embed

def _cmd_embed(args):
    from .model import HFBackbone

    if min(args.batch, args.save_every) < 1:
        raise ValueError("--batch and --save-every must be positive")
    rows_by_file = {path: load_decisions_ids(path) for path in args.decisions}
    model = HFBackbone(args.backbone, device=args.device)
    model.eval().requires_grad_(False)
    cache_path = Path(args.cache)
    if cache_path.exists():
        payload = load_cache(cache_path)
        if payload["backbone"] != args.backbone or payload["layer"] != args.layer:
            raise ValueError(
                "existing cache was built with a different backbone/layer; "
                "use a fresh --cache path")
        cache, done_files = payload["vectors"], set(payload["source_files"])
        print(json.dumps({"resumed_cache": str(cache_path),
                          "vectors": len(cache)}), flush=True)
    else:
        cache, done_files = {}, set()
    saves = {"n": 0}

    def persist(live_cache, done, total):
        saves["n"] += 1
        if saves["n"] % args.save_every == 0 and done < total:
            save_cache(cache_path, live_cache, backbone=args.backbone,
                       layer=args.layer, source_files=sorted(done_files),
                       complete=False)

    for path, rows in rows_by_file.items():
        before = len(cache)
        cache = embed_dataset(model, rows, args.device, layer=args.layer,
                              existing=cache, batch=args.batch,
                              on_batch=persist)
        done_files.add(file_fingerprint(path)["sha256"])
        save_cache(cache_path, cache, backbone=args.backbone, layer=args.layer,
                   source_files=sorted(done_files), complete=False)
        print(json.dumps({"file": path, "rows": len(rows),
                          "new_embeddings": len(cache) - before}), flush=True)
    save_cache(cache_path, cache, backbone=args.backbone, layer=args.layer,
               source_files=sorted(done_files), complete=True)
    print(json.dumps({"cache": str(args.cache), "vectors": len(cache)}), flush=True)


# ---------------------------------------------------------------- CLI: train

def _require_choice_rows(rows, name):
    if any(row.get("kind") not in (None, "choice") for row in rows):
        raise ValueError(f"{name} must contain only choice decisions")


def _train(args):
    if min(args.steps, args.bs, args.warmup, args.eval_every, args.proj_dim,
           args.depth, args.science_repeats) < 1 or args.select_limit < 0:
        raise ValueError("steps/bs/warmup/eval-every/proj-dim/depth/science-repeats must be positive")
    if args.lr <= 0 or args.grad_clip <= 0 or args.max_seconds <= 0:
        raise ValueError("lr, grad-clip and max-seconds must be positive")
    out = Path(args.out).resolve()
    ensure_fresh_output(out)
    out.mkdir(parents=True, exist_ok=True)
    from .qa_train import require_training_data

    rows = load_decisions_ids(args.decisions_train)
    if args.science_train:
        rows += load_decisions_ids(args.science_train) * args.science_repeats
    selection = load_decisions_ids(args.selection_dev)
    heldout = {Path(path).name: load_decisions_ids(path) for path in args.heldout}
    require_training_data(rows)
    require_training_data(selection)
    _require_choice_rows(rows, "train")
    _require_choice_rows(selection, "selection")
    assert_disjoint_splits({"train": rows, "selection": selection})
    # heldout files are pairwise-checked against train/selection only —
    # they may legitimately overlap each other (gpqa_diamond ⊂ gpqa_main)
    for name, held_rows in heldout.items():
        assert_disjoint_splits({"train": rows, "selection": selection,
                                name: held_rows})
    cache, cache_files = merge_caches(args.cache)
    train_vectors = _row_vectors(cache, rows)
    select_vectors = _row_vectors(cache, selection)
    if args.select_limit:
        from .qa_train import selection_subset
        chosen = set(id(row) for row in selection_subset(
            selection, args.select_limit, args.seed))
        select_vectors = [vector for vector, row in zip(select_vectors, selection)
                          if id(row) in chosen]
    by_k = defaultdict(list)
    for item in train_vectors:
        by_k[item[1].shape[0]].append(item)
    dim = train_vectors[0][0].numel()
    torch.manual_seed(args.seed)
    heads = BiEncoderHeads(dim, args.proj_dim, args.width, args.depth).to(args.device)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr)
    config = {"format": CHECKPOINT_FORMAT, "seed": args.seed, "steps": args.steps,
              "bs": args.bs, "lr": args.lr, "warmup": args.warmup,
              "grad_clip": args.grad_clip, **{f"cfg_{k}": v for k, v in heads.cfg.items()},
              "input_files": [file_fingerprint(p) for p in
                              [args.decisions_train, args.selection_dev,
                               args.science_train, *args.heldout] if p],
              "embedding_caches": cache_files,
              "training_data": dataset_contract(rows),
              "selection_data": dataset_contract(selection),
              "code": [file_fingerprint(Path(__file__)),
                       file_fingerprint(Path(__file__).with_name("protocol.py"))]}
    _atomic_json(out / "training_config.json", config)
    rng = torch.Generator().manual_seed(args.seed)
    progress = {"completed_steps": 0, "best_step": None, "best_selection": None}
    started = time.perf_counter()

    def evaluate(step):
        metrics = eval_vectors(heads, select_vectors)
        record = {key: metrics[key] for key in ("n", "acc", "nll", "brier", "ece")}
        _atomic_json(out / f"selection_{step:06d}.json", metrics)
        previous = progress["best_selection"]
        score = (metrics["acc"], -(metrics["nll"] or float("inf")))
        best_score = (-1, -float("inf")) if previous is None else (
            previous["acc"], -(previous["nll"] if previous["nll"] is not None else float("inf")))
        if score > best_score:
            save_heads(out / "decision.pt", heads, step=step, selection=record,
                       **{key: config[key] for key in
                          ("input_files", "embedding_caches", "training_data",
                           "selection_data", "code")})
            progress.update(best_step=step, best_selection=record)

    for step in range(1, args.steps + 1):
        group = by_k[sorted(by_k)[step % len(by_k)]]
        picks = torch.randint(len(group), (min(args.bs, len(group)),), generator=rng)
        batch = [group[int(i)] for i in picks]
        states = torch.stack([item[0] for item in batch]).to(args.device)
        actions = torch.stack([item[1] for item in batch]).to(args.device)
        golds = torch.tensor([item[2] for item in batch], device=args.device)
        loss = infonce_loss(heads, states, actions, golds)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        lr_scale = min(1.0, step / max(1, args.warmup))
        for group_lr in opt.param_groups:
            group_lr["lr"] = args.lr * lr_scale
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), args.grad_clip,
                                       error_if_nonfinite=True)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            progress["completed_steps"] = step
            progress["elapsed_s"] = time.perf_counter() - started
            _atomic_json(out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if time.perf_counter() - started >= args.max_seconds:
            break
    progress["budget_complete"] = progress["completed_steps"] == args.steps
    progress["elapsed_s"] = time.perf_counter() - started
    _atomic_json(out / "progress.json", progress)
    print("COMPLETE" if progress["budget_complete"] else "RESUME_REQUIRED", flush=True)
    if not progress["budget_complete"]:
        raise SystemExit(75)


# ------------------------------------------------------------------ CLI: eval

def _eval(args):
    out = Path(args.out_dir).resolve()
    heads, checkpoint = load_heads(args.heads, device=args.device)
    cache, cache_files = merge_caches(args.cache)
    temperature_rows = load_decisions_ids(args.calibration_dev)
    policy_rows = load_decisions_ids(args.policy_dev) if args.policy_dev else None
    from .qa_train import require_training_data

    require_training_data(temperature_rows)
    if policy_rows is not None:
        require_training_data(policy_rows)
    evaluations = {}
    for setting in args.evaluation:
        name, separator, path = setting.partition("=")
        if not separator or not name or Path(name).name != name or name in evaluations:
            raise ValueError("evaluation must be a unique name=path without separators")
        evaluations[name] = Path(path)
    if not evaluations:
        raise ValueError("at least one --evaluation name=path is required")
    if any((out / f"{name}.json").exists() or (out / "calibration.json").exists()
           for name in evaluations):
        raise FileExistsError("evaluation artifacts already exist; use a fresh out-dir")
    checks = {}
    for name, path in evaluations.items():
        rows = load_decisions_ids(path)
        roles = {"temperature": temperature_rows, "evaluation": rows}
        if policy_rows is not None:
            roles["policy"] = policy_rows
        assert_disjoint_splits(roles)
        checks[name] = {key: value for key, value in
                        assert_checkpoint_disjoint(checkpoint, rows).items()
                        if key != "data"}
        selection_checkpoint = {"training_data": checkpoint.get("selection_data")}
        checks[name]["model_selection_check"] = {
            key: value for key, value in
            assert_checkpoint_disjoint(selection_checkpoint, rows,
                                       role="model_selection").items()
            if key != "data"}
    temperature_vectors = _row_vectors(cache, temperature_rows)
    temperature = fit_bi_temperature(heads, temperature_vectors)
    policy = None
    if policy_rows is not None:
        policy_vectors = _row_vectors(cache, policy_rows)
        metrics = eval_vectors(heads, policy_vectors, temperature=temperature,
                               return_predictions=True)
        policy = fit_acceptance_policy(
            [row["max_probability"] for row in metrics["predictions"]],
            [row["correct"] for row in metrics["predictions"]])
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "calibration.json", {
        "temperature": temperature, "acceptance_policy": policy,
        "temperature_dev": file_fingerprint(args.calibration_dev),
        "policy_dev": file_fingerprint(args.policy_dev) if args.policy_dev else None,
        "logit_scale": checkpoint["logit_scale"], "heads": file_fingerprint(args.heads)})
    for name, path in evaluations.items():
        rows = load_decisions_ids(path)
        vectors = _row_vectors(cache, rows)
        metrics = eval_vectors(heads, vectors, temperature=temperature,
                               return_predictions=True)
        metrics["selective"] = (evaluate_acceptance_policy(
            [row["max_probability"] for row in metrics["predictions"]],
            [row["correct"] for row in metrics["predictions"]], policy)
            if policy else None)
        report = {"protocol_version": 2, "role": "evaluation",
                  "architecture": "biencoder_contrastive",
                  "checkpoint": file_fingerprint(args.heads),
                  "input_file": file_fingerprint(path),
                  "temperature_file": file_fingerprint(args.calibration_dev),
                  "calibration_file": file_fingerprint(out / "calibration.json"),
                  "training_overlap": checks[name],
                  "calibration_regime": "held-out temperature and acceptance; transferred unchanged to each test suite",
                  "code": [file_fingerprint(Path(__file__))],
                  "embedding_caches": cache_files,
                  "decisions_eval": {"file": str(path), **metrics}}
        with (out / f"{name}.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({"suite": name, **{key: metrics[key] for key in
                                            ("n", "acc", "nll", "ece", "brier")}}), flush=True)
    print("ALL_DONE", flush=True)


# ---------------------------------------------------------------- CLI: recall

def _recall(args):
    out = Path(args.out).resolve()
    heads, _ = load_heads(args.heads, device=args.device)
    cache, cache_files = merge_caches(args.cache)
    results = {}
    for path in args.decisions:
        rows = load_decisions_ids(path)
        vectors = _row_vectors(cache, rows)
        metrics = recall_at_k(heads, vectors, args.ks, pool=args.pool)
        results[Path(path).name] = metrics
        print(json.dumps({"file": Path(path).name, **metrics}), flush=True)
    report = {"protocol_version": 2, "role": "diagnostic",
              "architecture": "biencoder_contrastive",
              "heads": file_fingerprint(args.heads),
              "embedding_caches": cache_files,
              "ks": sorted(set(args.ks)), "pool": args.pool,
              "code": [file_fingerprint(Path(__file__))],
              "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"wrote": str(out)}), flush=True)


def main():
    ap = argparse.ArgumentParser(prog="sciev.biencoder")
    sub = ap.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed")
    embed.add_argument("--backbone", default="GSAI-ML/LLaDA-8B-Instruct")
    embed.add_argument("--decisions", nargs="+", required=True)
    embed.add_argument("--cache", required=True)
    embed.add_argument("--layer", type=int, default=-1)
    embed.add_argument("--batch", type=int, default=64)
    embed.add_argument("--save-every", type=int, default=20,
                       help="persist the resumable cache every N batches")
    embed.add_argument("--device", default="cuda")

    train = sub.add_parser("train")
    train.add_argument("--decisions-train", required=True)
    train.add_argument("--selection-dev", required=True)
    train.add_argument("--heldout", action="append", default=[])
    train.add_argument("--science-train")
    train.add_argument("--science-repeats", type=int, default=15)
    train.add_argument("--cache", action="append", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--steps", type=int, default=4000)
    train.add_argument("--bs", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--warmup", type=int, default=200)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--eval-every", type=int, default=500)
    train.add_argument("--select-limit", type=int, default=512)
    train.add_argument("--proj-dim", type=int, default=512)
    train.add_argument("--width", type=int, default=None)
    train.add_argument("--depth", type=int, default=2)
    train.add_argument("--max-seconds", type=float, default=16500)
    train.add_argument("--seed", type=int, default=7331)
    train.add_argument("--device", default="cuda")

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--heads", required=True)
    evaluate.add_argument("--cache", action="append", required=True)
    evaluate.add_argument("--calibration-dev", required=True)
    evaluate.add_argument("--policy-dev")
    evaluate.add_argument("--evaluation", action="append", default=[])
    evaluate.add_argument("--out-dir", required=True)
    evaluate.add_argument("--device", default="cpu")

    recall = sub.add_parser("recall")
    recall.add_argument("--heads", required=True)
    recall.add_argument("--cache", action="append", required=True)
    recall.add_argument("--decisions", nargs="+", required=True)
    recall.add_argument("--pool", choices=("row", "suite"), default="row")
    recall.add_argument("--ks", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16])
    recall.add_argument("--out", required=True)
    recall.add_argument("--device", default="cpu")

    args = ap.parse_args()
    {"embed": _cmd_embed, "train": _train, "eval": _eval,
     "recall": _recall}[args.command](args)


if __name__ == "__main__":
    main()
