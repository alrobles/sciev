"""reverse_jev.eval — decision evaluation harness.

Modes:
  E0 pairs  : --pairs pairs_L3.jsonl   (ecoreasoner ids; R1 vs denoise_loss)
  decisions : --data decisions.jsonl   (System-One labels; acc/Brier/ECE)
  remote    : --remote URL             (same decisions against any
              /v1/systemone endpoint — Jev API or a local reverse-jev server)

Examples:
  python -m reverse_jev.eval --ckpt runs/f0/checkpoint-g10000/model.pt \
      --pairs /beegfs/.../pairs_L3.jsonl --tokenizer GSAI-ML/LLaDA-8B-Instruct
  python -m reverse_jev.eval --remote https://api.typesafe.ai \
      --api-key-file ~/env/typesafe-key --data evals/eco_decisions.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_pairs, iter_decisions, split_dev_test


# ---------------- metrics ----------------

def ece(confs, corrects, n_bins=10):
    """Expected calibration error: mean |bucket_conf - bucket_acc|."""
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1]) if i else \
            (confs >= edges[i]) & (confs <= edges[i + 1])
        if m.any():
            total += m.mean() * abs(confs[m].mean() - corrects[m].mean())
    return float(total)


def brier_multi(probs, labels):
    """Mean squared error between predicted distribution and one-hot label.
    Ragged-safe: option counts differ across question types."""
    total = 0.0
    for p_row, gold in zip(probs, labels):
        p = np.asarray(p_row, dtype=float)
        y = np.zeros_like(p)
        y[gold] = 1.0
        total += float(((p - y) ** 2).sum())
    return total / max(len(labels), 1)


def automation_rate(confs, corrects, err_budget=0.05):
    """Largest fraction of decisions auto-accepted while keeping error <= budget.

    Sort by confidence desc, find the largest prefix with mean error <= budget.
    """
    order = np.argsort(-np.asarray(confs, dtype=float))
    errs = 1.0 - np.asarray(corrects, dtype=float)[order]
    cum = np.cumsum(errs) / np.arange(1, len(errs) + 1)
    ok = np.where(cum <= err_budget)[0]
    return float((ok[-1] + 1) / len(errs)) if len(ok) else 0.0


# ---------------- local model paths ----------------

def _clamp_oob(seq, model):
    """ecoreasoner corpus has stray ids >= vocab (e.g. 126082); the embedding
    only covers 0..vocab (MASK). Clamp like the trainer's GUARDIA."""
    return seq.clamp(0, model.mask_id)


def denoise_loss(model, seq, mask_p, rng, mask_id):
    """Legacy ecoreasoner pairwise scorer (suite_smoke_v2): CE over masked
    positions of ctx+candidate."""
    seq = _clamp_oob(seq, model)
    T = seq.shape[0]
    n = max(1, int(mask_p * T))
    idx = torch.tensor(rng.sample(range(T), n), dtype=torch.long, device=seq.device)
    masked = seq.clone()
    masked[idx] = mask_id
    logits = model(masked.unsqueeze(0)).squeeze(0)
    target = seq[idx].clamp(0, model.vocab - 1)
    return F.cross_entropy(logits[idx], target)


def span_logprob(model, ctx, cand, device):
    """r1_span: ctx + [MASK]*len(cand) -> mean logprob of true candidate tokens
    at the masked positions. The honest dLLM analog of 'score the option':
    every candidate token predicted simultaneously, conditioned on ctx."""
    mask_id = model.mask_id
    seq = torch.tensor(ctx + [mask_id] * len(cand), dtype=torch.long,
                       device=device)
    seq = _clamp_oob(seq, model)
    logits = model(seq.unsqueeze(0)).squeeze(0)
    pos = torch.arange(len(ctx), len(seq), device=device)
    tgt = torch.tensor([min(t, model.vocab - 1) for t in cand],
                       dtype=torch.long, device=device)
    lp = F.log_softmax(logits[pos].float(), dim=-1)
    return lp.gather(-1, tgt.unsqueeze(-1)).mean().item()


def _option_logits(model, head, ctx, ok, bad, device, mode="marker",
                   layers=(-1,)):
    """R2 forward per order. Returns [logits_a, logits_b] (2-dim tensors)."""
    from .decisions import decision_logits
    out = []
    for opts, _gold in (((ok, bad), 0), ((bad, ok), 1)):
        # fit markers + options inside seq_len
        logits, _ = decision_logits(model, head, ctx, opts, device, mode, layers)
        out.append(logits)
    return out


def _option_probs(model, head, ctx, ok, bad, device, mode="marker",
                  temperature=1.0, layers=(-1,)):
    """R2: one forward per order. Returns p(gold) for each order and the
    debiased (order-averaged) probability that 'ok' is the correct option."""
    logits = _option_logits(model, head, ctx, ok, bad, device, mode, layers)
    p_ok, order_correct = [], []
    for lg, gold in zip(logits, (0, 1)):
        p = torch.softmax(lg / temperature, dim=-1)
        p_ok.append(p[gold].item())
        order_correct.append(int(p.argmax().item() == gold))
    return float(np.mean(p_ok)), order_correct


def fit_r2_temperature(model, head, pairs, device, mode="spanpool",
                       max_ctx=None, max_cand=None, seed=7331,
                       layers=(-1,)):
    """Fit scalar T minimizing binary NLL over per-order option logits.

    Dev data only — never the test battery. Gold index is 0 for order A
    (ok first), 1 for order B (bad first).
    """
    max_ctx = max_ctx or model.seq_len // 2
    max_cand = max_cand or model.seq_len // 4
    logit_rows, golds = [], []
    with torch.no_grad():
        for ctx, ok, bad in pairs:
            ctx, ok, bad = ctx[:max_ctx], ok[:max_cand], bad[:max_cand]
            if len(ctx) < 2 or not ok or not bad:
                continue
            for lg, gold in zip(
                    _option_logits(model, head, ctx, ok, bad, device, mode,
                                   layers),
                    (0, 1)):
                logit_rows.append(lg)
                golds.append(gold)
    if not logit_rows:
        return 1.0
    L = torch.stack(logit_rows)          # (2N, 2)
    y = torch.tensor(golds, device=L.device)  # (2N,)

    def nll(t):
        return float(F.cross_entropy(L / t, y).item())

    # coarse grid + local refine
    grid = np.concatenate([np.linspace(0.2, 5.0, 97),
                           np.linspace(5.5, 20.0, 30)])
    t_best = min(grid, key=nll)
    lo, hi = max(0.05, t_best * 0.5), t_best * 2.0
    for _ in range(40):                  # golden-section on log-T
        a, b = np.log(lo), np.log(hi)
        c, d = b - (b - a) * 0.618, a + (b - a) * 0.618
        if nll(np.exp(c)) < nll(np.exp(d)):
            hi = np.exp(d)
        else:
            lo = np.exp(c)
    return float(np.exp((np.log(lo) + np.log(hi)) / 2))


def _r2_row_logits(model, head, ctx, opts, device, mode, layers=(-1,)):
    """One forward -> (K,) head logits for options in given order."""
    from .decisions import decision_logits
    logits, _ = decision_logits(model, head, ctx, opts, device, mode, layers)
    return logits


def fit_r2_temperature_decisions(model, head, rows, device,
                                 mode="spanpool", layers=(-1,),
                                 canonical=False, strict=None, return_logits=False):
    """Fit scalar T minimizing NLL over K-way decision rows (dev only)."""
    from .decisions import ENCODING_VERSION, decision_logits, validate_decision_row
    rows = [validate_decision_row(row) for row in rows]
    if not rows:
        raise ValueError("temperature fitting requires at least one dev decision")
    encodings = {row.get("encoding", "legacy_ids") for row in rows}
    if len(encodings) != 1:
        raise ValueError("temperature fitting cannot mix encoding profiles")
    if strict is None:
        strict = next(iter(encodings)) == ENCODING_VERSION
    examples, grouped = [], {}
    with torch.no_grad():
        for row in rows:
            logits, prepared = decision_logits(
                model, head, row["ctx"], row["opts"], device, mode, layers,
                canonical=canonical, strict=strict)
            examples.append((logits, row["gold"]))
            ordered = logits[prepared.order]
            grouped.setdefault(len(ordered), []).append((ordered, prepared.order.index(row["gold"])))
    batches = [(torch.stack([logits for logits, _ in group]).double(),
                torch.tensor([gold for _, gold in group], device=group[0][0].device))
               for group in grouped.values()]

    def nll(t):
        return sum(F.cross_entropy(logits / t, golds, reduction="sum").item()
                   for logits, golds in batches)

    grid = np.concatenate([np.linspace(0.2, 5.0, 97),
                           np.linspace(5.5, 20.0, 30)])
    t_best = min(grid, key=nll)
    lo, hi = max(0.05, t_best * 0.5), t_best * 2.0
    for _ in range(40):
        a, b = np.log(lo), np.log(hi)
        c, d = b - (b - a) * 0.618, a + (b - a) * 0.618
        if nll(np.exp(c)) < nll(np.exp(d)):
            hi = np.exp(d)
        else:
            lo = np.exp(c)
    temperature = float(np.exp((np.log(lo) + np.log(hi)) / 2))
    return (temperature, examples) if return_logits else temperature


def eval_pairs(model, pairs, device, mask_p=0.15, seed=7331, head=None,
               r2_mode="marker", r2_temp=1.0, layers=(-1,)):
    """E0/R2: compare readouts on (ctx, ok, bad) id triples.

    r1_first : ctx + [MASK] -> logit[ok[0]] vs logit[bad[0]]  (degenerate if
               first tokens coincide — reported as diagnostic)
    r1_span  : mean logprob of cand tokens under all-masked candidate region
    r2_marker: (only if head given) Laya-style marker head, both option
               orders; reports order sensitivity + calibration
    legacy   : denoise_loss(ctx+ok) < denoise_loss(ctx+bad)  (Fase-3 style)
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    mask_id = model.mask_id
    # same truncation as suite_smoke_v2: ctx <= seq_len//2, cand <= seq_len//4
    max_ctx = model.seq_len // 2
    max_cand = model.seq_len // 4
    r1_wins = sp_wins = leg_wins = 0
    r1_deltas, sp_deltas, leg_deltas = [], [], []
    n_first_token_diff = 0
    r2_p, r2_corr, r2_ord_corr, r2_flips = [], [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for ctx, ok, bad in pairs:
            ctx, ok, bad = ctx[:max_ctx], ok[:max_cand], bad[:max_cand]
            if len(ctx) < 2 or not ok or not bad:
                continue
            if ok[0] != bad[0]:
                n_first_token_diff += 1
            # R1: single mask slot right after ctx; option = first token
            seq = torch.tensor(ctx + [mask_id], dtype=torch.long, device=device)
            seq = _clamp_oob(seq, model)
            row = model(seq.unsqueeze(0)).squeeze(0)[-1]
            ok_tok = min(ok[0], model.vocab - 1)
            bad_tok = min(bad[0], model.vocab - 1)
            d_r1 = (row[ok_tok] - row[bad_tok]).item()
            r1_wins += d_r1 > 0
            r1_deltas.append(d_r1)
            # r1_span: all-masked candidate scoring
            lp_ok = span_logprob(model, ctx, ok, device)
            lp_bad = span_logprob(model, ctx, bad, device)
            d_sp = lp_ok - lp_bad
            sp_wins += d_sp > 0
            sp_deltas.append(d_sp)
            # r2: trained head over option features, both orders
            if head is not None:
                p_avg, order_correct = _option_probs(
                    model, head, ctx, ok, bad, device, mode=r2_mode,
                    temperature=r2_temp, layers=layers)
                r2_p.append(p_avg)
                r2_corr.append(int(p_avg > 0.5))
                r2_ord_corr.append(order_correct)
                r2_flips += int(order_correct[0] != order_correct[1])
            # legacy pseudo-likelihood
            l_ok = denoise_loss(model, torch.tensor(ctx + ok, dtype=torch.long,
                                                    device=device), mask_p, rng, mask_id).item()
            l_bad = denoise_loss(model, torch.tensor(ctx + bad, dtype=torch.long,
                                                     device=device), mask_p, rng, mask_id).item()
            leg_wins += l_ok < l_bad
            leg_deltas.append(l_bad - l_ok)
    n = len(r1_deltas)
    out = {
        "n_pairs": n,
        "first_token_differs": n_first_token_diff,
        "r1_first_token": {"pairwise_acc": round(r1_wins / n, 4),
                           "mean_delta": round(float(np.mean(r1_deltas)), 5)},
        "r1_span": {"pairwise_acc": round(sp_wins / n, 4),
                    "mean_delta": round(float(np.mean(sp_deltas)), 5)},
        "legacy_denoise": {"pairwise_acc": round(leg_wins / n, 4),
                           "mean_delta": round(float(np.mean(leg_deltas)), 5)},
        "elapsed_s": round(time.time() - t0, 2),
    }
    if head is not None:
        p = np.asarray(r2_p)
        corr = np.asarray(r2_corr)
        conf = np.maximum(p, 1.0 - p)
        oc = np.asarray(r2_ord_corr)
        out[f"r2_{r2_mode}"] = {
            "pairwise_acc": round(float(corr.mean()), 4),
            "acc_order_a": round(float(oc[:, 0].mean()), 4),
            "acc_order_b": round(float(oc[:, 1].mean()), 4),
            "flip_rate": round(r2_flips / n, 4),
            "mean_p_gold": round(float(p.mean()), 4),
            "brier": round(float(np.mean((p - 1.0) ** 2)), 4),
            "nll": round(float(-np.log(np.clip(p, 1e-9, 1)).mean()), 4),
            "ece": round(ece(conf, corr), 4),
            "automation_5pct": round(automation_rate(conf, corr), 4),
            "temperature": round(r2_temp, 4),
            "calib_curve": _calib_curve(conf, corr),
        }
    return out


def eval_decisions_ids(model, head, rows, device, mode="spanpool",
                       temperature=1.0, seed=7331, layers=(-1,),
                       canonical=False, decision_type=None, fixed_labels=False,
                       acceptance_policy=None, return_predictions=False, strict=None):
    """K-way labeled decisions {ctx, opts[K], gold}: two orders per row.

    canonical=True sorts each presented option list deterministically by
    content, so both eval orders collapse to the same sequence
    (flip_rate -> 0 by construction).

    Returns acc (debiased mean-p argmax), per-order acc, flip_rate,
    mean p_gold, brier (K-dim), nll, ece, automation, calib_curve.
    """
    from .decisions import (ENCODING_VERSION, decision_logits, decision_prediction,
                            decision_probabilities, validate_decision_row)
    from .metrics import classification_metrics, evaluate_acceptance_policy

    rows = [validate_decision_row(row) for row in rows]
    if not rows:
        raise ValueError("evaluation requires at least one decision; input is empty")
    if decision_type not in (None, "choice", "noul", "score"):
        raise ValueError("decision_type must be choice, noul, or score")
    kinds = {row["kind"] for row in rows if row.get("kind") is not None}
    if decision_type is not None and kinds - {decision_type}:
        raise ValueError("decision_type conflicts with dataset kinds")
    if decision_type is None and len(kinds) == 1:
        decision_type = next(iter(kinds))
    encodings = {row.get("encoding", "legacy_ids") for row in rows}
    if len(encodings) != 1:
        raise ValueError("evaluation cannot mix encoding profiles")
    if strict is None:
        strict = next(iter(encodings)) == ENCODING_VERSION
    decision_probabilities(torch.zeros(2), temperature)
    if decision_type == "noul" and any(len(row["opts"]) != 2 for row in rows):
        raise ValueError("noul requires exactly two options")
    fixed_labels = fixed_labels or decision_type in ("noul", "score") or all(
        row.get("label_space") is not None for row in rows)
    label_names = None
    if fixed_labels and any(row.get("option_keys") is not None for row in rows):
        label_names = rows[0].get("label_space", rows[0].get("option_keys"))
        if (not isinstance(label_names, list) or len(set(label_names)) != len(label_names)
                or any(row.get("option_keys") is None or
                       set(row["option_keys"]) != set(label_names) for row in rows)):
            raise ValueError("fixed label metrics require the same named label space")
    rng = random.Random(seed)
    prob_rows, golds, predictions, second_correct, records = [], [], [], [], []
    flips = context_truncated = options_truncated = input_tokens = 0
    started = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            perm = list(range(len(row["opts"])))
            rng.shuffle(perm)
            first_selected = None
            for order in (None, perm):
                logits, prepared = decision_logits(
                    model, head, row["ctx"], row["opts"], device, mode, layers,
                    canonical=canonical, order=order, strict=strict)
                selected = decision_prediction(logits, prepared)
                input_tokens += len(prepared.ids)
                if order is None:
                    first_selected = selected
                    probs = decision_probabilities(logits, temperature)
                    prob_rows.append(probs)
                    golds.append(row["gold"])
                    predictions.append(selected)
                    context_truncated += int(prepared.context_truncated > 0)
                    options_truncated += int(any(prepared.option_truncated))
                    records.append({
                        **{key: row[key] for key in ("qid", "kind", "pid", "split_group",
                                                    "group_id", "option_keys") if key in row},
                        "gold": row["gold"], "prediction": selected,
                        "probabilities": probs, "max_probability": max(probs),
                        "correct": int(selected == row["gold"]),
                        "context_truncated": prepared.context_truncated,
                        "option_truncated": prepared.option_truncated})
                else:
                    flips += int(selected != first_selected)
                    second_correct.append(int(selected == row["gold"]))
    metric_probs, metric_golds, metric_predictions = prob_rows, golds, predictions
    if label_names is not None:
        metric_probs, metric_golds, metric_predictions = [], [], []
        for row, probs, gold, predicted in zip(rows, prob_rows, golds, predictions):
            keys = row["option_keys"]
            metric_probs.append([probs[keys.index(key)] for key in label_names])
            metric_golds.append(label_names.index(keys[gold]))
            metric_predictions.append(label_names.index(keys[predicted]))
    report = classification_metrics(metric_probs, metric_golds,
                                    ordinal=decision_type == "score", fixed_labels=fixed_labels,
                                    predictions=metric_predictions)
    confidences = [max(probs) for probs in prob_rows]
    corrects = [int(predicted == gold) for predicted, gold in zip(predictions, golds)]
    report.update({
        "metrics_version": 2, "decision_type": decision_type,
        "encoding": next(iter(encodings)), "label_space": "fixed" if fixed_labels else "per_example",
        "prediction_protocol": "first presentation; second permutation is diagnostic",
        "acc_order_a": report["acc"], "acc_order_b": sum(second_correct) / len(rows),
        "flip_rate": flips / len(rows),
        "mean_p_gold": float(np.mean([probs[gold] for probs, gold in zip(prob_rows, golds)])),
        "automation_5pct": report["oracle_automation_5pct"],
        "automation_note": "retrospective tie-safe oracle; not a deployment policy",
        "temperature": float(temperature), "calib_curve": _calib_curve(confidences, corrects),
        "truncation": {"context_rows": context_truncated, "option_rows": options_truncated},
        "forward_passes": 2 * len(rows), "model_input_tokens": input_tokens,
        "elapsed_s": time.perf_counter() - started})
    if label_names is not None:
        report["label_names"] = list(label_names)
    if acceptance_policy is not None:
        report["selective"] = evaluate_acceptance_policy(confidences, corrects, acceptance_policy)
    if return_predictions:
        report["predictions"] = records
    return report


def _calib_curve(confs, corrects, n_bins=10):
    """Reliability bins: [{conf, acc, n}] for plotting."""
    confs = np.asarray(confs, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1]) if i else \
            (confs >= edges[i]) & (confs <= edges[i + 1])
        if m.any():
            rows.append({"conf": round(float(confs[m].mean()), 4),
                         "acc": round(float(corrects[m].mean()), 4),
                         "n": int(m.sum())})
    return rows


def eval_decisions_local(model, tok, rows, device, temperature=1.0, max_len=768):
    """Score labeled System-One decisions with the R1 readout."""
    from .readout import (predict_choice, predict_noul, predict_score)

    per_type = {}
    recs = []
    for state, qid, q, label in rows:
        qt = q["type"]
        if qt == "choice":
            names = list(q["criteria"].keys())
            ans = predict_choice(model, tok, state, q["instructions"],
                                 q["criteria"], temperature, max_len, device)
            probs = [ans["probabilities"][n] for n in names]
            pred, gold = names.index(ans["choice"]), names.index(label)
        elif qt == "noul":
            ans = predict_noul(model, tok, state, q["instructions"],
                               temperature, max_len, device)
            probs = [ans["noul"], 1 - ans["noul"]]
            gold = 0 if label in (True, "true", "yes") else 1
            pred = 0 if ans["noul"] >= 0.5 else 1
        elif qt == "score":
            ans = predict_score(model, tok, state, q["instructions"],
                                q["criteria"], temperature, max_len, device)
            probs = [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]
            pred = int(np.argmax(probs))
            gold = int(label)
        else:
            continue
        conf = max(probs)
        correct = int(pred == gold)
        recs.append({"type": qt, "probs": probs, "gold": gold,
                     "conf": conf, "correct": correct})
        per_type.setdefault(qt, []).append(recs[-1])

    def summarize(rs):
        return {
            "n": len(rs),
            "accuracy": round(float(np.mean([r["correct"] for r in rs])), 4),
            "brier": round(brier_multi([r["probs"] for r in rs],
                                       [r["gold"] for r in rs]), 4),
            "ece": round(ece([r["conf"] for r in rs],
                             [r["correct"] for r in rs]), 4),
            "automation@5%err": round(automation_rate(
                [r["conf"] for r in rs], [r["correct"] for r in rs]), 4),
        }

    out = {"overall": summarize(recs), "temperature": temperature}
    for t, rs in per_type.items():
        out[t] = summarize(rs)
    return out, recs


def fit_temperature(model, tok, dev_rows, device, max_len=768):
    """Grid-search temperature minimizing CE on dev decisions (held-out)."""
    from .readout import r1_logits, build_sequence, option_id_sets, noul_id_sets

    examples = []
    with torch.no_grad():
        for state, qid, q, label in dev_rows:
            qt = q["type"]
            ids, mpos = build_sequence(tok, state, q["instructions"],
                                       model.mask_id, max_len)
            seq = torch.tensor(ids, dtype=torch.long, device=device)
            if qt == "choice":
                names = list(q["criteria"].keys())
                sets = [option_id_sets(tok, n) for n in names]
                gold = names.index(label)
            elif qt == "noul":
                yes, no = noul_id_sets(tok)
                sets = [yes, no]
                gold = 0 if label in (True, "true", "yes") else 1
            elif qt == "score":
                sets = [option_id_sets(tok, str(i))
                        for i in range(len(q["criteria"]))]
                gold = int(label)
            else:
                continue
            lg = r1_logits(model, seq, mpos, sets, temperature=1.0)
            examples.append((lg, gold))
    if not examples:
        return 1.0
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.25, 8.0, 64):
        l = 0.0
        for lg, gold in examples:
            l += F.cross_entropy((lg / t).unsqueeze(0),
                                 torch.tensor([gold], device=lg.device)).item()
        if l < best_loss:
            best_loss, best_t = l, float(t)
    return best_t


# ---------------- remote (Jev / any /v1/systemone) ----------------

def eval_decisions_remote(rows, base_url, api_key, model_name="jev-latest",
                          timeout=120):
    """Run labeled decisions against any System One endpoint."""
    import urllib.request

    url = base_url.rstrip("/") + "/v1/systemone"
    by_state = {}
    for state, qid, q, label in rows:
        by_state.setdefault(state, {"questions": {}, "labels": {}})
        qq = {k: v for k, v in q.items() if k != "label"}
        by_state[state]["questions"][qid] = qq
        by_state[state]["labels"][qid] = label

    per_type, recs, errors = {}, [], []
    for state, pack in by_state.items():
        body = json.dumps({"state": state, "model": model_name,
                           "questions": pack["questions"]}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
        except Exception as e:
            errors.append({"state": state[:80], "error": str(e)})
            continue
        for qid, ans in resp["answers"].items():
            q = pack["questions"][qid]
            label = pack["labels"][qid]
            qt, gold = q["type"], None
            if qt == "choice":
                names = list(q["criteria"].keys())
                probs = [ans["probabilities"].get(n, 0.0) for n in names]
                pred, gold = names.index(ans["choice"]), names.index(label)
            elif qt == "noul":
                probs = [ans["noul"], 1 - ans["noul"]]
                gold = 0 if label in (True, "true", "yes") else 1
                pred = 0 if ans["noul"] >= 0.5 else 1
            elif qt == "score":
                k = len(q["criteria"])
                probs = [ans["probabilities"].get(str(i), 0.0) for i in range(k)]
                pred, gold = int(np.argmax(probs)), int(label)
            else:
                continue
            r = {"type": qt, "probs": probs, "gold": gold,
                 "conf": max(probs), "correct": int(pred == gold)}
            recs.append(r)
            per_type.setdefault(qt, []).append(r)

    def summarize(rs):
        return {"n": len(rs),
                "accuracy": round(float(np.mean([r["correct"] for r in rs])), 4),
                "brier": round(brier_multi([r["probs"] for r in rs],
                                           [r["gold"] for r in rs]), 4),
                "ece": round(ece([r["conf"] for r in rs],
                                 [r["correct"] for r in rs]), 4),
                "automation@5%err": round(automation_rate(
                    [r["conf"] for r in rs], [r["correct"] for r in rs]), 4)}

    out = {"overall": summarize(recs), "remote": base_url, "model": model_name,
           "n_request_errors": len(errors), "errors": errors[:20]}
    for t, rs in per_type.items():
        out[t] = summarize(rs)
    return out, recs


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="ecoreasoner checkpoint (model.pt)")
    ap.add_argument("--hf-backbone", default=None,
                    help="HF model name/path (e.g. GSAI-ML/LLaDA-8B-Instruct) "
                         "instead of an ecoreasoner --ckpt")
    ap.add_argument("--lora-adapter", default=None,
                    help="LoRA adapter dir (e.g. DAPT output) merged into the "
                         "HF backbone before eval")
    ap.add_argument("--head", help="trained DecisionHead state (enables r2 eval)")
    ap.add_argument("--r2-mode", choices=["marker", "spanpool"],
                    default="marker")
    ap.add_argument("--r2-layers", default="-1",
                    help="hidden-layer indices for attnpool heads "
                         "(comma list, negatives ok)")
    ap.add_argument("--r2-temp", type=float, default=1.0,
                    help="fixed temperature for r2 softmax")
    ap.add_argument("--canonical-order", action="store_true",
                    help="sort options deterministically by content before "
                         "layout (must match training)")
    ap.add_argument("--decision-type", choices=["choice", "noul", "score"],
                    default=None, help="task type asserted for --decisions-eval")
    ap.add_argument("--fixed-labels", action="store_true",
                    help="compute metrics on the shared named label space")
    ap.add_argument("--allow-truncation", dest="strict_inputs",
                    action="store_false", default=None,
                    help="explicitly permit truncation (default: strict for "
                         "systemone-v2 encoded rows)")
    ap.add_argument("--strict-inputs", dest="strict_inputs",
                    action="store_true",
                    help="reject any input truncation")
    ap.add_argument("--calibration-in", default=None,
                    help="frozen r2_calibration artifact (temperature + "
                         "acceptance policy); verifies checkpoint fingerprint "
                         "and eval/calibration disjointness")
    ap.add_argument("--train-reference", default=None,
                    help="training decisions jsonl; fails on known identity "
                         "overlap with the evaluation set")
    ap.add_argument("--r2-temp-fit", default=None,
                    help="dev pairs dir: fit T on it, then evaluate test")
    ap.add_argument("--r2-temp-fit-decisions", default=None,
                    help="dev decisions jsonl: fit T on it, then evaluate")
    ap.add_argument("--config", help="harness yaml with model section (optional)")
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--pairs", help="ecoreasoner pairs jsonl (E0 mode)")
    ap.add_argument("--pairs-dir", help="dir with pairs_L*.jsonl (battery)")
    ap.add_argument("--decisions-eval", default=None,
                    help="K-way decisions jsonl {ctx,opts,gold} (needs --head)")
    ap.add_argument("--data", help="System-One decisions jsonl (labeled)")
    ap.add_argument("--remote", help="base URL of a /v1/systemone endpoint")
    ap.add_argument("--api-key-file", help="file with API key (remote mode)")
    ap.add_argument("--model-name", default="jev-latest")
    ap.add_argument("--mask-p", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=None,
                    help="fixed T; if omitted with --data, fit on dev split")
    ap.add_argument("--no-temp-fit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"args": {k: v for k, v in vars(args).items() if k != "api_key_file"}}

    if args.remote:
        key = Path(args.api_key_file).read_text().strip() if args.api_key_file \
            else __import__("os").environ.get("TYPESAFE_API_KEY", "local")
        rows = list(iter_decisions(args.data))
        out, recs = eval_decisions_remote(rows, args.remote, key, args.model_name)
        report.update(out)
        print(json.dumps(out, indent=2))
    else:
        if not args.ckpt and not args.hf_backbone:
            ap.error("--ckpt or --hf-backbone required for local eval")
        from .model import (load_backbone, load_decision, DecisionHead,
                            HFBackbone)
        cfg = None
        if args.config:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text()).get("model", {})
        head = None
        if args.head:
            model = (HFBackbone(args.hf_backbone, device=args.device,
                                lora_adapter=args.lora_adapter)
                     if args.hf_backbone
                     else load_backbone(args.ckpt, cfg, device=args.device))
            head = DecisionHead(model.tok_emb.embedding_dim).to(args.device)
            hsd = torch.load(args.head, map_location="cpu")
            head.load_state_dict(hsd.get("head", hsd))
            head.eval()
        else:
            model, head = load_decision(args.ckpt, cfg, device=args.device)

        layers = [int(x) for x in args.r2_layers.split(",")]
        r2_temp = args.r2_temp
        if head is not None and args.r2_temp_fit:
            from .data import load_pairs_dir
            dev_pairs = [p for ps in load_pairs_dir(args.r2_temp_fit).values()
                         for p in ps]
            r2_temp = fit_r2_temperature(model, head, dev_pairs, args.device,
                                         mode=args.r2_mode, layers=layers)
            report["r2_temp_fitted"] = round(r2_temp, 4)
            print(f"[temp] fitted T={r2_temp:.3f} on "
                  f"{len(dev_pairs)} dev pairs", file=sys.stderr)
        fitted_dev_rows = None
        if head is not None and args.r2_temp_fit_decisions:
            from .data import load_decisions_ids
            dev_rows = load_decisions_ids(args.r2_temp_fit_decisions)
            fitted_dev_rows = dev_rows
            r2_temp = fit_r2_temperature_decisions(
                model, head, dev_rows, args.device, mode=args.r2_mode,
                layers=layers, canonical=args.canonical_order)
            report["r2_temp_fitted"] = round(r2_temp, 4)
            print(f"[temp] fitted T={r2_temp:.3f} on "
                  f"{len(dev_rows)} dev decisions", file=sys.stderr)

        if args.pairs or args.pairs_dir:
            if args.pairs_dir:
                from .data import load_pairs_dir
                levels = {}
                for lvl, pairs in load_pairs_dir(args.pairs_dir).items():
                    levels[lvl] = eval_pairs(model, pairs, args.device,
                                             args.mask_p, args.seed, head=head,
                                             r2_mode=args.r2_mode,
                                             r2_temp=r2_temp, layers=layers)
                    print(json.dumps({"level": lvl, **levels[lvl]}))
                report["battery"] = levels
            else:
                pairs = load_pairs(args.pairs)
                out = eval_pairs(model, pairs, args.device, args.mask_p,
                                 args.seed, head=head, r2_mode=args.r2_mode,
                                 r2_temp=r2_temp, layers=layers)
                report["pairs_eval"] = out
                print(json.dumps(out, indent=2))

        if args.decisions_eval:
            if head is None:
                ap.error("--decisions-eval requires a trained head "
                         "(--head or decision.pt with head)")
            from .data import load_decisions_ids
            rows = load_decisions_ids(args.decisions_eval)
            if fitted_dev_rows is not None:
                from . import protocol
                report["dev_eval_disjointness"] = protocol.assert_disjoint_splits(
                    {"temperature_dev": fitted_dev_rows, "evaluation": rows})
            acceptance_policy = None
            if args.calibration_in:
                from .calibration import load_calibration
                if not args.head:
                    ap.error("--calibration-in verifies the checkpoint given "
                             "by --head")
                expected = {"mode": args.r2_mode, "layers": layers,
                            "canonical_order": args.canonical_order}
                if args.decision_type:
                    expected["decision_type"] = args.decision_type
                artifact = load_calibration(
                    args.calibration_in, args.head, evaluation_rows=rows,
                    expected_inference=expected)
                r2_temp = artifact["temperature"]
                acceptance_policy = artifact["acceptance_policy"]
                report["calibration"] = {
                    "file": args.calibration_in,
                    "temperature": artifact["temperature"],
                    "encoding": artifact["inference"]["encoding"],
                    "training_overlap_check":
                        artifact["provenance"]["training_overlap_check"]["status"]}
            if args.train_reference:
                from . import protocol
                ref_rows = load_decisions_ids(args.train_reference)
                report["train_overlap_check"] = protocol.assert_disjoint_splits(
                    {"train_reference": ref_rows, "evaluation": rows})
            out = eval_decisions_ids(model, head, rows, args.device,
                                     mode=args.r2_mode, temperature=r2_temp,
                                     seed=args.seed, layers=layers,
                                     canonical=args.canonical_order,
                                     decision_type=args.decision_type,
                                     fixed_labels=args.fixed_labels,
                                     acceptance_policy=acceptance_policy,
                                     strict=args.strict_inputs)
            report["decisions_eval"] = {
                "file": args.decisions_eval, **out}
            print(json.dumps(out, indent=2))

        if args.data:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            dev, test = split_dev_test(args.data)
            temp = args.temperature
            if temp is None and not args.no_temp_fit:
                temp = fit_temperature(model, tok, dev, args.device)
                print(f"[temp] fitted T={temp:.3f} on {len(dev)} dev decisions")
            temp = temp or 1.0
            out, recs = eval_decisions_local(model, tok, test, args.device,
                                             temperature=temp)
            report["decisions"] = out
            print(json.dumps(out, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[wrote] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
