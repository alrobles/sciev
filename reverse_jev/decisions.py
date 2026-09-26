import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real


ENCODING_VERSION = "systemone-v2"


class InputOverflow(ValueError):
    """Encoded input exceeds the declared token budget."""


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _token_ids(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of token IDs")
    return [_integer(token, f"{name} token") for token in value]


def _distinct_options(options):
    if len({tuple(option) for option in options}) != len(options):
        raise ValueError("option encodings must be distinct; identical options are indistinguishable")


def validate_decision_row(row, require_gold=True):
    if not isinstance(row, Mapping):
        raise ValueError("decision must be an object")
    if "ctx" not in row or "opts" not in row:
        raise ValueError("decision requires ctx and opts")
    ctx = _token_ids(row["ctx"], "ctx")
    if not isinstance(row["opts"], (list, tuple)) or len(row["opts"]) < 2:
        raise ValueError("opts must contain at least two options")
    opts = [_token_ids(option, f"opts[{index}]") for index, option in enumerate(row["opts"])]
    if any(not option for option in opts):
        raise ValueError("options must have at least one token")
    _distinct_options(opts)
    result = dict(row, ctx=ctx, opts=opts)
    if require_gold and "gold" not in row:
        raise ValueError("labeled decision requires gold")
    if "gold" in row:
        gold = _integer(row["gold"], "gold")
        if gold >= len(opts):
            raise ValueError("gold is outside the option range")
        result["gold"] = gold
    if row.get("kind") is not None and row["kind"] not in ("choice", "noul", "score"):
        raise ValueError("kind must be choice, noul, or score")
    if row.get("qid") is not None and (not isinstance(row["qid"], str) or not row["qid"].strip()):
        raise ValueError("qid must be a nonempty string")
    keys = row.get("option_keys")
    if keys is not None:
        if (not isinstance(keys, (list, tuple)) or len(keys) != len(opts)
                or any(not isinstance(key, str) or not key.strip() for key in keys)
                or len(set(keys)) != len(keys)):
            raise ValueError("option_keys must uniquely name each option")
        result["option_keys"] = list(keys)
    soft = row.get("soft")
    if soft is not None:
        if not isinstance(soft, (list, tuple)) or len(soft) != len(opts):
            raise ValueError("soft must contain one probability per option")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or not 0 <= value <= 1 for value in soft):
            raise ValueError("soft must contain finite probabilities in [0, 1]")
        if not math.isclose(sum(soft), 1.0, abs_tol=1e-4):
            raise ValueError("soft probabilities must sum to one")
        result["soft"] = [float(value) for value in soft]
    return result


def render_value(value, name="value", allow_empty=False):
    if isinstance(value, str):
        if not allow_empty and not value.strip():
            raise ValueError(f"{name} must not be empty")
        return value
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{name} must be a string, object, or array")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error


def encode_question(tokenizer, state, question, max_ctx=640, max_opt=120,
                    overflow="error"):
    max_ctx = _integer(max_ctx, "max_ctx", 1)
    max_opt = _integer(max_opt, "max_opt", 1)
    if overflow not in ("error", "truncate"):
        raise ValueError("overflow must be error or truncate")
    if not isinstance(question, Mapping):
        raise ValueError("question must be an object")
    kind = question.get("type")
    if kind not in ("choice", "noul", "score"):
        raise ValueError("question type must be choice, noul, or score")
    if "instructions" not in question:
        raise ValueError("question requires instructions")
    instructions = render_value(question["instructions"], "instructions")
    state_text = render_value(state, "state", allow_empty=True)
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= 255:
            raise ValueError("choice criteria must contain 2..255 options")
        keys = list(criteria)
        if any(not isinstance(key, str) or not key.strip() for key in keys):
            raise ValueError("choice option names must be nonempty strings")
        options = [key if value is None else
                   f"{key}: {render_value(value, 'choice description', allow_empty=True)}"
                   for key, value in criteria.items()]
    elif kind == "noul":
        keys, options = ["true", "false"], ["yes", "no"]
        if criteria is not None:
            if not isinstance(criteria, Mapping) or set(criteria) != {"true", "false"}:
                raise ValueError("noul criteria must define true and false")
            options = [f"{option}: {render_value(criteria[key], 'noul criterion')}"
                       for option, key in zip(options, keys)]
    else:
        if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must contain 2..10 ordered levels")
        options = [render_value(value, "score criterion") for value in criteria]
        keys = [str(index) for index in range(len(options))]
    ctx = _token_ids(tokenizer.encode(f"{state_text}\nQuestion: {instructions}",
                                      add_special_tokens=False), "encoded ctx")
    opts = [_token_ids(tokenizer.encode(option, add_special_tokens=False), "encoded option")
            for option in options]
    if overflow == "error":
        if len(ctx) > max_ctx:
            raise InputOverflow(f"context exceeds max_ctx={max_ctx}; truncation was not authorized")
        if any(len(option) > max_opt for option in opts):
            raise InputOverflow(f"option exceeds max_opt={max_opt}; truncation was not authorized")
    row = {"ctx": ctx[:max_ctx], "opts": [option[:max_opt] for option in opts],
           "kind": kind, "option_keys": keys, "encoding": ENCODING_VERSION,
           "schema_version": 2,
           "truncation": {"context_tokens": max(0, len(ctx) - max_ctx),
                          "option_tokens": [max(0, len(option) - max_opt) for option in opts]}}
    return validate_decision_row(row, require_gold=False)


@dataclass(frozen=True)
class PreparedDecision:
    ids: list[int]
    positions: list
    order: list[int]
    context_truncated: int
    option_truncated: list[int]

    @property
    def truncated(self):
        return bool(self.context_truncated or any(self.option_truncated))


def prepare_decision(model, ctx, opts, mode="spanpool", canonical=False,
                     order=None, strict=False):
    from .model import marker_layout, spanpool_layout

    row = validate_decision_row({"ctx": ctx, "opts": opts}, require_gold=False)
    ctx, opts = row["ctx"], row["opts"]
    if mode not in ("marker", "spanpool"):
        raise ValueError("mode must be marker or spanpool")
    if not isinstance(canonical, bool) or not isinstance(strict, bool):
        raise ValueError("canonical and strict must be booleans")
    seq_len = _integer(model.seq_len, "model.seq_len", 1)
    vocab_size = getattr(getattr(model, "tok_emb", None), "num_embeddings", None)
    if vocab_size is None:
        vocab_size = max(getattr(model, "vocab", 0), getattr(model, "mask_id", -1) + 1)
    vocab_size = _integer(vocab_size, "vocabulary size", 1)
    if any(token >= vocab_size for sequence in [ctx, *opts] for token in sequence):
        raise ValueError(f"token ID outside model vocabulary [0, {vocab_size})")
    count = len(opts)
    markers = count if mode == "marker" else 0
    if count + markers > seq_len:
        raise ValueError("options cannot fit inside the model token budget")
    ctx_cap = min(seq_len // 2, seq_len - count - markers)
    trimmed_ctx = ctx[:ctx_cap]
    option_cap = (seq_len - len(trimmed_ctx) - markers) // count
    trimmed_opts = [option[:option_cap] for option in opts]
    context_truncated = len(ctx) - len(trimmed_ctx)
    option_truncated = [len(option) - len(trimmed) for option, trimmed in zip(opts, trimmed_opts)]
    if strict and (context_truncated or any(option_truncated)):
        raise ValueError("decision exceeds token budget and would require truncation")
    _distinct_options(trimmed_opts)
    if order is None:
        permutation = list(range(count))
    else:
        if not isinstance(order, (list, tuple)):
            raise ValueError("order must be a permutation of option indices")
        permutation = [_integer(index, "option index") for index in order]
        if sorted(permutation) != list(range(count)):
            raise ValueError("order must be a permutation of option indices")
    if canonical:
        permutation.sort(key=lambda index: tuple(trimmed_opts[index]))
    ordered = [trimmed_opts[index] for index in permutation]
    if mode == "marker":
        mask_id = _integer(model.mask_id, "mask_id")
        if mask_id >= vocab_size:
            raise ValueError("mask_id is outside the model vocabulary")
        ids, positions = marker_layout(trimmed_ctx, ordered, mask_id)
    else:
        ids, positions = spanpool_layout(trimmed_ctx, ordered)
    return PreparedDecision(ids, positions, permutation, context_truncated, option_truncated)


def decision_logits(model, head, ctx, opts, device, mode="spanpool", layers=(-1,),
                    canonical=False, order=None, strict=False):
    import torch
    from .model import AttnPoolHead, forward_feats

    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("layers must be a nonempty sequence of integer indices")
    if any(isinstance(index, bool) or not isinstance(index, Integral) for index in layers):
        raise ValueError("layers must contain integer indices")
    if isinstance(head, AttnPoolHead):
        if mode != "spanpool":
            raise ValueError("AttnPoolHead requires spanpool mode")
        if len(layers) != head.n_layers:
            raise ValueError("layer count does not match AttnPoolHead")
    prepared = prepare_decision(model, ctx, opts, mode, canonical, order, strict)
    ids = torch.tensor(prepared.ids, dtype=torch.long, device=device)
    logits = forward_feats(model, head, ids, mode, prepared.positions, layers).float()
    if logits.shape != (len(opts),):
        raise ValueError("head must produce exactly one logit per option")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("head produced nonfinite logits")
    restored = torch.empty_like(logits)
    restored[prepared.order] = logits
    return restored, prepared


def decision_prediction(logits, prepared):
    import torch

    if not isinstance(logits, torch.Tensor) or logits.shape != (len(prepared.order),):
        raise ValueError("prediction requires one logit per original option")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("cannot select from nonfinite logits")
    return prepared.order[logits[prepared.order].argmax().item()]


def decision_probabilities(logits, temperature=1.0):
    import torch

    if (isinstance(temperature, bool) or not isinstance(temperature, Real)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("temperature must be positive and finite")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("probabilities require a vector of at least two logits")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("cannot normalize nonfinite logits")
    values = logits.detach().double().cpu().tolist()
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    total = math.fsum(weights)
    return [weight / total for weight in weights]
