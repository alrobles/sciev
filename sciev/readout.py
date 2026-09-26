"""R1 readout — token-slot decision scoring (zero extra params).

Sequence:  <state tokens> <question tokens> [MASK]
P(option) = softmax over per-option scores at the MASK position, where each
option's score is logsumexp over its candidate token-id set (first-token
variants: casing / leading space). Multi-token option text is only read
through its first token — the R2 marker head (model.DecisionHead) covers
full-text options.

This is the E0 experiment readout: runs on existing ecoreasoner checkpoints
with no training.
"""
import torch
import torch.nn.functional as F

ANSWER_PROMPT = "\nAnswer:"


def option_id_sets(tok, text):
    """Candidate first-token ids for an option string (casing/space variants)."""
    variants = {text, " " + text, text.capitalize(), " " + text.capitalize(),
                text.upper(), " " + text.upper(), text.lower(), " " + text.lower()}
    ids = set()
    for v in variants:
        enc = tok.encode(v, add_special_tokens=False)
        if enc:
            ids.add(enc[0])
    return sorted(ids)


NOUL_YES = ("yes", "true", "si", "sí")
NOUL_NO = ("no", "false")


def noul_id_sets(tok):
    yes, no = set(), set()
    for w in NOUL_YES:
        yes.update(option_id_sets(tok, w))
    for w in NOUL_NO:
        no.update(option_id_sets(tok, w))
    return sorted(yes), sorted(no)


def _logsumexp_option(logits_row, id_set):
    if not id_set:
        return torch.tensor(float("-inf"), device=logits_row.device)
    return torch.logsumexp(logits_row[list(id_set)], dim=-1)


def r1_logits(model, ids, mask_pos, option_sets, temperature=1.0):
    """One forward pass -> per-option logit vector (temperature-scaled)."""
    with torch.no_grad():
        out = model(ids.unsqueeze(0)).squeeze(0)
    row = out[mask_pos]
    logits = torch.stack([_logsumexp_option(row, s) for s in option_sets])
    return logits / max(temperature, 1e-6)


def build_sequence(tok, state, instructions, mask_id, max_len):
    text = f"{state}{ANSWER_PROMPT} {instructions}" if instructions else str(state)
    ids = tok.encode(text, add_special_tokens=False)[: max_len - 1]
    return ids + [mask_id], len(ids)


def predict_choice(model, tok, state, instructions, criteria, temperature=1.0,
                   max_len=None, device="cpu"):
    """criteria: {option_name: description|None} -> Jev-format choice answer."""
    names = list(criteria.keys())
    sets = [option_id_sets(tok, n) for n in names]
    max_len = max_len or model.pos.weight.shape[0] if model.pos is not None else 2048
    ids, mpos = build_sequence(tok, state, instructions, model.mask_id, max_len)
    seq = torch.tensor(ids, dtype=torch.long, device=device)
    logits = r1_logits(model, seq, mpos, sets, temperature)
    probs = torch.softmax(logits.float(), dim=-1)
    k = len(names)
    p_max = probs.max().item()
    conf = (p_max - 1.0 / k) / (1.0 - 1.0 / k) if k > 1 else 1.0
    return {
        "type": "choice",
        "choice": names[int(probs.argmax())],
        "confidence": round(conf, 4),
        "probabilities": {n: round(probs[i].item(), 4) for i, n in enumerate(names)},
    }


def predict_noul(model, tok, state, instructions, temperature=1.0,
                 max_len=None, device="cpu"):
    max_len = max_len or model.pos.weight.shape[0] if model.pos is not None else 2048
    ids, mpos = build_sequence(tok, state, instructions, model.mask_id, max_len)
    seq = torch.tensor(ids, dtype=torch.long, device=device)
    yes, no = noul_id_sets(tok)
    logits = r1_logits(model, seq, mpos, [yes, no], temperature)
    p_yes = torch.softmax(logits.float(), dim=-1)[0].item()
    return {"type": "noul", "noul": round(p_yes, 4)}


def predict_score(model, tok, state, instructions, criteria, temperature=1.0,
                  max_len=None, device="cpu"):
    """criteria: ordered list of level descriptions -> distribution over level
    index tokens ("0".."K-1"); score = expected level index."""
    k = len(criteria)
    sets = [option_id_sets(tok, str(i)) for i in range(k)]
    max_len = max_len or model.pos.weight.shape[0] if model.pos is not None else 2048
    ids, mpos = build_sequence(tok, state, instructions, model.mask_id, max_len)
    seq = torch.tensor(ids, dtype=torch.long, device=device)
    logits = r1_logits(model, seq, mpos, sets, temperature)
    probs = torch.softmax(logits.float(), dim=-1)
    levels = torch.arange(k, dtype=torch.float)
    score = float((probs.cpu() * levels).sum())
    p_max = probs.max().item()
    conf = (p_max - 1.0 / k) / (1.0 - 1.0 / k) if k > 1 else 1.0
    return {
        "type": "score",
        "score": round(score, 4),
        "confidence": round(conf, 4),
        "legend": {str(i): c for i, c in enumerate(criteria)},
        "probabilities": {str(i): round(probs[i].item(), 4) for i in range(k)},
    }


def answer_questions(model, tok, state, questions, temperature=1.0,
                     max_len=None, device="cpu"):
    """questions: {id: {type, instructions, criteria}} -> {id: Jev-format answer}."""
    out = {}
    for qid, q in questions.items():
        qt = q.get("type")
        ins = q.get("instructions", "")
        crit = q.get("criteria")
        if qt == "noul":
            out[qid] = predict_noul(model, tok, state, ins, temperature, max_len, device)
        elif qt == "choice":
            out[qid] = predict_choice(model, tok, state, ins, crit or {},
                                      temperature, max_len, device)
        elif qt == "score":
            out[qid] = predict_score(model, tok, state, ins, crit or [],
                                     temperature, max_len, device)
        else:
            raise ValueError(f"unknown question type: {qt}")
    return out
