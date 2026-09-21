"""Data loaders.

Two input shapes:
1. ecoreasoner pairs — pairs_L*.jsonl, one {"ctx","ok","bad"} of token ids per
   line (the L0-L3 discrimination battery format).
2. System-One JSONL — one request per line, API-shaped plus a "label" per
   question (same format kev trains on):

   {"state": "...", "questions": {"team": {"type": "choice",
        "instructions": "...", "criteria": {"a": "...", "b": "..."},
        "label": "a"}}}
"""
import json
from pathlib import Path


def load_pairs(path, max_ctx=None, max_cand=None):
    """pairs jsonl -> [(ctx_ids, ok_ids, bad_ids)]."""
    pairs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ctx = rec["ctx"][:max_ctx] if max_ctx else rec["ctx"]
        ok = rec["ok"][:max_cand] if max_cand else rec["ok"]
        bad = rec["bad"][:max_cand] if max_cand else rec["bad"]
        if len(ctx) >= 2 and ok and bad:
            pairs.append((ctx, ok, bad))
    return pairs


def load_pairs_dir(pairs_dir):
    """Directory of pairs_L{0..3}.jsonl -> {level: pairs}."""
    out = {}
    for fp in sorted(Path(pairs_dir).glob("pairs_L*.jsonl")):
        lvl = fp.stem.split("_L")[-1]
        if lvl in ("0", "1", "2", "3"):
            out[f"L{lvl}"] = load_pairs(fp)
    return out


def iter_decisions(path):
    """System-One JSONL -> (state, qid, question_dict, label)."""
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        state = rec["state"]
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        for qid, q in rec["questions"].items():
            if "label" not in q:
                continue
            yield state, qid, q, q["label"]


def split_dev_test(path, dev_frac=0.5, seed=7331):
    """Deterministic dev/test split over decisions (dev is for temperature
    fitting; test is read once for the reported number)."""
    import random
    rows = list(iter_decisions(path))
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_dev = int(len(rows) * dev_frac)
    return rows[:n_dev], rows[n_dev:]
