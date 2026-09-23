#!/usr/bin/env python3
"""Build K-way *scientific* decision batteries from ecoreasoner QA data.

Source: qa_devin_elite.jsonl — {pid, type, q, a, passage} where type in
{numerical, causal, multihop, negation, definitional}.

Decision tasks (same shape as the tool-call battery):
  sci_choice_{train,dev,eval}.jsonl   K=4 answers, gold correct
  sci_noul_{train,dev,eval}.jsonl     K=2, is the proposed answer supported?
  sci_score_{train,dev,eval}.jsonl    K=3 rubric: unrelated / related-but-wrong
                                    / correct
  sci_decisions_{tag}_text.jsonl      text System-One rows for --remote eval

Leakage control: split by pid — no passage appears in two splits.

Distractor/negative strategies:
  - same-passage swap: answer to a different question on the same pid
    (topically right context, factually wrong for this question)
  - number perturb: for answers containing numerals, multiply by {0.1,2,10}
    or perturb digits — hard negatives for the `numerical` type
  - cross-passage: answer to a question on a different pid (easy negative)

Record format (model path): {"ctx":[ids], "opts":[[ids]...], "gold":i, "qid":str}

v2 sources: --qa-raw GLOB accepts qa_raw_shard*.jsonl
{"pid","passage","qa":[{q,a}]} (ecoreasoner qa_gen output). Light
groundedness filters from qa_filter.py keep bad teacher generations out.
"""
import argparse
import glob
import json
import random
import re
from pathlib import Path

INSTR_CHOICE = "Which answer is supported by the passage?"
INSTR_NOUL = "Is the proposed answer supported by the passage?"
SCORE_LEGEND = ["unrelated answer", "related but wrong answer",
                "correct answer"]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-záéíóúñü][a-záéíóúñü\-']*", re.I)
_STOP = set("""a an the of in on at to for and or but is are was were be been
it its this that these those with by from as we i you they he she not no do
does did have has had can could will would should may might than then so such
""".split())
MAX_CTX = 640          # leave headroom for opts inside seq_len=768
MAX_OPT = 120


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip()]


def _content_words(text):
    return [w.lower() for w in _WORD.findall(text)
            if len(w) > 3 and w.lower() not in _STOP]


def _norm_num(s):
    return s.replace(",", "").rstrip("0.")


def grounded(q, a, passage):
    """qa_filter-style gates: schema, answer recall >=0.5, number binding."""
    if not q or not a or len(q) > 400 or not (8 <= len(a) <= 512):
        return False
    pset = set(_content_words(passage))
    aw = _content_words(a)
    if aw and sum(1 for w in aw if w in pset) / len(aw) < 0.5:
        return False
    pnums = {_norm_num(n) for n in re.findall(r"\d[\d.,]*", passage)}
    return all(_norm_num(n) in pnums
               for n in re.findall(r"\d[\d.,]*", a))


def load_qa_raw(pattern, rng, qas_per_pid=2, min_recall=True):
    """qa_raw_shard*.jsonl -> flat recs {pid, passage, q, a}, filtered."""
    recs = []
    for fp in sorted(glob.glob(pattern)):
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            pas = r.get("passage", "").strip()
            if not pas:
                continue
            qas = [qa for qa in r.get("qa", [])
                   if grounded(qa.get("q", "").strip(),
                               qa.get("a", "").strip(), pas)]
            rng.shuffle(qas)
            for qa in qas[:qas_per_pid]:
                recs.append({"pid": r["pid"], "passage": pas,
                             "q": qa["q"].strip(), "a": qa["a"].strip()})
    return recs


def perturb_number(ans, rng):
    """Return a copy of `ans` with one number changed, or None."""
    ms = list(NUM_RE.finditer(ans))
    if not ms:
        return None
    m = rng.choice(ms)
    x = float(m.group())
    factor = rng.choice([0.1, 0.5, 2.0, 10.0])
    y = x * factor
    rep = f"{y:g}"
    if rep == m.group():          # degenerate (x==0 etc.)
        rep = f"{x + rng.choice([1, 2, 5]):g}"
    return ans[:m.start()] + rep + ans[m.end():]


def truncate_ids(ids, n):
    return ids[:n]


def build(recs, by_pid, all_answers, tok, rng, tag):
    id_rows, text_by_state = [], {}
    for i, r in enumerate(recs):
        passage, q, gold_a = r["passage"].strip(), r["q"].strip(), r["a"].strip()
        if not passage or not q or not gold_a:
            continue
        state = f"Passage: {passage}\nQuestion: {q}"

        # ---------- choice: K=4 ----------
        sibs = [o["a"].strip() for o in by_pid.get(r["pid"], [])
                if o["a"].strip() != gold_a]
        distr = []
        rng.shuffle(sibs)
        distr += sibs[:2]                          # hard: same passage
        pert = perturb_number(gold_a, rng)
        if pert and pert not in distr:
            distr.append(pert)                     # hard: perturbed gold
        while len(distr) < 3:                      # fill: other passages
            c = rng.choice(all_answers)
            if c != gold_a and c not in distr:
                distr.append(c)
        opts = distr[:3] + [gold_a]
        rng.shuffle(opts)
        gold_i = opts.index(gold_a)
        ctx = truncate_ids(tok.encode(
            f"{state}\nQuestion: {INSTR_CHOICE}",
            add_special_tokens=False), MAX_CTX)
        id_rows.append({"ctx": ctx,
                        "opts": [truncate_ids(tok.encode(
                            o, add_special_tokens=False), MAX_OPT)
                                 for o in opts],
                        "gold": gold_i, "kind": "choice",
                        "qid": f"choice_{tag}_{i}"})
        ts = text_by_state.setdefault(
            state, {"state": state, "questions": {}})
        ts["questions"][f"choice_{tag}_{i}"] = {
            "type": "choice", "instructions": INSTR_CHOICE,
            "criteria": {o: None for o in opts}, "label": gold_a}

        # ---------- noul + score: gold vs related-wrong vs unrelated ----
        if sibs:
            related_wrong = sibs[0]
        elif pert:
            related_wrong = pert
        else:
            related_wrong = rng.choice(
                [a for a in all_answers if a != gold_a])
        unrelated = rng.choice([a for a in all_answers
                                if a != gold_a and a != related_wrong])
        variants = [(gold_a, True, 2), (related_wrong, False, 1),
                    (unrelated, False, 0)]
        for j, (ans, ok, sc) in enumerate(variants):
            st2 = f"{state}\nProposed answer: {ans}"
            ctx = truncate_ids(tok.encode(
                f"{st2}\nQuestion: {INSTR_NOUL}",
                add_special_tokens=False), MAX_CTX)
            yesno = [tok.encode("yes", add_special_tokens=False),
                     tok.encode("no", add_special_tokens=False)]
            id_rows.append({"ctx": ctx, "opts": yesno,
                            "gold": 0 if ok else 1, "kind": "noul",
                            "qid": f"noul_{tag}_{i}_{j}"})
            id_rows.append({"ctx": ctx, "opts": [
                truncate_ids(tok.encode(x, add_special_tokens=False),
                             MAX_OPT) for x in SCORE_LEGEND],
                "gold": sc, "kind": "score",
                "qid": f"score_{tag}_{i}_{j}"})
            ts = text_by_state.setdefault(
                st2, {"state": st2, "questions": {}})
            ts["questions"][f"noul_{tag}_{i}_{j}"] = {
                "type": "noul", "instructions": INSTR_NOUL, "label": ok}
            ts["questions"][f"score_{tag}_{i}_{j}"] = {
                "type": "score",
                "instructions": "Rate the proposed answer.",
                "criteria": SCORE_LEGEND, "label": sc}
    return id_rows, list(text_by_state.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=None, help="qa_devin_elite.jsonl")
    ap.add_argument("--qa-raw", action="append", default=[],
                    help="glob of qa_raw_shard*.jsonl {pid,passage,qa}; "
                         "repeatable. Alternative to --qa")
    ap.add_argument("--qas-per-pid", type=int, default=2,
                    help="max QAs kept per passage (qa-raw mode)")
    ap.add_argument("--train-pids", type=int, default=None,
                    help="explicit train split size in pids (else fractions)")
    ap.add_argument("--dev-pids", type=int, default=None)
    ap.add_argument("--eval-pids", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--eval-frac", type=float, default=0.20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.qa_raw:
        rows = []
        for pat in args.qa_raw:
            rs = load_qa_raw(pat, rng, qas_per_pid=args.qas_per_pid)
            print(f"[load] {pat}: {len(rs)} recs")
            rows += rs
    elif args.qa:
        rows = load_rows(args.qa)
    else:
        ap.error("--qa or --qa-raw required")
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["pid"], []).append(r)
    all_answers = sorted({r["a"].strip() for r in rows if r["a"].strip()})

    pids = sorted(by_pid, key=str)
    rng.shuffle(pids)
    if args.train_pids is not None:
        n_eval = args.eval_pids or 0
        n_dev = args.dev_pids or 0
        n_train = args.train_pids
    else:
        n_eval = int(len(pids) * args.eval_frac)
        n_dev = int(len(pids) * args.dev_frac)
        n_train = len(pids) - n_eval - n_dev
    pid_eval = set(pids[:n_eval])
    pid_dev = set(pids[n_eval:n_eval + n_dev])
    pid_train = set(pids[n_eval + n_dev:n_eval + n_dev + n_train])
    splits = {"eval": [], "dev": [], "train": []}
    for r in rows:
        tag = ("eval" if r["pid"] in pid_eval
               else "dev" if r["pid"] in pid_dev
               else "train" if r["pid"] in pid_train else None)
        if tag:
            splits[tag].append(r)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for tag, recs in splits.items():
        id_rows, text_rows = build(recs, by_pid, all_answers, tok, rng, tag)
        by_kind = {}
        for r in id_rows:
            by_kind.setdefault(r.pop("kind"), []).append(r)
        for kind, rs in by_kind.items():
            fp = out / f"sci_{kind}_{tag}.jsonl"
            fp.write_text("\n".join(json.dumps(r) for r in rs) + "\n")
            print(f"[wrote] {fp.name}: {len(rs)}")
        fp = out / f"sci_decisions_{tag}_text.jsonl"
        fp.write_text("\n".join(json.dumps(r) for r in text_rows) + "\n")
        print(f"[wrote] {fp.name}: {len(text_rows)}")
    print("pids:", {t: len({r["pid"] for r in rs})
                    for t, rs in splits.items()})


if __name__ == "__main__":
    main()
