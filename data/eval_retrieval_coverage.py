"""Measure explanation-term coverage of a retrieved-evidence variant.

Cheap CPU proxy for retrieval quality: per row, what fraction of the
gold-explanation content terms (oracle state minus clean state) appear in
the retrieved state (retrieved state minus clean state)?

    python data/eval_retrieval_coverage.py \
        --base gpqa_main_choice_eval --dir data/bench_external/gpqa/systemone-v2
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STOPWORDS = frozenset(
    "a an the and or of to in is are was were be been for on at by with "
    "that this these those which what when where who whom how why not no "
    "it its he she they we you i do does did have has had can could will "
    "would should may might must shall from as if then than so such into "
    "about between through during before after above below up down out "
    "over under again further once each other some any all both few more "
    "most only own same very just also".split())


def terms(text):
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())
            if w not in STOPWORDS}


def load(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def coverage(directory, stem):
    base = Path(directory)
    oracle = load(base / f"{stem}_oracle_text.jsonl")
    retrieved = load(base / f"{stem}_retrieved_text.jsonl")
    clean = load(base / f"{stem}_clean_text.jsonl")
    ratios = []
    for ora, ret, cln in zip(oracle, retrieved, clean):
        oev = terms(ora["state"]) - terms(cln["state"])
        rev = terms(ret["state"]) - terms(cln["state"])
        ratios.append(len(oev & rev) / max(1, len(oev)))
    return ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--base", default="gpqa_main_choice_eval",
                    help="file stem: <base>_{oracle,retrieved,clean}_text.jsonl")
    ap.add_argument("--retrieved-tag", default=None,
                    help="use <base>_retrieved_<tag>_text.jsonl instead")
    args = ap.parse_args()
    if args.retrieved_tag:
        stem = f"{args.base}_retrieved_{args.retrieved_tag}"
        directory = Path(args.dir)
        oracle = load(directory / f"{args.base}_oracle_text.jsonl")
        retrieved = load(directory / f"{stem}_text.jsonl")
        clean = load(directory / f"{args.base}_clean_text.jsonl")
        ratios = []
        for ora, ret, cln in zip(oracle, retrieved, clean):
            oev = terms(ora["state"]) - terms(cln["state"])
            rev = terms(ret["state"]) - terms(cln["state"])
            ratios.append(len(oev & rev) / max(1, len(oev)))
    else:
        ratios = coverage(args.dir, args.base)
    ratios.sort()
    n = len(ratios)
    print(json.dumps({
        "dir": args.dir, "base": args.base, "retrieved_tag": args.retrieved_tag,
        "n": n, "median": ratios[n // 2], "mean": sum(ratios) / n,
        "ge_30pct": sum(r > 0.3 for r in ratios),
        "ge_50pct": sum(r > 0.5 for r in ratios)}))


if __name__ == "__main__":
    main()
