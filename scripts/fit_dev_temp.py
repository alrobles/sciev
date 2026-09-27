"""Fit the R2 temperature on a decisions file and print it.

Usage:
    python scripts/fit_dev_temp.py --ckpt <decision.pt> \
        --dev <decisions.jsonl> [--r2-layers=-1,-9,-17,-25] [--canonical-order]

Prints the fitted temperature to stdout (last line) for capture in shell.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from sciev.data import load_decisions_ids
from sciev.eval import fit_r2_temperature_decisions
from sciev.model import load_decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--r2-layers", default="-1,-9,-17,-25")
    ap.add_argument("--canonical-order", action="store_true")
    ap.add_argument("--r2-mode", default="spanpool")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layers = [int(x) for x in args.r2_layers.split(",")]
    model, head = load_decision(args.ckpt, device=args.device)
    if head is None:
        raise SystemExit("checkpoint has no decision head")

    rows = load_decisions_ids(args.dev)
    temp = fit_r2_temperature_decisions(
        model, head, rows, args.device, mode=args.r2_mode,
        layers=layers, canonical=args.canonical_order)
    print(json.dumps({"dev": args.dev, "n": len(rows), "T": temp}),
          file=sys.stderr)
    print(f"{temp:.6f}")


if __name__ == "__main__":
    main()
