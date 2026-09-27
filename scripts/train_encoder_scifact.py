"""Encoder baseline for SciFact: DeBERTa-v3 sequence classifier.

Trains a plain bidirectional encoder on the same evidence+claim state the
decision heads consume, then reports noul (SUPPORT vs rest) and legacy
score (SUPPORT=2, CONTRADICT=1, NEI=0) accuracy on the validation split —
the comparable baselines for the frozen-diffusion heads.

Usage:
    python train_encoder_scifact.py --train train.parquet \
        --dev validation.parquet --out eval_encoder_scifact.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

MODEL = "microsoft/deberta-v3-base"
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]
SCORE_MAP = {"SUPPORT": 2, "CONTRADICT": 1, "NEI": 0}


class Claims(Dataset):
    def __init__(self, df, tok, max_len=512):
        self.df = df
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        abstract = " ".join(list(r["abstract"]))
        text = f"Title: {r['title']}\nAbstract: {abstract}\nClaim: {r['claim']}"
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        return {"input_ids": enc.input_ids[0],
                "attention_mask": enc.attention_mask[0],
                "label": LABELS.index(r["verdict"])}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    lab = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["input_ids"])] = b["input_ids"]
        att[i, : len(b["attention_mask"])] = b["attention_mask"]
        lab[i] = b["label"]
    return ids, att, lab


def predict(model, loader, device):
    model.eval()
    preds, golds, probs = [], [], []
    with torch.no_grad():
        for ids, att, lab in loader:
            logits = model(input_ids=ids.to(device),
                           attention_mask=att.to(device)).logits
            probs.append(torch.softmax(logits.float(), -1).cpu())
            preds += logits.argmax(-1).tolist()
            golds += lab.tolist()
    return np.array(preds), np.array(golds), torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=7331)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    train = pd.read_parquet(args.train)
    dev = pd.read_parquet(args.dev)
    train = train[train["verdict"].isin(LABELS)].reset_index(drop=True)
    dev = dev[dev["verdict"].isin(LABELS)].reset_index(drop=True)

    pad = tok.pad_token_id or 0
    tr_loader = DataLoader(Claims(train, tok), batch_size=args.bs,
                           shuffle=True, collate_fn=lambda b: collate(b, pad))
    dv_loader = DataLoader(Claims(dev, tok), batch_size=args.bs * 2,
                           collate_fn=lambda b: collate(b, pad))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS)).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        opt, int(0.1 * len(tr_loader) * args.epochs),
        len(tr_loader) * args.epochs)

    t0 = time.time()
    model.train()
    for _ in range(args.epochs):
        for ids, att, lab in tr_loader:
            loss = model(input_ids=ids.to(args.device),
                         attention_mask=att.to(args.device),
                         labels=lab.to(args.device)).loss
            loss.backward()
            opt.step()
            sched.step()
            model.zero_grad()

    preds, golds, probs = predict(model, dv_loader, args.device)
    acc3 = float((preds == golds).mean())
    noul_pred = (preds == 0).astype(int)
    noul_gold = (golds == 0).astype(int)
    score_pred = np.array([SCORE_MAP[LABELS[p]] for p in preds])
    score_gold = np.array([SCORE_MAP[LABELS[g]] for g in golds])
    out = {
        "model": args.model, "n_dev": len(dev), "n_train": len(train),
        "epochs": args.epochs, "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 2),
        "verdict3_acc": round(acc3, 4),
        "noul_acc": round(float((noul_pred == noul_gold).mean()), 4),
        "noul_baseline_majority": round(float(max(noul_gold.mean(),
                                                  1 - noul_gold.mean())), 4),
        "score_acc": round(float((score_pred == score_gold).mean()), 4),
        "confusion": {LABELS[g]: {LABELS[p]: int(((golds == g) &
                                                 (preds == p)).sum())
                                 for p in range(3)} for g in range(3)},
        "predictions": [{"verdict_pred": LABELS[p], "verdict_gold": LABELS[g],
                         "p": probs[i].tolist()}
                        for i, (p, g) in enumerate(zip(preds, golds))],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "predictions"},
                     indent=2))


if __name__ == "__main__":
    main()
