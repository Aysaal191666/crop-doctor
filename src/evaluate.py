"""Evaluate a saved checkpoint on a split (test by default)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.train import SPLITS_CSV, LeafDataset, build_model, get_transforms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/baseline/best.pt")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    df = pd.read_csv(SPLITS_CSV)
    df = df[df["split"] == args.split]
    dl = DataLoader(LeafDataset(df, get_transforms(ckpt["size"], False)),
                    batch_size=32, num_workers=2)

    preds, true = [], []
    with torch.no_grad():
        for x, y in dl:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
            true += y.tolist()
    preds, true = np.array(preds), np.array(true)

    print(classification_report(true, preds, target_names=classes, digits=3))
    print(confusion_matrix(true, preds))

    healthy = [i for i, c in enumerate(classes) if "healthy" in c.lower()][0]
    diseased = true != healthy
    missed = int(((preds == healthy) & diseased).sum())
    acc = float((preds == true).mean())
    macro_f1 = float(f1_score(true, preds, average="macro"))
    print(f"Accuracy {acc:.3f} | macro F1 {macro_f1:.3f} | "
          f"diseased called healthy {missed}/{int(diseased.sum())}")

    out = Path(args.ckpt).parent / f"metrics_{args.split}.json"
    json.dump({"accuracy": acc, "macro_f1": macro_f1, "diseased_called_healthy": missed,
               "diseased_total": int(diseased.sum()), "n": int(len(true))},
              open(out, "w"), indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()