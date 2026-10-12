"""Train on lab-style photos (source A) plus field photos (source B).
Evaluates on the source-A test set and the field test set separately."""
import argparse
import json
import time
from pathlib import Path 

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.cross_eval import FOLDER_MAP
from src.train import SEED, LeafDataset, build_model, get_transforms, run_epoch

A_CSV = Path("data/splits.csv")
B_CSV = Path("data/field_splits.csv")
KNOWN = ["_BrownSpot", "_Healthy", "_Hispa", "_LeafBlast"]
OTHER = "_Other"


def load(classes, use_other):
    a = pd.read_csv(A_CSV)[["path", "class_name", "split"]].copy()
    b = pd.read_csv(B_CSV)[["path", "folder", "split"]].copy()
    b["class_name"] = b["folder"].map(FOLDER_MAP)
    if use_other:
        b["class_name"] = b["class_name"].fillna(OTHER)
    else:
        b = b[b["class_name"].notna()]
    b = b[["path", "class_name", "split"]].copy()
    idx = {c: i for i, c in enumerate(classes)}
    a["label"] = a["class_name"].map(idx)
    b["label"] = b["class_name"].map(idx)
    return a, b


def predict(model, df, size, device):
    dl = DataLoader(LeafDataset(df, get_transforms(size, False)), batch_size=32, num_workers=2)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            preds += model(x.to(device)).argmax(1).cpu().tolist()
    return np.array(preds)


def report(name, df, preds, classes):
    y = df["label"].to_numpy()
    present = sorted(set(y.tolist()))
    print(f"\n=== {name} ({len(y)} photos) ===")
    print(classification_report(y, preds, labels=present,
                                target_names=[classes[i] for i in present],
                                digits=3, zero_division=0))
    print(confusion_matrix(y, preds, labels=list(range(len(classes)))))
    healthy = classes.index("_Healthy")
    diseased = y != healthy
    missed = int(((preds == healthy) & diseased).sum())
    acc = float((preds == y).mean())
    macro = float(f1_score(y, preds, labels=present, average="macro", zero_division=0))
    print(f"Accuracy {acc:.3f} | macro F1 {macro:.3f} | "
          f"diseased called healthy {missed}/{int(diseased.sum())}")
    return {"n": int(len(y)), "accuracy": acc, "macro_f1": macro,
            "diseased_called_healthy": missed, "diseased_total": int(diseased.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--aug", choices=["flip", "rotate"], default="rotate")
    ap.add_argument("--other", action="store_true",
                    help="add an Other class for the diseases the model was never trained on")
    ap.add_argument("--out", default="runs/mixed_a")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    classes = KNOWN + ([OTHER] if args.other else [])
    a, b = load(classes, args.other)
    train = pd.concat([a[a["split"] == "train"], b[b["split"] == "train"]], ignore_index=True)
    val = pd.concat([a[a["split"] == "val"], b[b["split"] == "val"]], ignore_index=True)
    tests = {"source A test (lab-style)": a[a["split"] == "test"].reset_index(drop=True),
             "field test": b[b["split"] == "test"].reset_index(drop=True)}
    print("Classes:", classes)
    print(f"Train {len(train)} | Val {len(val)}")
    print(train["class_name"].value_counts().to_string())

    train_dl = DataLoader(LeafDataset(train, get_transforms(args.size, True, args.aug)),
                          batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(LeafDataset(val, get_transforms(args.size, False)),
                        batch_size=args.batch, num_workers=2)

    model = build_model(args.model, len(classes)).to(device)
    counts = np.bincount(train["label"], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best_acc, log, start = 0.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, loss_fn, device, optimizer)
        vl, vacc = run_epoch(model, val_dl, loss_fn, device)
        print(f"epoch {epoch}: train acc {ta:.3f} | val loss {vl:.3f} | val acc {vacc:.3f}")
        log.append({"epoch": epoch, "train_acc": ta, "val_loss": vl, "val_acc": vacc})
        if vacc >= best_acc:
            best_acc = vacc
            torch.save({"model": args.model, "size": args.size, "classes": classes,
                        "state_dict": model.state_dict()}, out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    results = {}
    for name, df in tests.items():
        results[name] = report(name, df, predict(model, df, args.size, device), classes)

    json.dump({"args": vars(args), "log": log, "best_val_acc": best_acc,
               "results": results, "seconds": time.time() - start},
              open(out / "mixed_results.json", "w"), indent=2)
    print(f"\nDone. Best val acc {best_acc:.3f}. Saved to {out}")


if __name__ == "__main__":
    main()