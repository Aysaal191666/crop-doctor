"""Train an image classifier on the grouped split in data/splits.csv."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SPLITS_CSV = Path("data/splits.csv")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SEED = 42
PAPER_GREY = (235, 235, 235)  # fill colour for rotated corners (matches the paper background)


def get_transforms(size: int, train: bool, aug: str = "flip"):
    """aug='flip': flips only. aug='rotate': flips + rotation + colour jitter.
    No random crop in either, so tiny lesions are never cropped out."""
    steps = [transforms.Resize((size, size))]
    if train:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        if aug == "rotate":
            steps += [transforms.RandomRotation(20, fill=PAPER_GREY),
                      transforms.ColorJitter(0.2, 0.2, 0.2)]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


class LeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="DEFAULT")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="DEFAULT")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def run_epoch(model, dl, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--aug", choices=["flip", "rotate"], default="flip")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--cosine", action="store_true", help="cosine learning-rate schedule")
    ap.add_argument("--out", default="runs/baseline")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    df = pd.read_csv(SPLITS_CSV)
    classes = df.drop_duplicates("label").sort_values("label")["class_name"].tolist()
    tr = df[df["split"] == "train"]
    va = df[df["split"] == "val"]

    train_dl = DataLoader(LeafDataset(tr, get_transforms(args.size, True, args.aug)),
                          batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(LeafDataset(va, get_transforms(args.size, False)),
                        batch_size=args.batch, num_workers=2)

    model = build_model(args.model, len(classes)).to(device)
    counts = np.bincount(tr["label"], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.cosine else None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log, best_acc = [], 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, loss_fn, device, optimizer)
        vl, vacc = run_epoch(model, val_dl, loss_fn, device)
        if scheduler:
            scheduler.step()
        print(f"epoch {epoch}: train acc {ta:.3f} | val loss {vl:.3f} | val acc {vacc:.3f}")
        log.append({"epoch": epoch, "train_acc": ta, "val_loss": vl, "val_acc": vacc})
        if vacc >= best_acc:
            best_acc = vacc
            torch.save({"model": args.model, "size": args.size, "classes": classes,
                        "state_dict": model.state_dict()}, out / "best.pt")

    json.dump({"args": vars(args), "log": log, "seconds": time.time() - start},
              open(out / "train_log.json", "w"), indent=2)
    print(f"Done. Best val acc {best_acc:.3f}. Saved to {out}/best.pt")


if __name__ == "__main__":
    main()