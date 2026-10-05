"""Error analysis and confidence thresholds.
Thresholds are chosen on the validation set and checked once on the test set."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.train import SPLITS_CSV, LeafDataset, build_model, get_transforms

T_DISEASE_GRID = [0.4, 0.5, 0.6, 0.7]
T_HEALTHY_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
MIN_COVERAGE = 0.70  # the app must answer at least 70% of photos


def get_probs(model, df, size, device):
    dl = DataLoader(LeafDataset(df, get_transforms(size, False)), batch_size=32, num_workers=2)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
            labels += y.tolist()
    return np.vstack(probs), np.array(labels)


def apply_rule(probs, t_dis, t_healthy, healthy_idx):
    """Answer only if confidence passes the threshold (higher bar for 'Healthy')."""
    pred = probs.argmax(1)
    thr = np.where(pred == healthy_idx, t_healthy, t_dis)
    return pred, probs.max(1) >= thr


def score(probs, labels, t_dis, t_healthy, healthy_idx):
    pred, answered = apply_rule(probs, t_dis, t_healthy, healthy_idx)
    coverage = float(answered.mean())
    acc = float((pred[answered] == labels[answered]).mean()) if answered.any() else float("nan")
    diseased = labels != healthy_idx
    missed = int(((pred == healthy_idx) & answered & diseased).sum())
    return coverage, acc, missed, int(diseased.sum())


def save_mistakes(model_probs, labels, paths, classes, out_png, n=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = model_probs.argmax(1)
    wrong = np.where(pred != labels)[0]
    rng = np.random.RandomState(0)
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.axis("off")
        ax.set_title(f"true: {classes[labels[i]]}\npred: {classes[pred[i]]} "
                     f"({model_probs[i].max():.0%})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/effb0_aug/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    healthy = [i for i, c in enumerate(classes) if "healthy" in c.lower()][0]

    df = pd.read_csv(SPLITS_CSV)
    val_df, test_df = df[df["split"] == "val"], df[df["split"] == "test"]
    pv, yv = get_probs(model, val_df, ckpt["size"], device)
    pt, yt = get_probs(model, test_df, ckpt["size"], device)

    print("VALIDATION (choose thresholds from this)")
    rows = []
    for td in T_DISEASE_GRID:
        for th in T_HEALTHY_GRID:
            cov, acc, miss, nd = score(pv, yv, td, th, healthy)
            rows.append((td, th, cov, acc, miss, nd))
            print(f"disease>={td}, healthy>={th}: answers {cov:.0%} | acc {acc:.3f} | "
                  f"diseased called healthy {miss}/{nd}")

    ok = [r for r in rows if r[2] >= MIN_COVERAGE] or rows
    best = sorted(ok, key=lambda r: (r[4], -r[2]))[0]
    td, th = best[0], best[1]
    print(f"\nChosen on validation: disease>={td}, healthy>={th}")

    cov, acc, miss, nd = score(pt, yt, td, th, healthy)
    print(f"TEST (checked once): answers {cov:.0%} | acc {acc:.3f} | "
          f"diseased called healthy {miss}/{nd}")
    cov0, acc0, miss0, nd0 = score(pt, yt, 0.0, 0.0, healthy)
    print(f"TEST without thresholds: answers {cov0:.0%} | acc {acc0:.3f} | "
          f"diseased called healthy {miss0}/{nd0}")

    out = Path(args.ckpt).parent
    json.dump({"t_disease": td, "t_healthy": th, "min_coverage": MIN_COVERAGE,
               "test_coverage": cov, "test_accuracy_answered": acc,
               "test_diseased_called_healthy": miss, "test_diseased_total": nd},
              open(out / "thresholds.json", "w"), indent=2)
    save_mistakes(pt, yt, test_df["path"].tolist(), classes, out / "mistakes.png")
    print("Saved thresholds.json and mistakes.png in", out)


if __name__ == "__main__":
    main()