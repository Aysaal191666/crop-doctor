"""Cross-source evaluation: test a trained model on photos from a different dataset.
Classes shared with training give normal metrics. Diseases the model never saw
show whether it wrongly answers with confidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.train import LeafDataset, build_model, get_transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# External folder name -> our class name (None = disease the model was never trained on)
FOLDER_MAP = {
    "Brown Spot": "_BrownSpot",
    "Healthy Rice Leaf": "_Healthy",
    "Rice Hispa": "_Hispa",
    "Leaf Blast": "_LeafBlast",
    "Bacterial Leaf Blight": None,
    "Leaf scald": None,
    "Narrow Brown Leaf Spot": None,
    "Sheath Blight": None,
}


def list_images(root: Path) -> pd.DataFrame:
    rows = []
    for folder, mapped in FOLDER_MAP.items():
        d = root / folder
        if not d.is_dir():
            raise FileNotFoundError(f"Missing folder: {d}")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                rows.append({"path": str(f), "folder": folder, "mapped": mapped})
    return pd.DataFrame(rows)


def get_probs(model, df, size, device):
    tmp = df[["path"]].copy()
    tmp["label"] = 0  # placeholder, labels are not used here
    dl = DataLoader(LeafDataset(tmp, get_transforms(size, False)), batch_size=32, num_workers=2)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            out.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.vstack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/effb0_aug/best.pt")
    ap.add_argument("--root", default="/content/ext")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    healthy = classes.index("_Healthy")

    ckpt_dir = Path(args.ckpt).parent
    thr = json.load(open(ckpt_dir / "thresholds.json"))
    t_dis, t_h = thr["t_disease"], thr["t_healthy"]

    df = list_images(Path(args.root))
    print("Images found per folder:")
    print(df.groupby("folder").size().to_string())

    probs = get_probs(model, df, ckpt["size"], device)
    pred, conf = probs.argmax(1), probs.max(1)
    answered = conf >= np.where(pred == healthy, t_h, t_dis)

    df["pred"] = [classes[i] for i in pred]
    df["conf"] = conf
    df["answered"] = answered
    df.to_csv(ckpt_dir / "cross_source_predictions.csv", index=False)

    # ---- In-scope: classes the model was trained on ----
    inscope = df["mapped"].notna().to_numpy()
    y = np.array([classes.index(m) for m in df.loc[inscope, "mapped"]])
    p, a = pred[inscope], answered[inscope]
    labels = list(range(len(classes)))
    diseased = y != healthy
    acc = float((p == y).mean())
    macro = float(f1_score(y, p, labels=labels, average="macro", zero_division=0))
    missed = int(((p == healthy) & diseased).sum())

    print(f"\nIN-SCOPE PHOTOS ({inscope.sum()}), no thresholds")
    print(classification_report(y, p, labels=labels, target_names=classes,
                                digits=3, zero_division=0))
    print(confusion_matrix(y, p, labels=labels))
    print(f"Accuracy {acc:.3f} | macro F1 {macro:.3f} | "
          f"diseased called healthy {missed}/{int(diseased.sum())}")

    cov = float(a.mean())
    acc_a = float((p[a] == y[a]).mean()) if a.any() else float("nan")
    missed_a = int(((p == healthy) & diseased & a).sum())
    print(f"\nWith thresholds (disease>={t_dis}, healthy>={t_h}): answers {cov:.0%} | "
          f"accuracy on answered {acc_a:.3f} | "
          f"diseased called healthy {missed_a}/{int(diseased.sum())}")

    # ---- Compare with the internal test set ----
    mt = ckpt_dir / "metrics_test.json"
    if mt.exists():
        m = json.load(open(mt))
        print(f"\nINTERNAL TEST (same source as training): accuracy {m['accuracy']:.3f} | "
              f"macro F1 {m['macro_f1']:.3f} | diseased called healthy "
              f"{m['diseased_called_healthy']}/{m['diseased_total']}")
        print(f"EXTERNAL (different source):              accuracy {acc:.3f} | "
              f"macro F1 {macro:.3f} | diseased called healthy {missed}/{int(diseased.sum())}")

    # ---- Out-of-scope: diseases the model never saw ----
    oos = ~inscope
    po, ao = pred[oos], answered[oos]
    folders = df.loc[oos, "folder"].to_numpy()
    print(f"\nDISEASES THE MODEL NEVER SAW ({oos.sum()} photos)")
    for f in sorted(set(folders)):
        mk = folders == f
        print(f"{f}: n={mk.sum()} | answered confidently {ao[mk].mean():.0%} | "
              f"confidently called Healthy {((po[mk] == healthy) & ao[mk]).mean():.0%}")
    print(f"All: answered confidently {ao.mean():.0%} | "
          f"confidently called Healthy {((po == healthy) & ao).mean():.0%}")

    json.dump({"in_scope_n": int(inscope.sum()), "accuracy": acc, "macro_f1": macro,
               "diseased_called_healthy": missed, "diseased_total": int(diseased.sum()),
               "thr_coverage": cov, "thr_accuracy_answered": acc_a,
               "thr_diseased_called_healthy": missed_a,
               "unseen_n": int(oos.sum()),
               "unseen_answered_confidently": float(ao.mean()),
               "unseen_confidently_called_healthy": float(((po == healthy) & ao).mean())},
              open(ckpt_dir / "cross_source.json", "w"), indent=2)
    print("\nSaved cross_source.json and cross_source_predictions.csv in", ckpt_dir)


if __name__ == "__main__":
    main()