"""Data pipeline for Crop Doctor: download, index, group near-duplicates, split."""
import argparse
import subprocess
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

DATASET = "nizorogbezuode/rice-leaf-images"
RAW_DIR = Path("data/raw")
OUT_CSV = Path("data/splits.csv")
IMG_EXT = {".jpg", ".jpeg", ".png"}
SEED = 42
HASH_THRESHOLD = 4  # max hash distance for two images to count as near-duplicates


def download(raw_dir: Path = RAW_DIR) -> None:
    """Download and unzip the Kaggle dataset (needs KAGGLE_API_TOKEN)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(raw_dir), "--unzip"],
        check=True,
    )


def find_image_root(raw_dir: Path = RAW_DIR) -> Path:
    """Return the folder whose sub-folders are the class folders."""
    candidates = {}
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            candidates.setdefault(p.parent.parent, set()).add(p.parent.name)
    if not candidates:
        raise FileNotFoundError(f"No images found under {raw_dir}")
    return max(candidates.items(), key=lambda kv: len(kv[1]))[0]


def build_index(root: Path) -> pd.DataFrame:
    """One row per image: path, label, class_name."""
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    rows = []
    for label, cname in enumerate(classes):
        for f in sorted((root / cname).iterdir()):
            if f.suffix.lower() in IMG_EXT:
                rows.append({"path": str(f), "label": label, "class_name": cname})
    return pd.DataFrame(rows)


def compute_hashes(paths) -> np.ndarray:
    """Perceptual hash (64 bits) for each image."""
    return np.array(
        [imagehash.phash(Image.open(p).convert("RGB")).hash.flatten() for p in paths]
    )


def group_duplicates(hashes: np.ndarray, threshold: int = HASH_THRESHOLD) -> np.ndarray:
    """Give near-identical images the same group id (union-find on hash distance)."""
    n = len(hashes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n - 1):
        dist = (hashes[i + 1:] != hashes[i]).sum(axis=1)
        for j in np.where(dist <= threshold)[0]:
            a, b = find(i), find(i + 1 + j)
            if a != b:
                parent[a] = b
    return np.array([find(i) for i in range(n)])


def make_splits(df: pd.DataFrame, seed: int = SEED, n_splits: int = 7) -> pd.DataFrame:
    """Group-aware split (about 71/14/14): duplicates never cross train/val/test."""
    df = df.reset_index(drop=True).copy()
    folds = list(
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        .split(df["path"], df["label"], df["group"])
    )
    df["split"] = "train"
    df.loc[df.index[folds[0][1]], "split"] = "test"
    df.loc[df.index[folds[1][1]], "split"] = "val"

    g = {s: set(df.loc[df["split"] == s, "group"]) for s in ("train", "val", "test")}
    if g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"]:
        raise ValueError("Leakage: a duplicate group appears in more than one split")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download the dataset first")
    args = ap.parse_args()

    if args.download:
        download()
    root = find_image_root()
    df = build_index(root)
    df["group"] = group_duplicates(compute_hashes(df["path"]))
    df = make_splits(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Root: {root}")
    print(f"Images: {len(df)} | Unique groups: {df['group'].nunique()}")
    print(df.groupby(["split", "class_name"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()