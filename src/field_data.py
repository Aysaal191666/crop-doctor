"""Index the field-photo dataset (RiceyLeafDisease originals), group near-duplicate
photos, and make a group-aware train/val/test split."""
import argparse
from pathlib import Path

from src.cross_eval import FOLDER_MAP, list_images
from src.data import SEED, compute_hashes, group_duplicates, make_splits

OUT_CSV = Path("data/field_splits.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/ext")
    args = ap.parse_args()

    df = list_images(Path(args.root))
    folder_ids = {f: i for i, f in enumerate(sorted(FOLDER_MAP))}
    df["label"] = df["folder"].map(folder_ids)
    df["group"] = group_duplicates(compute_hashes(df["path"]))
    df = make_splits(df, seed=SEED)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    mixed = df.groupby("group")["folder"].nunique()
    print(f"Photos: {len(df)} | Groups: {df['group'].nunique()}")
    print(f"Photos that have a near-duplicate: {len(df) - df['group'].nunique()}")
    print(f"Groups containing photos from more than one folder: {(mixed > 1).sum()}")
    print(df.groupby(["split", "folder"]).size().unstack(fill_value=0).T)
    print("Saved", OUT_CSV)


if __name__ == "__main__":
    main()