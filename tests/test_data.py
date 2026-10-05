import numpy as np
import pandas as pd

from src.data import group_duplicates, make_splits


def test_group_duplicates_merges_identical_images():
    zeros = np.zeros(64, dtype=bool)
    ones = np.ones(64, dtype=bool)
    groups = group_duplicates(np.array([zeros, zeros.copy(), ones]))
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def _fake_df():
    rows, gid = [], 0
    for label in range(4):
        for k in range(20):          # 20 groups per class
            for copy in range(2):    # 2 near-identical images per group
                rows.append({"path": f"img_{label}_{k}_{copy}.jpg", "label": label, "group": gid})
            gid += 1
    return pd.DataFrame(rows)


def test_splits_have_no_group_leakage():
    df = make_splits(_fake_df())
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ga = set(df.loc[df["split"] == a, "group"])
        gb = set(df.loc[df["split"] == b, "group"])
        assert not (ga & gb)


def test_every_image_gets_a_split():
    df = make_splits(_fake_df())
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["split"].notna().all()


def test_splits_are_reproducible():
    a = make_splits(_fake_df(), seed=42)
    b = make_splits(_fake_df(), seed=42)
    assert (a["split"] == b["split"]).all()