"""Confidence thresholds for the mixed-source model (with the Other class).
Chosen on the combined validation set; checked once on each test set."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.analyze import MIN_COVERAGE, T_DISEASE_GRID, T_HEALTHY_GRID, get_probs, score
from src.train import build_model
from src.train_mixed import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mixed_b/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = ckpt["classes"]
    model = build_model(ckpt["model"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    healthy = classes.index("_Healthy")

    a, b = load(classes, use_other="_Other" in classes)
    sets = {
        "val lab": a[a["split"] == "val"],
        "val field": b[b["split"] == "val"],
        "test lab": a[a["split"] == "test"],
        "test field": b[b["split"] == "test"],
    }
    probs, labels = {}, {}
    for name, df in sets.items():
        probs[name], labels[name] = get_probs(model, df.reset_index(drop=True),
                                              ckpt["size"], device)

    pv = np.vstack([probs["val lab"], probs["val field"]])
    yv = np.concatenate([labels["val lab"], labels["val field"]])

    print("VALIDATION (choose thresholds from this)")
    rows = []
    for td in T_DISEASE_GRID:
        for th in T_HEALTHY_GRID:
            cov, acc, miss, nd = score(pv, yv, td, th, healthy)
            cl = score(probs["val lab"], labels["val lab"], td, th, healthy)[0]
            cf = score(probs["val field"], labels["val field"], td, th, healthy)[0]
            rows.append((td, th, cov, acc, miss, nd))
            print(f"disease>={td}, healthy>={th}: answers {cov:.0%} "
                  f"(lab {cl:.0%}, field {cf:.0%}) | acc {acc:.3f} | "
                  f"diseased called healthy {miss}/{nd}")

    ok = [r for r in rows if r[2] >= MIN_COVERAGE] or rows
    best = sorted(ok, key=lambda r: (r[4], -r[2]))[0]
    td, th = best[0], best[1]
    print(f"\nChosen on validation: disease>={td}, healthy>={th}")

    out = {"t_disease": td, "t_healthy": th, "min_coverage": MIN_COVERAGE,
           "classes": classes}
    for name in ("test lab", "test field"):
        cov, acc, miss, nd = score(probs[name], labels[name], td, th, healthy)
        cov0, acc0, miss0, nd0 = score(probs[name], labels[name], 0.0, 0.0, healthy)
        print(f"{name.upper()} with thresholds: answers {cov:.0%} | acc {acc:.3f} | "
              f"diseased called healthy {miss}/{nd}")
        print(f"{name.upper()} without thresholds: answers {cov0:.0%} | acc {acc0:.3f} | "
              f"diseased called healthy {miss0}/{nd0}")
        out[name.replace(" ", "_")] = {
            "n": int(len(labels[name])), "coverage": cov, "accuracy_answered": acc,
            "diseased_called_healthy": miss, "diseased_total": nd,
            "coverage_no_threshold": cov0, "accuracy_no_threshold": acc0,
            "diseased_called_healthy_no_threshold": miss0}

    json.dump(out, open(Path(args.ckpt).parent / "thresholds.json", "w"), indent=2)
    print("Saved thresholds.json next to", args.ckpt)


if __name__ == "__main__":
    main()