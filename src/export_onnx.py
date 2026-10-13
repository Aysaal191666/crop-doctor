"""Export a trained checkpoint to ONNX, check it matches PyTorch, and benchmark size and speed."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from src.train import MEAN, STD, LeafDataset, build_model, get_transforms
from src.train_mixed import load


def export(model, size, path):
    dummy = torch.randn(1, 3, size, size)
    kwargs = dict(input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except Exception as e:
        print("Legacy exporter failed, trying the default exporter:", repr(e)[:200])
        torch.onnx.export(model, dummy, str(path), **kwargs)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compare(model, sess, df, size):
    dl = DataLoader(LeafDataset(df, get_transforms(size, False)), batch_size=32, num_workers=2)
    ys, p_t, p_o = [], [], []
    with torch.no_grad():
        for x, y in dl:
            p_t.append(torch.softmax(model(x), 1).numpy())
            p_o.append(softmax(sess.run(None, {"image": x.numpy()})[0]))
            ys += y.tolist()
    pt, po, y = np.vstack(p_t), np.vstack(p_o), np.array(ys)
    return {"n": int(len(y)),
            "pytorch_accuracy": float((pt.argmax(1) == y).mean()),
            "onnx_accuracy": float((po.argmax(1) == y).mean()),
            "same_prediction": float((pt.argmax(1) == po.argmax(1)).mean()),
            "max_prob_difference": float(np.abs(pt - po).max())}


def time_it(fn, runs=50, warmup=5):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/mixed_b/best.pt")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out = ckpt_path.parent
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes, size = ckpt["classes"], ckpt["size"]
    model = build_model(ckpt["model"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = out / "model.onnx"
    export(model, size, onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("Exported:", onnx_path)

    a, b = load(classes, use_other="_Other" in classes)
    tests = {"lab-style test": a[a["split"] == "test"].reset_index(drop=True),
             "field test": b[b["split"] == "test"].reset_index(drop=True)}
    report = {"compare": {}}
    for name, df in tests.items():
        r = compare(model, sess, df, size)
        report["compare"][name] = r
        print(f"{name} ({r['n']} photos): PyTorch accuracy {r['pytorch_accuracy']:.3f} | "
              f"ONNX accuracy {r['onnx_accuracy']:.3f} | "
              f"same prediction {r['same_prediction']:.1%} | "
              f"max probability difference {r['max_prob_difference']:.6f}")

    dummy = torch.randn(1, 3, size, size)
    with torch.no_grad():
        t_med, t_p95 = time_it(lambda: model(dummy))
    o_med, o_p95 = time_it(lambda: sess.run(None, {"image": dummy.numpy()}))
    pt_mb = ckpt_path.stat().st_size / 1e6
    onnx_mb = onnx_path.stat().st_size / 1e6
    print(f"\nSize: PyTorch file {pt_mb:.1f} MB | ONNX file {onnx_mb:.1f} MB")
    print(f"Speed, 1 photo on this machine's CPU (median / 95th percentile): "
          f"PyTorch {t_med:.1f} / {t_p95:.1f} ms | ONNX Runtime {o_med:.1f} / {o_p95:.1f} ms")
    report.update({"pytorch_mb": pt_mb, "onnx_mb": onnx_mb,
                   "pytorch_ms": [t_med, t_p95], "onnx_ms": [o_med, o_p95]})

    thr = json.load(open(out / "thresholds.json"))
    meta = {"classes": classes, "size": size, "mean": MEAN, "std": STD,
            "preprocess": ("RGB, resize to size x size (no aspect-ratio crop), "
                           "scale to 0-1, normalise with mean/std, NCHW float32"),
            "thresholds": {"disease": thr["t_disease"], "healthy": thr["t_healthy"]},
            "onnx_input": "image", "onnx_output": "logits (apply softmax)"}
    json.dump(meta, open(out / "model_meta.json", "w"), indent=2)
    json.dump(report, open(out / "export_report.json", "w"), indent=2)
    print("Saved model.onnx, model_meta.json and export_report.json in", out)


if __name__ == "__main__":
    main()