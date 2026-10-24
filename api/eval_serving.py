"""Run the real API pipeline (preprocess + ONNX + decision rule) over the lab-style test set.
Needs no PyTorch. Results should match the Colab run of the same model."""
import time

import numpy as np
import onnxruntime as ort
import pandas as pd

import api.main as m


def main():
    meta = m.load_meta()
    classes = meta["classes"]
    t_dis, t_h = meta["thresholds"]["disease"], meta["thresholds"]["healthy"]
    idx = {c: i for i, c in enumerate(classes)}
    sess = ort.InferenceSession(str(m.MODEL_PATH), providers=["CPUExecutionProvider"])

    df = pd.read_csv("data/splits.csv")
    df = df[df["split"] == "test"].reset_index(drop=True)

    true, pred, answered = [], [], []
    start = time.time()
    for _, row in df.iterrows():
        with open(row["path"], "rb") as f:
            data = f.read()
        x = m.preprocess(data, meta["size"], meta["mean"], meta["std"])
        probs = m.softmax(sess.run(None, {meta["onnx_input"]: x})[0][0])
        r = m.decide(probs, classes, t_dis, t_h)
        true.append(idx[row["class_name"]])
        pred.append(int(np.argmax(probs)))
        answered.append(r["status"] != "retake")

    true, pred, ans = np.array(true), np.array(pred), np.array(answered)
    healthy = idx["_Healthy"]
    diseased = true != healthy
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1

    print(f"Photos: {len(true)} | {time.time() - start:.0f} seconds")
    print("Classes:", classes)
    print(cm)
    print(f"No thresholds: accuracy {(pred == true).mean():.3f} | "
          f"diseased called healthy {int(((pred == healthy) & diseased).sum())}/{int(diseased.sum())}")
    print(f"With thresholds: answers {ans.mean():.0%} | "
          f"accuracy on answered {(pred[ans] == true[ans]).mean():.3f} | "
          f"diseased called healthy {int(((pred == healthy) & diseased & ans).sum())}/{int(diseased.sum())}")
    print("Colab run of the same model: accuracy 0.775, 25/267 | "
          "with thresholds: answers 69%, accuracy 0.751, 3/267")


if __name__ == "__main__":
    main()