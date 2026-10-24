"""Crop Doctor API: rice leaf disease prediction with ONNX Runtime."""
import io
import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, closing
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.onnx"))
META_PATH = Path(os.environ.get("META_PATH", "models/model_meta.json"))
DB_PATH = Path(os.environ.get("DB_PATH", "data/app.db"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs

DISCLAIMER = ("This is a decision aid, not a diagnosis. For serious cases, "
              "contact your local agriculture officer.")
DISPLAY = {"_BrownSpot": "Brown Spot", "_Healthy": "Healthy", "_Hispa": "Hispa",
           "_LeafBlast": "Leaf Blast", "_Other": "Other disease"}


def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # phone photos are often stored sideways
    return img.convert("RGB")


def preprocess(data: bytes, size: int, mean, std) -> np.ndarray:
    """Same steps as training: RGB, resize straight to size x size, 0-1, normalise."""
    img = load_image(data).resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def decide(probs, classes, t_disease, t_healthy):
    """Answer only if confidence passes the threshold (higher bar for Healthy)."""
    i = int(np.argmax(probs))
    name, conf = classes[i], float(probs[i])
    threshold = t_healthy if name == "_Healthy" else t_disease
    if conf < threshold:
        return {"status": "retake", "label": None, "display_name": None,
                "confidence": conf,
                "message": ("Not sure. Please retake the photo in daylight "
                            "with the leaf filling the frame.")}
    shown = DISPLAY.get(name, name.strip("_"))
    if name == "_Other":
        status = "other"
        msg = ("This does not look like one of the conditions I know. "
               "Please ask an agriculture expert to check the plant.")
    elif name == "_Healthy":
        status = "result"
        msg = "The leaf looks healthy. If you still see problems, ask an agriculture officer."
    else:
        status = "result"
        msg = f"This may be {shown}. Please confirm with an agriculture officer before treating."
    return {"status": status, "label": name.strip("_"), "display_name": shown,
            "confidence": conf, "message": msg}


def db_run(sql, params=()):
    with closing(sqlite3.connect(DB_PATH)) as db:
        rows = db.execute(sql, params).fetchall()
        db.commit()
        return rows


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db_run("CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, ts REAL, "
           "status TEXT, label TEXT, confidence REAL)")
    db_run("CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, "
           "prediction_id TEXT, ts REAL, correct INTEGER, true_label TEXT, comment TEXT)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.meta = load_meta()
    app.state.session = ort.InferenceSession(str(MODEL_PATH),
                                             providers=["CPUExecutionProvider"])
    init_db()
    yield


app = FastAPI(title="Crop Doctor API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "classes": app.state.meta["classes"]}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Photo is too large (limit 10 MB).")
    meta = app.state.meta
    try:
        x = preprocess(data, meta["size"], meta["mean"], meta["std"])
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise HTTPException(400, "Could not read this file as a photo.")
    logits = app.state.session.run(None, {meta["onnx_input"]: x})[0][0]
    probs = softmax(logits)
    result = decide(probs, meta["classes"],
                    meta["thresholds"]["disease"], meta["thresholds"]["healthy"])
    pid = uuid.uuid4().hex
    db_run("INSERT INTO predictions (id, ts, status, label, confidence) VALUES (?, ?, ?, ?, ?)",
           (pid, time.time(), result["status"], result["label"], result["confidence"]))
    return {"prediction_id": pid, **result,
            "probabilities": {c.strip("_"): float(p) for c, p in zip(meta["classes"], probs)},
            "disclaimer": DISCLAIMER}


class Feedback(BaseModel):
    prediction_id: str
    correct: bool
    true_label: str | None = None
    comment: str | None = Field(default=None, max_length=500)


@app.post("/feedback", status_code=201)
def feedback(fb: Feedback):
    known = {c.strip("_") for c in app.state.meta["classes"]}
    if fb.true_label is not None and fb.true_label not in known:
        raise HTTPException(422, f"true_label must be one of {sorted(known)}")
    if not db_run("SELECT 1 FROM predictions WHERE id = ?", (fb.prediction_id,)):
        raise HTTPException(404, "Unknown prediction_id.")
    db_run("INSERT INTO feedback (prediction_id, ts, correct, true_label, comment) "
           "VALUES (?, ?, ?, ?, ?)",
           (fb.prediction_id, time.time(), int(fb.correct), fb.true_label, fb.comment))
    return {"saved": True}