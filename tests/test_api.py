import io
import sqlite3
from contextlib import closing

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.main as api_main

CLASSES = ["_BrownSpot", "_Healthy", "_Hispa", "_LeafBlast", "_Other"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "DB_PATH", tmp_path / "test.db")
    with TestClient(api_main.app) as c:
        yield c


def png_bytes(size=(300, 200), color=(60, 140, 60)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def post_photo(client, data=None, name="leaf.png", ctype="image/png"):
    return client.post("/predict", files={"file": (name, data or png_bytes(), ctype)})


def feedback_rows():
    with closing(sqlite3.connect(api_main.DB_PATH)) as db:
        return db.execute("SELECT prediction_id, correct, true_label FROM feedback").fetchall()


# ---- decision rule ----
def test_confident_disease_is_answered():
    r = api_main.decide(np.array([0.8, 0.05, 0.05, 0.05, 0.05]), CLASSES, 0.4, 0.9)
    assert r["status"] == "result" and r["label"] == "BrownSpot"


def test_low_confidence_asks_for_retake():
    r = api_main.decide(np.array([0.35, 0.2, 0.15, 0.15, 0.15]), CLASSES, 0.4, 0.9)
    assert r["status"] == "retake" and r["label"] is None


def test_healthy_needs_higher_confidence():
    low = api_main.decide(np.array([0.03, 0.85, 0.04, 0.04, 0.04]), CLASSES, 0.4, 0.9)
    high = api_main.decide(np.array([0.01, 0.95, 0.01, 0.01, 0.02]), CLASSES, 0.4, 0.9)
    assert low["status"] == "retake"
    assert high["status"] == "result" and high["label"] == "Healthy"


def test_other_class_refers_to_expert():
    r = api_main.decide(np.array([0.05, 0.05, 0.05, 0.05, 0.8]), CLASSES, 0.4, 0.9)
    assert r["status"] == "other"


def test_softmax_sums_to_one():
    p = api_main.softmax([1000.0, 1001.0, 999.0])
    assert abs(p.sum() - 1.0) < 1e-9 and p.argmax() == 1


# ---- preprocessing ----
def test_preprocess_shape_and_type():
    meta = api_main.load_meta()
    x = api_main.preprocess(png_bytes(), meta["size"], meta["mean"], meta["std"])
    assert x.shape == (1, 3, meta["size"], meta["size"]) and x.dtype == np.float32


def test_exif_rotation_is_applied():
    img = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90 degrees to display"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert api_main.load_image(buf.getvalue()).size == (20, 40)


def test_preprocess_matches_training():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.train import get_transforms

    meta = api_main.load_meta()
    rng = np.random.RandomState(0)
    img = Image.fromarray(rng.randint(0, 256, (200, 300, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ours = api_main.preprocess(buf.getvalue(), meta["size"], meta["mean"], meta["std"])
    theirs = get_transforms(meta["size"], False)(img).unsqueeze(0).numpy()
    assert np.abs(ours - theirs).max() < 1e-4


# ---- endpoints ----
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_expected_fields(client):
    classes = api_main.load_meta()["classes"]
    r = post_photo(client)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"result", "other", "retake"}
    assert set(body["probabilities"]) == {c.strip("_") for c in classes}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["prediction_id"] and body["disclaimer"]


def test_predict_rejects_non_image(client):
    r = post_photo(client, data=b"hello", name="x.txt", ctype="text/plain")
    assert r.status_code == 400


def test_predict_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 1000)
    assert post_photo(client, data=b"x" * 2000).status_code == 413


def test_predict_requires_a_file(client):
    assert client.post("/predict").status_code == 422


# ---- feedback ----
def test_feedback_is_stored_and_linked(client):
    pid = post_photo(client).json()["prediction_id"]
    r = client.post("/feedback", json={"prediction_id": pid, "correct": False,
                                       "true_label": "Hispa"})
    assert r.status_code == 201
    assert feedback_rows() == [(pid, 0, "Hispa")]


def test_feedback_unknown_prediction(client):
    r = client.post("/feedback", json={"prediction_id": "nope", "correct": True})
    assert r.status_code == 404


def test_feedback_rejects_unknown_label(client):
    r = client.post("/feedback", json={"prediction_id": "nope", "correct": False,
                                       "true_label": "Nonsense"})
    assert r.status_code == 422