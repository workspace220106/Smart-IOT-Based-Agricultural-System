"""Lazy-loading registry for all 7 trained models."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@lru_cache(maxsize=1)
def yolo_disease():
    from ultralytics import YOLO
    return YOLO(str(MODELS_DIR / "crop_disease_yolov8.pt"))


@lru_cache(maxsize=1)
def yolo_pest():
    from ultralytics import YOLO
    return YOLO(str(MODELS_DIR / "pest_yolov8.pt"))


@lru_cache(maxsize=1)
def crop_health_tflite():
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(MODELS_DIR / "crop_health.tflite"))
    interp.allocate_tensors()
    return interp


@lru_cache(maxsize=1)
def irrigation_rf():
    return joblib.load(MODELS_DIR / "irrigation_rf.joblib")


@lru_cache(maxsize=1)
def fertilizer_xgb():
    return joblib.load(MODELS_DIR / "fertilizer_xgb.joblib")  # dict: model, crop_le, y_le


@lru_cache(maxsize=1)
def water_quality():
    return joblib.load(MODELS_DIR / "water_quality.joblib")


@lru_cache(maxsize=1)
def anomaly_iforest():
    return joblib.load(MODELS_DIR / "anomaly_iforest.joblib")


HEALTH_CLASSES = ["healthy", "stressed", "diseased"]


def tflite_predict(interp, x: np.ndarray) -> np.ndarray:
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    return interp.get_tensor(out["index"])
