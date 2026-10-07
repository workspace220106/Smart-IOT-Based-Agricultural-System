"""FastAPI inference service — one endpoint per model.

Run:  uvicorn inference.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from .model_registry import (
    yolo_disease, yolo_pest, crop_health_tflite, tflite_predict,
    irrigation_rf, fertilizer_xgb, water_quality, anomaly_iforest,
    HEALTH_CLASSES,
)
from .schemas.sensors import (
    SensorReading, IrrigationResponse, FertilizerResponse,
    WaterQualityResponse, AnomalyResponse, VisionResponse,
)

app = FastAPI(title="IOT LAB Inference API", version="0.1")


@app.get("/health")
def health():
    return {"status": "ok"}


# -------- Vision --------

def _read_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")


@app.post("/predict/disease", response_model=VisionResponse)
def predict_disease(file: UploadFile = File(...)):
    img = _read_image(file)
    res = yolo_disease().predict(img, verbose=False)[0]
    if len(res.boxes) == 0:
        return VisionResponse(label="unknown", confidence=0.0, boxes=[])
    best = int(res.boxes.conf.argmax())
    return VisionResponse(
        label=res.names[int(res.boxes.cls[best])],
        confidence=float(res.boxes.conf[best]),
        boxes=res.boxes.xyxy.tolist(),
    )


@app.post("/predict/pest", response_model=VisionResponse)
def predict_pest(file: UploadFile = File(...)):
    img = _read_image(file)
    res = yolo_pest().predict(img, verbose=False)[0]
    if len(res.boxes) == 0:
        return VisionResponse(label="none", confidence=0.0, boxes=[])
    best = int(res.boxes.conf.argmax())
    return VisionResponse(
        label=res.names[int(res.boxes.cls[best])],
        confidence=float(res.boxes.conf[best]),
        boxes=res.boxes.xyxy.tolist(),
    )


@app.post("/predict/health", response_model=VisionResponse)
def predict_health(file: UploadFile = File(...)):
    img = _read_image(file).resize((224, 224))
    x = np.expand_dims(np.array(img, dtype=np.float32) / 127.5 - 1.0, 0)
    probs = tflite_predict(crop_health_tflite(), x)[0]
    idx = int(np.argmax(probs))
    return VisionResponse(label=HEALTH_CLASSES[idx], confidence=float(probs[idx]))


# -------- Tabular --------

@app.post("/predict/irrigation", response_model=IrrigationResponse)
def predict_irrigation(r: SensorReading):
    clf = irrigation_rf()
    X = pd.DataFrame([{
        "soil_moisture": r.soil_moisture, "temperature": r.temperature,
        "humidity": r.humidity, "rainfall": r.rainfall,
    }])
    # Align with training columns if present
    if hasattr(clf, "feature_names_in_"):
        X = X.reindex(columns=clf.feature_names_in_, fill_value=0)
    proba = clf.predict_proba(X)[0]
    pred = int(clf.predict(X)[0])
    return IrrigationResponse(irrigate=bool(pred), confidence=float(proba.max()))


@app.post("/predict/fertilizer", response_model=FertilizerResponse)
def predict_fertilizer(r: SensorReading):
    bundle = fertilizer_xgb()
    model, crop_le, y_le = bundle["model"], bundle["crop_le"], bundle["y_le"]
    crop = r.crop_type or "unknown"
    try:
        crop_code = int(crop_le.transform([crop])[0])
    except ValueError:
        crop_code = 0
    X = pd.DataFrame([{
        "N": r.N, "P": r.P, "K": r.K, "ph": r.ph,
        "soil_moisture": r.soil_moisture, "crop_type": crop_code,
    }])
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return FertilizerResponse(recommendation=y_le.classes_[idx], confidence=float(proba[idx]))


@app.post("/predict/water", response_model=WaterQualityResponse)
def predict_water(r: SensorReading):
    clf = water_quality()
    X = pd.DataFrame([{"ph": r.ph, "tds": r.tds, "turbidity": r.turbidity}])
    pred = clf.predict(X)[0]
    return WaterQualityResponse(quality=str(pred))


@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(r: SensorReading):
    model = anomaly_iforest()
    X = pd.DataFrame([{
        "soil_moisture": r.soil_moisture, "temp": r.temperature,
        "humidity": r.humidity, "ph": r.ph, "tds": r.tds,
        "turbidity": r.turbidity, "N": r.N, "P": r.P, "K": r.K,
    }])
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    score = float(model.score_samples(X)[0])
    is_anom = int(model.predict(X)[0]) == -1
    return AnomalyResponse(is_anomaly=is_anom, score=score)
