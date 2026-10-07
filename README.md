# Smart IoT-Based Agricultural System

An end-to-end Smart IoT and Machine Learning system for intelligent agriculture, crop health monitoring, automated irrigation, pest & disease detection, fertilizer recommendation, and real-time dashboard analytics.

---

## 🌾 Project Overview

This repository contains edge firmware, FastAPI inference services, Streamlit monitoring dashboards, machine learning model training notebooks, and utility pipelines for automated precision farming.

### 🌟 Key Features
- **3 Vision Models**: Crop Disease Detection (YOLOv8n), Pest Detection (YOLOv8n), Crop Health Classification (MobileNetV2).
- **4 Tabular Models**: Irrigation Prediction (Random Forest), NPK Fertilizer Recommendation (XGBoost), Water Quality Assessment (LogReg/SVM/RF), and Sensor Anomaly Detection (Isolation Forest).
- **Edge Clients**: ESP32 C++ sensor firmware (`sensor_client.ino`) and Raspberry Pi leaf scanner client (`leaf_scanner.py`).
- **REST Inference Server**: FastAPI service with Pydantic validation and model registries.
- **Interactive Dashboard**: Streamlit-based live monitoring and decision-support control center.

---

## 📁 Repository Structure

```
.
├── .gitignore
├── README.md
└── ml/
    ├── dashboard/            # Streamlit interactive live dashboard
    │   └── app.py
    ├── datasets/             # Dataset staging directory (gitignored)
    ├── edge/                 # Edge microcontroller and SBC clients
    │   ├── esp32/            # ESP32 sensor firmware
    │   │   └── sensor_client.ino
    │   └── pi_client/        # Raspberry Pi leaf scanner
    │       └── leaf_scanner.py
    ├── inference/            # FastAPI inference service & schemas
    │   ├── app.py
    │   ├── Dockerfile
    │   ├── model_registry.py
    │   └── schemas/
    │       └── sensors.py
    ├── models/               # Model artifacts directory (.tflite, .onnx, .joblib)
    ├── notebooks/            # Colab & Jupyter training notebooks
    │   ├── 01_crop_disease_yolov8.ipynb
    │   ├── 02_pest_detection_yolov8.ipynb
    │   ├── 03_crop_health_mobilenet.ipynb
    │   ├── 04_irrigation_rf.ipynb
    │   ├── 05_fertilizer_xgboost.ipynb
    │   ├── 06_water_quality.ipynb
    │   └── 07_anomaly_iforest.ipynb
    ├── scripts/              # Dataset download & build utilities
    │   ├── build_notebooks.py
    │   └── download_datasets.py
    ├── tests/                # API and unit test suite
    │   └── test_api.py
    ├── utils/                # Synthetic data generator & Roboflow fetchers
    │   ├── roboflow_fetch.py
    │   └── synth_sensor_data.py
    ├── requirements.txt      # Python dependencies
    ├── README.md             # ML sub-module documentation
    └── EXTRAS.md             # Additional architecture and deployment guides
```

---

## 🚀 Quick Start

### 1. Installation
```bash
cd ml
pip install -r requirements.txt
```

### 2. Download Datasets
Ensure your Kaggle API key is located in `~/.kaggle/kaggle.json`:
```bash
python scripts/download_datasets.py
```

### 3. Run FastAPI Inference Service
```bash
uvicorn inference.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Models Summary

| # | Model | Type | Framework | Export Target |
|---|---|---|---|---|
| 1 | Crop Disease Detection | YOLOv8n | Ultralytics | TFLite |
| 2 | Pest Detection | YOLOv8n | Ultralytics | TFLite + ONNX |
| 3 | Crop Health Classification | MobileNetV2 | TF/Keras | TFLite |
| 4 | Irrigation Prediction | Random Forest | scikit-learn | joblib |
| 5 | NPK Fertilizer Recommendation | XGBoost | XGBoost | joblib |
| 6 | Water Quality Assessment | LogReg / SVM / RF | scikit-learn | joblib |
| 7 | Sensor Anomaly Detection | Isolation Forest | scikit-learn | joblib |

---

## 📄 License
This project is licensed under the MIT License.
