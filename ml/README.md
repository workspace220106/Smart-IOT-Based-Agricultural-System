# IOT LAB — ML Models

Seven models powering the smart-agriculture IoT lab: 3 vision models (leaf images) + 4 tabular models (sensor data).

## Structure

```
ml/
├── notebooks/        # Colab-ready .ipynb for each model
├── datasets/         # Download target (gitignored)
├── models/           # Trained model artifacts (.tflite, .onnx, .joblib)
├── scripts/          # Dataset download + training helpers
└── requirements.txt
```

## Models

| # | Model | Type | Framework | Export |
|---|-------|------|-----------|--------|
| 1 | Crop Disease Detection | YOLOv8n | Ultralytics | TFLite |
| 2 | Pest Detection | YOLOv8n | Ultralytics | TFLite + ONNX |
| 3 | Crop Health Classification | MobileNetV2 | TF/Keras | TFLite |
| 4 | Irrigation Prediction | Random Forest | sklearn | joblib |
| 5 | NPK Fertilizer Recommendation | XGBoost | xgboost | joblib |
| 6 | Water Quality Assessment | LogReg/SVM/RF | sklearn | joblib |
| 7 | Sensor Anomaly Detection | Isolation Forest | sklearn | joblib |

## Datasets — where to get them

### Vision datasets

**1. PlantVillage (Crop Disease)** — 54K leaf images, 38 classes
- Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Roboflow (YOLO format): https://universe.roboflow.com/plant-disease-detection
- HuggingFace: https://huggingface.co/datasets/plant_village
- Direct (raw): https://github.com/spMohanty/PlantVillage-Dataset

**2. IP102 (Pest Detection)** — 75K images, 102 insect pest classes
- Official: https://github.com/xpwu95/IP102
- Kaggle mirror: https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset
- Roboflow (annotated for YOLO): https://universe.roboflow.com/search?q=pest

**3. Crop Health (healthy/stressed/diseased)** — derive from PlantVillage by regrouping classes, or:
- Kaggle "New Plant Diseases Dataset": https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Roboflow "Crop Health": https://universe.roboflow.com/search?q=crop+health

### Tabular (sensor) datasets

**4. Irrigation Prediction**
- Kaggle "Smart Irrigation System": https://www.kaggle.com/datasets/nelakurthisudheer/dataset-for-predicting-watering-the-plants
- Kaggle "Crop Recommendation": https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset (has moisture, temp, humidity, rainfall)

**5. NPK Fertilizer Recommendation**
- Kaggle "Fertilizer Prediction": https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction
- Kaggle "Crop and Fertilizer Dataset": https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra

**6. Water Quality**
- Kaggle "Water Quality": https://www.kaggle.com/datasets/adityakadiwal/water-potability
- Kaggle "Water Quality and Potability": https://www.kaggle.com/datasets/mssmartypants/water-quality

**7. Sensor Anomaly Detection** — unsupervised, no labels needed
- Reuse datasets 4–6, or collect your own ESP32 logs
- Kaggle "IoT Agriculture 2024": https://www.kaggle.com/datasets/wisam1985/agriculture-dataset-2024

## Quick start

```bash
pip install -r requirements.txt
# Download datasets via Kaggle API (needs ~/.kaggle/kaggle.json)
python scripts/download_datasets.py
# Open any notebook in Colab and run top-to-bottom
```

## Colab tip

Each notebook has a first cell that mounts Google Drive and pip-installs its deps. Upload the `notebooks/` folder to Drive, open in Colab, run.
