"""Generate 7 Colab-ready notebooks under ml/notebooks/, wired to the real downloaded datasets."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

GITHUB_RAW = "https://raw.githubusercontent.com/workspace220106/Smart-IOT-Based-Agricultural-System/main/ml/datasets"

def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "accelerator": "GPU", "colab": {"provenance": []}},
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": c.splitlines(keepends=True)} if isinstance(c, str)
                  else {"cell_type": "markdown", "metadata": {}, "source": c["md"].splitlines(keepends=True)}
                  for c in cells],
    }

# ---------- 1. Crop Disease (YOLOv8n) on new_plant_diseases ----------
n1 = [
 {"md": "# 1. Crop Disease Detection — YOLOv8n (classification mode)\nDataset: Kaggle `vipoooool/new-plant-diseases-dataset` (87K imgs, 38 classes)."},
 "!pip -q install ultralytics kaggle",
 """import os; os.environ['KAGGLE_API_TOKEN'] = 'YOUR_KAGGLE_TOKEN'  # or upload kaggle.json
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p /content --unzip
!ls /content | head""",
 """# The archive contains 'New Plant Diseases Dataset(Augmented)' with train/valid subfolders
import glob
root = glob.glob('/content/New Plant Diseases Dataset*')[0] + '/New Plant Diseases Dataset(Augmented)'
print(root)""",
 """from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')  # classification variant since labels are per-image
model.train(data=root, epochs=20, imgsz=224, batch=32)""",
 """model.val()
model.export(format='tflite', imgsz=224, int8=True)""",
]

# ---------- 2. Pest Detection (YOLOv8n) on roboflow-100/insects-mytwu ----------
n2 = [
 {"md": "# 2. Pest Detection — YOLOv8n\nDataset: Roboflow-100 `roboflow-100/insects-mytwu` v2 (10 crop-pest classes, YOLO-ready)."},
 "!pip -q install ultralytics roboflow",
 """import os; os.environ['ROBOFLOW_API_KEY'] = 'YOUR_ROBOFLOW_KEY'
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
ds = rf.workspace('roboflow-100').project('insects-mytwu').version(2).download('yolov8')
DATA_YAML = f'{ds.location}/data.yaml'""",
 """from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data=DATA_YAML, epochs=100, imgsz=640, batch=16)""",
 """model.val()
model.export(format='onnx', imgsz=640)
model.export(format='tflite', imgsz=640, int8=True)""",
]

# ---------- 3. Crop Health (MobileNetV2) — regrouped new_plant_diseases ----------
n3 = [
 {"md": "# 3. Crop Health — MobileNetV2\nRegroup PlantVillage classes into **healthy / stressed / diseased**."},
 "!pip -q install tensorflow kaggle",
 """import os; os.environ['KAGGLE_API_TOKEN'] = 'YOUR_KAGGLE_TOKEN'
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p /content --unzip -q
import glob, shutil, os
src = glob.glob('/content/New Plant Diseases Dataset*')[0] + '/New Plant Diseases Dataset(Augmented)/train'
dst = '/content/health'
STRESSED = {'nutrient', 'deficiency', 'stressed'}  # none in PlantVillage -> all non-healthy = diseased
for cls in os.listdir(src):
    target = 'healthy' if 'healthy' in cls.lower() else 'diseased'
    os.makedirs(f'{dst}/{target}', exist_ok=True)
    for f in os.listdir(f'{src}/{cls}')[:800]:  # cap per class
        shutil.copy(f'{src}/{cls}/{f}', f'{dst}/{target}/{cls}_{f}')
print({c: len(os.listdir(f'{dst}/{c}')) for c in os.listdir(dst)})""",
 """import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

IMG=(224,224); BATCH=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/health', image_size=IMG, batch_size=BATCH, validation_split=0.2,
    subset='training', seed=42)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/health', image_size=IMG, batch_size=BATCH, validation_split=0.2,
    subset='validation', seed=42)
classes = train_ds.class_names; print(classes)

aug = tf.keras.Sequential([layers.RandomFlip('horizontal'), layers.RandomRotation(0.1),
                           layers.Rescaling(1./127.5, offset=-1)])
base = MobileNetV2(input_shape=IMG+(3,), include_top=False, weights='imagenet'); base.trainable=False
inp = tf.keras.Input(IMG+(3,))
x = aug(inp); x = base(x, training=False); x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x); out = layers.Dense(len(classes), activation='softmax')(x)
m = models.Model(inp, out)
m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
m.fit(train_ds, validation_data=val_ds, epochs=10)""",
 """conv = tf.lite.TFLiteConverter.from_keras_model(m)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
open('crop_health.tflite', 'wb').write(conv.convert())""",
]

# ---------- 4. Irrigation (RandomForest) on TARP.csv ----------
n4 = [
 {"md": "# 4. Irrigation Prediction — Random Forest\nDataset: `datasets/irrigation/TARP.csv` (Kaggle `nelakurthisudheer/dataset-for-predicting-watering-the-plants`)."},
 "!pip -q install scikit-learn pandas joblib",
 f"!wget -q {GITHUB_RAW}/irrigation/TARP.csv -O /content/irrigation.csv",
 """import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('/content/irrigation.csv')
print(df.columns.tolist()); print(df.head(3))

# TARP cols: Soil Moisture, Temperature, Air temperature (C), Wind speed (Km/h),
#            Air humidity (%), Wind gust (Km/h), Pressure (KPa), ph, rainfall, N, P, K, Status
y = df['Status']
X = df.drop(columns=['Status'])
X = X.select_dtypes(include='number').fillna(0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte)))
joblib.dump(clf, 'irrigation_rf.joblib')""",
]

# ---------- 5. Fertilizer (XGBoost) on Fertilizer Prediction.csv ----------
n5 = [
 {"md": "# 5. NPK Fertilizer Recommendation — XGBoost\nDataset: `datasets/fertilizer/Fertilizer Prediction.csv`."},
 "!pip -q install xgboost scikit-learn pandas joblib",
 f"!wget -q '{GITHUB_RAW}/fertilizer/Fertilizer%20Prediction.csv' -O /content/fertilizer.csv",
 """import pandas as pd, joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv('/content/fertilizer.csv')
print(df.columns.tolist()); print(df.head(3))
# Cols: Temparature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous, Fertilizer Name
le_soil = LabelEncoder(); le_crop = LabelEncoder(); le_y = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
y = le_y.fit_transform(df['Fertilizer Name'])
X = df.drop(columns=['Fertilizer Name'])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = XGBClassifier(n_estimators=300, max_depth=6, eval_metric='mlogloss').fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte), target_names=le_y.classes_))
joblib.dump({'model': clf, 'soil_le': le_soil, 'crop_le': le_crop, 'y_le': le_y},
            'fertilizer_xgb.joblib')""",
]

# ---------- 6. Water Quality on water_potability.csv ----------
n6 = [
 {"md": "# 6. Water Quality — binary potability + 3-class derived\nDataset: `datasets/water_quality/water_potability.csv`."},
 "!pip -q install scikit-learn pandas joblib",
 f"!wget -q {GITHUB_RAW}/water_quality/water_potability.csv -O /content/water.csv",
 """import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv('/content/water.csv').fillna(0)
print(df.columns.tolist()); print(df.head(3))
# Cols: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon,
#       Trihalomethanes, Turbidity, Potability
X = df.drop(columns=['Potability']); y = df['Potability']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

best, score = None, 0
for name, m in [('logreg', LogisticRegression(max_iter=1000)),
                ('svm',    SVC(kernel='rbf', probability=True)),
                ('rf',     RandomForestClassifier(n_estimators=200, n_jobs=-1))]:
    m.fit(Xtr, ytr); f1 = f1_score(yte, m.predict(Xte), average='macro')
    print(name, f1)
    if f1 > score: best, score = m, f1
print(classification_report(yte, best.predict(Xte)))
joblib.dump(best, 'water_quality.joblib')""",
]

# ---------- 7. Anomaly (Isolation Forest) on smart_farming ----------
n7 = [
 {"md": "# 7. Sensor Anomaly Detection — Isolation Forest\nDataset: `datasets/smart_farming/Smart_Farming_Crop_Yield_2024.csv`."},
 "!pip -q install scikit-learn pandas joblib",
 f"!wget -q {GITHUB_RAW}/smart_farming/Smart_Farming_Crop_Yield_2024.csv -O /content/sf.csv",
 """import pandas as pd, joblib
from sklearn.ensemble import IsolationForest

df = pd.read_csv('/content/sf.csv')
print(df.columns.tolist())
features = ['soil_moisture_%','soil_pH','temperature_C','rainfall_mm',
            'humidity_%','sunlight_hours','pesticide_usage_ml','NDVI_index']
X = df[features].fillna(df[features].median())
model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1).fit(X)
df['anomaly'] = model.predict(X)  # -1 anomaly, 1 normal
print(df['anomaly'].value_counts())
# Correlate anomalies with disease status
print(pd.crosstab(df['anomaly'], df['crop_disease_status']))
joblib.dump({'model': model, 'features': features}, 'anomaly_iforest.joblib')""",
]

NOTEBOOKS = {
    "01_crop_disease_yolov8.ipynb": n1,
    "02_pest_detection_yolov8.ipynb": n2,
    "03_crop_health_mobilenet.ipynb": n3,
    "04_irrigation_rf.ipynb": n4,
    "05_fertilizer_xgboost.ipynb": n5,
    "06_water_quality.ipynb": n6,
    "07_anomaly_iforest.ipynb": n7,
}

for name, cells in NOTEBOOKS.items():
    (OUT / name).write_text(json.dumps(nb(cells), indent=1))
    print("wrote", name)
