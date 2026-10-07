"""Generate 7 Colab-ready notebooks under ml/notebooks/."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

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

# ---------- 1. Crop Disease (YOLOv8n) ----------
n1 = [
 {"md": "# 1. Crop Disease Detection — YOLOv8n\nDataset: PlantVillage (Roboflow YOLO export recommended)."},
 "!pip -q install ultralytics roboflow",
 """# Option A: Roboflow (easiest, pre-annotated)
from roboflow import Roboflow
# rf = Roboflow(api_key='YOUR_KEY')
# project = rf.workspace('workspace').project('plant-disease-detection')
# dataset = project.version(1).download('yolov8')
# DATA_YAML = f'{dataset.location}/data.yaml'

# Option B: Kaggle PlantVillage (classification -> convert to YOLO)
# !kaggle datasets download -d abdallahalidev/plantvillage-dataset --unzip -p /content/pv
DATA_YAML = '/content/dataset/data.yaml'  # set after download""",
 """from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data=DATA_YAML, epochs=50, imgsz=640, batch=16,
            augment=True, mosaic=1.0, fliplr=0.5, degrees=15)""",
 """metrics = model.val()
print(metrics.box.map)  # mAP50-95""",
 """# Export TFLite for edge deployment
model.export(format='tflite', imgsz=640, int8=True)""",
]

# ---------- 2. Pest Detection (YOLOv8n + IP102) ----------
n2 = [
 {"md": "# 2. Pest Detection — YOLOv8n on IP102\nUse a Roboflow annotated subset for YOLO format."},
 "!pip -q install ultralytics roboflow",
 """# IP102 raw: https://github.com/xpwu95/IP102 (classification labels)
# Easier: Roboflow pest datasets with YOLO annotations
DATA_YAML = '/content/pests/data.yaml'""",
 """from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data=DATA_YAML, epochs=100, imgsz=640, batch=16, conf=0.4)""",
 """model.export(format='onnx', imgsz=640)
model.export(format='tflite', imgsz=640, int8=True)""",
]

# ---------- 3. Crop Health (MobileNetV2) ----------
n3 = [
 {"md": "# 3. Crop Health Classification — MobileNetV2\n3 classes: healthy / stressed / diseased."},
 "!pip -q install tensorflow",
 """import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224); BATCH = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/health/train', image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/health/val', image_size=IMG_SIZE, batch_size=BATCH)
class_names = train_ds.class_names; print(class_names)""",
 """aug = tf.keras.Sequential([
    layers.RandomFlip('horizontal'), layers.RandomRotation(0.1),
    layers.RandomZoom(0.1), layers.Rescaling(1./127.5, offset=-1)])

base = MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights='imagenet')
base.trainable = False
inputs = tf.keras.Input(shape=IMG_SIZE+(3,))
x = aug(inputs); x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.2)(x)
out = layers.Dense(3, activation='softmax')(x)
model = models.Model(inputs, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=30)""",
 """# TFLite quantized export
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
open('crop_health.tflite', 'wb').write(conv.convert())""",
]

# ---------- 4. Irrigation (RandomForest) ----------
n4 = [
 {"md": "# 4. Irrigation Prediction — Random Forest\nFeatures: soil_moisture, temp, humidity, rainfall, crop_type."},
 "!pip -q install scikit-learn pandas joblib",
 """import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('/content/irrigation.csv')
df = pd.get_dummies(df, columns=['crop_type'], drop_first=True) if 'crop_type' in df else df
X = df.drop('irrigate', axis=1); y = df['irrigate']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte)))
joblib.dump(clf, 'irrigation_rf.joblib')""",
]

# ---------- 5. Fertilizer (XGBoost) ----------
n5 = [
 {"md": "# 5. NPK Fertilizer Recommendation — XGBoost\nClasses: urea, DAP, MOP, NPK_complex, none_needed."},
 "!pip -q install xgboost scikit-learn pandas joblib",
 """import pandas as pd, joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/content/fertilizer.csv')
le_crop = LabelEncoder(); le_y = LabelEncoder()
df['crop_type'] = le_crop.fit_transform(df['crop_type'])
y = le_y.fit_transform(df['fertilizer'])
X = df.drop('fertilizer', axis=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
clf = XGBClassifier(n_estimators=200, max_depth=6, eval_metric='mlogloss').fit(Xtr, ytr)
print(classification_report(yte, clf.predict(Xte), target_names=le_y.classes_))
joblib.dump({'model': clf, 'crop_le': le_crop, 'y_le': le_y}, 'fertilizer_xgb.joblib')""",
]

# ---------- 6. Water Quality ----------
n6 = [
 {"md": "# 6. Water Quality — LogReg vs SVM vs RF\nFeatures: pH, TDS, turbidity -> good/acceptable/poor."},
 "!pip -q install scikit-learn pandas joblib",
 """import pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('/content/water.csv')
X = df[['ph','tds','turbidity']]; y = df['quality']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
best = (None, 0)
for name, m in [('logreg', LogisticRegression(max_iter=1000)),
                ('svm', SVC(kernel='rbf')),
                ('rf', RandomForestClassifier(n_estimators=100))]:
    m.fit(Xtr, ytr); f1 = f1_score(yte, m.predict(Xte), average='macro')
    print(name, f1)
    if f1 > best[1]: best = (m, f1)
joblib.dump(best[0], 'water_quality.joblib')""",
]

# ---------- 7. Anomaly (Isolation Forest) ----------
n7 = [
 {"md": "# 7. Sensor Anomaly Detection — Isolation Forest\nUnsupervised; flags sensor malfunctions."},
 "!pip -q install scikit-learn pandas joblib",
 """import pandas as pd, joblib
from sklearn.ensemble import IsolationForest

df = pd.read_csv('/content/sensor_logs.csv')
features = ['soil_moisture','temp','humidity','ph','tds','turbidity','N','P','K']
model = IsolationForest(contamination=0.05, random_state=42).fit(df[features])
df['anomaly'] = model.predict(df[features])  # -1 = anomaly
print(df['anomaly'].value_counts())
joblib.dump(model, 'anomaly_iforest.joblib')""",
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
