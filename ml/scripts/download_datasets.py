"""Download all datasets via Kaggle API. Requires ~/.kaggle/kaggle.json credentials."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets"
ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    # Vision (new_plant_diseases is a cleaner PlantVillage superset)
    "ip102":               "rtlmhjbn/ip02-dataset",
    "new_plant_diseases":  "vipoooool/new-plant-diseases-dataset",
    # Tabular / sensor
    "irrigation":          "nelakurthisudheer/dataset-for-predicting-watering-the-plants",
    "crop_recommend":      "atharvaingle/crop-recommendation-dataset",
    "fertilizer":          "gdabhishek/fertilizer-prediction",
    "water_quality":       "adityakadiwal/water-potability",
    # Anomaly / IoT sensor stream
    "smart_farming":       "atharvasoundankar/smart-farming-sensor-data-for-yield-prediction",
}

for name, slug in DATASETS.items():
    out = ROOT / name
    if out.exists():
        print(f"[skip] {name}")
        continue
    out.mkdir()
    print(f"[get ] {slug} -> {out}")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out), "--unzip"], check=False)

print("Done. Datasets under:", ROOT)
