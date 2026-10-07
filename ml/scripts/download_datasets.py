"""Download all datasets via Kaggle API. Requires ~/.kaggle/kaggle.json credentials."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "datasets"
ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "plantvillage":        "abdallahalidev/plantvillage-dataset",
    "ip102":               "rtlmhjbn/ip02-dataset",
    "new_plant_diseases":  "vipoooool/new-plant-diseases-dataset",
    "irrigation":          "nelakurthisudheer/dataset-for-predicting-watering-the-plants",
    "crop_recommend":      "atharvaingle/crop-recommendation-dataset",
    "fertilizer":          "gdabhishek/fertilizer-prediction",
    "water_quality":       "adityakadiwal/water-potability",
    "iot_agri":            "wisam1985/agriculture-dataset-2024",
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
