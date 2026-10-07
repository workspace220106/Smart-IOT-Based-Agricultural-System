"""Download YOLO-formatted datasets from Roboflow Universe for notebooks 01 & 02.

Usage:
  python utils/roboflow_fetch.py disease
  python utils/roboflow_fetch.py pest
"""
import os
import sys
from pathlib import Path
from roboflow import Roboflow

OUT = Path(__file__).resolve().parent.parent / "datasets"

# Edit these to the exact Roboflow project versions you prefer.
TARGETS = {
    "disease": {"workspace": "plant-disease-detection", "project": "plant-village",
                "version": 1, "format": "yolov8"},
    "pest":    {"workspace": "pest-detection",          "project": "ip102",
                "version": 1, "format": "yolov8"},
}


def main(key: str):
    cfg = TARGETS[key]
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("set ROBOFLOW_API_KEY first")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(cfg["workspace"]).project(cfg["project"])
    dataset = proj.version(cfg["version"]).download(cfg["format"], location=str(OUT / key))
    print("downloaded to:", dataset.location)
    print("data.yaml:", Path(dataset.location) / "data.yaml")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        sys.exit("usage: roboflow_fetch.py {disease|pest}")
    main(sys.argv[1])
