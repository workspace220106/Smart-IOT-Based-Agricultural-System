"""Raspberry Pi camera -> leaf disease/pest/health prediction via API.

Usage:
  python leaf_scanner.py --api http://192.168.1.10:8000 --mode disease
  python leaf_scanner.py --image path/to/leaf.jpg --mode health
"""
import argparse
import time
from pathlib import Path
import requests


def capture_frame(out_path: str = "/tmp/frame.jpg") -> str:
    try:
        from picamera2 import Picamera2  # Pi camera module
        cam = Picamera2()
        cam.configure(cam.create_still_configuration())
        cam.start(); time.sleep(1); cam.capture_file(out_path); cam.stop()
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read(); cap.release()
        if not ok: raise RuntimeError("camera read failed")
        cv2.imwrite(out_path, frame)
    return out_path


def predict(api: str, mode: str, image_path: str) -> dict:
    with open(image_path, "rb") as f:
        r = requests.post(f"{api}/predict/{mode}", files={"file": f}, timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["disease", "pest", "health"], default="disease")
    ap.add_argument("--image", help="skip camera; use this image")
    ap.add_argument("--interval", type=float, default=0.0, help="loop every N sec (0=once)")
    args = ap.parse_args()

    while True:
        img = args.image or capture_frame()
        try:
            print(predict(args.api, args.mode, img))
        except Exception as e:
            print("error:", e)
        if args.interval <= 0: break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
