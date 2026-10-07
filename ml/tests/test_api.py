"""Smoke tests for the inference API. Requires the API to be running on localhost:8000."""
import io
import pytest
import requests
from PIL import Image

API = "http://localhost:8000"
SENSOR = dict(soil_moisture=30, temperature=29, humidity=70, ph=6.5,
              tds=350, turbidity=2.0, rainfall=0, crop_type="tomato",
              N=40, P=20, K=30)


def _api_up():
    try: return requests.get(f"{API}/health", timeout=2).ok
    except Exception: return False


pytestmark = pytest.mark.skipif(not _api_up(), reason="API not running")


def test_irrigation():
    r = requests.post(f"{API}/predict/irrigation", json=SENSOR).json()
    assert "irrigate" in r and "confidence" in r


def test_fertilizer():
    r = requests.post(f"{API}/predict/fertilizer", json=SENSOR).json()
    assert "recommendation" in r


def test_water():
    r = requests.post(f"{API}/predict/water", json=SENSOR).json()
    assert r["quality"] in {"good", "acceptable", "poor"}


def test_anomaly():
    r = requests.post(f"{API}/predict/anomaly", json=SENSOR).json()
    assert "is_anomaly" in r


def test_disease():
    img = Image.new("RGB", (640, 640), (34, 139, 34))
    buf = io.BytesIO(); img.save(buf, "JPEG"); buf.seek(0)
    r = requests.post(f"{API}/predict/disease",
                      files={"file": ("leaf.jpg", buf, "image/jpeg")}).json()
    assert "label" in r
