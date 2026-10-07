# Extras — everything beyond the Colab notebooks

| Dir | Purpose |
|-----|---------|
| `inference/` | FastAPI service exposing all 7 models over HTTP |
| `edge/esp32/` | Arduino sketch — reads sensors, hits API, drives relay |
| `edge/pi_client/` | Raspberry Pi camera client for leaf scanning |
| `dashboard/` | Streamlit live dashboard |
| `utils/` | Roboflow dataset fetcher + synthetic data generator |
| `tests/` | Smoke tests against the running API |

## Run the inference API locally

```bash
pip install -r requirements.txt fastapi "uvicorn[standard]" python-multipart
uvicorn inference.app:app --reload --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for Swagger UI.

## Run the dashboard

```bash
pip install streamlit
streamlit run dashboard/app.py
```

## Train before you have real data

```bash
python utils/synth_sensor_data.py       # writes datasets/synthetic/*.csv
# Then point notebooks 4-7 at /content/synthetic/*.csv in Colab.
```

## Roboflow datasets

```bash
python utils/roboflow_fetch.py disease   # writes datasets/disease/
python utils/roboflow_fetch.py pest      # writes datasets/pest/
```

Edit the `TARGETS` dict in `utils/roboflow_fetch.py` with the exact workspace / project / version slugs you want.

## Docker

```bash
docker build -f inference/Dockerfile -t iot-lab-infer .
docker run -p 8000:8000 iot-lab-infer
```

## Deploy flow

1. Train on Colab → download artifacts to `ml/models/`
2. `uvicorn inference.app:app` on a Pi / server
3. Flash `edge/esp32/sensor_client.ino` to ESP32, set WiFi + API URL
4. `python edge/pi_client/leaf_scanner.py --api http://pi.local:8000 --mode disease --interval 30`
5. `streamlit run dashboard/app.py` for ops view
