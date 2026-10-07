"""Generate synthetic sensor CSVs so notebooks 4-7 can train before real data lands.

Writes:
  datasets/synthetic/irrigation.csv
  datasets/synthetic/fertilizer.csv
  datasets/synthetic/water.csv
  datasets/synthetic/sensor_logs.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "datasets" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
N = 5000

# ---- irrigation ----
soil = rng.uniform(5, 80, N)
temp = rng.uniform(15, 42, N)
hum  = rng.uniform(20, 95, N)
rain = rng.exponential(2, N).clip(0, 50)
crop = rng.choice(["tomato", "rice", "wheat", "cotton", "maize"], N)
irrigate = ((soil < 35) & (rain < 5) & (temp > 25)).astype(int)
pd.DataFrame({"soil_moisture": soil, "temperature": temp, "humidity": hum,
              "rainfall": rain, "crop_type": crop, "irrigate": irrigate}
             ).to_csv(OUT / "irrigation.csv", index=False)

# ---- fertilizer ----
N_v = rng.uniform(0, 150, N); P_v = rng.uniform(0, 100, N); K_v = rng.uniform(0, 100, N)
ph = rng.uniform(4.5, 8.5, N); sm = rng.uniform(5, 80, N)
fert = np.where(N_v < 30, "urea",
         np.where(P_v < 20, "DAP",
         np.where(K_v < 20, "MOP",
         np.where((N_v + P_v + K_v) < 90, "NPK_complex", "none_needed"))))
pd.DataFrame({"N": N_v, "P": P_v, "K": K_v, "ph": ph, "soil_moisture": sm,
              "crop_type": rng.choice(["tomato","rice","wheat","cotton","maize"], N),
              "fertilizer": fert}
             ).to_csv(OUT / "fertilizer.csv", index=False)

# ---- water ----
ph_w = rng.uniform(4, 10, N); tds = rng.uniform(50, 1500, N); turb = rng.uniform(0, 20, N)
quality = np.where((ph_w.between(6.5, 8.5) if hasattr(ph_w,'between') else (ph_w>=6.5)&(ph_w<=8.5))
                   & (tds < 500) & (turb < 5), "good",
            np.where((tds < 1000) & (turb < 10), "acceptable", "poor"))
pd.DataFrame({"ph": ph_w, "tds": tds, "turbidity": turb, "quality": quality}
             ).to_csv(OUT / "water.csv", index=False)

# ---- sensor logs (for anomaly) ----
base = pd.DataFrame({
    "soil_moisture": rng.normal(45, 8, N).clip(0, 100),
    "temp": rng.normal(28, 4, N),
    "humidity": rng.normal(65, 10, N).clip(0, 100),
    "ph": rng.normal(6.8, 0.3, N),
    "tds": rng.normal(400, 60, N),
    "turbidity": rng.normal(2, 0.5, N).clip(0, None),
    "N": rng.normal(40, 10, N), "P": rng.normal(25, 7, N), "K": rng.normal(30, 8, N),
})
# inject 5% anomalies
mask = rng.random(N) < 0.05
base.loc[mask, "soil_moisture"] = rng.uniform(-20, 150, mask.sum())
base.loc[mask, "ph"] = rng.uniform(0, 14, mask.sum())
base.to_csv(OUT / "sensor_logs.csv", index=False)

print("synthetic data ->", OUT)
