"""Streamlit dashboard: live sensor readings + model predictions.

Run:  streamlit run dashboard/app.py
"""
import os
import requests
import streamlit as st
from datetime import datetime

API = os.environ.get("INFERENCE_API", "http://localhost:8000")

st.set_page_config(page_title="IOT Lab", layout="wide")
st.title("🌱 IOT Lab — Smart Agriculture Dashboard")

tab1, tab2, tab3 = st.tabs(["Sensor predictions", "Leaf scan", "Health check"])

with tab1:
    st.subheader("Sensor inputs")
    c1, c2, c3, c4 = st.columns(4)
    soil = c1.slider("Soil moisture (%)", 0.0, 100.0, 35.0)
    temp = c2.slider("Temperature (°C)", -10.0, 60.0, 28.0)
    hum  = c3.slider("Humidity (%)", 0.0, 100.0, 65.0)
    ph   = c4.slider("pH", 0.0, 14.0, 6.8)
    c5, c6, c7, c8 = st.columns(4)
    tds  = c5.slider("TDS (ppm)", 0.0, 2000.0, 400.0)
    turb = c6.slider("Turbidity (NTU)", 0.0, 100.0, 2.0)
    rain = c7.slider("Rainfall (mm)", 0.0, 100.0, 0.0)
    crop = c8.selectbox("Crop", ["tomato", "rice", "wheat", "cotton", "maize"])
    c9, c10, c11 = st.columns(3)
    N = c9.slider("N", 0.0, 200.0, 40.0)
    P = c10.slider("P", 0.0, 200.0, 20.0)
    K = c11.slider("K", 0.0, 200.0, 30.0)

    body = dict(soil_moisture=soil, temperature=temp, humidity=hum, ph=ph,
                tds=tds, turbidity=turb, rainfall=rain, crop_type=crop,
                N=N, P=P, K=K)

    if st.button("Run predictions"):
        for name, path in [("Irrigation", "irrigation"), ("Fertilizer", "fertilizer"),
                           ("Water quality", "water"), ("Anomaly", "anomaly")]:
            try:
                r = requests.post(f"{API}/predict/{path}", json=body, timeout=10).json()
                st.success(f"**{name}** — {r}")
            except Exception as e:
                st.error(f"{name}: {e}")

with tab2:
    img = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    mode = st.radio("Model", ["disease", "pest"], horizontal=True)
    if img and st.button("Scan"):
        r = requests.post(f"{API}/predict/{mode}", files={"file": img.getvalue()})
        st.image(img)
        st.json(r.json())

with tab3:
    try:
        st.json(requests.get(f"{API}/health", timeout=3).json())
    except Exception as e:
        st.error(f"API unreachable: {e}")
    st.caption(f"Last checked: {datetime.now():%H:%M:%S}  ·  API: {API}")
