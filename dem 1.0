import os, io, json, time, zipfile, urllib.request
from typing import Dict, List


import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf


# ML
from sklearn.base import BaseEstimator
try:
  import joblib
except Exception:
  joblib = None


# ASR (Vosk CPU)
try:
from vosk import Model, KaldiRecognizer
except Exception:
Model = None
KaldiRecognizer = None


st.set_page_config(page_title="NC/MCI — Upload & Predict", page_icon="🧠", layout="wide")


# ---------------- UI THEME (small medical background) ---------------- #
st.markdown(
"""
<style>
.stApp {
background:
radial-gradient(900px 500px at 10% 5%, rgba(99,102,241,0.10), transparent),
radial-gradient(900px 500px at 90% 95%, rgba(16,185,129,0.10), transparent);
}
header[data-testid="stHeader"] { background: transparent; }
.watermark { position: fixed; right: -40px; bottom: -40px; width: 300px; height: 300px; opacity: 0.10; pointer-events: none; }
</style>
<svg class="watermark" viewBox="0 0 200 200">
<defs>
<linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
<stop offset="0%" stop-color="#6366F1"/>
<stop offset="100%" stop-color="#10B981"/>
</linearGradient>
</defs>
<!-- stylized brain + waveform -->
<path d="M60,110 C40,100 40,70 60,60 C60,40 90,40 95,55 C110,40 140,55 135,80 C155,90 150,120 130,125 C120,150 85,150 75,130 C60,140 45,130 50,115" fill="none" stroke="url(#g)" stroke-width="6" stroke-linecap="round"/>
<polyline points="20,170 50,150 70,160 90,120 110,160 130,135 150,155 180,140" fill="none" stroke="url(#g)" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>
</svg>
""",
unsafe_allow_html=True,
)


st.title("🧠 NC vs MCI — Upload Audio → Transcript → Prediction")
st.caption("Research demo • Not diagnostic • Audio processed in-session only")


# ---------------- Download/Load Vosk ---------------- #
VOSK_ZIP_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_DIRNAME = "vosk-model-small-en-us-0.15"


@st.cache_resource(show_spinner=True)
def ensure_vosk_model() -> str:
if Model is None:
st.error("Vosk not installed. Add 'vosk' to requirements.txt.")
st.stop()
if os.path.isdir(VOSK_DIRNAME):
return VOSK_DIRNAME
zip_path = "vosk_model.zip"
st.info("Downloading Vosk model (~50MB) — first run only…")
urllib.request.urlretrieve(VOSK_ZIP_URL, zip_path)
with zipfile.ZipFile(zip_path, 'r') as zf:
zf.extractall(".")
os.remove(zip_path)
return VOSK_DIRNAME


# ---------------- Utility: audio + features ---------------- #
DISFLUENCIES = {"um","uh","er","ah","hmm","mmm"}
FEATURE_LIST = ["wpm","mean_pause","pause_rate","ttr","disfluency_rate"]


class DummyModel(BaseEstimator):
def predict_proba(self, X):
wpm = X[:,0]
p_mci = 0.35 + 0.15*np.tanh((120 - wpm)/60.0)
p_mci = np.clip(p_mci, 0.05, 0.95)
return np.c_[1-p_mci, p_mci]


@st.cache_resource(show_spinner=False)
st.caption("© Your Org")
