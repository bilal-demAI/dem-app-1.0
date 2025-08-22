import os, io, json, time, zipfile, urllib.request
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf

# ML
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

# ---- small themed background ----
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
      <path d="M60,110 C40,100 40,70 60,60 C60,40 90,40 95,55 C110,40 140,55 135,80 C155,90 150,120 130,125 C120,150 85,150 75,130 C60,140 45,130 50,115" fill="none" stroke="url(#g)" stroke-width="6" stroke-linecap="round"/>
      <polyline points="20,170 50,150 70,160 90,120 110,160 130,135 150,155 180,140" fill="none" stroke="url(#g)" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 NC vs MCI — Upload Audio → Transcript → Prediction")
st.caption("Research demo • Not diagnostic • Audio processed in-session only")

# ---- Vosk model ----
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
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")
    os.remove(zip_path)
    return VOSK_DIRNAME

# ---- features ----
DISFLUENCIES = {"um", "uh", "er", "ah", "hmm", "mmm"}
FEATURE_LIST = ["wpm", "mean_pause", "pause_rate", "ttr", "disfluency_rate"]

class DummyModel(BaseEstimator):
    def predict_proba(self, X):
        wpm = X[:, 0]
        p_mci = 0.35 + 0.15 * np.tanh((120 - wpm) / 60.0)
        p_mci = np.clip(p_mci, 0.05, 0.95)
        return np.c_[1 - p_mci, p_mci]

@st.cache_resource(show_spinner=False)
def load_model_from_bytes(b: bytes) -> BaseEstimator:
    if not joblib:
        st.warning("joblib unavailable; using Dummy model")
        return DummyModel()
    try:
        return joblib.load(io.BytesIO(b))
    except Exception as e:
        st.warning(f"Could not load uploaded model: {e}. Using Dummy model.")
        return DummyModel()

def downsample_to_16k(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr == 16000:
        return data.astype(np.float32)
    ratio = sr / 16000.0
    idx = (np.arange(int(len(data) / ratio)) * ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(data) - 1)
    return data[idx].astype(np.float32)

def compute_features(words: List[Dict]) -> Dict[str, float]:
    if not words:
        return {k: 0.0 for k in FEATURE_LIST}
    toks = [w["word"].lower() for w in words]
    start = float(words[0].get("start", 0.0))
    end = float(words[-1].get("end", start))
    dur = max(1e-6, end - start)
    wpm = len(words) / (dur / 60.0)
    gaps, long_pauses = [], 0
    for i in range(len(words) - 1):
        g = float(words[i + 1].get("start", 0)) - float(words[i].get("end", 0))
        if g > 0:
            gaps.append(g)
            if g >= 0.5:
                long_pauses += 1
    mean_pause = float(np.mean(gaps)) if gaps else 0.0
    pause_rate = long_pauses / (dur / 60.0)
    ttr = len(set(toks)) / max(1, len(toks))
    disfluency_rate = sum(t in DISFLUENCIES for t in toks) / max(1, len(toks))
    return {
        "wpm": float(wpm),
        "mean_pause": mean_pause,
        "pause_rate": pause_rate,
        "ttr": ttr,
        "disfluency_rate": float(disfluency_rate),
    }

def transcribe_incremental(audio_bytes: bytes):
    model_path = ensure_vosk_model()
    rec = KaldiRecognizer(Model(model_path), 16000)
    rec.SetWords(True)

    data, sr = sf.read(io.BytesIO(audio_bytes))
    data = data.astype(np.float32)
    samples16k = downsample_to_16k(data, sr)
    int16 = (np.clip(samples16k, -1, 1) * 32767).astype(np.int16)

    chunk = 16000 * 2  # 2s chunks
    text_full = ""
    words: List[Dict] = []
    placeholder = st.empty()
    prog = st.progress(0.0, text="Transcribing…")

    N = len(int16)
    for i in range(0, N, chunk):
        piece = int16[i : i + chunk].tobytes()
        if rec.AcceptWaveform(piece):
            j = json.loads(rec.Result())
            if j.get("text"):
                text_full += (" " + j["text"])
            for w in j.get("result", []):
                words.append({"word": w.get("word"), "start": float(w.get("start", 0)), "end": float(w.get("end", 0))})
        else:
            pj = json.loads(rec.PartialResult())
            partial = pj.get("partial", "")
            placeholder.text_area("Live Transcript", value=(text_full + (" " + partial if partial else "")).strip(), height=240)
        prog.progress(min(0.999, (i + chunk) / N))
        time.sleep(0.01)

    j = json.loads(rec.FinalResult())
    if j.get("text"):
        text_full += (" " + j["text"])
    for w in j.get("result", []):
        words.append({"word": w.get("word"), "start": float(w.get("start", 0)), "end": float(w.get("end", 0))})

    placeholder.text_area("Live Transcript", value=text_full.strip(), height=240)
    prog.progress(1.0, text="Done")
    return text_full.strip(), words

# Pretty blue→teal palette that matches your background vibes
COOL_PALETTE = ["#6366F1", "#60A5FA", "#22D3EE", "#06B6D4", "#10B981"]
cool_cmap = LinearSegmentedColormap.from_list("cool_brand", COOL_PALETTE)

# Feature ranges for normalization (adjust to your dataset if needed)
FEATURE_RANGES = {
    "wpm": (60, 180),             # words per minute
    "mean_pause": (0.0, 1.0),     # seconds
    "pause_rate": (0.0, 20.0),    # pauses/min (>0.5s)
    "ttr": (0.2, 0.9),            # type–token ratio
    "disfluency_rate": (0.0, 0.2) # fraction of tokens
}

def _normalize(name, value):
    lo, hi = FEATURE_RANGES[name]
    return max(0.0, min(1.0, (value - lo) / (hi - lo + 1e-9)))

def feature_bar_chart(feats: dict):
    labels = FEATURE_LIST
    values = [feats[k] for k in labels]
    # color per bar via colormap on normalized value
    colors = [cool_cmap(_normalize(k, feats[k])) for k in labels]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.barh(labels, values, edgecolor="none", color=colors)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.set_xlabel("Value")
    # write value on bar
    for b, v in zip(bars, values):
        ax.text(b.get_width() + (max(values)*0.02 if max(values)>0 else 0.02), b.get_y() + b.get_height()/2,
                f"{v:.3g}", va="center", fontsize=9)
    fig.tight_layout()
    return fig

def feature_radar_chart(feats: dict):
    import numpy as np
    labels = FEATURE_LIST
    values = [ _normalize(k, feats[k]) for k in labels ]
    values += values[:1]  # close the loop

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(4.8, 4.8))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    # grid & ticks
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([".25", ".5", ".75", "1"], fontsize=8)
    ax.grid(alpha=0.2)

    # fill + outline (cool colors)
    ax.plot(angles, values, color=COOL_PALETTE[0], linewidth=2)
    ax.fill(angles, values, color=COOL_PALETTE[-1], alpha=0.25)
    fig.tight_layout()
    return fig


# ---- sidebar: model ----
st.sidebar.header("Model")
choice = st.sidebar.radio("Select model source", ["Dummy model (built-in)", "Upload .joblib"], index=0)
model = DummyModel()
if choice == "Upload .joblib":
    model_file = st.sidebar.file_uploader("Upload scikit-learn model (.joblib)", type=["joblib", "pkl"])
    if model_file is not None:
        model = load_model_from_bytes(model_file.read())
    else:
        st.sidebar.info("Using Dummy model until a file is uploaded.")
threshold = st.sidebar.slider("Decision threshold (MCI)", 0.05, 0.95, 0.50, 0.05)
st.sidebar.caption("Prediction label switches to MCI if P(MCI) ≥ threshold")


# ---- main: upload → play → transcribe → predict ----
left, right = st.columns([2, 1])

with left:
    st.subheader("1) Upload audio file (WAV/FLAC/OGG recommended)")
    audio = st.file_uploader("Choose a file", type=["wav", "flac", "ogg"], accept_multiple_files=False)
    if audio is not None:
        audio_bytes = audio.read()
        st.audio(audio_bytes)

        if st.button("Transcribe & Predict", type="primary"):
            try:
                text, words = transcribe_incremental(audio_bytes)
            except Exception as e:
                st.error(f"Could not transcribe audio: {e}")
                st.stop()

            feats = compute_features(words)
            X = np.array([[feats[k] for k in FEATURE_LIST]], dtype=float)
            proba = model.predict_proba(X)[0]  # [P(NC), P(MCI)]
            p_nc, p_mci = float(proba[0]), float(proba[1])
            pred = "MCI" if p_mci >= threshold else "NC"

            st.success("Transcription & prediction complete.")

            with right:
                st.subheader("2) Prediction")
                st.metric("Prediction", pred)
                st.progress(min(0.999, p_mci), text=f"MCI likelihood: {p_mci:.2%}")
                st.caption(f"NC: {p_nc:.2%} • Threshold: {threshold:.2f}")

                st.subheader("3) Features")
                st.dataframe(pd.DataFrame([feats]), use_container_width=True)
                
                st.subheader("Feature visuals")
                fig_bar = feature_bar_chart(feats)
                st.pyplot(fig_bar, use_container_width=True)
            
                fig_radar = feature_radar_chart(feats)
                st.pyplot(fig_radar, use_container_width=True)

            st.download_button("Download transcript (.txt)", data=text, file_name="transcript.txt")
            out = {"features": feats, "probabilities": {"NC": p_nc, "MCI": p_mci}, "prediction": pred}
            st.download_button("Download results (.json)", data=json.dumps(out, indent=2), file_name="results.json")
    else:
        st.info("Upload a short WAV/FLAC/OGG (1–3 minutes) for a smooth demo.")

with right:
    st.subheader("About")
    st.write("This app transcribes uploaded audio on CPU (Vosk) and runs a classification model (scikit-learn) on basic prosody/lexical features. It is for research and education only — not a diagnosis.")
    st.write("Features: words/minute, mean pause, pause rate, type–token ratio, disfluency rate.")
    st.caption("© Your Org")
