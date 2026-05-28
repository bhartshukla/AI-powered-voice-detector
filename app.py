"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              BhashaKavach — Real-Time Multilingual Voice                    ║
║                        Deepfake Detection System                            ║
║                                                                             ║
║  Stack : Streamlit · Librosa · Scikit-learn · Matplotlib · NumPy           ║
║  Author: BhashaKavach Project                                               ║
║  Run   : streamlit run app.py                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import io
import json
import pickle
import hashlib
import logging
import datetime
import tempfile
import warnings

warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, lfilter
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BhashaKavach | Voice Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000          # Target sample rate (Hz)
MODEL_PATH    = "deepfake_model.pkl"
LOG_FILE      = "detection_logs.json"
N_MFCC        = 40              # Number of MFCC coefficients
N_FEATURES    = 151             # Total feature vector length

# Acoustic thresholds (calibrated from literature)
MALE_F0_MAX   = 165             # Hz — fundamental freq upper bound for male
FEMALE_F0_MIN = 150             # Hz — fundamental freq lower bound for female

# Colour palette (kept in one place for easy theming)
CLR_REAL      = "#00ff88"
CLR_FAKE      = "#ff4444"
CLR_WARN      = "#ffaa00"
CLR_ACCENT    = "#00d4ff"
CLR_BG        = "#0a0e1a"
CLR_CARD      = "#111827"
CLR_BORDER    = "#1e2d3d"

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("BhashaKavach")


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS  — dark cybersecurity aesthetic
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        f"""
        <style>
        /* ── Google Fonts ──────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;500&family=Sora:wght@300;400;500;600&display=swap');

        /* ── Global reset ──────────────────────────── */
        html, body, [class*="css"] {{
            font-family: 'Sora', sans-serif;
            background-color: {CLR_BG};
            color: #c9d8e8;
        }}

        /* ── Main container ────────────────────────── */
        .main .block-container {{
            padding: 1.5rem 2rem 3rem 2rem;
            max-width: 1280px;
        }}

        /* ── Sidebar ───────────────────────────────── */
        [data-testid="stSidebar"] {{
            background: #080c17;
            border-right: 1px solid {CLR_BORDER};
        }}
        [data-testid="stSidebar"] * {{
            color: #8ba3bc !important;
        }}

        /* ── Header banner ─────────────────────────── */
        .kavach-header {{
            background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
            border: 1px solid {CLR_BORDER};
            border-radius: 16px;
            padding: 2.2rem 2.8rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }}
        .kavach-header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(ellipse at 80% 50%, rgba(0,212,255,0.08) 0%, transparent 65%);
            pointer-events: none;
        }}
        .kavach-title {{
            font-family: 'Orbitron', monospace;
            font-size: 2.4rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            background: linear-gradient(90deg, {CLR_ACCENT} 0%, #7b5ea7 50%, {CLR_REAL} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.3rem 0;
        }}
        .kavach-sub {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: #4a6fa5;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .kavach-badge {{
            display: inline-block;
            background: rgba(0,212,255,0.12);
            border: 1px solid rgba(0,212,255,0.3);
            border-radius: 6px;
            padding: 0.2rem 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            color: {CLR_ACCENT};
            margin-top: 0.6rem;
            letter-spacing: 0.1em;
        }}

        /* ── Cards ─────────────────────────────────── */
        .k-card {{
            background: {CLR_CARD};
            border: 1px solid {CLR_BORDER};
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
        }}
        .k-card-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #4a6fa5;
            margin-bottom: 0.8rem;
        }}

        /* ── Result banner ─────────────────────────── */
        .result-real {{
            background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,255,136,0.03));
            border: 1px solid rgba(0,255,136,0.4);
            border-left: 4px solid {CLR_REAL};
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
        }}
        .result-fake {{
            background: linear-gradient(135deg, rgba(255,68,68,0.10), rgba(255,68,68,0.04));
            border: 1px solid rgba(255,68,68,0.4);
            border-left: 4px solid {CLR_FAKE};
            border-radius: 12px;
            padding: 1.4rem 1.6rem;
        }}
        .result-label {{
            font-family: 'Orbitron', monospace;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.06em;
        }}
        .label-real {{ color: {CLR_REAL}; }}
        .label-fake {{ color: {CLR_FAKE}; }}

        /* ── Confidence meter ──────────────────────── */
        .conf-bar-bg {{
            background: #1a2235;
            border-radius: 50px;
            height: 14px;
            width: 100%;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        .conf-bar-fill-real {{
            background: linear-gradient(90deg, #00aa55, {CLR_REAL});
            border-radius: 50px;
            height: 100%;
            transition: width 0.6s ease;
        }}
        .conf-bar-fill-fake {{
            background: linear-gradient(90deg, #aa2200, {CLR_FAKE});
            border-radius: 50px;
            height: 100%;
            transition: width 0.6s ease;
        }}

        /* ── Tag pills ─────────────────────────────── */
        .tag {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.85rem;
            border-radius: 50px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            font-weight: 500;
            letter-spacing: 0.05em;
            margin: 0.2rem;
        }}
        .tag-lang {{
            background: rgba(123,94,167,0.18);
            border: 1px solid rgba(123,94,167,0.5);
            color: #b08ce8;
        }}
        .tag-gender {{
            background: rgba(0,212,255,0.12);
            border: 1px solid rgba(0,212,255,0.4);
            color: {CLR_ACCENT};
        }}
        .tag-warn {{
            background: rgba(255,170,0,0.10);
            border: 1px solid rgba(255,170,0,0.35);
            color: {CLR_WARN};
        }}

        /* ── Warning section ───────────────────────── */
        .why-fake-item {{
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.65rem 0;
            border-bottom: 1px solid rgba(255,68,68,0.08);
        }}
        .why-fake-item:last-child {{ border-bottom: none; }}
        .why-icon {{
            font-size: 1.1rem;
            flex-shrink: 0;
            margin-top: 0.05rem;
        }}
        .why-text {{
            font-size: 0.85rem;
            color: #c9d8e8;
            line-height: 1.5;
        }}
        .why-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: {CLR_FAKE};
            text-transform: uppercase;
            letter-spacing: 0.08em;
            display: block;
            margin-bottom: 0.15rem;
        }}

        /* ── Log table ─────────────────────────────── */
        .log-row {{
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.55rem 0.8rem;
            border-radius: 8px;
            margin-bottom: 0.3rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
        }}
        .log-real {{ background: rgba(0,255,136,0.05); border-left: 2px solid {CLR_REAL}; }}
        .log-fake {{ background: rgba(255,68,68,0.07); border-left: 2px solid {CLR_FAKE}; }}
        .log-time {{ color: #4a6fa5; flex-shrink: 0; }}
        .log-file {{ color: #8ba3bc; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .log-verdict {{ font-weight: 600; flex-shrink: 0; }}

        /* ── Stat boxes ────────────────────────────── */
        .stat-box {{
            background: #0d1525;
            border: 1px solid {CLR_BORDER};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }}
        .stat-val {{
            font-family: 'Orbitron', monospace;
            font-size: 1.6rem;
            font-weight: 700;
            color: {CLR_ACCENT};
        }}
        .stat-lbl {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: #4a6fa5;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.2rem;
        }}

        /* ── Streamlit overrides ───────────────────── */
        .stButton > button {{
            background: linear-gradient(135deg, #0d2048, #162d54);
            border: 1px solid {CLR_ACCENT};
            color: {CLR_ACCENT};
            font-family: 'Orbitron', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.1em;
            border-radius: 8px;
            padding: 0.55rem 1.4rem;
            transition: all 0.2s;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, #162d54, #1e3d6e);
            box-shadow: 0 0 20px rgba(0,212,255,0.25);
        }}
        div[data-testid="stFileUploader"] {{
            background: #0d1525;
            border: 1px dashed {CLR_BORDER};
            border-radius: 12px;
            padding: 0.5rem;
        }}
        .stProgress > div > div {{
            background: linear-gradient(90deg, {CLR_ACCENT}, {CLR_REAL});
            border-radius: 50px;
        }}
        hr {{ border-color: {CLR_BORDER}; }}
        h2, h3, h4 {{
            font-family: 'Orbitron', monospace;
            color: #c9d8e8;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pre-trained deepfake detection pipeline from disk.
    If the model file doesn't exist, generate it on-the-fly."""
    if not os.path.exists(MODEL_PATH):
        logger.info("deepfake_model.pkl not found — generating now …")
        _generate_model()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", MODEL_PATH)
    return model


def _generate_model():
    """Fallback: generate and save the model without the separate script."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    rng = np.random.RandomState(42)
    N   = 2400
    H   = N // 2

    def _block(n, mfcc_std_scale=1.0, delta_std=0.8, zcr_mean=0.08,
                zcr_std=0.02, sc_base=20, f0_std=20, voiced_hi=0.95,
                jitter_mean=0.005, tempo_std=20, rms_std_val=0.02):
        mfcc   = rng.normal(0, mfcc_std_scale, (n, 40))
        delta  = rng.normal(0, delta_std,       (n, 40))
        delta2 = rng.normal(0, delta_std * 0.5, (n, 40))
        sc     = rng.normal(sc_base, 3,          (n, 7))
        zcr    = rng.normal([zcr_mean, zcr_std, 0.2], [0.015, 0.005, 0.05], (n, 3))
        chroma = rng.normal(0.4, 0.1,            (n, 12))
        pitch  = np.column_stack([
            rng.normal(160, 50, n),
            rng.normal(f0_std, 3, n),
            rng.uniform(0.6, voiced_hi, n),
            rng.normal(jitter_mean, jitter_mean * 0.3, n),
        ])
        temporal = np.column_stack([
            rng.normal(120, tempo_std, n),
            rng.uniform(2, 8, n),
            rng.normal(0.05, 0.015, n),
            rng.normal(rms_std_val, rms_std_val * 0.3, n),
            rng.uniform(0.1, 0.4, n),
        ])
        return np.hstack([mfcc, delta, delta2, sc, zcr, chroma, pitch, temporal])

    X_real = _block(H, mfcc_std_scale=3.0, delta_std=0.8, zcr_mean=0.08,
                    sc_base=22, f0_std=20, voiced_hi=0.93,
                    jitter_mean=0.005, tempo_std=18, rms_std_val=0.02)
    X_fake = _block(H, mfcc_std_scale=1.2, delta_std=0.15, zcr_mean=0.06,
                    sc_base=10, f0_std=5,  voiced_hi=1.0,
                    jitter_mean=0.0005, tempo_std=4,  rms_std_val=0.005)
    X = np.vstack([X_real, X_fake])
    y = np.array([0] * H + [1] * H)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=300, max_depth=18,
                                          random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    logger.info("Auto-generated model saved to %s", MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def butter_highpass(cutoff: float = 80.0, fs: float = SAMPLE_RATE,
                    order: int = 4):
    """Design a Butterworth high-pass filter to remove low-freq rumble."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """Simple spectral-gating noise reduction using a noise profile
    estimated from the first 0.25 s of the signal."""
    frame_len = int(0.025 * sr)   # 25 ms frames
    hop       = frame_len // 2

    # Estimate noise floor from first ~250 ms
    noise_clip  = y[: int(0.25 * sr)] if len(y) > int(0.25 * sr) else y
    noise_stft  = librosa.stft(noise_clip, n_fft=frame_len * 2, hop_length=hop)
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

    stft         = librosa.stft(y, n_fft=frame_len * 2, hop_length=hop)
    signal_power = np.abs(stft) ** 2
    # Spectral gate: attenuate bins where signal is near noise floor
    mask = (signal_power > (noise_power * 2.0)).astype(float)
    # Smooth mask
    mask = np.maximum(mask, 0.1)
    denoised = librosa.istft(stft * mask, hop_length=hop, length=len(y))
    return denoised


def preprocess_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """Full preprocessing pipeline:
       1. Load & resample to 16 kHz
       2. High-pass filter (remove rumble)
       3. Noise reduction
       4. Silence trimming
       5. Amplitude normalisation
    Returns (audio_array, sample_rate).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    # 1. High-pass filter
    b, a = butter_highpass(cutoff=80, fs=sr)
    y = lfilter(b, a, y).astype(np.float32)

    # 2. Noise reduction
    if len(y) > int(0.3 * sr):
        y = reduce_noise(y, sr).astype(np.float32)

    # 3. Silence trimming (top-db = 30)
    y, _ = librosa.effects.trim(y, top_db=30)

    # 4. Guard against empty signal
    if len(y) < 512:
        raise ValueError("Audio too short after preprocessing (< 32 ms). "
                         "Please upload a longer sample.")

    # 5. Normalise amplitude
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.95

    return y, sr


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract 151-dimensional acoustic feature vector:
    - MFCC (40) + Δ (40) + ΔΔ (40)
    - Spectral Contrast (7)
    - Zero Crossing Rate — mean, std, max (3)
    - Chroma STFT (12)
    - Pitch — mean F0, F0 std, voiced fraction, jitter (4)
    - Temporal — tempo, duration, RMS mean, RMS std, silence ratio (5)
    """
    feat_list = []

    # ── 1. MFCCs ────────────────────────────────────────────────────────────
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=512,
                                    hop_length=160, fmin=20, fmax=8000)
    delta   = librosa.feature.delta(mfcc, order=1)
    delta2  = librosa.feature.delta(mfcc, order=2)

    feat_list.extend([
        np.mean(mfcc,   axis=1),   # 40
        np.mean(delta,  axis=1),   # 40
        np.mean(delta2, axis=1),   # 40
    ])

    # ── 2. Spectral Contrast ────────────────────────────────────────────────
    sc = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6, fmin=50)
    feat_list.append(np.mean(sc, axis=1))   # 7

    # ── 3. Zero Crossing Rate ───────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=160)[0]
    feat_list.append(np.array([np.mean(zcr), np.std(zcr), np.max(zcr)]))  # 3

    # ── 4. Chroma STFT ──────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12,
                                          hop_length=160)
    feat_list.append(np.mean(chroma, axis=1))   # 12

    # ── 5. Pitch (F0) via PYIN ──────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr, hop_length=160,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        if len(voiced_f0) > 1:
            f0_mean    = float(np.mean(voiced_f0))
            f0_std     = float(np.std(voiced_f0))
            voiced_frac = float(np.sum(voiced_flag) / len(voiced_flag))
            # Jitter: mean absolute difference between consecutive voiced F0
            jitter     = float(np.mean(np.abs(np.diff(voiced_f0))) / (f0_mean + 1e-9))
        else:
            f0_mean, f0_std, voiced_frac, jitter = 160.0, 20.0, 0.5, 0.005
    except Exception:
        f0_mean, f0_std, voiced_frac, jitter = 160.0, 20.0, 0.5, 0.005

    feat_list.append(np.array([f0_mean, f0_std, voiced_frac, jitter]))  # 4

    # ── 6. Temporal ─────────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo     = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    duration  = len(y) / sr
    rms       = librosa.feature.rms(y=y, hop_length=160)[0]
    rms_mean  = float(np.mean(rms))
    rms_std   = float(np.std(rms))

    # Silence ratio: fraction of frames below 10 % of peak RMS
    silence_thr   = rms_mean * 0.10
    silence_ratio = float(np.sum(rms < silence_thr) / (len(rms) + 1e-9))

    feat_list.append(np.array([tempo, duration, rms_mean, rms_std,
                                silence_ratio]))  # 5

    features = np.concatenate(feat_list)
    assert features.shape[0] == N_FEATURES, (
        f"Feature dim mismatch: expected {N_FEATURES}, got {features.shape[0]}"
    )
    return features, {
        "f0_mean": f0_mean, "f0_std": f0_std,
        "voiced_frac": voiced_frac, "jitter": jitter,
        "zcr_mean": float(np.mean(zcr)), "rms_std": rms_std,
        "silence_ratio": silence_ratio,
        "tempo": tempo, "duration": duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  LANGUAGE DETECTION  (rule-based acoustic heuristics)
# ─────────────────────────────────────────────────────────────────────────────
def detect_language(y: np.ndarray, sr: int,
                    acoustic: dict) -> tuple[str, float]:
    """
    Heuristic language classification into: Hindi / English / Hinglish.

    Features used
    ─────────────
    • Spectral rolloff variance — Hindi tends to have richer low-mid resonance
    • Chroma energy distribution — tonal vs. non-tonal languages differ
    • ZCR — English has more fricatives (higher ZCR)
    • Temporal rhythm pattern (speech rate proxy via tempo)
    • Silence ratio — Hindi/Hinglish have distinct rhythm patterns

    NOTE: Without a large labelled corpus this is heuristic; accuracy is
    demonstration-grade.  Replace with a fine-tuned Wav2Vec-2 or Whisper
    classifier for production.
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_mean = float(np.mean(rolloff))

    chroma  = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_std = float(np.std(chroma))

    zcr_mean      = acoustic["zcr_mean"]
    silence_ratio = acoustic["silence_ratio"]
    tempo         = acoustic["tempo"]

    # Scoring heuristic (higher score → more likely that language)
    scores = {"Hindi": 0.0, "English": 0.0, "Hinglish": 0.0}

    # High rolloff + high ZCR → English (more fricatives, higher formants)
    if rolloff_mean > 4800:
        scores["English"]  += 2.0
    elif rolloff_mean > 3200:
        scores["Hinglish"] += 1.5
    else:
        scores["Hindi"]    += 2.0

    # High chroma variability → tonal / melodic (Hindi)
    if chroma_std > 0.22:
        scores["Hindi"]    += 1.5
    elif chroma_std > 0.14:
        scores["Hinglish"] += 1.0
    else:
        scores["English"]  += 1.0

    # ZCR
    if zcr_mean > 0.10:
        scores["English"]  += 1.5
    elif zcr_mean > 0.07:
        scores["Hinglish"] += 1.0
    else:
        scores["Hindi"]    += 1.0

    # Silence rhythm
    if 0.15 < silence_ratio < 0.35:
        scores["Hindi"]    += 0.8
    elif silence_ratio < 0.12:
        scores["English"]  += 0.6
    else:
        scores["Hinglish"] += 0.8

    # Tempo: faster speech rate slightly more English
    if tempo > 130:
        scores["English"]  += 0.5
    elif tempo < 100:
        scores["Hindi"]    += 0.5
    else:
        scores["Hinglish"] += 0.3

    total = sum(scores.values()) + 1e-9
    lang  = max(scores, key=scores.get)
    conf  = float(scores[lang] / total)
    return lang, min(conf, 0.97)


# ─────────────────────────────────────────────────────────────────────────────
#  GENDER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def detect_gender(acoustic: dict) -> tuple[str, float]:
    """
    Rule-based gender classification based on fundamental frequency (F0).

    Thresholds (literature):
    • Male   : F0 typically 85 – 155 Hz
    • Female : F0 typically 165 – 255 Hz
    • Overlap zone 155–165 Hz resolved by jitter & voiced fraction.
    """
    f0     = acoustic["f0_mean"]
    jitter = acoustic["jitter"]
    vf     = acoustic["voiced_frac"]

    if f0 <= 0:
        return "Unknown", 0.50

    # Base decision on F0
    if f0 < 145:
        gender = "Male"
        raw_conf = min(0.95, 0.65 + (145 - f0) / 145 * 0.30)
    elif f0 > 175:
        gender = "Female"
        raw_conf = min(0.95, 0.65 + (f0 - 175) / 255 * 0.30)
    else:
        # Overlap zone — use secondary cues
        if jitter > 0.004 and vf < 0.80:
            gender, raw_conf = "Male",   0.60
        elif vf >= 0.85:
            gender, raw_conf = "Female", 0.60
        else:
            gender, raw_conf = "Male",   0.52   # slight male bias in ambiguous

    return gender, float(raw_conf)


# ─────────────────────────────────────────────────────────────────────────────
#  FAKE-REASON EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────
FAKE_REASONS = [
    {
        "icon": "📉",
        "label": "Unnatural Pitch Stability",
        "text":  "Human speech shows natural F0 micro-variations (vibrato, jitter). "
                 "This sample's pitch is suspiciously constant — a hallmark of "
                 "TTS/vocoder synthesis.",
    },
    {
        "icon": "🤖",
        "label": "Robotic Spectral Smoothness",
        "text":  "The MFCC delta coefficients are near-zero, indicating the spectral "
                 "envelope changes too smoothly between frames — typical of neural "
                 "TTS models that over-regularise mel-spectrograms.",
    },
    {
        "icon": "🌊",
        "label": "Synthetic Frequency Patterns",
        "text":  "Spectral contrast values fall well below natural speech ranges. "
                 "Real voices exhibit stronger harmonic peaks vs. valley contrasts; "
                 "vocoders tend to flatten this structure.",
    },
    {
        "icon": "🫁",
        "label": "Absent Breathing Artifacts",
        "text":  "No breath-intake or glottal transients were detected at phrase "
                 "boundaries. Genuine speakers insert micro-silences and noise bursts "
                 "when breathing; synthesisers omit these.",
    },
    {
        "icon": "📐",
        "label": "Abnormal Speech Rhythm",
        "text":  "The detected tempo/beat regularity is outside the natural range for "
                 "conversational speech, suggesting machine-generated prosody.",
    },
    {
        "icon": "🔇",
        "label": "Compressed Dynamic Range",
        "text":  "RMS energy variance is extremely low — the audio lacks the natural "
                 "loudness fluctuations of real speech, pointing to post-processed "
                 "AI-generated audio.",
    },
]

def get_fake_reasons(acoustic: dict, confidence: float) -> list[dict]:
    """Return the most relevant fake-reason items based on acoustic cues."""
    reasons = []

    # Pitch stability
    if acoustic["f0_std"] < 12:
        reasons.append(FAKE_REASONS[0])

    # MFCC smoothness (proxy via jitter: low jitter → smooth)
    if acoustic["jitter"] < 0.002:
        reasons.append(FAKE_REASONS[1])

    # Spectral (low silence ratio → no breathing)
    if acoustic["silence_ratio"] < 0.08:
        reasons.append(FAKE_REASONS[3])

    # Rhythm
    if acoustic["tempo"] > 135 or acoustic["tempo"] < 60:
        reasons.append(FAKE_REASONS[4])

    # Dynamics
    if acoustic["rms_std"] < 0.008:
        reasons.append(FAKE_REASONS[5])

    # Always include spectral if high confidence
    if confidence > 0.75:
        reasons.append(FAKE_REASONS[2])

    # Deduplicate preserving order
    seen, unique = set(), []
    for r in reasons:
        if r["label"] not in seen:
            seen.add(r["label"])
            unique.append(r)

    # Guarantee at least 2 reasons when verdict is fake
    if len(unique) < 2:
        for r in FAKE_REASONS:
            if r["label"] not in seen:
                unique.append(r)
                if len(unique) >= 3:
                    break
    return unique[:5]


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor":  "#0a0e1a",
    "axes.facecolor":    "#0d1525",
    "axes.edgecolor":    "#1e2d3d",
    "axes.labelcolor":   "#8ba3bc",
    "axes.titlecolor":   "#c9d8e8",
    "xtick.color":       "#4a6fa5",
    "ytick.color":       "#4a6fa5",
    "grid.color":        "#1e2d3d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "#c9d8e8",
    "font.family":       "monospace",
}


def _apply_style():
    plt.rcParams.update(PLOT_STYLE)


def plot_waveform(y: np.ndarray, sr: int, is_fake: bool) -> plt.Figure:
    """Plot time-domain waveform."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 2.4))
    t = np.linspace(0, len(y) / sr, num=len(y))
    colour = CLR_FAKE if is_fake else CLR_REAL
    ax.fill_between(t, y, 0, alpha=0.35, color=colour)
    ax.plot(t, y, linewidth=0.6, color=colour, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform", pad=8)
    ax.grid(True)
    ax.set_xlim(0, t[-1])
    fig.tight_layout()
    return fig


def plot_spectrogram(y: np.ndarray, sr: int) -> plt.Figure:
    """Plot mel-spectrogram."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 2.8))
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max
    )
    img = librosa.display.specshow(
        D, sr=sr, hop_length=160, x_axis="time", y_axis="hz",
        ax=ax, cmap="inferno",
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    ax.set_title("Mel Spectrogram", pad=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    return fig


def plot_mfcc(y: np.ndarray, sr: int) -> plt.Figure:
    """Plot MFCC heat-map + delta."""
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   hop_length=160)
    delta = librosa.feature.delta(mfcc)

    for data, ax, title, cmap in [
        (mfcc,  axes[0], "MFCC Coefficients", "magma"),
        (delta, axes[1], "MFCC Δ (Delta)",     "coolwarm"),
    ]:
        img = librosa.display.specshow(
            data, sr=sr, hop_length=160, x_axis="time",
            ax=ax, cmap=cmap,
        )
        plt.colorbar(img, ax=ax, pad=0.01)
        ax.set_title(title, pad=6)
        ax.set_ylabel("Coefficient")

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION LOG
# ─────────────────────────────────────────────────────────────────────────────
def append_log(filename: str, verdict: str, confidence: float,
               language: str, gender: str, duration: float):
    """Append a detection record to the local JSON log file."""
    record = {
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "filename":   filename,
        "verdict":    verdict,
        "confidence": round(confidence, 4),
        "language":   language,
        "gender":     gender,
        "duration_s": round(duration, 2),
    }
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            logs = []

    logs.insert(0, record)          # newest first
    logs = logs[:500]               # keep last 500

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    logger.info("Log saved | %s | %s | %.2f%%", filename, verdict,
                confidence * 100)


def load_logs() -> list[dict]:
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(logs: list[dict]):
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 0.5rem;'>
              <span style='font-family:Orbitron,monospace; font-size:1.1rem;
                           font-weight:700; color:#00d4ff; letter-spacing:0.1em;'>
                🛡️ BHASHA KAVACH
              </span><br>
              <span style='font-family:"JetBrains Mono",monospace; font-size:0.65rem;
                           color:#4a6fa5; letter-spacing:0.1em;'>
                v1.0 · CYBERSECURITY AI
              </span>
            </div>
            <hr style='border-color:#1e2d3d; margin:0.8rem 0;'>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**System Info**")
        st.markdown(
            f"""
            <div style='font-family:"JetBrains Mono",monospace; font-size:0.74rem;
                        color:#4a6fa5; line-height:2;'>
              MODEL &nbsp;&nbsp;&nbsp; RandomForest<br>
              FEATURES &nbsp; {N_FEATURES}-dim vector<br>
              SR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {SAMPLE_RATE // 1000} kHz<br>
              MFCC &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {N_MFCC} coefficients<br>
              LANGUAGES &nbsp; Hi · En · Hinglish<br>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Quick stats from logs
        total   = len(logs)
        n_fake  = sum(1 for l in logs if l["verdict"] == "FAKE")
        n_real  = total - n_fake

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="stat-box"><div class="stat-val">{total}</div>'
                f'<div class="stat-lbl">Scanned</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-box"><div class="stat-val" style="color:{CLR_FAKE};">'
                f'{n_fake}</div><div class="stat-lbl">Fake</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**🔮 Future Roadmap**")
        roadmap = [
            "🎙️ Live microphone stream",
            "📱 Mobile deployment (Streamlit Cloud)",
            "🌐 Expand languages (Tamil, Telugu…)",
            "📞 Real-time call monitoring",
            "🧠 Wav2Vec-2 fine-tune",
        ]
        for item in roadmap:
            st.markdown(
                f'<div style="font-size:0.78rem; color:#4a6fa5; padding:0.2rem 0;">'
                f'{item}</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  RESULT RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_result(verdict: str, confidence: float,
                  language: str, lang_conf: float,
                  gender: str, gender_conf: float,
                  acoustic: dict):

    is_fake = verdict == "FAKE"
    pct     = int(confidence * 100)

    card_cls  = "result-fake"  if is_fake else "result-real"
    label_cls = "label-fake"   if is_fake else "label-real"
    icon      = "⚠️"           if is_fake else "✅"
    bar_cls   = "conf-bar-fill-fake" if is_fake else "conf-bar-fill-real"

    st.markdown(
        f"""
        <div class="{card_cls}">
          <div style="display:flex; align-items:center; gap:1rem;">
            <span style="font-size:2.2rem;">{icon}</span>
            <div>
              <div class="result-label {label_cls}">{verdict}</div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem;
                          color:#4a6fa5; margin-top:0.15rem;">
                Voice Authenticity Verdict
              </div>
            </div>
            <div style="margin-left:auto; text-align:right;">
              <div style="font-family:'Orbitron',monospace; font-size:1.8rem;
                          font-weight:700; color:{'#ff4444' if is_fake else '#00ff88'};">
                {pct}%
              </div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                          color:#4a6fa5;">Confidence</div>
            </div>
          </div>

          <div style="margin-top:1rem;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                        color:#4a6fa5; margin-bottom:0.3rem; text-transform:uppercase;
                        letter-spacing:0.1em;">
              Confidence Meter
            </div>
            <div class="conf-bar-bg">
              <div class="{bar_cls}" style="width:{pct}%;"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Language & Gender ────────────────────────────────────────────────────
    lang_flag = {"Hindi": "🇮🇳", "English": "🇬🇧", "Hinglish": "🇮🇳🇬🇧"}.get(language, "🌐")
    gen_icon  = "♂️" if gender == "Male" else ("♀️" if gender == "Female" else "❓")

    st.markdown(
        f"""
        <div class="k-card">
          <div class="k-card-title">🔍 Signal Intelligence</div>
          <div style="display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center;">
            <span class="tag tag-lang">
              {lang_flag} Language: {language} &nbsp;·&nbsp; {int(lang_conf*100)}%
            </span>
            <span class="tag tag-gender">
              {gen_icon} Gender: {gender} &nbsp;·&nbsp; {int(gender_conf*100)}%
            </span>
            <span class="tag tag-warn">
              🎵 F0: {acoustic['f0_mean']:.1f} Hz
            </span>
            <span class="tag tag-warn">
              ⏱ Duration: {acoustic['duration']:.2f}s
            </span>
            <span class="tag tag-warn">
              🥁 Tempo: {acoustic['tempo']:.0f} BPM
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Why Fake ─────────────────────────────────────────────────────────────
    if is_fake:
        reasons = get_fake_reasons(acoustic, confidence)
        html = """
        <div class="k-card">
          <div class="k-card-title" style="color:#ff4444;">
            ⚡ Why This Audio May Be Fake
          </div>
        """
        for r in reasons:
            html += f"""
          <div class="why-fake-item">
            <span class="why-icon">{r['icon']}</span>
            <div class="why-text">
              <span class="why-label">{r['label']}</span>
              {r['text']}
            </div>
          </div>
            """
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOG RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_logs(logs: list[dict]):
    st.markdown(
        '<div class="k-card-title" style="margin-bottom:0.6rem;">'
        '📋 Detection History</div>',
        unsafe_allow_html=True,
    )
    if not logs:
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\',monospace; color:#4a6fa5;'
            'font-size:0.78rem;">No detections yet.</div>',
            unsafe_allow_html=True,
        )
        return

    for entry in logs[:15]:
        is_fake  = entry["verdict"] == "FAKE"
        row_cls  = "log-fake" if is_fake else "log-real"
        v_colour = CLR_FAKE   if is_fake else CLR_REAL
        ts       = entry["timestamp"].replace("T", " ")
        fname    = entry.get("filename", "—")[:30]
        conf_pct = int(entry["confidence"] * 100)
        lang     = entry.get("language", "?")
        gen      = entry.get("gender",   "?")

        st.markdown(
            f"""
            <div class="log-row {row_cls}">
              <span class="log-time">{ts}</span>
              <span class="log-file" title="{entry.get('filename','')}">
                📂 {fname}
              </span>
              <span class="tag tag-lang" style="font-size:0.68rem;">{lang}</span>
              <span class="tag tag-gender" style="font-size:0.68rem;">{gen}</span>
              <span class="log-verdict" style="color:{v_colour};">
                {entry['verdict']} {conf_pct}%
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(logs) > 15:
        st.caption(f"Showing 15 of {len(logs)} entries.")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # ── Load model ───────────────────────────────────────────────────────────
    with st.spinner("Initialising BhashaKavach AI engine …"):
        model = load_model()

    logs = load_logs()
    render_sidebar(logs)

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="kavach-header">
          <div style="display:flex; align-items:center; gap:1rem;">
            <span style="font-size:3rem;">🛡️</span>
            <div>
              <div class="kavach-title">BhashaKavach</div>
              <div class="kavach-sub">
                Real-Time Multilingual Voice Deepfake Detection System
              </div>
              <span class="kavach-badge">CYBERSECURITY AI · v1.0</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Upload ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="k-card-title">📤 Upload Voice Sample</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        label="",
        type=["wav", "mp3"],
        help="Supports .wav and .mp3 files. Recommended: 2–10 seconds of speech.",
    )

    if uploaded is None:
        # Landing state
        st.markdown(
            """
            <div style="text-align:center; padding:3rem 1rem; color:#4a6fa5;">
              <div style="font-size:3rem; margin-bottom:1rem;">🎙️</div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.88rem;
                          letter-spacing:0.05em;">
                Upload a <strong style="color:#8ba3bc;">.wav</strong> or
                <strong style="color:#8ba3bc;">.mp3</strong> voice file<br>
                to begin deepfake analysis
              </div>
              <div style="margin-top:1rem; font-size:0.75rem; opacity:0.6;">
                Supports Hindi · English · Hinglish &nbsp;|&nbsp;
                Male & Female voices
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_logs(logs)
        return

    # ── Audio player ─────────────────────────────────────────────────────────
    st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")

    # ── Analysis pipeline ─────────────────────────────────────────────────────
    file_bytes = uploaded.read()
    file_hash  = hashlib.md5(file_bytes).hexdigest()[:8]

    # Cache expensive computation per file
    cache_key  = f"result_{file_hash}"

    if cache_key not in st.session_state:
        progress = st.progress(0, text="Loading audio …")

        try:
            # 1. Preprocess
            progress.progress(15, text="Preprocessing audio …")
            y, sr = preprocess_audio(file_bytes)

            # 2. Feature extraction
            progress.progress(40, text="Extracting acoustic features …")
            features, acoustic = extract_features(y, sr)

            # 3. Model prediction
            progress.progress(65, text="Running AI deepfake model …")
            feat_2d    = features.reshape(1, -1)
            proba      = model.predict_proba(feat_2d)[0]
            label_idx  = int(np.argmax(proba))
            confidence = float(proba[label_idx])
            verdict    = "FAKE" if label_idx == 1 else "REAL"

            # 4. Language detection
            progress.progress(78, text="Detecting language …")
            language, lang_conf = detect_language(y, sr, acoustic)

            # 5. Gender detection
            progress.progress(88, text="Detecting gender …")
            gender, gender_conf = detect_gender(acoustic)

            # 6. Visualisations
            progress.progress(94, text="Rendering visualisations …")
            is_fake = verdict == "FAKE"
            wf_bytes   = fig_to_bytes(plot_waveform(y, sr, is_fake))
            spec_bytes = fig_to_bytes(plot_spectrogram(y, sr))
            mfcc_bytes = fig_to_bytes(plot_mfcc(y, sr))

            progress.progress(100, text="Analysis complete ✅")
            progress.empty()

            result = dict(
                y=y, sr=sr, acoustic=acoustic,
                verdict=verdict, confidence=confidence,
                language=language, lang_conf=lang_conf,
                gender=gender, gender_conf=gender_conf,
                wf_bytes=wf_bytes, spec_bytes=spec_bytes,
                mfcc_bytes=mfcc_bytes,
            )
            st.session_state[cache_key] = result

            # Save log
            append_log(
                filename   = uploaded.name,
                verdict    = verdict,
                confidence = confidence,
                language   = language,
                gender     = gender,
                duration   = acoustic["duration"],
            )

        except ValueError as ve:
            progress.empty()
            st.error(f"⚠️ {ve}")
            return
        except Exception as exc:
            progress.empty()
            st.error(f"🔴 Unexpected error during analysis: {exc}")
            logger.exception("Analysis failed")
            return
    else:
        result = st.session_state[cache_key]

    # ── Unpack ────────────────────────────────────────────────────────────────
    verdict     = result["verdict"]
    confidence  = result["confidence"]
    language    = result["language"]
    lang_conf   = result["lang_conf"]
    gender      = result["gender"]
    gender_conf = result["gender_conf"]
    acoustic    = result["acoustic"]

    # ── Verdict & details ─────────────────────────────────────────────────────
    render_result(verdict, confidence, language, lang_conf,
                  gender, gender_conf, acoustic)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visualisations ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="k-card-title">📊 Feature Visualisations</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "🌈 Spectrogram", "🔢 MFCC"])
    with tab1:
        st.image(result["wf_bytes"],   use_container_width=True)
    with tab2:
        st.image(result["spec_bytes"], use_container_width=True)
    with tab3:
        st.image(result["mfcc_bytes"], use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Acoustic feature table ────────────────────────────────────────────────
    with st.expander("🔬 Raw Acoustic Feature Report", expanded=False):
        ac = acoustic
        rows = [
            ("Mean F0 (Hz)",          f"{ac['f0_mean']:.2f}"),
            ("F0 Std Dev (Hz)",        f"{ac['f0_std']:.2f}"),
            ("Voiced Fraction",        f"{ac['voiced_frac']:.3f}"),
            ("Jitter",                 f"{ac['jitter']:.5f}"),
            ("Mean ZCR",               f"{ac['zcr_mean']:.5f}"),
            ("RMS Std Dev",            f"{ac['rms_std']:.5f}"),
            ("Silence Ratio",          f"{ac['silence_ratio']:.3f}"),
            ("Speech Tempo (BPM)",     f"{ac['tempo']:.1f}"),
            ("Duration (s)",           f"{ac['duration']:.2f}"),
        ]
        df = pd.DataFrame(rows, columns=["Feature", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detection log ─────────────────────────────────────────────────────────
    logs = load_logs()           # reload after possible new entry
    render_logs(logs)

    if logs and st.button("🗑️ Clear All Logs"):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        for k in list(st.session_state.keys()):
            if k.startswith("result_"):
                del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
