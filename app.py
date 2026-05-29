"""
╔══════════════════════════════════════════════════════════════════╗
║           BhashaKavach – AI Voice Deepfake Detection             ║
║                         app.py  v2                               ║
║                                                                  ║
║  Run: streamlit run app.py                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import io
import json
import pickle
import tempfile
import warnings

import numpy as np
import streamlit as st

# ── Optional imports (graceful degradation) ───────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

from feature_extractor import (
    load_and_preprocess,
    extract_features,
    detect_gender,
    is_valid_audio,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
MODEL_PATH   = "deepfake_model.pkl"
METRICS_PATH = "training_metrics.json"

SUPPORTED_FORMATS = ["wav", "mp3", "ogg", "flac", "m4a"]

# Confidence thresholds
HIGH_CONF   = 0.85   # display "HIGH CONFIDENCE"
MED_CONF    = 0.65   # display "MEDIUM CONFIDENCE"

# ──────────────────────────────────────────────────────────────────
# Page config – must be first Streamlit call
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "BhashaKavach – Voice Deepfake Detector",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ──────────────────────────────────────────────────────────────────
# CSS – cybersecurity dark theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

  :root {
    --bg-primary   : #0a0e1a;
    --bg-card      : #0f1629;
    --bg-panel     : #141c2e;
    --accent-cyan  : #00d4ff;
    --accent-green : #00ff88;
    --accent-red   : #ff2b4e;
    --accent-amber : #ffb627;
    --border       : #1e2d4a;
    --text-primary : #e2eaf8;
    --text-muted   : #5a7a9a;
    --font-mono    : 'Share Tech Mono', monospace;
    --font-ui      : 'Rajdhani', sans-serif;
  }

  html, body, [class*="css"] {
    font-family : var(--font-ui);
    background  : var(--bg-primary) !important;
    color       : var(--text-primary) !important;
  }

  /* Header */
  .bk-header {
    text-align      : center;
    padding         : 2rem 0 1.2rem;
    border-bottom   : 1px solid var(--border);
    margin-bottom   : 1.8rem;
  }
  .bk-logo {
    font-family : var(--font-mono);
    font-size   : 2.4rem;
    color       : var(--accent-cyan);
    letter-spacing : 0.08em;
    text-shadow : 0 0 24px #00d4ff88;
  }
  .bk-sub {
    font-size   : 0.95rem;
    color       : var(--text-muted);
    letter-spacing : 0.15em;
    text-transform : uppercase;
    margin-top  : 0.3rem;
  }

  /* Cards */
  .bk-card {
    background    : var(--bg-card);
    border        : 1px solid var(--border);
    border-radius : 8px;
    padding       : 1.4rem 1.6rem;
    margin-bottom : 1rem;
  }
  .bk-card-title {
    font-family   : var(--font-mono);
    color         : var(--accent-cyan);
    font-size     : 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom : 0.8rem;
    border-bottom : 1px solid var(--border);
    padding-bottom: 0.4rem;
  }

  /* Verdict banners */
  .verdict-real {
    background    : linear-gradient(135deg, #00291a 0%, #0f2018 100%);
    border        : 2px solid var(--accent-green);
    border-radius : 10px;
    padding       : 1.8rem;
    text-align    : center;
    box-shadow    : 0 0 30px #00ff8822;
  }
  .verdict-fake {
    background    : linear-gradient(135deg, #2a0014 0%, #200812 100%);
    border        : 2px solid var(--accent-red);
    border-radius : 10px;
    padding       : 1.8rem;
    text-align    : center;
    box-shadow    : 0 0 30px #ff2b4e22;
  }
  .verdict-label {
    font-family   : var(--font-mono);
    font-size     : 2.6rem;
    font-weight   : bold;
    letter-spacing: 0.12em;
  }
  .verdict-conf {
    font-size     : 0.95rem;
    color         : var(--text-muted);
    margin-top    : 0.3rem;
    letter-spacing: 0.08em;
  }

  /* Confidence bar */
  .conf-bar-outer {
    background    : #1a2035;
    border-radius : 4px;
    height        : 10px;
    width         : 100%;
    margin-top    : 0.6rem;
  }
  .conf-bar-inner-real {
    background    : linear-gradient(90deg, #00a855, #00ff88);
    height        : 100%;
    border-radius : 4px;
    transition    : width 0.4s ease;
  }
  .conf-bar-inner-fake {
    background    : linear-gradient(90deg, #c0001e, #ff2b4e);
    height        : 100%;
    border-radius : 4px;
    transition    : width 0.4s ease;
  }

  /* Metric chips */
  .chip-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.6rem; }
  .chip {
    background    : var(--bg-panel);
    border        : 1px solid var(--border);
    border-radius : 20px;
    padding       : 0.25rem 0.9rem;
    font-family   : var(--font-mono);
    font-size     : 0.80rem;
    color         : var(--accent-cyan);
  }
  .chip-label { color: var(--text-muted); font-size: 0.72rem; display: block; }
  .chip-val   { color: var(--text-primary); font-weight: 600; }

  /* Upload zone */
  .stFileUploader > div { border: 1px dashed var(--border) !important; border-radius: 8px !important; }
  .stFileUploader label { color: var(--text-muted) !important; }

  /* Buttons */
  .stButton > button {
    background    : transparent !important;
    border        : 1px solid var(--accent-cyan) !important;
    color         : var(--accent-cyan) !important;
    font-family   : var(--font-mono) !important;
    letter-spacing: 0.1em !important;
    border-radius : 6px !important;
    padding       : 0.5rem 1.6rem !important;
    transition    : all 0.2s !important;
  }
  .stButton > button:hover {
    background    : #00d4ff18 !important;
    box-shadow    : 0 0 14px #00d4ff44 !important;
  }

  /* Dividers */
  hr { border-color: var(--border) !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-primary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# Model loading  (cached – loads once per session)
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load deepfake_model.pkl. Returns (pipeline, error_message)."""
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file '{MODEL_PATH}' not found.\n"
            "Run: python train_model.py"
        )
    try:
        with open(MODEL_PATH, "rb") as fh:
            pipeline = pickle.load(fh)
        return pipeline, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load training_metrics.json if present."""
    if not os.path.exists(METRICS_PATH):
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ──────────────────────────────────────────────────────────────────
# Language detection via faster-whisper
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_whisper():
    if not WHISPER_OK:
        return None
    try:
        return WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception:
        return None


def detect_language(audio_path: str) -> str:
    """
    Return detected language string ('Hindi', 'English', 'Hinglish', …)
    Falls back to 'Unknown' if Whisper is unavailable.
    """
    model = _load_whisper()
    if model is None:
        return "Unknown (Whisper unavailable)"
    try:
        segments, info = model.transcribe(audio_path, beam_size=1, language=None)
        lang_code = info.language
        mapping = {
            "hi": "Hindi",
            "en": "English",
            "ur": "Urdu",
        }
        detected = mapping.get(lang_code, lang_code.upper() if lang_code else "Unknown")
        # Hinglish heuristic: if transcription mixes Devanagari & Latin
        text = " ".join(seg.text for seg in segments)
        has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in text)
        has_latin      = any("a" <= ch.lower() <= "z" for ch in text)
        if has_devanagari and has_latin:
            detected = "Hinglish"
        return detected
    except Exception as exc:
        return f"Error ({exc})"


# ──────────────────────────────────────────────────────────────────
# Prediction  (fast – loads audio once, reuses preprocessed signal)
# ──────────────────────────────────────────────────────────────────

def run_prediction(audio_path: str, pipeline) -> dict:
    """
    Returns a dict with all detection results.
    Never raises – errors are captured and returned in 'error' key.
    """
    result = {
        "label"    : None,
        "confidence": None,
        "prob_real": None,
        "prob_fake": None,
        "gender"   : "Unknown",
        "language" : "Unknown",
        "duration" : None,
        "error"    : None,
    }
    try:
        y, sr = load_and_preprocess(audio_path)

        if not is_valid_audio(y):
            result["error"] = "Audio is too short or silent after preprocessing."
            return result

        result["duration"] = round(len(y) / sr, 2)
        result["gender"]   = detect_gender(y, sr)

        # Feature extraction + prediction
        feat      = extract_features(y, sr).reshape(1, -1)
        proba     = pipeline.predict_proba(feat)[0]
        pred      = int(np.argmax(proba))

        result["prob_real"]   = float(proba[0])
        result["prob_fake"]   = float(proba[1])
        result["confidence"]  = float(max(proba))
        result["label"]       = "REAL" if pred == 0 else "FAKE"

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ──────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────

def _conf_badge(conf: float) -> str:
    if conf >= HIGH_CONF:
        return "HIGH CONFIDENCE"
    elif conf >= MED_CONF:
        return "MEDIUM CONFIDENCE"
    return "LOW CONFIDENCE"


def _render_verdict(result: dict):
    label = result["label"]
    conf  = result["confidence"]
    is_real = (label == "REAL")

    card_cls  = "verdict-real"  if is_real else "verdict-fake"
    color     = "#00ff88"       if is_real else "#ff2b4e"
    icon      = "✅"            if is_real else "🚨"
    bar_cls   = "conf-bar-inner-real" if is_real else "conf-bar-inner-fake"

    st.markdown(f"""
    <div class="{card_cls}">
      <div class="verdict-label" style="color:{color}">{icon} {label}</div>
      <div class="verdict-conf">{_conf_badge(conf)} &nbsp;|&nbsp; {conf*100:.1f}% confidence</div>
      <div class="conf-bar-outer">
        <div class="{bar_cls}" style="width:{conf*100:.1f}%"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("<div class='bk-card' style='margin-top:1rem'>", unsafe_allow_html=True)
    st.markdown("<div class='bk-card-title'>◈ PROBABILITY BREAKDOWN</div>", unsafe_allow_html=True)

    col_r, col_f = st.columns(2)
    with col_r:
        st.markdown(
            f"<div style='font-family:var(--font-mono);color:#00ff88;font-size:0.85rem'>"
            f"REAL VOICE<br>"
            f"<span style='font-size:1.8rem;font-weight:bold'>{result['prob_real']*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(result["prob_real"])
    with col_f:
        st.markdown(
            f"<div style='font-family:var(--font-mono);color:#ff2b4e;font-size:0.85rem'>"
            f"FAKE / AI<br>"
            f"<span style='font-size:1.8rem;font-weight:bold'>{result['prob_fake']*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(result["prob_fake"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Metadata chips
    chips_html = "<div class='chip-row'>"
    chips = {
        "GENDER"   : result.get("gender", "Unknown"),
        "LANGUAGE" : result.get("language", "Unknown"),
        "DURATION" : f"{result.get('duration', 0):.2f}s",
    }
    for k, v in chips.items():
        chips_html += (
            f"<div class='chip'>"
            f"<span class='chip-label'>{k}</span>"
            f"<span class='chip-val'>{v}</span>"
            f"</div>"
        )
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)


def _render_metrics_sidebar(metrics: dict):
    if not metrics:
        st.sidebar.markdown("_No training metrics found._")
        return

    st.sidebar.markdown("### 📊 Model Performance")
    kv = [
        ("Model",    metrics.get("model", "RF")),
        ("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%"),
        ("F1 Score", f"{metrics.get('f1_score', 0)*100:.2f}%"),
        ("ROC AUC",  f"{metrics.get('roc_auc', 0)*100:.2f}%"),
        ("Trained",  str(metrics.get("training_date", ""))[:10]),
    ]
    for k, v in kv:
        st.sidebar.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"font-family:monospace;font-size:0.82rem;margin:0.25rem 0;"
            f"color:#8899aa'>"
            f"<span>{k}</span><span style='color:#00d4ff'>{v}</span></div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────

def main():
    # ── Header ────────────────────────────────────────────────
    st.markdown("""
    <div class='bk-header'>
      <div class='bk-logo'>🛡 BHASHAKAVACH</div>
      <div class='bk-sub'>AI Voice Deepfake Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────
    pipeline, model_error = load_model()
    metrics               = load_metrics()

    if model_error:
        st.error(f"⚠ {model_error}")
        st.info("Train the model first by running: `python train_model.py`")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙ System Status")
        st.markdown(
            f"<div style='color:#00ff88;font-family:monospace;font-size:0.85rem'>"
            f"✓ Model loaded<br>"
            f"{'✓ Whisper ready' if WHISPER_OK else '○ Whisper off (pip install faster-whisper)'}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        _render_metrics_sidebar(metrics)

    # ── Two-column layout ─────────────────────────────────────
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown("<div class='bk-card'>", unsafe_allow_html=True)
        st.markdown("<div class='bk-card-title'>◈ UPLOAD AUDIO</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop a voice recording",
            type=SUPPORTED_FORMATS,
            label_visibility="collapsed",
        )

        if uploaded:
            st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
            st.markdown(
                f"<div style='font-family:monospace;font-size:0.8rem;color:#5a7a9a;margin-top:0.5rem'>"
                f"File: {uploaded.name}  |  {uploaded.size/1024:.1f} KB"
                f"</div>",
                unsafe_allow_html=True,
            )

        analyze_btn = st.button("⚡  ANALYZE", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # How it works
        with st.expander("ℹ How it works", expanded=False):
            st.markdown("""
**BhashaKavach** uses a 151-dimensional acoustic feature vector:
- **MFCC** (Mel-Frequency Cepstral Coefficients) + Deltas
- **Spectral Contrast** & **Chroma** features
- **YIN Pitch** analysis (F0, jitter, voiced fraction)
- **Spectral Rolloff & Centroid** (speech quality markers)

A **RandomForest classifier** (or XGBoost if available) was trained on real vs. AI-generated voices and is used for inference.
""")

    with col_result:
        if not uploaded:
            st.markdown("""
            <div style='text-align:center;padding:4rem 2rem;color:#2a3a5a;
                        border:1px dashed #1e2d4a;border-radius:8px;'>
              <div style='font-size:3rem'>🎙</div>
              <div style='font-family:monospace;font-size:0.85rem;margin-top:1rem'>
                Upload an audio file to begin analysis
              </div>
            </div>
            """, unsafe_allow_html=True)

        elif analyze_btn:
            with st.spinner("Analysing audio …"):
                # Write to temp file (librosa needs a path)
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                result = run_prediction(tmp_path, pipeline)

                # Language detection (Whisper, if available)
                result["language"] = detect_language(tmp_path)

                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if result["error"]:
                st.error(f"Analysis failed: {result['error']}")
            else:
                _render_verdict(result)

                # Downloadable JSON report
                report_json = json.dumps(
                    {
                        "file"      : uploaded.name,
                        "verdict"   : result["label"],
                        "confidence": result["confidence"],
                        "prob_real" : result["prob_real"],
                        "prob_fake" : result["prob_fake"],
                        "gender"    : result["gender"],
                        "language"  : result["language"],
                        "duration"  : result["duration"],
                    },
                    indent=4,
                )
                st.download_button(
                    "⬇  Download Report (JSON)",
                    data       = report_json,
                    file_name  = f"bhashakavach_{uploaded.name.rsplit('.', 1)[0]}.json",
                    mime       = "application/json",
                    use_container_width = True,
                )


if __name__ == "__main__":
    main()
