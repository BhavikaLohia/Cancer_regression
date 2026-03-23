# ============================================================
#  app.py  –  Cancer Risk Classifier  |  Streamlit GUI
#  Usage:  streamlit run app.py
#  Requires: model.pkl downloaded from Google Colab
#            Place model.pkl in the SAME folder as app.py
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import streamlit as st

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Risk Classifier",
    page_icon="🧬",
    layout="wide",
)

# ── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f172a; color: #e2e8f0; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2137 100%);
        border: 1px solid #1e40af;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 28px;
    }
    .main-header h1 { color: #ffffff; font-size: 1.8rem; margin: 0; }
    .main-header p  { color: #94a3b8; margin: 6px 0 0; font-size: 0.95rem; }

    /* Section headings */
    .section-title {
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px;
        margin-bottom: 4px;
    }

    /* Result cards */
    .result-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.04));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-high h2 { color: #f87171; font-size: 2rem; margin: 0; }
    .result-low  h2 { color: #34d399; font-size: 2rem; margin: 0; }
    .result-high p, .result-low p { color: #cbd5e1; margin: 8px 0 0; }

    /* Metric boxes */
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-box .m-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .metric-box .m-value { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; margin-top: 4px; }

    /* Disclaimer */
    .disclaimer {
        background: #1e293b;
        border-left: 3px solid #475569;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 16px;
        line-height: 1.6;
    }

    /* Streamlit widget label overrides */
    label { color: #94a3b8 !important; font-size: 0.88rem !important; }
    .stNumberInput > div > div > input { background: #1e293b !important; color: #e2e8f0 !important; border-color: #334155 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6);
        color: white; border: none; font-weight: 600;
        border-radius: 8px; padding: 10px 28px;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #2563eb, #60a5fa); }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL (plain Colab pickle) ─────────────────────────
# Your Colab saved ONLY the model:  pickle.dump(model, open("model.pkl","wb"))
# So we load it directly, then rebuild the scaler & imputer
# from cancer_reg.csv (same preprocessing your notebook used).

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "cancer_reg.csv")

MEDIAN_VAL   = 178.1   # fallback if CSV not present

@st.cache_resource
def load_everything():
    # ── 1. Load raw model from Colab pickle ──
    with open(MODEL_PATH, "rb") as f:
        content = pickle.load(f)

    # Handle both plain model and bundle dict (just in case)
    if isinstance(content, dict) and "model" in content:
        model        = content["model"]
        scaler       = content.get("scaler")
        imputer      = content.get("imputer")
        feature_cols = content.get("feature_cols")
        median_val   = content.get("median_val", MEDIAN_VAL)
        # If bundle is complete, return immediately
        if all(v is not None for v in [scaler, imputer, feature_cols]):
            return model, scaler, imputer, feature_cols, median_val
    else:
        model = content

    # ── 2. Rebuild scaler + imputer from cancer_reg.csv ──
    df = pd.read_csv(DATASET_PATH)

    # Drop non-numeric / target columns — NO cfr added here.
    # The Colab model was trained on the original 30 columns only.
    drop_cols    = ["Geography", "binnedInc", "target_deathrate"]
    feature_cols = [c for c in df.columns
                    if c not in drop_cols and df[c].dtype != object]

    median_val = df["target_deathrate"].median()

    X = df[feature_cols].values

    imputer = KNNImputer(n_neighbors=5)
    X_imp   = imputer.fit_transform(X)

    scaler  = RobustScaler()
    scaler.fit(X_imp)

    return model, scaler, imputer, feature_cols, median_val

# ── Try loading ──────────────────────────────────────────────
model_loaded   = False
missing_csv    = False
load_error_msg = ""

if not os.path.exists(MODEL_PATH):
    load_error_msg = f"model.pkl not found at: `{MODEL_PATH}`"
elif not os.path.exists(DATASET_PATH):
    # Load only the model; skip scaler rebuild — use identity transforms
    try:
        with open(MODEL_PATH, "rb") as f:
            raw = pickle.load(f)
        model        = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        scaler       = None
        imputer      = None
        feature_cols = list(DEFAULTS.keys())   # defined below
        model_loaded = True
        missing_csv  = True
    except Exception as e:
        load_error_msg = str(e)
else:
    try:
        model, scaler, imputer, feature_cols, MEDIAN_VAL = load_everything()
        model_loaded = True
    except Exception as e:
        load_error_msg = str(e)

# ── HEADER ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧬 Cancer Risk Classifier</h1>
  <p>Predicts whether a county's cancer death-rate is <strong>High Risk</strong> or <strong>Low Risk</strong>
     based on demographic, socioeconomic, and healthcare indicators.<br>
     Dataset: <em>cancer_reg.csv</em> (Kaggle) · Model: HistGradientBoosting</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️  Could not load model. {load_error_msg}\n\n"
             "**Fix:** Place `model.pkl` (downloaded from Colab) in the **same folder** as `app.py`, "
             "and also place `cancer_reg.csv` there for scaling.")
    st.stop()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Options")
    load_example = st.button("📋 Load Example Values")
    st.markdown("---")
    st.caption(f"**Median threshold:** {MEDIAN_VAL:.2f} deaths / 100k")
    st.caption(f"**Features:** {len(feature_cols)}")
    if missing_csv:
        st.warning("⚠️ `cancer_reg.csv` not found — scaler/imputer skipped. Predictions may differ slightly.")

# Example defaults (median US county)
DEFAULTS = {
    "avganncount":              1397.0,
    "avgdeathsperyear":          469.0,
    "incidencerate":             489.8,
    "medincome":               45207.0,
    "popest2015":             164171.0,
    "povertypercent":             19.6,
    "studypercap":               152.0,
    "binnedinc":                   0.0,
    "medianage":                  39.5,
    "medianagemale":              37.8,
    "medianagefemale":            41.2,
    "geography":                   0.0,
    "percentmarried":             50.2,
    "pctnohs18_24":               17.4,
    "pcths18_24":                 34.8,
    "pctsomecol18_24":            38.2,
    "pctbachdeg18_24":             5.3,
    "pcths25over":                33.7,
    "pctbachdeg25over":           14.5,
    "pctemployed16_over":         55.8,
    "pctunemployed16_over":        8.4,
    "pctprivatecoverage":         60.4,
    "pctprivatecoveragealone":    47.1,
    "pctempprivcoverage":         41.2,
    "pctpubliccoverage":          35.7,
    "pctpubliccoveragealone":     18.6,
    "pctwhite":                   82.6,
    "pctblack":                    8.2,
    "pctasian":                    1.1,
    "pctotherrace":                1.8,
    "pctmarriedhouseholds":       50.8,
    "birthrate":                   5.5,
    "percapitainc":            23694.0,
}

def get_default(col):
    return DEFAULTS.get(col.lower(), 0.0)

# ── INPUT FORM ──────────────────────────────────────────────
st.markdown("### 📝 Enter County Data")

inputs = {}

# Group features into logical sections for readability
sections = {
    "📊 Incidence & Mortality": [
        "avganncount", "avgdeathsperyear", "incidencerate"
    ],
    "👥 Demographics": [
        "popest2015", "medianage", "medianagemale", "medianagefemale",
        "pctwhite", "pctblack", "pctasian", "pctotherrace",
        "birthrate", "percentmarried", "pctmarriedhouseholds"
    ],
    "💰 Socioeconomic": [
        "medincome", "percapitainc", "povertypercent", "studypercap"
    ],
    "🎓 Education": [
        "pctnohs18_24", "pcths18_24", "pctsomecol18_24", "pctbachdeg18_24",
        "pcths25over", "pctbachdeg25over"
    ],
    "💼 Employment": [
        "pctemployed16_over", "pctunemployed16_over"
    ],
    "🏥 Healthcare Coverage": [
        "pctprivatecoverage", "pctprivatecoveragealone", "pctempprivcoverage",
        "pctpubliccoverage", "pctpubliccoveragealone"
    ],
}

# Remaining features not in any group
grouped = [f for grp in sections.values() for f in grp]
remaining = [c for c in feature_cols if c.lower() not in grouped]

with st.form("prediction_form"):

    for section_name, fields in sections.items():
        st.markdown(f'<div class="section-title">{section_name}</div>', unsafe_allow_html=True)
        visible = [c for c in feature_cols if c.lower() in [f.lower() for f in fields]]

        if visible:
            cols = st.columns(3)
            for i, col_name in enumerate(visible):
                default_val = get_default(col_name)
                if load_example:
                    default_val = get_default(col_name)
                with cols[i % 3]:
                    inputs[col_name] = st.number_input(
                        label=col_name,
                        value=float(default_val),
                        format="%.2f",
                        key=col_name,
                    )

    # Any leftover features
    if remaining:
        st.markdown('<div class="section-title">🔧 Other Features</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, col_name in enumerate(remaining):
            with cols[i % 3]:
                inputs[col_name] = st.number_input(
                    label=col_name,
                    value=float(get_default(col_name)),
                    format="%.2f",
                    key=col_name,
                )

    st.markdown("---")
    submitted = st.form_submit_button("⚡  Predict Risk", use_container_width=True)

# ── PREDICTION ──────────────────────────────────────────────
if submitted:
    # Build feature vector in the exact order the model was trained on
    row = np.array([[inputs.get(c, get_default(c)) for c in feature_cols]], dtype=float)

    # Apply imputer → scaler → model (skip if not available)
    if imputer is not None:
        row = imputer.transform(row)
    if scaler is not None:
        row = scaler.transform(row)
    prediction = model.predict(row)[0]

    # Probability if model supports it
    try:
        proba = model.predict_proba(row)[0]
        confidence = float(max(proba)) * 100
        risk_score = float(proba[1]) * 100
    except AttributeError:
        confidence = None
        risk_score = None

    # Derived CFR — computed from inputs for display only (not a model feature)
    cfr_val = inputs.get("avgdeathsperyear", 0) / (inputs.get("avganncount", 1) + 1)

    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    # Result card
    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
            <h2>⚠️ HIGH RISK</h2>
            <p>This county profile indicates a cancer death-rate <strong>above</strong> the national median ({MEDIAN_VAL:.1f} per 100k).</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <h2>✅ LOW RISK</h2>
            <p>This county profile indicates a cancer death-rate <strong>below</strong> the national median ({MEDIAN_VAL:.1f} per 100k).</p>
        </div>
        """, unsafe_allow_html=True)

    # Metric row
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="m-label">Classification</div>
            <div class="m-value" style="color:{'#f87171' if prediction==1 else '#34d399'}">
                {'High Risk' if prediction==1 else 'Low Risk'}
            </div>
        </div>""", unsafe_allow_html=True)

    with m2:
        val = f"{risk_score:.1f}%" if risk_score is not None else "N/A"
        st.markdown(f"""
        <div class="metric-box">
            <div class="m-label">Risk Score</div>
            <div class="m-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        val = f"{confidence:.1f}%" if confidence is not None else "N/A"
        st.markdown(f"""
        <div class="metric-box">
            <div class="m-label">Confidence</div>
            <div class="m-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="m-label">Case Fatality Rate</div>
            <div class="m-value">{cfr_val:.3f}</div>
        </div>""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This prediction is generated by a machine learning model trained on the
        <em>cancer_reg</em> dataset (Kaggle). It is intended for <strong>educational and research purposes only</strong>
        and should not be used for any clinical or medical decision-making.
        Consult qualified healthcare professionals for guidance.
    </div>
    """, unsafe_allow_html=True)
