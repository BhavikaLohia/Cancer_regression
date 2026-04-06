import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Cancer Risk Screener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

/* ── Base reset ────────────────────────────────────────────────────────── */
html, body, [class*="css"], [data-testid="stApp"],
[data-testid="stAppViewContainer"], .main {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #f5f3ef !important;
    color: #1c1c2e !important;
}
[data-testid="stHeader"]         { display: none !important; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 0 3rem !important; max-width: 100% !important; }

/* ── Hero banner ────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(120deg, #1c1c2e 0%, #2d4a47 60%, #1c3a36 100%);
    padding: 3.2rem max(5vw, 2rem) 3rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 80% 50%, rgba(45,122,111,0.25) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(255,180,100,0.1) 0%, transparent 60%);
    pointer-events: none;
}
.hero-inner {
    position: relative;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1.5rem;
}
.hero-brand {
    display: flex; align-items: center; gap: 0.55rem;
    margin-bottom: 1rem;
}
.hero-dot {
    width: 9px; height: 9px;
    background: #4ecdc4;
    border-radius: 50%;
    box-shadow: 0 0 0 3px rgba(78,205,196,0.25);
}
.hero-brand-name {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #4ecdc4;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 3.5vw, 3rem);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.15;
    margin-bottom: 0.6rem;
}
.hero-title em { font-style: italic; color: #8de8de; }
.hero-sub {
    font-size: 0.92rem; color: rgba(255,255,255,0.55);
    line-height: 1.65; max-width: 520px;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 99px;
    padding: 0.45rem 1.1rem;
    font-size: 0.78rem; font-weight: 600; color: #c8f5f2;
    white-space: nowrap; align-self: flex-start; margin-top: 0.4rem;
}
.hero-badge-dot { width: 6px; height: 6px; background: #4ecdc4; border-radius: 50%; }

/* ── Main content wrapper ────────────────────────────────────────────────── */
.content-wrap {
    padding: 2.8rem max(5vw, 1.5rem) 4rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* ── Section label ────────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #a09e9e; margin-bottom: 0.9rem; margin-top: 2rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-label::after {
    content: ''; flex: 1; height: 1px; background: #e4e0d8;
}

/* ── Input cards ──────────────────────────────────────────────────────────── */
.input-card {
    background: #ffffff;
    border: 1px solid #e8e4dc;
    border-radius: 18px;
    padding: 1.8rem 2.2rem 1.6rem;
    box-shadow: 0 2px 16px rgba(28,28,46,0.05);
    margin-bottom: 1.2rem;
}
.input-card-title {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #b0aeae; margin-bottom: 1.1rem;
}

/* ── Widget overrides ─────────────────────────────────────────────────────── */
[data-testid="stWidgetLabel"] p, label,
.stSelectbox label, .stNumberInput label {
    color: #3d3d55 !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
input[type="number"] {
    background: #faf9f7 !important;
    border: 1.5px solid #e4e0d8 !important;
    border-radius: 10px !important;
    color: #1c1c2e !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.8rem !important;
}
input[type="number"]:focus {
    border-color: #2d7a6f !important;
    box-shadow: 0 0 0 3px rgba(45,122,111,0.12) !important;
    outline: none !important;
}
.stSelectbox > div > div {
    background: #faf9f7 !important;
    border: 1.5px solid #e4e0d8 !important;
    border-radius: 10px !important;
    color: #1c1c2e !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── Predict button ───────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1c1c2e 0%, #2d4a47 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.85rem 2rem !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2d4a47 0%, #1c6b61 100%) !important;
    box-shadow: 0 8px 24px rgba(45,122,111,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Result section ───────────────────────────────────────────────────────── */
.result-high {
    background: linear-gradient(135deg, #fff5f5 0%, #ffeded 100%);
    border: 2px solid #fca5a5;
    border-radius: 20px;
    padding: 2.8rem 2.8rem;
    display: flex; align-items: center; gap: 2rem;
    box-shadow: 0 8px 32px rgba(239,68,68,0.1);
    margin-bottom: 1.4rem;
}
.result-low {
    background: linear-gradient(135deg, #f0fdf8 0%, #e5faf3 100%);
    border: 2px solid #6ee7c0;
    border-radius: 20px;
    padding: 2.8rem 2.8rem;
    display: flex; align-items: center; gap: 2rem;
    box-shadow: 0 8px 32px rgba(16,185,129,0.1);
    margin-bottom: 1.4rem;
}
.result-icon-wrap {
    width: 80px; height: 80px; flex-shrink: 0;
    border-radius: 50%; display: flex;
    align-items: center; justify-content: center;
    font-size: 2.2rem;
}
.result-high .result-icon-wrap { background: rgba(239,68,68,0.1); }
.result-low  .result-icon-wrap { background: rgba(16,185,129,0.1); }
.result-text {}
.result-eyebrow {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.result-high .result-eyebrow { color: #dc2626; }
.result-low  .result-eyebrow { color: #059669; }
.result-headline {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem; font-weight: 700; line-height: 1.1;
    margin-bottom: 0.5rem;
}
.result-high .result-headline { color: #991b1b; }
.result-low  .result-headline { color: #065f46; }
.result-desc { font-size: 0.88rem; color: #6b7280; line-height: 1.6; max-width: 440px; }

/* ── Probability card ─────────────────────────────────────────────────────── */
.prob-card {
    background: #ffffff;
    border: 1px solid #e8e4dc;
    border-radius: 18px;
    padding: 2rem 2.4rem;
    box-shadow: 0 2px 16px rgba(28,28,46,0.05);
    margin-bottom: 1.2rem;
}
.prob-card-title {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #b0aeae; margin-bottom: 1.2rem;
}
.pbar-wrap { margin-bottom: 1.1rem; }
.pbar-row {
    display: flex; justify-content: space-between;
    align-items: center; margin-bottom: 6px;
}
.pbar-label { font-size: 0.86rem; font-weight: 500; color: #3d3d55; }
.pbar-value { font-size: 0.92rem; font-weight: 700; color: #1c1c2e; }
.pbar-bg { background: #f0ede8; border-radius: 99px; height: 8px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 99px; }

/* ── Disclaimer ───────────────────────────────────────────────────────────── */
.disclaimer {
    background: #fffceb;
    border: 1px solid #fde68a;
    border-radius: 12px;
    padding: 1.1rem 1.6rem;
    font-size: 0.8rem; color: #92400e; line-height: 1.6;
}

/* ── Empty state ──────────────────────────────────────────────────────────── */
.empty-state {
    background: #ffffff;
    border: 1.5px dashed #d8d4cc;
    border-radius: 20px;
    padding: 5rem 3rem;
    text-align: center;
}
.empty-icon  { font-size: 3rem; margin-bottom: 1rem; opacity: 0.35; }
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; color: #9a98a8; margin-bottom: 0.5rem;
}
.empty-sub { font-size: 0.84rem; color: #b8b5c5; line-height: 1.6; }

/* ── Column gap fix ───────────────────────────────────────────────────────── */
[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
div[data-testid="column"] > div   { padding: 0 !important; }

/* scrollbar */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: #f5f3ef; }
::-webkit-scrollbar-thumb { background: #ccc8c0; border-radius: 99px; }
</style>
""", unsafe_allow_html=True)


# ── Load pickles ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open("models.pkl", "rb") as f:
            models = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler_bundle = pickle.load(f)
        return models, scaler_bundle, None
    except FileNotFoundError as e:
        return None, None, str(e)

models, scaler_bundle, err = load_artifacts()

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div>
      <div class="hero-brand">
        <div class="hero-dot"></div>
        <div class="hero-brand-name">OncoScreen AI</div>
      </div>
      <div class="hero-title">Cancer Risk<br><em>Assessment Tool</em></div>
      <div class="hero-sub">
        Complete the patient profile below to receive an instant AI-powered
        cancer risk evaluation based on clinical and lifestyle indicators.
      </div>
    </div>
    <div class="hero-badge">
      <div class="hero-badge-dot"></div>
      ⭐ Random Forest · Best Model
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

if err:
    st.error(f"**Model files not found:** `{err}`\n\nPlace `models.pkl` and `scaler.pkl` in the same folder as `app.py`.")
    st.stop()

feature_names = scaler_bundle['feature_names']
MODEL_NAME    = "Random Forest"
selected      = models[MODEL_NAME]

vals = {}

# ── Row 1: Demographics ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    vals['Age'] = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
with c2:
    gender_map = {"Male": 1, "Female": 0}
    vals['Gender'] = gender_map[st.selectbox("Gender", list(gender_map.keys()))]
with c3:
    vals['BMI'] = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")
with c4:
    vals['PhysicalActivity'] = st.number_input("Physical Activity (hrs/wk)", min_value=0.0, max_value=20.0, value=3.0, step=0.5, format="%.1f")

# ── Row 2: Risk Factors ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Risk Factors</div>', unsafe_allow_html=True)
c5, c6, c7, c8 = st.columns(4)
with c5:
    smoking_map = {"Non-Smoker": 0, "Smoker": 1}
    vals['Smoking'] = smoking_map[st.selectbox("Smoking Status", list(smoking_map.keys()))]
with c6:
    vals['AlcoholIntake'] = st.number_input("Alcohol Intake (units/wk)", min_value=0.0, max_value=40.0, value=2.0, step=0.5, format="%.1f")
with c7:
    genetic_map = {"Low": 0, "Medium": 1, "High": 2}
    vals['GeneticRisk'] = genetic_map[st.selectbox("Genetic Risk", list(genetic_map.keys()), index=1)]
with c8:
    history_map = {"No": 0, "Yes": 1}
    vals['CancerHistory'] = history_map[st.selectbox("Family Cancer History", list(history_map.keys()))]

# Extra features if any
shown  = {'Age','Gender','BMI','Smoking','GeneticRisk','CancerHistory','PhysicalActivity','AlcoholIntake'}
extras = [f for f in feature_names if f not in shown]
if extras:
    st.markdown('<div class="section-label">Additional</div>', unsafe_allow_html=True)
    ecols = st.columns(min(len(extras), 4))
    for i, feat in enumerate(extras):
        with ecols[i % 4]:
            vals[feat] = st.number_input(feat, value=0.0, format="%.2f")

# ── Predict button ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("Run Risk Assessment →", use_container_width=True, type="primary")

# ── Divider ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

# ── Results ────────────────────────────────────────────────────────────────
if predict_btn:
    X_in = pd.DataFrame([[vals.get(f, 0) for f in feature_names]], columns=feature_names)

    if selected['needs_scaling']:
        X_in = pd.DataFrame(
            scaler_bundle['scaler'].transform(X_in),
            columns=feature_names
        )

    pred   = selected['model'].predict(X_in)[0]
    probs  = selected['model'].predict_proba(X_in)[0]
    p_high = probs[1]
    p_low  = probs[0]

    # Result banner — full width
    if pred == 1:
        st.markdown(f"""
        <div class="result-high">
            <div class="result-icon-wrap">⚠️</div>
            <div class="result-text">
                <div class="result-eyebrow">Assessment Result</div>
                <div class="result-headline">High Risk</div>
                <div class="result-desc">
                    This patient profile shows elevated indicators associated with cancer risk.
                    Further clinical evaluation and specialist referral is strongly advised.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <div class="result-icon-wrap">✅</div>
            <div class="result-text">
                <div class="result-eyebrow">Assessment Result</div>
                <div class="result-headline">Low Risk</div>
                <div class="result-desc">
                    No significant cancer risk indicators detected in this patient profile.
                    Routine check-ups and a healthy lifestyle are recommended.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Probability bars — two columns
    pb1, pb2 = st.columns(2)
    with pb1:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-card-title">Low Risk Probability</div>
            <div class="pbar-wrap">
                <div class="pbar-row">
                    <span class="pbar-label">Low Risk</span>
                    <span class="pbar-value">{p_low:.1%}</span>
                </div>
                <div class="pbar-bg">
                    <div class="pbar-fill" style="width:{p_low*100:.1f}%; background: linear-gradient(90deg,#10b981,#34d399);"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
    with pb2:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-card-title">High Risk Probability</div>
            <div class="pbar-wrap">
                <div class="pbar-row">
                    <span class="pbar-label">High Risk</span>
                    <span class="pbar-value">{p_high:.1%}</span>
                </div>
                <div class="pbar-bg">
                    <div class="pbar-fill" style="width:{p_high*100:.1f}%; background: linear-gradient(90deg,#ef4444,#f87171);"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Educational use only.</strong> This tool does not constitute medical advice and
        should not replace professional clinical judgement. Always consult a qualified healthcare
        professional for diagnosis and treatment decisions.
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔬</div>
        <div class="empty-title">Awaiting Assessment</div>
        <div class="empty-sub">
            Complete the patient profile above and click<br>
            <strong>Run Risk Assessment</strong> to see results here.
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
