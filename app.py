import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Insurance Cost Estimator",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "insurance_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.pkl")

# Reference costs (illustrative)
REF_AVG_NON_SMOKER = 8400
REF_AVG_SMOKER = 32000

# ---------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&display=swap');

:root {
  --ink: #12303B;
  --muted: #5B7078;
  --paper: #F3F6F5;
  --card: #FFFFFF;
  --line: #D5DEDB;
  --teal: #0F7B6C;
  --teal-soft: #DDF0EC;
  --amber: #B86E00;
  --amber-soft: #FBEBD2;
}

html, body, [class*="css"], .stApp { font-family: 'Figtree', sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

h1, h2, h3, .display { font-family: 'Bricolage Grotesque', sans-serif !important; color: var(--ink); letter-spacing: -0.02em; }

/* Page heading */
.page-title { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.3rem; font-weight: 700; margin: 0; letter-spacing: -0.03em; }
.page-sub { color: var(--muted); font-size: 1.02rem; margin: .35rem 0 1.6rem; max-width: 60ch; }

/* Form panel */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card); border: 1px solid var(--line) !important;
  border-radius: 16px; padding: .6rem .5rem;
}
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .92rem; color: var(--ink); }
div[data-baseweb="select"] > div { border-radius: 10px; border-color: var(--line); background: var(--paper); }
[data-testid="stSlider"] [role="slider"] { background: var(--teal); box-shadow: none; }
[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: var(--teal); font-weight: 600; }

/* Button */
.stButton > button {
  background: var(--teal); color: #fff; border: 0; border-radius: 12px;
  padding: .75rem 1rem; font-weight: 600; font-size: 1rem;
  transition: background .15s ease, transform .1s ease;
}
.stButton > button:hover { background: #0B655A; color: #fff; }
.stButton > button:active { transform: scale(.99); }
.stButton > button:focus-visible { outline: 3px solid #7CC9BD; outline-offset: 2px; }

/* Result */
.result { background: var(--ink); color: #fff; border-radius: 16px; padding: 1.6rem 1.8rem; }
.result .label { color: #9FC3BC; font-size: .95rem; margin-bottom: .2rem; }
.result .amount { font-family: 'Bricolage Grotesque', sans-serif; font-size: 3.6rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1.05; }
.result .delta { margin-top: .6rem; font-size: .98rem; color: #D6E7E3; }

.panel { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 1.3rem 1.5rem; margin-top: 1rem; }
.panel h4 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem; margin: 0 0 1rem; }

.bar-row { margin-bottom: .9rem; }
.bar-head { display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: .3rem; }
.bar-head span:last-child { font-weight: 600; }
.track { background: var(--paper); border-radius: 999px; height: 12px; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.fill.you { background: var(--teal); }
.fill.ref { background: #9DB3B0; }

.chips { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .2rem; }
.chip { background: var(--teal-soft); color: #0A5A4E; padding: .3rem .75rem; border-radius: 999px; font-size: .86rem; font-weight: 500; }
.chip.warn { background: var(--amber-soft); color: var(--amber); }

.note { color: var(--muted); font-size: .85rem; margin-top: 1rem; }

/* Empty state */
.empty { border: 1.5px dashed var(--line); border-radius: 16px; padding: 3rem 1.8rem; text-align: center; color: var(--muted); background: rgba(255,255,255,.5); }
.empty .display { font-size: 1.25rem; color: var(--ink); margin-bottom: .4rem; }

@media (prefers-reduced-motion: reduce) { .stButton > button { transition: none; } }
@media (max-width: 640px) { .result .amount { font-size: 2.6rem; } .page-title { font-size: 1.8rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    return "obese"


def build_input_row(age, sex, bmi, children, smoker, region, feature_cols):
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "bmi_category": bmi_category(bmi),
    }
    return pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)


def comparison_bars(pred, ref, ref_label):
    top = max(pred, ref) * 1.05
    return f"""
    <div class="panel">
      <h4>How it compares</h4>
      <div class="bar-row">
        <div class="bar-head"><span>Your estimate</span><span>${pred:,.0f}</span></div>
        <div class="track"><div class="fill you" style="width:{pred / top * 100:.1f}%"></div></div>
      </div>
      <div class="bar-row">
        <div class="bar-head"><span>{ref_label}</span><span>${ref:,.0f}</span></div>
        <div class="track"><div class="fill ref" style="width:{ref / top * 100:.1f}%"></div></div>
      </div>
      <p class="note">The average is an illustrative reference value, not a personalised insurance quote.</p>
    </div>
    """


# ---------------------------------------------------------------- page
model, feature_cols = load_artifacts()

st.markdown('<h1 class="page-title">Insurance Cost Estimator</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-sub">Fill in the profile on the left to see the estimated yearly '
    "insurance charges and how they compare to a typical policyholder.</p>",
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "Model files not found. Add insurance_model.pkl and model_features.pkl "
        "next to this script, then reload the page."
    )
    st.stop()

left, right = st.columns([1, 1.2], gap="large")

with left:
    with st.container(border=True):
        st.markdown("### Profile")

        age = st.slider("Age", 18, 64, 30)

        c1, c2 = st.columns(2)
        with c1:
            sex = st.selectbox("Sex", ["female", "male"])
        with c2:
            smoker = st.selectbox("Smoker", ["no", "yes"])

        bmi = st.slider("BMI", 10.0, 70.0, 25.0, step=0.1)

        c3, c4 = st.columns(2)
        with c3:
            children = st.slider("Children", 0, 5, 0)
        with c4:
            region = st.selectbox(
                "Region", ["northeast", "northwest", "southeast", "southwest"]
            )

        predict_clicked = st.button("Estimate cost", type="primary", use_container_width=True)

with right:
    if predict_clicked:
        input_df = build_input_row(age, sex, bmi, children, smoker, region, feature_cols)

        try:
            prediction = model.predict(input_df)[0]
            # expm1 only if the model was trained on log1p(charges)
            pred_charges = float(np.expm1(prediction))

            is_smoker = smoker == "yes"
            average_charges = REF_AVG_SMOKER if is_smoker else REF_AVG_NON_SMOKER
            ref_label = "Average smoker" if is_smoker else "Average non-smoker"

            diff = pred_charges - average_charges
            pct = abs(diff) / average_charges * 100
            direction = "above" if diff > 0 else "below"

            st.markdown(
                f"""
                <div class="result">
                  <div class="label">Estimated yearly insurance cost</div>
                  <div class="amount">${pred_charges:,.2f}</div>
                  <div class="delta">{pct:.0f}% {direction} the {ref_label.lower()} reference.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(comparison_bars(pred_charges, average_charges, ref_label), unsafe_allow_html=True)

            cat = bmi_category(bmi)
            chips = [f'<span class="chip">{age} years old</span>',
                     f'<span class="chip{" warn" if cat in ("obese", "underweight") else ""}">BMI: {cat}</span>',
                     f'<span class="chip{" warn" if is_smoker else ""}">{"Smoker" if is_smoker else "Non-smoker"}</span>',
                     f'<span class="chip">{region.capitalize()}</span>']
            st.markdown(
                f'<div class="panel"><h4>Profile used</h4><div class="chips">{"".join(chips)}</div></div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.markdown(
            """
            <div class="empty">
              <div class="display">Your estimate will appear here</div>
              Set the profile, then select Estimate cost.
            </div>
            """,
            unsafe_allow_html=True,
        )