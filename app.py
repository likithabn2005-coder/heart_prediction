import streamlit as st
import numpy as np
import pickle

# ---------- Page setup ----------
st.set_page_config(
    page_title="Heart Health Analyzer",
    page_icon="💓",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center;'>💓 Heart Health Analyzer</h1>"
    "<p style='text-align:center;color:#666'>Fill the details below. "
    "Tap the ❓ on any field to learn what it means and which test it comes from.</p>",
    unsafe_allow_html=True
)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- Helper (pretty chips) ----------
def chip(label, color="#eef"):
    st.markdown(
        f"<span style='background:{color};padding:4px 8px;border-radius:8px;"
        f"font-size:12px;border:1px solid #ddd;margin-left:6px'>{label}</span>",
        unsafe_allow_html=True
    )

# ---------- Input form with tooltips ----------
with st.form("inputs"):
    st.subheader("Patient details")

    col1, col2 = st.columns(2)

    age = col1.number_input(
        "Age",
        min_value=1, max_value=120, value=25,
        help="Your age in years. 
        ℹ️ Source: Basic info / ID / records."
    )
    sex = col2.selectbox(
        "Sex",
        options=[("Male", 1), ("Female", 0)],
        format_func=lambda x: x[0],
        help="Biological sex coded as 1=Male, 0=Female. 
        ℹ️ Source: Basic info."
    )[1]

    cp = col1.selectbox(
        "Chest Pain Type (cp)",
        options=[
            ("0 – Typical angina", 0),
            ("1 – Atypical angina", 1),
            ("2 – Non-anginal pain", 2),
            ("3 – Asymptomatic", 3),
        ],
        index=2,
        format_func=lambda x: x[0],
        help="Clinical chest-pain category.
        ℹ️ Source: Doctor evaluation / symptom history."
    )[1]

    trestbps = col2.number_input(
        "Resting Blood Pressure (trestbps, mmHg)",
        min_value=60, max_value=250, value=120,
        help="Resting systolic BP measured in mmHg.
        ℹ️ Source: Blood pressure monitor."
    )

    chol = col1.number_input(
        "Cholesterol (chol, mg/dL)",
        min_value=80, max_value=700, value=200,
        help="Total serum cholesterol.
        ℹ️ Source: Blood test (lipid profile)."
    )

    fbs = col2.selectbox(
        "Fasting Blood Sugar > 120 mg/dL (fbs)",
        options=[("No (0)", 0), ("Yes (1)", 1)],
        format_func=lambda x: x[0],
        help="High fasting blood sugar flag.
        ℹ️ Source: Blood test (fasting glucose)."
    )[1]

    restecg = col1.selectbox(
        "Resting ECG (restecg)",
        options=[
            ("0 – Normal", 0),
            ("1 – ST-T wave abnormality", 1),
            ("2 – Left ventricular hypertrophy", 2),
        ],
        index=1,
        format_func=lambda x: x[0],
        help="Resting electrocardiogram result.
        ℹ️ Source: Resting ECG."
    )[1]

    thalach = col2.number_input(
        "Max Heart Rate (thalach)",
        min_value=60, max_value=240, value=150,
        help="Maximum heart rate achieved during exercise.
        ℹ️ Source: Treadmill stress test / exercise ECG."
    )

    exang = col1.selectbox(
        "Exercise-Induced Angina (exang)",
        options=[("No (0)", 0), ("Yes (1)", 1)],
        format_func=lambda x: x[0],
        help="Angina triggered by exercise.
        ℹ️ Source: Treadmill stress test."
    )[1]

    oldpeak = col2.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help="ST depression induced by exercise relative to rest.
        ℹ️ Source: Exercise ECG report."
    )

    slope = col1.selectbox(
        "Slope of ST Segment (slope)",
        options=[("0 – Upsloping", 0), ("1 – Flat", 1), ("2 – Downsloping", 2)],
        index=1,
        format_func=lambda x: x[0],
        help="Slope of the peak exercise ST segment. 
        ℹ️ Source: Exercise ECG."
    )[1]

    ca = col2.number_input(
        "Major Vessels Colored by Fluoroscopy (ca)",
        min_value=0, max_value=3, value=0,
        help="Number of major vessels (0–3) seen in angiography. 
        ℹ️ Source: Coronary angiogram."
    )

    thal = col1.selectbox(
        "Thalassemia (thal)",
        options=[("0 – Normal", 0), ("1 – Fixed defect", 1), ("2 – Reversible defect", 2), ("3 – Other", 3)],
        index=2,
        format_func=lambda x: x[0],
        help="Thallium stress test result (per UCI dataset coding).
        ℹ️ Source: Nuclear stress test."
    )[1]

    st.markdown(
        "<div style='font-size:13px;color:#666'>"
        "📎 Feature order for the model: age, sex, cp, trestbps, chol, fbs, restecg, thalach, "
        "exang, oldpeak, slope, ca, thal"
        "</div>", unsafe_allow_html=True
    )

    submitted = st.form_submit_button("🔍 Predict")

# ---------- Prediction ----------
if submitted:
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    try:
        prob_disease = float(model.predict_proba(features)[0][1])  # class 1 = disease
    except Exception:
        # Fallback if model has no predict_proba
        pred = int(model.predict(features)[0])
        prob_disease = 0.7 if pred == 1 else 0.1

    pred_label = "High risk of heart disease" if prob_disease >= 0.5 else "No heart disease"
    percent = round(prob_disease * 100, 1)

    # Nice colored card
    if prob_disease >= 0.5:
        st.error(f"❤️ **{pred_label}**  \nEstimated probability: **{percent}%**")
    else:
        st.success(f"💚 **{pred_label}**  \nEstimated probability: **{percent}%**")

    with st.expander("What do these fields mean? (tests & quick guide)"):
        st.markdown("""
- **Age** — years (basic info)  
- **Sex** — 1 male / 0 female (basic info)  
- **Chest Pain Type (cp)** — clinical category (doctor assessment)  
- **Resting BP (trestbps)** — mmHg (BP monitor)  
- **Cholesterol (chol)** — mg/dL (blood test, lipid profile)  
- **Fasting Blood Sugar (fbs)** — 1 if >120 mg/dL (blood test)  
- **Resting ECG (restecg)** — result code (resting ECG)  
- **Max Heart Rate (thalach)** — bpm (treadmill/exercise ECG)  
- **Exercise Angina (exang)** — 1 yes / 0 no (stress test)  
- **ST depression (oldpeak)** — value from exercise ECG  
- **Slope** — ST segment slope (exercise ECG)  
- **ca** — number of major vessels (0–3) (angiography)  
- **thal** — thallium stress test code (nuclear test)
        """)

st.caption("⚠️ This tool is for educational use and is **not** a medical diagnosis. Please consult a doctor for clinical decisions.")

