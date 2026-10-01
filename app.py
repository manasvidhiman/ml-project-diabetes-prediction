import joblib
import pandas as pd
import streamlit as st

# ---------- Page settings ----------
st.set_page_config(
    page_title="Diabetes Risk Assessment", page_icon="🩺", layout="centered"
)

# ---------- Load the saved model and scaler ----------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------- Professional Blue Palette ----------
PRIMARY_BLUE = "#2563EB"  # Main accent & primary button
DARK_TEXT = "#0F172A"  # Text & headers
BG_BLUE = "#F0F6FF"  # Plain light blue background
CARD_BG = "#FFFFFF"  # Form & card container background
BORDER_COLOR = "#E2E8F0"  # Subtle neutral borders

# ---------- Custom CSS ----------
css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@400;600;700&display=swap');

/* Plain blue background */
.stApp {{
    background-color: {BG_BLUE};
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}}

/* Hide default header and footer */
#MainMenu, footer, header {{ visibility: hidden; }}

.block-container {{
    max-width: 680px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}}

/* Heading Title - Pixelify Sans */
.hero-title {{
    font-family: 'Pixelify Sans', sans-serif;
    font-weight: 500;
    font-size: 2.8rem;
    color: {DARK_TEXT};
    text-align: center;
    margin-bottom: 0.25rem;
}}

.hero-sub {{
    text-align: center;
    color: #475569;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}}

/* Professional Standard Button */
.stButton button {{
    width: 100%;
    background-color: {PRIMARY_BLUE};
    color: #FFFFFF;
    font-weight: 600;
    font-size: 1rem;
    border-radius: 8px;
    border: none;
    padding: 0.65rem 1rem;
    transition: background-color 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}}
.stButton button:hover {{
    background-color: #1D4ED8;
    color: #FFFFFF;
    border: none;
}}

/* Result Card */
.result-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    padding: 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
}}
.result-number {{
    font-weight: 700;
    font-size: 3rem;
    line-height: 1;
}}
.result-label {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {DARK_TEXT};
    margin-top: 0.5rem;
}}
.bar-outer {{
    background: #E2E8F0;
    border-radius: 9999px;
    height: 12px;
    margin-top: 1rem;
    overflow: hidden;
}}
.bar-inner {{
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
}}
.small-note {{
    text-align: center;
    color: #64748B;
    font-size: 0.85rem;
    margin-top: 2rem;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---------- Title area ----------
st.markdown(
    '<div class="hero-title">Diabetes Risk Assessment</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Enter your health details below to estimate your medical risk level.</div>',
    unsafe_allow_html=True,
)

# ---------- Input Form Container ----------
with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1
        )
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=40, max_value=300, value=120
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=30, max_value=140, value=70
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=5, max_value=100, value=20
        )

    with col2:
        insulin = st.number_input(
            "Insulin (mu U/ml)", min_value=10, max_value=900, value=80
        )
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=70.0, value=28.0
        )
        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.05,
            max_value=2.5,
            value=0.4,
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=35)

st.write("")

# ---------- Prediction Logic ----------
if st.button("Calculate Risk"):
    person = pd.DataFrame(
        [
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age,
            }
        ]
    )

    person_scaled = scaler.transform(person)
    risk = model.predict_proba(person_scaled)[0][1] * 100

    if risk < 30:
        color, label = "#16A34A", "Low Risk"
    elif risk < 60:
        color, label = "#D97706", "Moderate Risk"
    else:
        color, label = "#DC2626", "High Risk"

    card = f"""
    <div class="result-card">
        <div class="result-number" style="color: {color};">{round(risk, 1)}%</div>
        <div class="result-label">{label}</div>
        <div class="bar-outer">
            <div class="bar-inner" style="background: {color}; width: {int(risk)}%;"></div>
        </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown(
    '<div class="small-note"><strong>Disclaimer:</strong> This tool is intended for informational purposes only and does not constitute medical advice. Consult a healthcare professional for clinical evaluation.</div>',
    unsafe_allow_html=True,
)