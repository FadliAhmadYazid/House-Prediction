import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS: Fonts, Colors, Cards, Animations ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    background-attachment: fixed;
}

/* Hide default streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Hero header */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    animation: fadeInDown 0.8s ease-out;
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    margin-bottom: 0.3rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
}

@keyframes shimmer {
    to { background-position: 200% center; }
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Glass-card style sections */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    animation: fadeInUp 0.6s ease-out;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(96, 165, 250, 0.15);
}

.section-header {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 1.25rem;
    color: #e2e8f0;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Sliders */
div[data-testid="stSlider"] label p {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #60a5fa, #a78bfa) !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] label p {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* Submit button */
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    padding: 0.75rem 0;
    border-radius: 14px;
    border: none;
    margin-top: 0.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.35);
}

div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 6px 28px rgba(139, 92, 246, 0.55);
    border: none;
    color: white;
}

div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0) scale(0.99);
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 22px;
    padding: 2.2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 10px 40px rgba(139, 92, 246, 0.25);
}

@keyframes popIn {
    0% { opacity: 0; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}

.result-label {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}

.result-price {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.result-note {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 0.6rem;
}

/* Divider styling */
hr {
    border-color: rgba(148, 163, 184, 0.15) !important;
}

/* Metric-like summary chips shown after prediction */
.chip-row {
    display: flex;
    gap: 0.7rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}

.chip {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)


# --- Load Data & Model ---
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('xgboost_model.pkl', 'rb'))
    encoders = pickle.load(open('label_encoder.pkl', 'rb'))
    scaler_fitur = pickle.load(open('scaler_fitur.pkl', 'rb'))
    target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))
    rentang = pickle.load(open('rentang_fitur.pkl', 'rb'))
    return model, encoders, scaler_fitur, target_scaler, rentang

try:
    model, encoders, scaler_fitur, target_scaler, rentang = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading model artifacts: {e}")
    st.stop()

# --- Hero Header ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🏠 Smart House Price Predictor</div>
    <div class="hero-subtitle">
        Powered by machine learning — enter your property details below and get an instant, data-driven market estimate.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🛏️ Physical Features</div>', unsafe_allow_html=True)

        bedrooms = st.slider("Bedrooms",
                             min_value=float(rentang['bedrooms']['min']),
                             max_value=float(rentang['bedrooms']['max']),
                             value=float(rentang['bedrooms']['min']))

        bathrooms = st.slider("Bathrooms",
                              min_value=float(rentang['bathrooms']['min']),
                              max_value=float(rentang['bathrooms']['max']),
                              value=float(rentang['bathrooms']['min']))

        floors = st.slider("Floors",
                           min_value=float(rentang['floors']['min']),
                           max_value=float(rentang['floors']['max']),
                           value=float(rentang['floors']['min']))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📐 Space & Location</div>', unsafe_allow_html=True)

        sqft_living = st.slider("Living Area (Sqft)",
                                min_value=float(rentang['sqft_living']['min']),
                                max_value=float(rentang['sqft_living']['max']),
                                value=float(rentang['sqft_living']['min']))

        sqft_above = st.slider("Above Ground Area (Sqft)",
                               min_value=float(rentang['sqft_above']['min']),
                               max_value=float(rentang['sqft_above']['max']),
                               value=float(rentang['sqft_above']['min']))

        city = st.selectbox("City", options=list(encoders['city'].classes_))
        statezip = st.selectbox("State Zip Code", options=list(encoders['statezip'].classes_))
        st.markdown('</div>', unsafe_allow_html=True)

    submit_button = st.form_submit_button("✨ Predict House Price")

if submit_button:
    # 1. Pre-processing: Categorical Encoding
    enc_city = encoders['city'].transform([city])[0]
    enc_zip = encoders['statezip'].transform([statezip])[0]

    # 2. Construct DataFrame with exact training order
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft_living': [sqft_living],
        'floors': [floors],
        'sqft_above': [sqft_above],
        'city': [enc_city],
        'statezip': [enc_zip]
    })

    # 3. Scaling features
    input_scaled = scaler_fitur.transform(input_data)

    # 4. Prediction
    pred_scaled = model.predict(input_scaled)

    # 5. Inverse Transform target
    price_final = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    # --- Result Display ---
    st.balloons()
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Market Price</div>
        <div class="result-price">${price_final:,.2f}</div>
        <div class="result-note">Based on the property details you provided</div>
        <div class="chip-row">
            <div class="chip">🛏️ {bedrooms:g} bed</div>
            <div class="chip">🛁 {bathrooms:g} bath</div>
            <div class="chip">🏢 {floors:g} floor(s)</div>
            <div class="chip">📐 {sqft_living:,.0f} sqft living</div>
            <div class="chip">📍 {city}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)