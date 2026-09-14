import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="House Price Estimator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background-color: #EDEAE3;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
}

.masthead {
    border-bottom: 2px solid #1F2A24;
    padding-bottom: 1rem;
    margin-bottom: 2.2rem;
}

.masthead-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6B6459;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
}

.masthead-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: #1F2A24;
    line-height: 1.15;
}

.masthead-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: #5A5348;
    margin-top: 0.4rem;
    max-width: 520px;
}

.field-group-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6B6459;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    border-bottom: 1px solid #C9C2B4;
    padding-bottom: 0.5rem;
    margin-bottom: 1.1rem;
    margin-top: 0.3rem;
}

div[data-testid="stSlider"] label p,
div[data-testid="stSelectbox"] label p {
    color: #3A362E !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

div[data-testid="stSlider"] > div > div > div > div {
    background-color: #1F2A24 !important;
}

/* The round drag handle: solid accent color */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: #C1443A !important;
    border: none !important;
}

/* Everything inside the handle (the value tooltip) should have no dark box, plain dark text */
div[data-baseweb="slider"] div[role="slider"] * {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    color: #1F2A24 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    box-shadow: none !important;
    border: none !important;
    opacity: 1 !important;
    visibility: visible !important;
}

div[data-testid="stThumbValue"],
div[data-testid="stSliderThumbValue"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #1F2A24 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    box-shadow: none !important;
    border: none !important;
    opacity: 1 !important;
    visibility: visible !important;
    top: -1.6rem !important;
}

div[data-testid="stTickBar"] {
    display: none;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: #F7F5F0 !important;
    border: 1px solid #C9C2B4 !important;
    border-radius: 3px !important;
    color: #1F2A24 !important;
}

div[data-testid="stSelectbox"] > div > div * {
    color: #1F2A24 !important;
}

div[data-testid="stFormSubmitButton"] {
    display: flex;
    justify-content: flex-end;
    margin-top: 0.5rem;
}

div[data-testid="stFormSubmitButton"] > button {
    width: auto;
    background-color: #1F2A24;
    color: #EDEAE3;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.85rem 2.4rem;
    line-height: 1;
    border-radius: 4px;
    border: none;
    margin-top: 0.8rem;
    transition: background-color 0.15s ease;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #34433A;
    color: #EDEAE3;
    border: none;
}

.result-block {
    margin-top: 2.2rem;
    padding-top: 1.6rem;
    border-top: 2px solid #1F2A24;
}

.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6B6459;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-price {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 3.2rem;
    color: #1F2A24;
    line-height: 1;
}

.result-table {
    margin-top: 1.4rem;
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}

.result-table td {
    padding: 0.5rem 0;
    border-bottom: 1px solid #DAD5C9;
    color: #3A362E;
}

.result-table td:first-child {
    color: #6B6459;
    width: 45%;
}

.result-table td:last-child {
    text-align: right;
    font-weight: 500;
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
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- Masthead ---
st.markdown("""
<div class="masthead">
    <div class="masthead-label">VALUATION TOOL</div>
    <div class="masthead-title">House Price Estimator</div>
    <div class="masthead-sub">
        Provide the property specifications below to generate a market value estimate from the trained pricing model.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="field-group-label">Physical Features</div>', unsafe_allow_html=True)

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

    with col2:
        st.markdown('<div class="field-group-label">Space &amp; Location</div>', unsafe_allow_html=True)

        sqft_living = st.slider("Living area (sqft)",
                                min_value=float(rentang['sqft_living']['min']),
                                max_value=float(rentang['sqft_living']['max']),
                                value=float(rentang['sqft_living']['min']))

        sqft_above = st.slider("Above-ground area (sqft)",
                               min_value=float(rentang['sqft_above']['min']),
                               max_value=float(rentang['sqft_above']['max']),
                               value=float(rentang['sqft_above']['min']))

        city = st.selectbox("City", options=list(encoders['city'].classes_))
        statezip = st.selectbox("State ZIP code", options=list(encoders['statezip'].classes_))

    submit_button = st.form_submit_button("Estimate price")

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
    st.snow()
    st.markdown(f"""
    <div class="result-block">
        <div class="result-label">Estimated market price</div>
        <div class="result-price">${price_final:,.0f}</div>
        <table class="result-table">
            <tr><td>Bedrooms</td><td>{bedrooms:g}</td></tr>
            <tr><td>Bathrooms</td><td>{bathrooms:g}</td></tr>
            <tr><td>Floors</td><td>{floors:g}</td></tr>
            <tr><td>Living area</td><td>{sqft_living:,.0f} sqft</td></tr>
            <tr><td>Above-ground area</td><td>{sqft_above:,.0f} sqft</td></tr>
            <tr><td>City</td><td>{city}</td></tr>
            <tr><td>State ZIP</td><td>{statezip}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)