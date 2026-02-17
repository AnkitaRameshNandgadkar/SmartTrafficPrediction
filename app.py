import streamlit as st
import pickle
import numpy as np
import base64

# ---------------- LOAD MODEL ----------------

model = pickle.load(open("traffic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- BACKGROUND IMAGE ----------------

def get_base64(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg = get_base64("traffic.jpg")  # make sure traffic.jpg exists

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="Traffic Predictor", layout="centered")

# ---------------- FULL CSS ----------------

st.markdown(f"""
<style>

/* BACKGROUND */

.stApp {{
    background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85)),
                      url("data:image/jpg;base64,{bg}");
    background-size: cover;
    background-position: center;
}}

/* TITLE */

h1 {{
    text-align: center;
    font-size: 60px !important;
    color: #00FFD1;
}}

h3 {{
    text-align: center;
    font-size: 30px !important;
    color: #00FFD1;
}}

/* LABELS */

label {{
    font-size: 50px !important;
    color: #00FFD1 !important;
}}

/* INPUT BOXES */

.stNumberInput input {{
    font-size: 20px !important;
    height: 75px !important;
    background-color: #111 !important;
    color: #00FFD1 !important;
    border: 2px solid #00FFD1 !important;
    border-radius: 12px !important;
}}

/* SELECT BOX FIXED */

.stSelectbox div[data-baseweb="select"] {{
    font-size: 20px !important;
    min-height: 80px !important;
    background-color: #111 !important;
    color: #00FFD1 !important;
    border: 2px solid #00FFD1 !important;
    border-radius: 12px !important;
    display: flex !important;
    align-items: center !important;
}}

.stSelectbox span {{
    font-size: 20px !important;
    line-height: 60px !important;
}}

/* BUTTON */

.stButton button {{
    font-size: 28px !important;
    height: 65px;
    width: 220px;
    border-radius: 12px;
    background: linear-gradient(45deg,#00FFD1,#0099FF);
    color: black;
    font-weight: bold;
    border: none;
}}

.stButton button:hover {{
    box-shadow: 0px 0px 25px #00FFD1;
    transform: scale(1.05);
}}

/* RESULT BOX */

.result {{
    text-align:center;
    font-size: 40px;
    font-weight: bold;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}}

.high {{
    background-color: red;
    color: white;
}}

.medium {{
    background-color: orange;
    color: black;
}}

.low {{
    background-color: green;
    color: white;
}}

</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------

st.markdown("<h1>🚦 Traffic Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI-based Congestion Prediction System</h3>", unsafe_allow_html=True)

hour = st.number_input("Enter Hour (0–23)", 0, 23, 8)

vehicles = st.number_input("Enter Vehicle Count", 0, 200, 50)

weather = st.selectbox(
    "Select Weather",
    ["Sunny", "Cloudy", "Rainy"]
)

# WEATHER ENCODING

weather_map = {
    "Sunny": 0,
    "Cloudy": 1,
    "Rainy": 2
}

weather_encoded = weather_map[weather]

# ---------------- PREDICTION ----------------

if st.button("Predict"):

    input_data = np.array([[hour, vehicles, weather_encoded]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        level = "Low"
        css = "low"
    elif prediction == 1:
        level = "Medium"
        css = "medium"
    else:
        level = "HIGH"
        css = "high"

    st.markdown(
        f'<div class="result {css}">Congestion Level: {level}</div>',
        unsafe_allow_html=True
    )
