import streamlit as st
import numpy as np
import pickle
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Traffic Predictor",
    page_icon="🚦",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
try:
    model = pickle.load(open("traffic_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or Scaler files not found.")

# ---------------- WEATHER MAP ----------------
weather_map = {"Sunny": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}

# ---------------- BACKGROUND & STYLING ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main-box {{
        background: rgba(255, 255, 255, 0.05);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        margin-bottom: 20px;
    }}

    /* 1st Change: Increasing Label Font Size */
    label p {{
        font-size: 22px !important;
        font-weight: 500 !important;
        color: white !important;
        margin-bottom: 10px !important;
    }}

    .title {{ font-size: 48px; font-weight: 800; color: white; margin-bottom: 5px; }}
    .subtitle {{ font-size: 18px; color: #00ffff; margin-bottom: 35px; }}

    .prediction-container {{
        margin-top: 30px;
        padding: 20px;
        border-radius: 15px;
        background: rgba(0, 0, 0, 0.4);
        border: 2px solid;
        display: inline-block;
        min-width: 85%;
    }}

    .res-text {{ font-size: 34px; font-weight: bold; text-transform: uppercase; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg("background.jpg")

# ---------------- UI LAYOUT ----------------

st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.markdown('<div class="title">🚦 Traffic Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Congestion Prediction System</div>', unsafe_allow_html=True)

# 2nd Change: Updated placeholders
hour = st.text_input("Enter Hour (0–23)", placeholder="Enter hour here...")
vehicles = st.text_input("Enter Vehicle Count", placeholder="Enter number of vehicles...")
weather = st.selectbox("Select Weather", ["Select Weather", "Sunny", "Rainy", "Foggy", "Snowy"])

predict_clicked = st.button("Predict")

# ---------------- PREDICTION LOGIC ----------------

if predict_clicked:
    if hour == "" or vehicles == "" or weather == "Select Weather":
        st.warning("⚠ Please fill in all fields.")
    else:
        try:
            h_val = int(hour)
            v_val = int(vehicles)
            
            if not (0 <= h_val <= 23):
                st.error("Hour must be between 0 and 23.")
            else:
                w_val = weather_map[weather]
                input_data = np.array([[h_val, v_val, w_val]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]

                if prediction == 0: level, color = "LOW", "#00ff00"
                elif prediction == 1: level, color = "MEDIUM", "#ffff00"
                else: level, color = "HIGH", "#ff4b4b"

                st.markdown(f"""
                    <div class="prediction-container" style="border-color: {color};">
                        <p style="color: white; margin-bottom: 5px; font-size: 16px;">PREDICTED RESULT</p>
                        <div class="res-text" style="color: {color};">Congestion Level: {level}</div>
                    </div>
                """, unsafe_allow_html=True)
        except ValueError:
            st.error("Please enter numbers only.")

st.markdown('</div>', unsafe_allow_html=True)