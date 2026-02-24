import sys
import os

# Ensure project root is on the path so `src` package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Water Quality Monitoring",
    page_icon="💧",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box-safe {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #2e7d32;
    }
    .result-box-unsafe {
        background-color: #ffebee;
        color: #c62828;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #c62828;
    }
    .result-box-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        border: 2px solid #856404;
    }
    </style>
    """, unsafe_allow_html=True)

# Generate Alarm Sound (Placeholder)
beep_audio = """
    <audio autoplay>
    <source src="data:audio/mp3;base64,//uQxAAAAANIAAAAAExBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq" type="audio/mpeg">
    </audio>
    """

# --- Load Model Bundle ---
MODEL_PATH = "water_model.pkl"


@st.cache_resource
def load_model_bundle():
    """Load the trained model bundle from pickle file."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


bundle = load_model_bundle()

# --- Import prediction logic from src package ---
from src.predict import detect_anomaly, predict_quality

# --- Sidebar: Model Selection ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "SVM"])

# --- Main Interface ---
st.title("💧 AI-Based Water Quality Monitoring")
st.markdown("Real-time monitoring using **Hybrid Logic** (AI + Safety Rules).")

if bundle is None:
    st.error("⚠️ Model file not found! Please run `python -m src.train` first.")
else:
    # --- Input Section ---
    st.subheader("Enter Water Parameters")

    col1, col2 = st.columns(2)
    with col1:
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1, help="Safe Range: 6.5 - 8.5")
    with col2:
        solids = st.number_input("TDS (Solids)", min_value=0.0, max_value=50000.0, value=20000.0, step=100.0, help="Safe Range: < 500 ppm (Drinking)")

    # --- Prediction Logic ---
    if st.button("Analyze Quality", type="primary"):
        prediction, probability, reason, is_anomaly = predict_quality(ph, solids, model_choice, bundle)

        # Display Results
        st.markdown("---")
        if prediction == 1:
            st.markdown(f'''
                <div class="result-box-safe">
                    <h2>✅ Safe for Drinking</h2>
                    <p>{reason}</p>
                    <p>Confidence: {probability:.1%}</p>
                </div>
            ''', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'''
                <div class="result-box-unsafe">
                    <h2>⚠️ Contaminated / Unsafe</h2>
                    <p>{reason}</p>
                    <p>Confidence: {1-probability:.1%}</p>
                </div>
            ''', unsafe_allow_html=True)
            if is_anomaly:
                st.markdown(beep_audio, unsafe_allow_html=True)

# --- Real-Time Simulation Section ---
st.markdown("---")
st.subheader("📡 Real-Time Simulation Monitoring")

if 'simulation' not in st.session_state:
    st.session_state.simulation = False

col_start, col_stop = st.columns(2)

with col_start:
    if st.button("Start Simulation"):
        st.session_state.simulation = True

with col_stop:
    if st.button("Stop Simulation"):
        st.session_state.simulation = False

# Placeholder for charts
chart_placeholder = st.empty()

if st.session_state.simulation:
    # Initialize data storage
    if 'data_log' not in st.session_state:
        st.session_state.data_log = pd.DataFrame(columns=['Time', 'pH', 'TDS'])

    # Initialize contamination state if not present
    if 'contamination_steps' not in st.session_state:
        st.session_state.contamination_steps = 0

    # Simulation Loop
    while st.session_state.simulation:

        # Decide state based on persistence
        if st.session_state.contamination_steps > 0:
            # Continue Contamination Event
            st.session_state.contamination_steps -= 1

            # Randomly fluctuate between "Critical" and "Subtle" to show both Logic & ML
            if np.random.rand() > 0.5:
                # Critical (Triggers Safety Rule)
                sim_ph = np.random.normal(3.5, 0.2)
                sim_tds = np.random.normal(3500, 50)
            else:
                # Subtle / Gray Zone (Triggers ML Model)
                sim_ph = np.random.normal(6.0, 0.2)
                sim_tds = np.random.normal(1500, 100)
        else:
            # Safe State
            sim_ph = np.random.normal(7.2, 0.1)
            sim_tds = np.random.normal(300, 10)

            # Randomly trigger new contamination event (5% chance)
            if np.random.rand() > 0.95:
                st.session_state.contamination_steps = np.random.randint(5, 15)

        current_time = pd.Timestamp.now().strftime('%H:%M:%S')
        new_row = pd.DataFrame({'Time': [current_time], 'pH': [sim_ph], 'TDS': [sim_tds]})
        st.session_state.data_log = pd.concat([st.session_state.data_log, new_row], ignore_index=True)

        # Keep last 50 points
        if len(st.session_state.data_log) > 50:
            st.session_state.data_log = st.session_state.data_log.iloc[1:]

        with chart_placeholder.container():
            # 1. Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("pH Level", f"{sim_ph:.2f}")
            m2.metric("TDS Level", f"{sim_tds:.0f} ppm")

            # Predict status (Hybrid Logic)
            if bundle:
                sim_pred, _, _, sim_anomaly = predict_quality(sim_ph, sim_tds, model_choice, bundle)

                if sim_anomaly:
                    status_text = "CRITICAL UNSAFE"
                    status_color = "inverse"
                else:
                    status_text = "Safe" if sim_pred == 1 else "Unsafe"
                    status_color = "normal" if sim_pred == 1 else "off"

                m3.metric("Status", status_text, delta_color=status_color)

                # Show error message if critical
                is_anomaly, anomaly_msg = detect_anomaly(sim_ph, sim_tds)
                if is_anomaly:
                    st.error(f"🚨 {anomaly_msg}")
                elif sim_pred == 0:
                    st.warning("⚠️ Contamination Detected")

            # 2. Altair Charts
            data = st.session_state.data_log.reset_index()

            # pH Chart
            ph_chart = alt.Chart(data).mark_line(color='blue').encode(
                x=alt.X('index', axis=alt.Axis(title='Time Step')),
                y=alt.Y('pH', scale=alt.Scale(domain=[0, 14]), title='pH Level'),
                tooltip=['pH', 'TDS']
            ).properties(height=200, title="Real-Time pH Trend")

            # Safe Zone Background for pH (6.5 - 8.5)
            safe_zone = pd.DataFrame({'y': [6.5], 'y2': [8.5]})
            ph_bg = alt.Chart(safe_zone).mark_rect(opacity=0.2, color='green').encode(
                y='y', y2='y2'
            )

            # TDS Chart
            tds_chart = alt.Chart(data).mark_line(color='brown').encode(
                x=alt.X('index', axis=alt.Axis(title='Time Step')),
                y=alt.Y('TDS', title='TDS (ppm)'),
                tooltip=['pH', 'TDS']
            ).properties(height=200, title="Real-Time TDS Trend")

            st.altair_chart(ph_bg + ph_chart, use_container_width=True)
            st.altair_chart(tds_chart, use_container_width=True)

        time.sleep(1)
