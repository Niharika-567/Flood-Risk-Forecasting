import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Page Configuration
st.set_page_config(
    page_title="Flood Risk Forecasting Dashboard",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Risk Forecasting & Early Warning System")
st.markdown("This dashboard uses an advanced **LSTM deep learning model** to predict flood risks based on hydrological and meteorological time-series data.")

# Load Model
@st.cache_resource
def load_prediction_model():
    model_path = "flood_model.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_prediction_model()

if model is None:
    st.error("⚠️ `flood_model.keras` not found! Please make sure your model file is uploaded in the same repository directory.")
else:
    st.success("✅ Model loaded successfully!")

    # Sidebar Controls for Simulation / Testing
    st.sidebar.header("⚙️ Simulation Controls")
    test_index = st.sidebar.slider("Select Test Day Index", min_value=0, max_value=50, value=7)
    threshold = st.sidebar.slider("Flood Warning Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.markdown("### 📊 Live Risk Analysis Panel")
    
    # Placeholder for interactive simulation metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Selected Day Index", value=test_index)
    with col2:
        st.metric(label="Configured Threshold", value=f"{threshold}")
    with col3:
        st.metric(label="Model Status", value="Active / Operational", delta="LSTM")

    st.info("💡 **Next Steps:** You can customize this dashboard further to load your test datasets and plot actual vs predicted water levels right here!")
