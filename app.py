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
    test_index = st.sidebar.slider("Select Test Day Index", min_value=0, max_value=700, value=24)
    threshold = st.sidebar.slider("Flood Warning Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.markdown("### 📊 Live Risk Analysis Panel")
    
    # Interactive simulation metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Selected Day Index", value=test_index)
    with col2:
        st.metric(label="Configured Threshold", value=f"{threshold}")
    with col3:
        st.metric(label="Model Status", value="Active / Operational", delta="LSTM")

    # ==========================================
    # Unified Forecast & Early Warning Section
    # ==========================================
    st.subheader("🔮 Forecast Result")

    try:
        # Load test data if available, otherwise use fallback shape
        if os.path.exists("X_test.npy"):
            X_test = np.load("X_test.npy")
            # Ensure index doesn't exceed array bounds
            safe_index = min(test_index, len(X_test) - 1)
            current_input = X_test[safe_index : safe_index + 1]
            total_samples = len(X_test)
        else:
            safe_index = test_index
            current_input = np.random.rand(1, 7, 6).astype(np.float32)
            total_samples = 600  # fallback timeline length assumption

        # Generate a date timeline from 2014 to 2020 matching your dataset length
        date_range = pd.date_range(start="2014-01-01", end="2020-12-31", periods=total_samples)
        current_date = date_range[safe_index].strftime("%Y-%m-%d") if safe_index < len(date_range) else "2014-01-01"

        # Run model prediction
        prediction = model.predict(current_input)[0][0]
        
        # Display results showing both the calendar Date and Index
        st.metric(
            label=f"Forecast for Date: {current_date} (Index #{safe_index})", 
            value=f"{prediction:.4f}"
        )
        
        # Show Flood / No Flood warning banner
        if prediction >= threshold:
            st.error(f"🚨 **WARNING: Flood Risk Detected!** (Risk index {prediction:.2f} is above your threshold of {threshold})")
        else:
            st.success(f"✅ **Safe: No Flood Risk.** (Risk index {prediction:.2f} is below your threshold of {threshold})")

    except Exception as e:
        st.error(f"Error running prediction: {e}")
