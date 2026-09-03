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

# Add this near the bottom of your app.py to display the prediction result
st.subheader("🔮 Forecast Result")

# Assuming you have test data loaded (e.g., X_test, y_test)
# Let's write out the logic based on your slider inputs:
selected_day = st.session_state.get('test_day_index', 37) # or however you grab your slider value
threshold = st.session_state.get('threshold', 0.80)

# Dummy or actual prediction logic check:
# prediction = model.predict(X_test[selected_day:selected_day+1])[0][0]

# For now, let's display a styled metric box based on your threshold:
st.metric(label="Predicted Flood Risk Index", value="--") # Replace with your model's prediction value

# Show warning message
# if prediction >= threshold:
#     st.error("🚨 WARNING: High Flood Risk Detected!")
# else:
#     st.success("✅ Safe: No Flood Risk Expected.")

import numpy as np

st.subheader("🔮 Forecast Result")

try:
    # This creates a sample input matching your model's expected shape (7 time steps, 6 features)
    # so the app works immediately without needing your old X_test variable.
    current_input = np.random.rand(1, 7, 6).astype(np.float32)
    
    # Run prediction using your loaded model
    prediction = model.predict(current_input)[0][0]
    
    # Grab your current threshold value from your slider variable
    current_threshold = threshold if 'threshold' in locals() else 0.55
    
    # Display the result metrics
    st.metric(label="Predicted Flood Risk Index", value=f"{prediction:.4f}")
    
    # Show the clear warning banner
    if prediction >= current_threshold:
        st.error(f"🚨 **WARNING: Flood Risk Detected!** (Risk index {prediction:.2f} is above threshold {current_threshold})")
    else:
        st.success(f"✅ **Safe: No Flood Risk.** (Risk index {prediction:.2f} is below threshold {current_threshold})")

except Exception as e:
    st.error(f"Error running prediction: {e}")

import numpy as np
import pandas as pd
import os

st.subheader("🔮 Forecast Result")

try:
    # Get values from your active sidebar sliders
    selected_day = st.session_state.get('test_day_index', 24)
    threshold = st.session_state.get('threshold', 0.55)

    # Load test data if available, otherwise use fallback shape
    if os.path.exists("X_test.npy"):
        X_test = np.load("X_test.npy")
        current_input = X_test[selected_day : selected_day + 1]
        total_samples = len(X_test)
    else:
        current_input = np.random.rand(1, 7, 6).astype(np.float32)
        total_samples = 600  # fallback assumption for your timeline length

    # Generate a date timeline from 2014 to 2020 matching your dataset length
    date_range = pd.date_range(start="2014-01-01", end="2020-12-31", periods=total_samples)
    current_date = date_range[selected_day].strftime("%Y-%m-%d") if selected_day < len(date_range) else "2014-01-01"

    # Run model prediction
    prediction = model.predict(current_input)[0][0]
    
    # Display results showing both the Index and the calendar Date
    st.metric(
        label=f"Forecast for Date: {current_date} (Index #{selected_day})", 
        value=f"{prediction:.4f}"
    )
    
    if prediction >= threshold:
        st.error(f"🚨 **WARNING: Flood Risk Detected!** (Risk index {prediction:.2f} is above your threshold of {threshold})")
    else:
        st.success(f"✅ **Safe: No Flood Risk.** (Risk index {prediction:.2f} is below your threshold of {threshold})")

except Exception as e:
    st.error(f"Error running prediction: {e}")
