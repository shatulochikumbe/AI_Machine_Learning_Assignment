import streamlit as st
import pandas as pd
import joblib # Load your saved model here

st.title("🌱 Agri-Predict: Zero Hunger AI")
rain = st.number_input("Average Rainfall (mm)", value=0.0)
temp = st.number_input("Average Temperature (C)", value=0.0)
# ... add other inputs ...

# try loading a saved model (adjust path as needed)
model = None
try:
    model = joblib.load("model.joblib")  # change filename/path if needed
except Exception as e:
    st.warning(f"Could not load model: {e}")

if st.button("Predict Yield"):
    if model is None:
        st.error("Model not loaded. Please provide a valid model file at 'model.joblib'.")
    else:
        # prepare input array matching the model's expected features
        X = [[rain, temp]]
        pred = model.predict(X)
        result = float(pred[0])
        st.success(f"Predicted Yield: {result:.2f} hg/ha")