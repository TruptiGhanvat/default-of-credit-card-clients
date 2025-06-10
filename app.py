import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Actual feature names used in model training
feature_columns = ['LIMIT_BAL', 'AGE', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'BILL_AMT1']

st.title("Credit Card Default Predictor")

# User input form
user_input = {}
for feature in feature_columns:
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Predict button
if st.button("Predict"):
    try:
        features = [user_input[feature] for feature in feature_columns]
        prediction = model.predict([features])[0]
        result = "Will Default" if prediction == 1 else "No Default"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
