import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Define expected features
feature_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6']

st.title("Credit Card Default Predictor")

# Create inputs dynamically
user_input = {}
for feature in feature_columns:
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Prediction button
if st.button("Predict"):
    try:
        features = [user_input[feature] for feature in feature_columns]
        prediction = model.predict([features])[0]
        result = "Will Default" if prediction == 1 else "No Default"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
