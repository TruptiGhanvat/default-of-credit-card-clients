from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# You need to know the order and names of features your model expects
feature_columns = ['feature1', 'feature2', 'feature3']  # Replace with your actual feature column names

@app.route('/')
def home():
    return "Welcome to the ML Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract features from JSON in the correct order
        features = [data[col] for col in feature_columns]
    except KeyError as e:
        return jsonify({"error": f"Missing feature in input: {e}"}), 400

    # Convert features into a 2D array for prediction
    prediction = model.predict([features])

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

