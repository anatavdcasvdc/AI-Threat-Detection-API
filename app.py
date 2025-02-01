from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model_path = "D:/AI_Threat_Detection/dataset/processed/random_forest_tuned.pkl"
clf = joblib.load(model_path)

@app.route('/')
def home():
    return "🚀 AI Threat Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert JSON to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure features match training data
        required_features = clf.feature_names_in_
        df = df[required_features]

        # Make prediction
        prediction = clf.predict(df)[0]
        
        # Interpret result
        result = "Attack" if prediction == 1 else "Normal"
        
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
