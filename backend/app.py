from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
import os

# Initialize Flask App
app = Flask(__name__)
# Allow requests from any origin for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Load Models at Startup ---
MODEL_PATH = "backend/models/cyber_model.pkl"
BERT_MODEL_NAME = 'all-MiniLM-L6-v2'

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    bert_model = SentenceTransformer(BERT_MODEL_NAME)
    print(f"SentenceTransformer '{BERT_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading models: {e}")
    model = None
    bert_model = None

# --- API Endpoints ---
@app.route("/")
def index():
    return "Cyberbully Detection API is running!"

@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    if not model or not bert_model:
        return jsonify({"error": "Models are not available."}), 503

    data = request.get_json()
    if not data or "text" not in data or not data["text"].strip():
        return jsonify({"error": "Invalid input: 'text' field is required."}), 400

    text_to_analyze = data["text"]
    
    try:
        # Determine the correct input format based on model type
        if isinstance(model, Pipeline):
            print("Processing with TF-IDF pipeline...")
            input_data = pd.DataFrame({'text': [text_to_analyze]})
            prediction = model.predict(input_data)[0]
        else:
            print("Processing with BERT embeddings...")
            embedding = bert_model.encode([text_to_analyze])
            input_data = embedding
            prediction = model.predict(input_data)[0]

        # --- THIS IS THE UPDATED LOGIC ---
        probability = None
        # Try to get probability with predict_proba (works for Logistic Regression, etc.)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            class_index = list(model.classes_).index(prediction)
            probability = float(probs[class_index])
        # Fallback to decision_function for models like LinearSVC
        elif hasattr(model, "decision_function"):
            score = model.decision_function(input_data)[0]
            # Convert the score to a pseudo-probability using the sigmoid function
            prob = 1 / (1 + np.exp(-score))
            # Ensure the probability matches the predicted class
            # (For binary classification, if the prediction is the first class, prob is 1-prob)
            if prediction == model.classes_[0]:
                 probability = 1 - prob
            else:
                 probability = prob
        
        response = {
            "text": text_to_analyze,
            "label": str(prediction),
            "probability": f"{probability:.4f}" if probability is not None else "N/A"
        }
        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "Failed to process the request."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
