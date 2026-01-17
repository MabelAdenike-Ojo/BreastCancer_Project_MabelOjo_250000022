from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model and feature columns
model = joblib.load("breast_cancer_ann_model.joblib")
model_columns = joblib.load("breast_cancer_columns.joblib")

@app.route("/")
def home():
    return render_template("index.html")  # HTML form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the frontend form
        data = request.get_json()
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure columns match training features
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Convert to numpy array
        input_array = input_df.to_numpy()
        
        # ANN prediction
        pred_prob = model.predict(input_array)
        pred_class = (pred_prob > 0.5).astype(int)[0][0]
        
        result = "Malignant" if pred_class == 1 else "Benign"
        
        return jsonify({
            "Prediction": result,
            "Probability": float(pred_prob[0][0])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)