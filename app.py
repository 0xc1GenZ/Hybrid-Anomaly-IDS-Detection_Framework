from flask import Flask, request, jsonify
import pandas as pd
from pathlib import Path
from src.hybrid_ids import HybridIDS

app = Flask(__name__)
model = HybridIDS()

# Load model on startup
@app.before_first_request
def load_model():
    try:
        model.load()
        print("✅ Model loaded")
    except:
        print("⚠️ Model not found – will train on first request")

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Hybrid IDS REST API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    df = pd.read_csv(file)
    
    preds, shap_vals = model.predict(df)
    flagged_count = int((preds > 0.5).sum())
    
    return jsonify({
        "total_flows": len(df),
        "flagged_anomalies": flagged_count,
        "flagged_percentage": round(flagged_count / len(df) * 100, 2),
        "predictions": preds.tolist(),
        "status": "ALERT" if flagged_count > 0 else "NORMAL"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
