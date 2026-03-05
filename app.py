from flask import Flask, request, jsonify
import pandas as pd
from src.hybrid_ids import HybridIDS  # Import your model
from pathlib import Path
import io  # For handling uploaded files
import numpy as np

app = Flask(__name__)

# Load or train model on startup
print("Loading/Training model...")
model_path = Path("models/hybrid_model.h5")
if model_path.exists():
    # Load saved model (add model.save/load methods to HybridIDS if needed)
    print("Loading saved model...")
    model = HybridIDS.load(model_path)  # Implement this in HybridIDS class
else:
    # Train on sample data for demo
    sample_path = Path("data/sample_flows.csv")
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        X = df.drop('label', axis=1)
        y = (df['label'] == 'attack').astype(int)
        model = HybridIDS()
        model.fit(X, y)
        print("Model trained on sample data.")
    else:
        raise FileNotFoundError("Sample data not found – train model first.")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'welcome': 'Hybrid IDS API',
        'endpoints': {
            'health': '/health (GET)',
            'predict': '/predict (POST with CSV file)'
        },
        'model_status': 'ready'
    })

@app.route('/predict', methods=['POST'])
def predict_anomaly():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read CSV from upload
    df = pd.read_csv(io.BytesIO(file.read()))
    X = df.drop('label', axis=1) if 'label' in df.columns else df  # Handle labeled/unlabeled
    
    try:
        preds, shap_vals = model.predict(X)
        response = {
            'predictions': preds.tolist(),  # 0=normal, 1=attack
            'flagged_count': int(np.sum(preds > 0.5)),  # Threshold for alerts
            'shap_summary': shap_vals[0].tolist() if shap_vals is not None else None  # Sample SHAP
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model.lstm is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
