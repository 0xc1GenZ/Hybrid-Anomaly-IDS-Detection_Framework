from flask import Flask, request, jsonify
from src.hybrid_ids import HybridIDS
import pandas as pd
import io

app = Flask(__name__)
model = HybridIDS()  # Load or train model here

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(io.BytesIO(file.read()))
    X = df.drop('label', axis=1)
    preds, _ = model.predict(X)
    return jsonify({'predictions': preds.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)