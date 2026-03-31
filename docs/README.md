# Hybrid-IDS Framework🛡️

**A Hybrid Deep Learning Framework for Anomaly-Based Intrusion Detection Systems (IDS)**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A practical two-stage hybrid IDS that dramatically reduces false positives while maintaining >98% detection accuracy. Combines Autoencoder flagging + SSA-optimised LSTM verification with LOF, SMOTE, and SHAP explainability. Ready for IoT and critical infrastructure.

---

## ✨ Key Features

- **Two-stage hybrid architecture**: Autoencoder for fast unsupervised flagging + SSA-optimised LSTM for accurate verification
- **Built-in explainability** with SHAP (see exactly which features triggered an alert)
- **Robust preprocessing**: LOF noise removal + SMOTE class balancing + dynamic handling of CIC dataset quirks
- **Professional deployment**: Streamlit dashboard + Flask REST API + Docker Compose
- **Real-world datasets**: Tested on CIC-IDS2017, CICIoT2023, UAVIDS-2025
- **Production-ready**: GitHub CI/CD, model persistence, and clean code structure

**Achieved Results**  
- Accuracy: **~98.7%**  
- False Positive Rate: **2.43%** (well below 5% target)  
- F1-Score: **99.16%**  
- 50%+ reduction in false alarms compared to single-stage models

---

## 📊 Architecture
Raw Network Flows
↓
Preprocessing (LOF + SMOTE + Scaling)
↓
Autoencoder → Anomaly Flagging (mean + 3σ)
↓
Only Flagged Sequences → SSA-LSTM Classifier
↓
SHAP Explainability → Final Alert + Feature Importance

<img width="404" height="685" alt="High-Level Architecture of the Proposed Hybrid Deep Learning Framework for Anomaly-Based IDS" src="https://github.com/user-attachments/assets/0761ff75-7a79-4373-ab03-ce37e92272ab" /> <img width="1483" height="1147" alt="Ablation Study mpact Removing Components" src="https://github.com/user-attachments/assets/4f920569-567e-4d6c-bb27-fe2e70a35eaa" />

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/hybrid-ids-framework.git
cd hybrid-ids-framework
Using Docker (Recommended)
Bashdocker compose up --build

Streamlit Dashboard → http://localhost:8501
Flask API → http://localhost:5000

Local Setup (without Docker)
Bashpython -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py

📁 Project Structure
texthybrid-ids-framework/
├── src/                  # Core source code
│   ├── hybrid_ids.py     # Main HybridIDS class
│   ├── preprocessor.py   # LOF + SMOTE + scaling
│   └── ...
├── models/               # Saved models (scaler, LOF, LSTM, etc.)
├── results/              # SHAP plots, metrics, confusion matrix
├── notebooks/            # EDA and experiments
├── docs/                 # Documentation
├── .streamlit/           # Streamlit config
├── app.py                # Flask REST API
├── dashboard.py          # Beautiful Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

📈 Results & Visualisations

Confusion Matrix (CIC-IDS2017 test set)
ROC Curve (AUC = 0.9974)
SHAP Summary Plots (feature importance)
Ablation Study (impact of each component)


All visualisations are automatically saved in the results/ folder when you run the dashboard.

🛠️ Technologies Used

Core: Python 3.12, TensorFlow/Keras, scikit-learn
Explainability: SHAP
Imbalance Handling: imbalanced-learn (SMOTE/ADASYN)
UI: Streamlit
API: Flask
Deployment: Docker + Docker Compose
CI/CD: GitHub Actions

📚 Datasets

CIC-IDS2017
CICIoT2023
UNSW-NB15
UAVIDS-2025

All datasets were preprocessed with dynamic encoding, inf/NaN handling, and 99th-percentile clipping.

📄 Citation
If you use this work in your research, please cite:
bibtex@misc{
  author = {Lalthan Sanga},
  title = {A Hybrid Deep Learning Framework for Anomaly-Based Intrusion Detection Systems},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/0xc1GenZ/hybrid-ids-framework}
}

🤝 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

🙏 Acknowledgments

Dr. M N Nachappa (Guide)
Jain (Deemed-to-be University)
All researchers whose work on CIC datasets and hybrid IDS inspired this project


Made with ❤️ for real-world cybersecurity
