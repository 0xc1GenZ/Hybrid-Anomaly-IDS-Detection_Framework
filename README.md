# Hybrid Anomaly-Based Intrusion Detection Framework

**🔎A Hybrid Deep Learning Framework for Anomaly-Based Intrusion Detection Systems (IDS)🔍**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A practical **two-stage hybrid IDS** that dramatically reduces false positives while maintaining **>98% detection accuracy**.  
> Combines Autoencoder flagging + SSA-optimised LSTM verification with LOF, SMOTE, and **SHAP explainability**.  
> Ready for IoT, critical infrastructure, and real-world 0-day & known attack detection.

---

## 🎥 Live Demonstration!🍿
🔴REC


https://github.com/user-attachments/assets/7b7c4f42-1a3c-4ab3-a147-2dc3cf1a9ef8

---

**Achieved Results**  
- **Accuracy**: ~98.7%  
- **False Positive Rate**: 2.43% (well below 5% target)  
- **F1-Score**: 99.16%  
- **50%+ reduction** in false alarms compared to single-stage models

## ✨ Key Features

- **Two-stage hybrid architecture**: Autoencoder for fast unsupervised flagging + SSA-optimised LSTM for accurate verification
- **Built-in explainability** with SHAP (see exactly which features triggered an alert)
- **Robust preprocessing**: LOF noise removal + SMOTE class balancing + dynamic handling of CIC dataset quirks
- **Professional deployment**: Streamlit dashboard + Flask REST API + Docker Compose
- **Real-world datasets**: Tested on CIC-IDS2017, CICIoT2023, UAVIDS-2025
- **Production-ready**: One-click scripts, logging, GitHub CI/CD, model persistence, and clean code structure

---

## 📊 Architecture

Raw Network Flows → Preprocessing (LOF + SMOTE + Scaling) → Autoencoder → Anomaly Flagging (mean + 3σ) → Only Flagged Sequences → SSA-LSTM Classifier → SHAP Explainability → Final Alert + Feature Importance
<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/0761ff75-7a79-4373-ab03-ce37e92272ab" alt="System Architecture of Proposed Hybrid Model" width="100%" style="max-width: 100%; height: auto;">
      <br><strong>System Architecture</strong><br>
      <em>Raw Network Flows → Preprocessing → Autoencoder + SSA-LSTM + SHAP</em>
    </td>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/4f920569-567e-4d6c-bb27-fe2e70a35eaa"alt="Ablation Study: Impact of Removing Components" width="100%" style="max-width: 100%; height: auto;">
      <br><strong>Ablation Study: Impact of Each Component</strong><br>
      <em>Full Model vs. Removing LOF / SMOTE / SSA / Autoencoder</em>
    </td>
  </tr>
</table>

---
## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/hybrid-ids-framework.git
cd hybrid-ids-framework
Local Setup (without Docker)
Bashpython -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py or executing run_dashboard.bat 
Streamlit Dashboard → http://localhost:8501
Flask API → http://localhost:5000
OR
Using Docker
### One-Click Production Deployment (Recommended)
**Windows**  
Double-click **`run_production.bat`**

**macOS / Linux**  
```bash
./run_production.sh

These scripts automatically:
- Check Docker status
- Stop old containers
- Build and start both services
- Save detailed logs to `logs/production.log`

### Manual Commands

```bash
docker compose up --build -d     # Start in background
docker compose down              # Stop all services

📁 Project Structure
texthybrid-ids-framework/
├── src/                  # Core source code
│   ├── hybrid_ids.py     # Main HybridIDS class
│   ├── preprocessor.py   # LOF + SMOTE + scaling
│   └── ...
├── models/               # Saved models (scaler, LOF, LSTM, etc.)
├── results/              # SHAP plots, metrics, confusion matrix
├── notebooks/            # EDA and experiments
├── logs/                 # Production run logs
├── .streamlit/           # Streamlit config
├── app.py                # Flask REST API
├── dashboard.py          # Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── run_production.bat    # Windows one-click start
├── run_production.sh     # macOS/Linux one-click start
├── stop_production.bat
├── stop_production.sh
├── requirements.txt
└── README.md

📈 Results & Visualisations
Confusion Matrix on(CIC-IDS2017 test set)
ROC Curve (AUC = 0.9974)
SHAP Summary Plots (feature importance)
All visualisations are automatically saved in the results/ folder when you run the dashboard.

🛠️ Technologies Used
+______________________________________________________+
| Category           | Technology                      |
+______________________________________________________+   
| Language           |  Python 3.12                    |
| Deep Learning      |  TensorFlow / Keras             |
| ML Tools/Imbalance-|  scikit-learn, imbalanced-learn,|
| Handling           |  (SMOTE/ADASYN)                 |
| Explainability     |  SHAP                           |
| UI                 |  Streamlit                      |
| API                |  Flask                          |
| Deployment         |  Docker + Docker Compose        |
| CI/CD              |  GitHub Actions                 |
+______________________________________________________+

📚 Datasets

CIC-IDS2017
CICIoT2023
UNSW-NB15
UAVIDS-2025

All datasets were preprocessed with dynamic encoding, inf/NaN handling, and 99th-percentile clipping.

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
Professor
Jain (Deemed-to-be University)
All researchers whose work on CIC datasets and hybrid IDS inspired this project


Made with ❤️ for real-world cybersecurity.
