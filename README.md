# Hybrid-IDS Framework

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
- **Real-world datasets**: Tested on CIC-IDS2017, CICIoT2023, and UNSW-NB15
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


*(Insert your architecture diagram here — the flowchart we created earlier)*

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/hybrid-ids-framework.git
cd hybrid-ids-framework
