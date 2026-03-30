# hybrid-ids-framework

Hybrid deep learning framework for anomaly-based IDS to reduce FPR in IoT/CI networks

# Hybrid IDS Framework 🛡️

A hybrid deep learning framework for anomaly-based intrusion detection systems (IDS) using Autoencoder + SSA-optimized LSTM, with LOF/SMOTE preprocessing and SHAP explainability. Designed for IoT/CI networks to reduce FPR <5%.

## Quick Start

1. Clone: `git clone https://github.com/0xc1GenZ/hybrid-ids-framework.git`
2. Install: `pip install -r requirements.txt`
3. Train: `python src/hybrid_ids.py --file data/sample_flows.csv`
4. API: `python app.py` (test at http://localhost:5000/predict)
5. Dashboard: `streamlit run dashboard.py`

## Structure

- `/src`: Core modules (hybrid_ids.py, app.py).
- `/notebooks`: Experiments (e.g., demo.ipynb).
- `/data`: Sample CSVs (add CIC-IDS2017 here).
- `/models`: Saved models (.pkl/.h5).
- `/results`: Plots/metrics.
- `/docs`: Usage guides.

## Datasets

- CIC-IDS2017: 2.8M flows, 14 attacks.
- CICIoT2023: Millions of IoT flows, 33 attacks.
- UNSW-NB15: 2.5M records, 9 attacks.

## Deployment

- Docker: `docker build -t hybrid-ids . && docker run -p 5000:5000 hybrid-ids`.
- Heroku: `git push heroku main`.

## License

MIT – Free to use/modify.

\*\*Master's Project @ Jain University | Author: oxcy
