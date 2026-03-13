import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
import argparse
import glob
import warnings

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from preprocessor    import Preprocessor
from autoencoder     import Autoencoder
from lstm_classifier import LSTMClassifier
from shap_explainer  import SHAPExplainer

warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LARGE_DATASET_ROWS = 100_000    # chunked CSV loading above this row count
_CHUNK_SIZE         = 50_000     # rows per chunk when loading large CSVs
_MAX_TRAIN_ROWS     = 200_000    # stratified subsample cap before model training
                                 # keeps RAM safe on multi-million row datasets
_MAX_LSTM_SEQUENCES = 50_000     # max sliding-window sequences fed to LSTM
_RANDOM_SEED        = 42

# Benign label strings found across CICIDS-2017/2018 and CICIoT-2023 datasets.
# Matched case-insensitively so 'BENIGN', 'BenignTraffic', 'benign' all work.
_BENIGN_LABELS = {'benign', 'benigntraffic', 'normal', 'background'}


# ---------------------------------------------------------------------------
# HybridIDS class
# ---------------------------------------------------------------------------
class HybridIDS:
    def __init__(self):
        self.preprocessor  = Preprocessor()
        self.autoencoder   = Autoencoder()
        self.lstm          = LSTMClassifier()
        self.explainer     = SHAPExplainer()
        self.threshold     = None
        self.is_supervised = True
        self.feature_cols  = None
        self.window_size   = 10   # overridden in fit()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        self.is_supervised = y is not None
        n_input = len(X)
        print(f"  Dataset rows fed to fit(): {n_input:,}")

        # --- Stratified subsample for very large datasets ---
        # Running LOF, Autoencoder, and LSTM on 5M+ rows causes OOM on a
        # standard workstation.  We stratify so every attack class is
        # represented even after subsampling.
        if n_input > _MAX_TRAIN_ROWS:
            X, y = _stratified_subsample(X, y, _MAX_TRAIN_ROWS, _RANDOM_SEED)
            print(f"  Subsampled to {len(X):,} rows (stratified, seed={_RANDOM_SEED}).")

        # --- Preprocess ---
        X_clean, y_clean = self.preprocessor.preprocess(X, y, fit=True)
        self.feature_cols = self.preprocessor.feature_cols
        n_clean = len(X_clean)
        print(f"  Rows after preprocessing: {n_clean:,}")

        if n_clean == 0:
            raise ValueError("No samples survived preprocessing. Check your data.")

        # --- Autoencoder (anomaly scorer) ---
        self.autoencoder.build(X_clean.shape[1])
        self.autoencoder.fit(X_clean)
        recon_errors   = self.autoencoder.predict(X_clean).mean(axis=1)
        self.threshold = recon_errors.mean() + 2 * recon_errors.std()
        print(f"  Autoencoder threshold (mean + 2std): {self.threshold:.4f}")

        if self.is_supervised:
            # --- Balance classes ---
            X_bal, y_bal = self.preprocessor.balance(X_clean, y_clean)

            # --- Adaptive window size ---
            self.window_size      = self._pick_window_size(len(X_bal))
            self.lstm.window_size = self.window_size

            # --- Build LSTM sequences (capped to avoid OOM) ---
            X_seq, y_seq = self._reshape_to_sequences(X_bal, y_bal, self.window_size)
            print(f"  LSTM sequences: {len(X_seq):,}  (window={self.window_size})")

            # --- Class weights ---
            classes = np.unique(np.asarray(y_bal))
            cw      = compute_class_weight('balanced', classes=classes, y=np.asarray(y_bal))
            cw_dict = dict(zip(classes.tolist(), cw.tolist()))
            print(f"  Class weights: {cw_dict}")

            # --- Train LSTM ---
            self.lstm.build(X_seq.shape[2])
            self.lstm.fit(X_seq, y_seq, class_weight=cw_dict)

            # --- Build SHAP explainer ---
            bg_size = min(50, len(X_seq))
            if bg_size >= 2:
                self.explainer.build(self.lstm.predict, X_seq[:bg_size])

        print("Model fitted successfully.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X):
        X_clean, _ = self.preprocessor.preprocess(X, fit=False)
        n_clean    = len(X_clean)

        if n_clean == 0:
            return np.array([]), None

        # Autoencoder reconstruction error flagging
        recon_errors = self.autoencoder.predict(X_clean).mean(axis=1)
        flagged_mask = recon_errors > self.threshold
        n_flagged    = int(flagged_mask.sum())

        if n_flagged == 0:
            return np.zeros(n_clean), None

        flagged_X = X_clean[flagged_mask]

        if self.is_supervised:
            # Expand each flagged row into (1, window_size, features) sequence
            flagged_seq   = np.repeat(
                flagged_X.values[:, np.newaxis, :], self.window_size, axis=1
            )
            flagged_preds = self.lstm.predict(flagged_seq)

            preds = np.zeros(n_clean)
            preds[flagged_mask] = flagged_preds.flatten()

            shap_values = self.explainer.explain(flagged_seq)
        else:
            preds       = flagged_mask.astype(float)
            shap_values = None

        return preds, shap_values

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_window_size(n_samples):
        """Adaptive window size — must always be < n_samples."""
        if n_samples < 20:
            return 1
        if n_samples < 100:
            return 3
        return 10

    def _reshape_to_sequences(self, X, y, window_size):
        """Sliding-window sequences, capped at _MAX_LSTM_SEQUENCES to avoid OOM."""
        n = len(X)

        if n <= window_size:
            X_seq = np.expand_dims(X.values, axis=1)
            y_seq = y.values if y is not None else None
            return X_seq, y_seq

        total_seq = n - window_size + 1

        if total_seq <= _MAX_LSTM_SEQUENCES:
            X_seq, y_seq = [], []
            for i in range(total_seq):
                X_seq.append(X.iloc[i:i + window_size].values)
                if y is not None:
                    y_seq.append(y.iloc[i + window_size - 1])
            return np.array(X_seq), (np.array(y_seq) if y is not None else None)

        # Large dataset: uniform index sampling
        print(f"  Capping sequences at {_MAX_LSTM_SEQUENCES:,} "
              f"(total possible: {total_seq:,}).")
        chosen = np.linspace(0, total_seq - 1, _MAX_LSTM_SEQUENCES, dtype=int)
        X_seq, y_seq = [], []
        for i in chosen:
            X_seq.append(X.iloc[i:i + window_size].values)
            if y is not None:
                y_seq.append(y.iloc[i + window_size - 1])
        return np.array(X_seq), (np.array(y_seq) if y is not None else None)


# ---------------------------------------------------------------------------
# Utility: stratified subsample
# ---------------------------------------------------------------------------
def _stratified_subsample(X, y, max_rows, seed=42):
    """
    Subsample X/y to at most max_rows while preserving the class ratio.
    Works for both binary and multi-class y.
    """
    rng     = np.random.default_rng(seed)
    y_arr   = np.asarray(y)
    n       = len(y_arr)
    ratio   = max_rows / n
    indices = []

    for cls in np.unique(y_arr):
        cls_idx  = np.where(y_arr == cls)[0]
        n_keep   = max(1, int(len(cls_idx) * ratio))
        chosen   = rng.choice(cls_idx, size=min(n_keep, len(cls_idx)), replace=False)
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)

    X_sub = X.iloc[indices].reset_index(drop=True)
    y_sub = pd.Series(y_arr[indices]).reset_index(drop=True)
    return X_sub, y_sub


# ---------------------------------------------------------------------------
# Utility: auto-detect the benign label string in a label column
# ---------------------------------------------------------------------------
def _detect_benign_label(series):
    """
    Return the exact label string that represents normal/benign traffic.
    Matches case-insensitively against _BENIGN_LABELS.
    Returns None if no benign label is found (dataset may be attack-only).
    """
    unique_labels = series.dropna().unique()
    for lbl in unique_labels:
        if str(lbl).strip().lower() in _BENIGN_LABELS:
            return lbl
    return None


# ---------------------------------------------------------------------------
# Utility: chunked CSV loader
# ---------------------------------------------------------------------------
def _load_csv(path):
    """
    Load a CSV.  For files estimated to be > _LARGE_DATASET_ROWS rows,
    read in chunks to keep peak RAM lower than a single pd.read_csv() call.
    """
    with open(path, 'rb') as f:
        n_lines = sum(1 for _ in f)
    estimated_rows = n_lines - 1   # subtract header row

    if estimated_rows > _LARGE_DATASET_ROWS:
        print(f"  Large file detected (~{estimated_rows:,} rows). Loading in chunks...")
        chunks = pd.read_csv(path, chunksize=_CHUNK_SIZE, low_memory=False)
        df     = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, low_memory=False)

    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid IDS Model")
    parser.add_argument(
        '--file', type=str, default='sample_flows.csv',
        help='CSV filename in /data folder (default: sample_flows.csv)'
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir     = project_root / "data"
    data_path    = data_dir / args.file

    print(f"Data directory : {data_dir}")
    print(f"Looking for    : {args.file}")

    if not data_path.exists():
        csv_files = glob.glob(str(data_dir / "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}.")
        print(f"'{args.file}' not found. Available CSVs:")
        for f in csv_files:
            print(f"  - {Path(f).name}")
        data_path = Path(csv_files[0])
        print(f"Defaulting to: {data_path.name}")

    print(f"Loading: {data_path}")
    df = _load_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # --- Auto-detect label column ---
    label_col = next(
        (c for c in df.columns if c.strip().lower() == 'label'), None
    )

    if label_col:
        print(f"Label column : '{label_col}'")
        print(f"Distribution :\n{df[label_col].value_counts()}\n")

        # FIX: detect the actual benign string instead of hardcoding 'BENIGN'.
        # CICIoT2023 uses 'BenignTraffic'; CICIDS-2017 uses 'BENIGN'.
        # Hardcoding 'BENIGN' labels every CICIoT row as attack (y=1) -> 1-class crash.
        benign_label = _detect_benign_label(df[label_col])

        if benign_label is None:
            print("  WARNING: No benign label found – treating all rows as attacks (unsupervised).")
            y = None
            X = df.drop(columns=[label_col])
        else:
            print(f"  Benign label detected: '{benign_label}'")
            y = (df[label_col] != benign_label).astype(int)
            X = df.drop(columns=[label_col])
            print(f"  y=0 (benign): {int((y == 0).sum()):,}   "
                  f"y=1 (attack): {int((y == 1).sum()):,}")
    else:
        print("  No label column found – unsupervised mode.")
        y = None
        X = df

    # --- Train ---
    print("\nTraining HybridIDS model...")
    model = HybridIDS()
    model.fit(X, y)

    # --- Predict (on the subsampled data to avoid RAM issues) ---
    print("\nMaking predictions...")
    X_pred = X.iloc[:min(len(X), _MAX_TRAIN_ROWS)]
    preds, shap_vals = model.predict(X_pred)

    n_preds = len(preds)
    if model.is_supervised:
        flagged_count = int(np.sum(preds > 0.5))
        print(f"Predictions sample : {preds[:10]}")
        print(f"Flagged anomalies  : {flagged_count}/{n_preds}  "
              f"({flagged_count / max(n_preds, 1) * 100:.1f}%)")
    else:
        flagged_count = int(np.sum(preds > 0))
        print(f"Anomaly flags      : {flagged_count}/{n_preds}  "
              f"({flagged_count / max(n_preds, 1) * 100:.1f}%)")

    # --- SHAP plot ---
    print("\nGenerating SHAP plot...")
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    if shap_vals is not None and len(shap_vals) > 0:
        shap_arr = np.array(shap_vals)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[0]

        n_plot        = min(20, shap_arr.shape[0])
        n_shap_feats  = shap_arr.shape[1]
        feature_names = model.feature_cols

        if n_shap_feats != len(feature_names):
            feature_names = [f"f{i}" for i in range(n_shap_feats)]

        plot_data = pd.DataFrame(shap_arr[:n_plot], columns=feature_names)
        shap.summary_plot(shap_arr[:n_plot], plot_data,
                          feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        out_path = results_dir / "shap_summary.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP plot saved to: {out_path}")
    else:
        print("  No SHAP data (small data, no flagged samples, or unsupervised mode).")

    print("\nDone! Model is ready.")
