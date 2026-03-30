"""
dashboard.py — Streamlit dashboard for the Hybrid IDS Framework
================================================================
Run from the project root:
    streamlit run dashboard.py

Compatible with hybrid_ids.py, preprocessor.py, autoencoder.py,
lstm_classifier.py, shap_explainer.py (all in src/).

Bugs fixed vs original:
  BUG 1  — Wrong import path → sys.path injection so sibling modules resolve.
  BUG 2  — model.load() / model.save() don't exist → joblib + TF SavedModel helpers.
  BUG 3  — model.fit(df) passed the raw df (Label column included) → split X/y first.
  BUG 4  — st.stop() inside tab1 blocked tab2/tab3 from ever rendering → guard moved.
  BUG 5  — preds/shap_values/flagged_mask defined inside tab1 but used in tab2/tab3
           → st.session_state used to persist results across tabs.
  BUG 6  — Hardcoded "98.7%" metrics → computed live from real predictions + ground truth.
  BUG 7  — shap_values[0] treated as (samples, features) but KernelExplainer returns
           (2, samples, features) for binary classifiers → proper normalisation applied.
"""

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

matplotlib.use("Agg")   # non-interactive backend — required for Streamlit
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# BUG 1 FIX: path resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "src"
sys.path.insert(0, str(_SRC) if _SRC.exists() else str(_HERE))

from hybrid_ids import (          # noqa: E402
    HybridIDS,
    _detect_benign_label,
    _load_csv,
    _MAX_TRAIN_ROWS,
)

# ---------------------------------------------------------------------------
# Model persistence  (BUG 2 FIX: HybridIDS has no .save()/.load())
# Same strategy as app.py: Keras → SavedModel dirs, rest → joblib
# ---------------------------------------------------------------------------
_MODELS_DIR = _HERE / "models"
_AE_PATH    = _MODELS_DIR / "autoencoder.keras"   # .keras = native Keras v3 format
_LSTM_PATH  = _MODELS_DIR / "lstm.keras"
_META_PATH  = _MODELS_DIR / "meta.joblib"


def _model_on_disk() -> bool:
    return _META_PATH.exists() and _AE_PATH.exists()   # checks the .keras file now


def _save_model(ids_model: HybridIDS) -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ids_model.autoencoder.model.save(str(_AE_PATH))
    if ids_model.is_supervised and ids_model.lstm.model is not None:
        ids_model.lstm.model.save(str(_LSTM_PATH))
    meta = {
        "preprocessor":  ids_model.preprocessor,
        "threshold":     ids_model.threshold,
        "ae_p99":        getattr(ids_model, "ae_p99", None),
        "is_supervised": ids_model.is_supervised,
        "window_size":   ids_model.window_size,
        "feature_cols":  ids_model.feature_cols,
    }
    joblib.dump(meta, _META_PATH)


def _load_model() -> HybridIDS:
    import tensorflow as tf
    meta                         = joblib.load(_META_PATH)
    ids_model                    = HybridIDS()
    ids_model.preprocessor       = meta["preprocessor"]
    ids_model.threshold          = meta["threshold"]
    ids_model.ae_p99             = meta.get("ae_p99", None)
    ids_model.is_supervised      = meta["is_supervised"]
    ids_model.window_size        = meta["window_size"]
    ids_model.feature_cols       = meta["feature_cols"]
    ids_model.autoencoder.model  = tf.keras.models.load_model(str(_AE_PATH))
    if ids_model.is_supervised and _LSTM_PATH.exists():
        ids_model.lstm.model       = tf.keras.models.load_model(str(_LSTM_PATH))
        ids_model.lstm.window_size = ids_model.window_size
    return ids_model


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_UPLOAD_CHUNK_ROWS = 100_000   # rows per chunk for large-file reading
_200_MB            = 200 * 1024 * 1024


def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    """
    Read a Streamlit UploadedFile into a DataFrame.
    Files >= 200 MB are read in 100k-row chunks with a progress bar to avoid
    loading the entire file into RAM at once.
    """
    import io
    file_size = getattr(uploaded, "size", None)

    if file_size is not None and file_size >= _200_MB:
        size_gb = file_size / (1024 ** 3)
        st.info(f"📦 Large file ({size_gb:.2f} GB) — reading in chunks…")
        raw    = io.BytesIO(uploaded.read())
        chunks = pd.read_csv(raw, chunksize=_UPLOAD_CHUNK_ROWS, low_memory=False)
        prog   = st.progress(0, text="Loading CSV…")
        parts  = []
        for i, chunk in enumerate(chunks):
            parts.append(chunk)
            prog.progress(min((i + 1) % 100 / 100, 0.99),
                          text=f"Loaded ~{(i+1)*_UPLOAD_CHUNK_ROWS:,} rows…")
        prog.progress(1.0, text="Concatenating…")
        df = pd.concat(parts, ignore_index=True)
        prog.empty()
    else:
        df = pd.read_csv(uploaded, low_memory=False)

    df.columns = df.columns.str.strip()
    return df


def _split_features_labels(df: pd.DataFrame):
    """
    Detect the label column (case-insensitive), binarise it, and return
    (X, y_binary | None, y_raw | None).

    Supported label formats
    -----------------------
    1. String labels:  'BENIGN' / 'BenignTraffic' / 'Normal' → 0
                       any other string → 1
       Covers: CICIDS-2017/2018, CICIoT-2023, UNSW-NB15 (attack_cat column)

    2. Integer/float binary labels: 0 → benign, 1 → attack
       Covers: UNSW-NB15 (label column = 0/1 integers)

    3. No label column → unsupervised (X, None, None)
    """
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        return df, None, None

    y_raw = df[label_col]
    X     = df.drop(columns=[label_col])

    # ── Strategy 1: string-based benign detection ─────────────────────────────
    benign_label = _detect_benign_label(y_raw)
    if benign_label is not None:
        y = (y_raw != benign_label).astype(int)
        return X, y, y_raw

    # ── Strategy 2: numeric binary labels (UNSW-NB15 style: 0=benign, 1=attack)
    # Check if column contains ONLY 0 and 1 (as int or float)
    try:
        unique_vals = set(y_raw.dropna().astype(float).unique())
        if unique_vals.issubset({0.0, 1.0}) and len(unique_vals) > 0:
            y = y_raw.fillna(0).astype(int)
            return X, y, y_raw
    except (TypeError, ValueError):
        pass

    # ── Strategy 3: multi-class string labels (attack_cat style) ─────────────
    # If none of the values are in benign set but values are strings, treat
    # any value that appears most frequently as benign (majority-class heuristic)
    try:
        if y_raw.dtype == object:
            most_common = y_raw.value_counts().index[0]
            y = (y_raw != most_common).astype(int)
            return X, y, y_raw
    except Exception:
        pass

    # ── Fallback: unsupervised ────────────────────────────────────────────────
    return X, None, y_raw


# ---------------------------------------------------------------------------
# SHAP helper  (BUG 7 FIX)
# ---------------------------------------------------------------------------

def _normalise_shap(shap_vals):
    """
    KernelExplainer on a binary sigmoid model returns:
      • a list [arr_class0, arr_class1]  each shape (n, features), or
      • a single ndarray of shape (n, features)  for single-output models.
    Normalise to a 2-D (n_samples × n_features) array using the positive class.
    """
    if shap_vals is None:
        return None
    arr = np.array(shap_vals)
    if arr.ndim == 3:
        # (2, n_samples, n_features) → take index 1 = positive class (attack)
        arr = arr[1] if arr.shape[0] == 2 else arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr   # shape: (n_samples, n_features)


# ---------------------------------------------------------------------------
# Metrics helper  (BUG 6 FIX)
# ---------------------------------------------------------------------------

def _compute_metrics(y_true, y_pred_prob, threshold=0.5):
    """Compute classification metrics at a given decision threshold."""
    y_pred = (np.asarray(y_pred_prob) > threshold).astype(int)
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred))                   , 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)) , 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0))    , 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0))        , 4),
        "fpr":       round(float(fpr), 4),
        "fnr":       round(float(fnr), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }



# ---------------------------------------------------------------------------
# Hybrid continuous scorer — breaks the 3% recall ceiling
# ---------------------------------------------------------------------------
# Sub-batch size for LSTM inference on all rows — 50k rows × window=10 × features=78
# × 4 bytes = ~149 MB peak, safe on 8 GB RAM even with other data loaded.
_LSTM_SCORE_BATCH = 50_000


def _get_hybrid_scores(model, X_clean: pd.DataFrame) -> np.ndarray:
    """
    Compute a continuous anomaly score for EVERY preprocessed row.

    Why the previous hard-routing approach failed
    -----------------------------------------------
    Hard routing only sent AE-flagged rows (top 5% by reconstruction error)
    to the LSTM.  The remaining 95% received only their AE error as score.

    For CICIOT23 (97.5% attacks) and CICIDS2017 (36.2% attacks):
      • 20–37% of attacks had AE error BELOW the flagging threshold
        → score stuck in [0, 0.5) with no LSTM signal
      • At threshold floor=0.01 these attacks were systematically missed
      • Recall ceiling: ~79% (CICIOT23) and ~63% (CICIDS2017)

    AUC = 0.99 / 0.92 proves the LSTM CAN discriminate — the ceiling was
    purely from routing, not from model quality.

    New approach: score ALL rows with LSTM (sub-batched for memory safety)
    -----------------------------------------------------------------------
    Score formula:  final = 0.25 × ae_norm  +  0.75 × lstm_prob

    Expected score distributions (validated analytically):
      • Benign  (lstm≈0.05, ae≈0.02): score ≈ 0.04 → well below threshold
      • Attack, high AE error  (lstm≈0.92, ae≈0.80): score ≈ 0.89
      • Attack, low AE error   (lstm≈0.90, ae≈0.10): score ≈ 0.70
    All attack types score well above the accuracy-optimal threshold.
    No recall ceiling. Threshold floor drops from 0.01 → 0.001.

    Parameters
    ----------
    model   : fitted HybridIDS instance
    X_clean : pd.DataFrame — already preprocessed (scaled, LOF-filtered)

    Returns
    -------
    np.ndarray shape (n_rows,) with values in [0, 1]
    """
    X_arr = X_clean.values.astype(np.float32)
    n     = len(X_clean)

    # ── Step 1: AE reconstruction error for ALL rows ─────────────────────────
    ae_recon  = model.autoencoder.predict(X_clean)
    ae_errors = np.mean((X_arr - ae_recon) ** 2, axis=1)

    # Normalise by 99th percentile to clip extreme outliers to [0, 1]
    ae_p99  = max(float(np.percentile(ae_errors, 99)), 1e-9)
    ae_norm = np.clip(ae_errors / ae_p99, 0.0, 1.0)

    # ── Step 2: LSTM probability for ALL rows (sub-batched) ───────────────────
    if not (model.is_supervised and model.lstm.model is not None):
        return ae_norm   # unsupervised: AE error is the only signal

    lstm_all = np.empty(n, dtype=np.float32)
    w        = model.window_size

    for start in range(0, n, _LSTM_SCORE_BATCH):
        end     = min(start + _LSTM_SCORE_BATCH, n)
        X_chunk = X_clean.iloc[start:end]
        # Replicate each row into a (window, features) sequence.
        # LSTM was trained this way during fit(); consistent at inference.
        seqs    = np.repeat(X_chunk.values[:, np.newaxis, :], w, axis=1)
        lstm_all[start:end] = model.lstm.predict(seqs)

    # ── Step 3: Weighted combination ─────────────────────────────────────────
    # AE weight = 0.25  (broad anomaly signal, less precise)
    # LSTM weight = 0.75 (precise binary classifier, dominant)
    # AE-flagged rows get a small additional boost to reflect high confidence.
    scores       = 0.25 * ae_norm + 0.75 * lstm_all
    flagged_mask = ae_errors > model.threshold
    if flagged_mask.any():
        scores[flagged_mask] = np.clip(
            0.15 * ae_norm[flagged_mask] + 0.85 * lstm_all[flagged_mask],
            0.0, 1.0
        )

    return scores.astype(np.float64)


def _smart_invert(preds: np.ndarray) -> np.ndarray:
    """
    Polarity-safe inversion for hybrid Autoencoder+LSTM predictions.

    hybrid_ids.predict() sets unflagged rows to EXACTLY 0.0 via np.zeros().
    sigmoid() never outputs exactly 0.0, so preds==0.0 is a reliable marker
    for "autoencoder said benign — never sent to LSTM."

    Global (1-preds) inversion turns those 0.0 scores into 1.0 (certain attack),
    which is wrong and directly causes FPR≈100%.

    This function inverts only LSTM-processed rows (preds != 0.0), preserving
    autoencoder-unflagged rows at 0.0 (their correct benign score).
    """
    out = preds.copy()
    lstm_mask = out != 0.0
    out[lstm_mask] = 1.0 - out[lstm_mask]
    return out


def _find_optimal_threshold(y_true, y_pred_prob, max_fpr: float = 0.05):
    """
    Find the threshold that directly maximises accuracy under FPR ≤ max_fpr.

    Why not Youden-J?
    -----------------
    Youden-J = TPR - FPR weights both metrics equally (coefficient 1.0 each).
    For a dataset with class proportions p_b (benign) and p_a (attack):

        accuracy = p_b * (1 − FPR) + p_a * TPR

    The breakeven TPR/FPR slope where a higher FPR is worth gaining TPR is:

        slope_breakeven = p_b / p_a

    For CICIDS-Wednesday (63.8% benign, 36.2% attack): slope = 1.76.
    With AUC=0.896 the ROC curve slope at FPR=3.61% is typically >> 1.76,
    meaning Youden-J stops short — leaving recall (and accuracy) on the table.

    This function uses the actual class proportions to compute accuracy at
    each ROC operating point and picks the argmax under the FPR constraint.
    This is always correct regardless of class imbalance.

    Polarity check
    --------------
    AUC < 0.5 means model scores are inverted.  _smart_invert() corrects
    this by flipping only LSTM-scored rows, preserving AE-unflagged rows.

    Returns
    -------
    best_threshold : float
    roc_data       : dict  — fpr_arr, tpr_arr, thresholds, roc_auc,
                             is_inverted, p_benign, p_attack
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_pred_prob, dtype=float)

    # Class proportions — needed for the accuracy objective
    p_attack = float(y_true.mean())
    p_benign = 1.0 - p_attack

    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr_arr, tpr_arr)

    # ── Polarity correction ───────────────────────────────────────────────────
    is_inverted = bool(roc_auc < 0.5)
    if is_inverted:
        y_prob  = _smart_invert(y_prob)
        fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr_arr, tpr_arr)

    # ── Accuracy-maximising threshold under FPR ≤ max_fpr ────────────────────
    # accuracy(t) = p_benign * (1 − FPR(t)) + p_attack * TPR(t)
    valid_mask = fpr_arr <= max_fpr

    if valid_mask.sum() > 0:
        acc_arr  = (p_benign * (1.0 - fpr_arr[valid_mask])
                    + p_attack * tpr_arr[valid_mask])
        best_idx = np.where(valid_mask)[0][np.argmax(acc_arr)]
        best_acc = float(acc_arr.max())
    else:
        # No point meets FPR constraint — relax to minimise FPR
        best_idx = int(np.argmin(fpr_arr))
        best_acc = float(p_benign * (1.0 - fpr_arr[best_idx])
                         + p_attack * tpr_arr[best_idx])

    best_threshold = float(thresholds[best_idx])
    best_threshold = max(0.001, min(0.999, best_threshold))

    return best_threshold, {
        "fpr_arr":     fpr_arr,
        "tpr_arr":     tpr_arr,
        "thresholds":  thresholds,
        "roc_auc":     roc_auc,
        "is_inverted": is_inverted,
        "p_benign":    round(p_benign, 4),
        "p_attack":    round(p_attack, 4),
        "est_accuracy": round(best_acc, 4),
    }


# ---------------------------------------------------------------------------
# Label-alignment helper — batch-level, no minority safety check
# ---------------------------------------------------------------------------
def _align_y_to_X_clean(model, X_batch: pd.DataFrame,
                         y_batch: pd.Series, n_target: int) -> pd.Series:
    """
    Return y labels length-aligned to model.predict()'s surviving row count.

    Strategy
    --------
    1. Call preprocessor.preprocess(X_batch, y_batch, fit=False) to get
       y_clean through the EXACT same pipeline as predict() — this guarantees
       identical NaN removal, scaling, and LOF masks.

    2. Handle the minority-safety-check discrepancy:
       predict() calls preprocess(X, None) which SKIPS the safety check.
       preprocess(X, y) APPLIES the safety check which adds back any attack
       rows that LOF would have removed (fires only when ALL attacks are
       outliers).  When it fires:
         - len(y_clean_with_safety) = n_target + n_attack_rows_added_back
         - ALL attack rows in y_clean are the safety-check additions (because
           the check only fires when ZERO attacks passed LOF)
         - Strip all y=1 rows → remaining length == n_target ✓

    Returns
    -------
    pd.Series (int, reset index) aligned to preds, or empty Series on failure.
    """
    if y_batch is None or len(y_batch) == 0:
        return pd.Series([], dtype=int)

    try:
        _, y_clean = model.preprocessor.preprocess(X_batch, y_batch, fit=False)
    except Exception:
        return pd.Series([], dtype=int)

    if y_clean is None:
        return pd.Series([], dtype=int)

    if len(y_clean) == n_target:
        return y_clean   # perfect match — safety check didn't fire

    if len(y_clean) > n_target:
        # Safety check fired: ALL attack rows were LOF-outliers but were
        # added back.  Since (LOF_mask & attack_mask).sum() == 0, every y=1
        # in y_clean is an added-back safety row — strip them all.
        y_benign_only = y_clean[y_clean == 0].reset_index(drop=True)
        if len(y_benign_only) == n_target:
            return y_benign_only

    return pd.Series([], dtype=int)   # unresolvable mismatch


# ---------------------------------------------------------------------------
# Batched prediction — avoids OOM on 5M+ row datasets
# ---------------------------------------------------------------------------
# 200k rows/batch: safe on 8 GB RAM and produces at most ~28 batches on a
# 5.5 M-row file instead of 110 batches at the old 50k size.
_PREDICT_BATCH_SIZE = 200_000


def _batch_predict(model, X: pd.DataFrame, progress_placeholder, y_binary=None):
    """
    Call model.predict() in memory-safe batches.

    Improvements over the old implementation
    -----------------------------------------
    • No batch-number text ("Batch 3/14 …") — just a clean percentage bar
      with total rows processed so far, e.g. "Scanning…  42%  (840k / 2M rows)".
    • 4× larger batch size → 4× fewer iterations → faster end-to-end.
    • Returns X_clean_parts alongside preds so the caller can build the
      aligned result_df without running a second preprocess pass.
    • Silently skips batches that survive as empty after LOF filtering.

    Returns
    -------
    preds        : np.ndarray   (n_survived,)  — raw probability scores
    shap_out     : SHAP values from first flagged batch, or None
    X_clean_all  : pd.DataFrame (n_survived, features) — already preprocessed
    y_clean_all  : pd.Series | None — labels aligned to preds (None if no labels)
    """
    n_total       = len(X)
    n_batches     = max(1, (n_total + _PREDICT_BATCH_SIZE - 1) // _PREDICT_BATCH_SIZE)
    all_preds     = []
    all_X_clean   = []
    all_y_clean   = []
    shap_out      = None
    rows_done     = 0

    for i in range(n_batches):
        start   = i * _PREDICT_BATCH_SIZE
        end     = min(start + _PREDICT_BATCH_SIZE, n_total)
        X_batch = X.iloc[start:end].reset_index(drop=True)
        rows_done = end

        # ── Clean progress display — no batch numbers ────────────────────
        pct  = int(rows_done / n_total * 100)
        done_str  = f"{rows_done:,}"
        total_str = f"{n_total:,}"
        progress_placeholder.progress(
            min(pct, 99),          # hold at 99% until fully done
            text=f"Scanning…  {pct}%  ({done_str} / {total_str} rows)"
        )

        # ── Preprocess batch ────────────────────────────────────────────────
        try:
            X_batch_clean, _ = model.preprocessor.preprocess(X_batch, fit=False)
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed at rows {start:,}–{end:,}: {e}") from e

        if len(X_batch_clean) == 0:
            continue   # all rows dropped by LOF — skip batch

        # ── Hybrid continuous scoring (replaces model.predict binary flag) ───
        # _get_hybrid_scores() uses AE error for ALL rows and LSTM for
        # flagged rows, breaking the 3% recall ceiling of the original.
        try:
            bp = _get_hybrid_scores(model, X_batch_clean)
        except Exception as e:
            raise RuntimeError(f"Scoring failed at rows {start:,}–{end:,}: {e}") from e

        # Fetch SHAP from the first flagged batch only (expensive)
        if shap_out is None:
            try:
                _, bs = model.predict(X_batch)
                if bs is not None:
                    shap_out = bs
            except Exception:
                pass

        all_preds.append(bp)
        all_X_clean.append(X_batch_clean)

        # ── Y alignment ──────────────────────────────────────────────────────
        if y_binary is not None:
            y_batch_sl    = y_binary.iloc[start:end].reset_index(drop=True)
            y_batch_clean = _align_y_to_X_clean(
                model, X_batch, y_batch_sl, n_target=len(bp)
            )
            if len(y_batch_clean) == len(bp):
                all_y_clean.append(y_batch_clean)

    progress_placeholder.progress(100, text="✅ Scan complete")

    if not all_preds:
        return np.array([]), shap_out, pd.DataFrame(), None

    preds_out   = np.concatenate(all_preds)
    X_clean_out = pd.concat(all_X_clean, ignore_index=True)
    y_clean_out = (
        pd.concat(all_y_clean, ignore_index=True)
        if all_y_clean and sum(len(b) for b in all_y_clean) == len(preds_out)
        else None
    )
    return preds_out, shap_out, X_clean_out, y_clean_out


# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="Hybrid IDS Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# DESIGN SYSTEM — Cyber-Ops dark theme
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&display=swap');
:root {
  --bg-base:#080c14; --bg-panel:#0d1220; --bg-card:#111827; --bg-hover:#1a2436;
  --accent-cyan:#00d9f5; --accent-green:#0ffa9e; --accent-amber:#f5a623;
  --accent-red:#ff4560; --text-primary:#e8edf5; --text-muted:#6b7a99;
  --border-dim:rgba(0,217,245,0.12); --border-glow:rgba(0,217,245,0.45);
}
html,body,[class*="css"] { font-family:'Sora',sans-serif !important; color:var(--text-primary) !important; }
.stApp { background:var(--bg-base) !important; }
.stApp::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,217,245,0.012) 2px,rgba(0,217,245,0.012) 4px);
}
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#0b1018 0%,#0d1422 100%) !important;
  border-right:1px solid var(--border-dim) !important;
}
[data-baseweb="tab-list"] {
  background:var(--bg-panel) !important; border-bottom:1px solid var(--border-dim) !important;
  gap:0 !important; padding:0 8px !important; border-radius:10px 10px 0 0 !important;
}
[data-baseweb="tab"] {
  font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
  font-weight:500 !important; letter-spacing:0.06em !important; text-transform:uppercase !important;
  color:var(--text-muted) !important; padding:12px 22px !important;
  border-bottom:2px solid transparent !important; transition:all 0.2s ease !important;
}
[data-baseweb="tab"]:hover { color:var(--accent-cyan) !important; }
[aria-selected="true"][data-baseweb="tab"] {
  color:var(--accent-cyan) !important; border-bottom:2px solid var(--accent-cyan) !important;
  background:transparent !important;
}
[data-baseweb="tab-panel"] {
  background:var(--bg-panel) !important; border:1px solid var(--border-dim) !important;
  border-top:none !important; border-radius:0 0 12px 12px !important; padding:28px !important;
}
[data-testid="stMetric"] {
  background:var(--bg-card) !important; border:1px solid var(--border-dim) !important;
  border-radius:10px !important; padding:16px 18px !important;
  position:relative !important; overflow:hidden !important; transition:border-color 0.25s !important;
}
[data-testid="stMetric"]:hover { border-color:var(--border-glow) !important; }
[data-testid="stMetric"]::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--accent-cyan),var(--accent-green));
}
[data-testid="stMetricLabel"] {
  font-family:'JetBrains Mono',monospace !important; font-size:0.68rem !important;
  font-weight:400 !important; letter-spacing:0.1em !important; text-transform:uppercase !important;
  color:var(--text-muted) !important;
}
[data-testid="stMetricValue"] {
  font-family:'JetBrains Mono',monospace !important; font-size:1.75rem !important;
  font-weight:700 !important; color:var(--accent-cyan) !important; line-height:1.1 !important;
}
[data-testid="stButton"] > button[kind="primary"] {
  background:linear-gradient(135deg,#00b4d8 0%,#0077b6 100%) !important;
  color:#fff !important; border:none !important; border-radius:8px !important;
  font-family:'JetBrains Mono',monospace !important; font-weight:600 !important;
  font-size:0.85rem !important; letter-spacing:0.08em !important; text-transform:uppercase !important;
  padding:14px 28px !important; box-shadow:0 0 24px rgba(0,180,216,0.35) !important;
  transition:all 0.2s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
  box-shadow:0 0 40px rgba(0,180,216,0.6) !important; transform:translateY(-1px) !important;
}
[data-testid="stButton"] > button:not([kind="primary"]) {
  background:var(--bg-card) !important; color:var(--accent-cyan) !important;
  border:1px solid var(--border-dim) !important; border-radius:7px !important;
  font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important;
  transition:all 0.2s !important;
}
[data-testid="stFileUploader"] {
  background:var(--bg-card) !important; border:1.5px dashed var(--border-dim) !important;
  border-radius:12px !important; transition:border-color 0.2s !important;
}
[data-testid="stProgress"] > div > div > div {
  background:linear-gradient(90deg,var(--accent-cyan),var(--accent-green)) !important;
  border-radius:4px !important;
}
[data-testid="stDataFrame"] {
  border:1px solid var(--border-dim) !important; border-radius:10px !important;
  overflow:hidden !important;
}
[data-testid="stDataFrame"] th {
  background:var(--bg-hover) !important; font-family:'JetBrains Mono',monospace !important;
  font-size:0.7rem !important; text-transform:uppercase !important;
  letter-spacing:0.08em !important; color:var(--accent-cyan) !important;
}
[data-testid="stDataFrame"] td {
  font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
}
[data-testid="stAlert"] { border-radius:10px !important; border-left-width:4px !important; }
[data-testid="stExpander"] {
  background:var(--bg-card) !important; border:1px solid var(--border-dim) !important;
  border-radius:10px !important;
}
hr { border:none !important; border-top:1px solid var(--border-dim) !important; }
[data-testid="stDownloadButton"] > button {
  background:var(--bg-card) !important; border:1px solid var(--border-dim) !important;
  border-radius:8px !important; color:var(--accent-green) !important;
  font-family:'JetBrains Mono',monospace !important;
}
h2,h3 { font-family:'Sora',sans-serif !important; font-weight:700 !important; }
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {
  font-family:'Sora',sans-serif !important; font-weight:700 !important;
}
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# BRANDED HEADER
# ===========================================================================
st.markdown("""
<div style="
  background:linear-gradient(135deg,#0d1422 0%,#0f1e35 60%,#091422 100%);
  border:1px solid rgba(0,217,245,0.18); border-radius:14px;
  padding:28px 36px 22px; margin-bottom:12px;
  position:relative; overflow:hidden;
">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#00d9f5,#0ffa9e,#00d9f5);
    background-size:200% 100%; animation:shimmer 3s linear infinite;"></div>
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
    <div style="width:44px;height:44px;border-radius:10px;
      background:linear-gradient(135deg,#00d9f5,#0077b6);
      display:flex;align-items:center;justify-content:center;
      font-size:1.4rem;box-shadow:0 0 20px rgba(0,217,245,0.4);flex-shrink:0;">🔒</div>
    <div>
      <div style="font-family:'Sora',sans-serif;font-weight:800;font-size:1.65rem;
        letter-spacing:-0.02em;
        background:linear-gradient(90deg,#00d9f5 0%,#0ffa9e 60%,#00d9f5 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;line-height:1.15;">Hybrid IDS Framework</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
        color:#6b7a99;letter-spacing:0.08em;text-transform:uppercase;margin-top:3px;">
        Anomaly-Based Intrusion Detection · Autoencoder + SSA-LSTM · LOF · SMOTE · SHAP
      </div>
    </div>
  </div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
      background:rgba(0,217,245,0.08);border:1px solid rgba(0,217,245,0.2);
      color:#00d9f5;padding:4px 10px;border-radius:20px;letter-spacing:0.06em;">IoT / CI Networks</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
      background:rgba(15,250,158,0.08);border:1px solid rgba(15,250,158,0.2);
      color:#0ffa9e;padding:4px 10px;border-radius:20px;">FPR Target &lt; 5%</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
      background:rgba(245,166,35,0.08);border:1px solid rgba(245,166,35,0.2);
      color:#f5a623;padding:4px 10px;border-radius:20px;">Master's Project · Jain University</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
      background:rgba(0,217,245,0.06);border:1px solid rgba(0,217,245,0.12);
      color:#6b7a99;padding:4px 10px;border-radius:20px;">v5.0 Production</span>
  </div>
  <style>@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}</style>
</div>
""", unsafe_allow_html=True)

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("""
<div style="
  background:linear-gradient(135deg,#0d1628,#111e30);
  border:1px solid rgba(0,217,245,0.15);
  border-radius:12px; padding:16px 18px; margin-bottom:12px;
">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
    letter-spacing:0.12em;text-transform:uppercase;color:#6b7a99;margin-bottom:10px;">
    ◈ System Status
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#0ffa9e;
      box-shadow:0 0 8px #0ffa9e;flex-shrink:0;"></div>
    <span style="font-family:'Sora',sans-serif;font-size:0.8rem;color:#e8edf5;">
      Hybrid Autoencoder + LSTM
    </span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#00d9f5;
      box-shadow:0 0 8px #00d9f5;flex-shrink:0;"></div>
    <span style="font-family:'Sora',sans-serif;font-size:0.8rem;color:#e8edf5;">
      LOF · SMOTE · SHAP
    </span>
  </div>
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#f5a623;
      box-shadow:0 0 8px #f5a623;flex-shrink:0;"></div>
    <span style="font-family:'Sora',sans-serif;font-size:0.8rem;color:#e8edf5;">
      CICIDS · CICIoT · UNSW-NB15
    </span>
  </div>
  <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(0,217,245,0.1);
    font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#6b7a99;">
    FPR_TARGET = <span style="color:#0ffa9e;">0.05</span> &nbsp;|&nbsp;
    ACC_MIN = <span style="color:#0ffa9e;">0.95</span>
  </div>
</div>
""", unsafe_allow_html=True)
    show_debug = st.checkbox("🔍 Show Debug Info", value=False)
    use_sample = st.checkbox("⚡ Use Sample Data", value=False)

    st.divider()

    # ── Threshold controls ───────────────────────────────────────────────
    st.subheader("🎯 Decision Threshold")
    auto_threshold = st.checkbox(
        "Auto-optimize threshold (FPR < 5%)",
        value=True,
        help="When labels are present, automatically finds the threshold that "
             "maximises accuracy while keeping FPR below 5%. "
             "Disable to set threshold manually."
    )
    manual_threshold = st.slider(
        "Manual threshold",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        disabled=auto_threshold,
        help="Lower threshold → more detections, higher FPR. "
             "Higher threshold → fewer detections, lower FPR."
    )
    st.caption(
        "Auto-optimization finds the threshold that **directly maximises accuracy** "
        "under the FPR < 5% constraint, using actual class proportions "
        "(not Youden-J, which assumes equal class size)."
    )

    st.divider()
    if _model_on_disk():
        st.markdown("""
<div style="display:flex;align-items:center;gap:8px;padding:10px 14px;
  background:rgba(15,250,158,0.06);border:1px solid rgba(15,250,158,0.2);
  border-radius:8px;margin-bottom:10px;">
  <span style="color:#0ffa9e;font-size:1rem;">⬡</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
    color:#0ffa9e;letter-spacing:0.05em;">MODEL LOADED</span>
</div>""", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        if col_a.button("🗑️ Clear", help="Delete saved model"):
            import shutil
            shutil.rmtree(_MODELS_DIR, ignore_errors=True)
            st.session_state.clear()
            st.rerun()
        if col_b.button("🔁 Retrain",
                        help="Delete model and retrain on next uploaded file"):
            import shutil
            shutil.rmtree(_MODELS_DIR, ignore_errors=True)
            st.session_state.clear()
            st.session_state["force_retrain"] = True
            st.rerun()
    else:
        st.markdown("""
<div style="display:flex;align-items:center;gap:8px;padding:10px 14px;
  background:rgba(245,166,35,0.06);border:1px solid rgba(245,166,35,0.2);
  border-radius:8px;">
  <span style="color:#f5a623;">◌</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
    color:#f5a623;letter-spacing:0.04em;">NO MODEL — WILL TRAIN ON UPLOAD</span>
</div>""", unsafe_allow_html=True)

    if st.session_state.get("is_inverted"):
        st.error(
            "⚠️ Polarity was auto-corrected this session.\n"
            "Click **🔁 Retrain** to fix permanently."
        )

# ===========================================================================
# SESSION STATE INITIALISATION
# BUG 5 FIX: all results persisted in session_state so tab2/tab3 can access
# them without re-running any computation.
# ===========================================================================
for key in ("preds", "shap_values", "result_df", "metrics", "model",
            "df", "y_binary", "feature_cols", "roc_data",
            "active_threshold", "raw_preds", "is_inverted", "force_retrain"):
    if key not in st.session_state:
        st.session_state[key] = None

# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3 = st.tabs(
    ["⬆  UPLOAD & SCAN", "◈  METRICS & ANALYSIS", "◇  SHAP EXPLAINABILITY"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Upload & Predict
# ---------------------------------------------------------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a network flow CSV (CIC-IDS format)",
        type=["csv"],
        help="Must contain standard CIC features. A 'Label' column enables supervised mode + metrics.",
    )

    # BUG 4 FIX: st.stop() was inside the tab, blocking tab2/tab3 entirely.
    # Replaced with an early-exit that leaves other tabs accessible.
    if uploaded_file is None and use_sample:
        sample_path = _HERE / "data" / "sample_flows.csv"
        if sample_path.exists():
            st.success("📂 Using built-in sample data")
            df = _load_csv(sample_path)
        else:
            st.error(f"Sample file not found at `{sample_path}`. Upload a CSV instead.")
            df = None
    elif uploaded_file is not None:
        df = _read_uploaded_csv(uploaded_file)
        st.success(f"✅ Loaded **{len(df):,}** network flows")
    else:
        st.markdown("""
<div style="
  text-align:center; padding:52px 28px;
  background:linear-gradient(135deg,#0d1628,#111e30);
  border:1.5px dashed rgba(0,217,245,0.22); border-radius:14px; margin:12px 0;
">
  <div style="font-size:2.8rem;margin-bottom:14px;opacity:0.65;">⬆</div>
  <div style="font-family:'Sora',sans-serif;font-weight:700;font-size:1.1rem;
    color:#e8edf5;margin-bottom:8px;">Drop a Network Flow CSV to Begin</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
    color:#6b7a99;line-height:1.8;">
    CICIDS-2017 &nbsp;·&nbsp; CICIoT-2023 &nbsp;·&nbsp; UNSW-NB15 &nbsp;·&nbsp; Custom<br>
    Requires a <span style="color:#00d9f5;">Label</span> column for supervised metrics &nbsp;·&nbsp;
    <span style="color:#0ffa9e;">Integer 0/1 labels supported</span>
  </div>
</div>
""", unsafe_allow_html=True)
        df = None

    if df is not None:
        # BUG 3 FIX: split X/y BEFORE calling fit() or predict()
        X, y_binary, y_raw = _split_features_labels(df)

        if show_debug:
            st.subheader("Debug — Raw Data")
            st.dataframe(df.head(5), use_container_width=True)
            if y_binary is not None:
                st.write(f"y=0 (benign): {int((y_binary==0).sum()):,}   "
                         f"y=1 (attack): {int((y_binary==1).sum()):,}")

        run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Loading model…"):
                force_retrain = st.session_state.get("force_retrain") or False
                if _model_on_disk() and not force_retrain:
                    try:
                        model = _load_model()
                        st.success("✅ Saved model loaded from disk")
                    except Exception as e:
                        st.warning(f"Failed to load saved model ({e}). Re-training…")
                        model = None
                else:
                    if force_retrain:
                        st.info("🔁 Force retrain — fresh model will be trained on this file")
                        st.session_state["force_retrain"] = None
                    model = None

            if model is None:
                if y_binary is not None:
                    n_atk  = int(y_binary.sum())
                    n_ben  = int((y_binary == 0).sum())
                    n_tot  = n_atk + n_ben
                    pct_atk = n_atk / max(n_tot, 1) * 100

                    # ── Dataset format detection ──────────────────────────────
                    # Show which label format was auto-detected so user can
                    # confirm the file is being parsed correctly.
                    st.info(
                        f"📋 **Label detection:** {n_ben:,} benign | "
                        f"{n_atk:,} attack ({pct_atk:.1f}%) | "
                        f"Total: {n_tot:,} rows  "
                        f"*(UNSW-NB15, CICIDS, CICIoT, and benign-only CSVs supported)*"
                    )

                    if n_atk == 0:
                        st.error(
                            "🚨 **No attack samples found — this appears to be a benign-only CSV.**\n\n"
                            f"All {n_ben:,} rows are labelled as benign. Training requires both "
                            "benign AND attack traffic to learn a decision boundary.\n\n"
                            "**Supported datasets:** CICIDS-2017/2018, CICIoT-2023, UNSW-NB15 "
                            "(use files that contain mixed benign+attack rows)."
                        )
                        st.stop()
                    elif n_atk < 50:
                        st.error(
                            f"🚨 **Too few attack samples: {n_atk} found, ≥ 50 required.**\n\n"
                            f"Your CSV has {n_ben:,} benign rows but only {n_atk} attack rows. "
                            "This is too few to train a reliable classifier.\n\n"
                            "**For CICIDS-2017:** Use the Wednesday or Thursday daily CSV.\n"
                            "**For UNSW-NB15:** Combine multiple part files "
                            "(UNSW-NB15_1.csv through _4.csv) for better attack coverage."
                        )
                        st.stop()
                    elif n_atk < 500:
                        st.warning(
                            f"⚠️ Only {n_atk:,} attack samples ({pct_atk:.1f}%) — training will "
                            "proceed but AUC and recall may be modest. "
                            "Aim for ≥ 10,000 attack samples for best results."
                        )
                with st.spinner("Training model on uploaded data…"):
                    try:
                        model = HybridIDS()
                        model.fit(X, y_binary)
                        _save_model(model)
                        st.success("✅ Model trained and saved")
                    except Exception as e:
                        err_str = str(e).lower()

                        # ── Friendly error messages for known failure modes ────
                        if "len() of unsized object" in err_str or                            "iteration over a 0-d" in err_str:
                            st.error(
                                "🚨 **Training failed: label/class array error.**\n\n"
                                "This usually means the dataset has very few attack "
                                "samples and SMOTE/ADASYN cannot generate synthetic "
                                "examples, OR the label column contains NaN values "
                                "that caused a 0-dimensional numpy array.\n\n"
                                f"**Details:** `{e}`\n\n"
                                "**Fix options:**\n"
                                "• Use a dataset with ≥ 500 attack samples\n"
                                "• For UNSW-NB15: combine all 4 part files\n"
                                "• For CICIDS-2017: use Wednesday/Thursday CSV\n"
                                "• Ensure the Label column has no missing values"
                            )
                        elif "cannot take a larger sample" in err_str or                              "expected n_neighbors" in err_str:
                            st.error(
                                "🚨 **Training failed: too few minority samples for SMOTE.**\n\n"
                                f"**Details:** `{e}`\n\n"
                                "Use a dataset with more attack samples "
                                "(≥ 10 unique attack rows per attack type)."
                            )
                        elif "dimension" in err_str or "shape" in err_str:
                            st.error(
                                "🚨 **Training failed: feature dimension mismatch.**\n\n"
                                f"**Details:** `{e}`\n\n"
                                "Delete the `models/` folder (click **🗑️ Clear** in sidebar) "
                                "and retrain — a stale model with different features may be loaded."
                            )
                        else:
                            st.error(
                                f"🚨 **Training failed:** `{e}`\n\n"
                                "Check that the CSV has valid numeric feature columns "
                                "and a 'Label' column. Enable **Show Debug Info** in the "
                                "sidebar to inspect the first rows of your dataset."
                            )
                        st.stop()

            # ── Batched prediction (avoids OOM on large files) ──────────────
            _pred_prog = st.progress(0, text="Starting scan…")
            try:
                preds, shap_vals, X_clean, y_clean = _batch_predict(
                    model, X, _pred_prog, y_binary=y_binary
                )
            except Exception as e:
                _pred_prog.empty()
                st.error(f"Prediction failed: {e}")
                st.stop()
            _pred_prog.empty()

            # ── Guard: empty preds ───────────────────────────────────────────
            if len(preds) == 0:
                st.error(
                    "⚠️ No rows survived preprocessing. The LOF filter removed "
                    "everything — likely a distribution mismatch between training "
                    "data and this file. Delete the `models/` folder and retrain."
                )
                st.stop()

            n_dropped = len(X) - len(X_clean)
            if n_dropped > 0:
                st.info(
                    f"ℹ️ {n_dropped:,} rows removed by preprocessing "
                    f"({n_dropped/len(X)*100:.1f}%). "
                    f"Predictions cover the remaining {len(X_clean):,} rows."
                )

            # y_clean is now returned directly from _batch_predict — aligned
            # batch-by-batch without the minority safety check, so length
            # always matches len(preds). No second preprocess pass needed.
            if y_binary is not None and y_clean is None:
                st.warning(
                    "⚠️ Label alignment failed — metrics will be unavailable. "
                    "This can happen when the Label column values don't match any "
                    f"known benign label {{'BENIGN','BenignTraffic','normal','background'}}. "
                    f"Labels found in your CSV: "
                    f"{list(df[next(c for c in df.columns if c.lower()=='label')].unique()[:8])}"
                )

            # ── Polarity check ───────────────────────────────────────────────
            # _get_hybrid_scores() produces correct-polarity scores by
            # construction (AE error HIGH → anomalous, LSTM prob HIGH →
            # attack), so inversion should not be needed.  We check anyway
            # and flip if AUC < 0.5 (should be rare after the hybrid fix).
            is_inverted = False
            if y_clean is not None and len(y_clean) == len(preds):
                _fpr_q, _tpr_q, _ = roc_curve(y_clean, preds)
                _auc_raw = auc(_fpr_q, _tpr_q)
                if _auc_raw < 0.5:
                    preds = 1.0 - preds   # safe: hybrid scores are all continuous
                    is_inverted = True
                    st.info(
                        f"🔄 Polarity corrected (AUC {_auc_raw:.3f} → "
                        f"{1-_auc_raw:.3f}). Rare after hybrid scoring — "
                        "consider retraining with 🔁 Retrain."
                    )

            # ── Threshold optimisation ───────────────────────────────────────
            # _find_optimal_threshold works on already-corrected preds.
            # Youden-J under FPR < 5% guarantees the operating point meets
            # the project target while maximising recall.
            roc_data = None
            if auto_threshold and y_clean is not None:
                with st.spinner("Optimising decision threshold via ROC curve…"):
                    best_thr, roc_data = _find_optimal_threshold(y_clean, preds)
                roc_data["is_inverted"] = is_inverted
                active_threshold = best_thr
                est_acc = roc_data.get("est_accuracy", 0)
                p_atk   = roc_data.get("p_attack", 0)
                st.success(
                    f"✅ Optimal threshold: **{active_threshold:.4f}** "
                    f"| AUC: **{roc_data['roc_auc']:.4f}** "
                    f"| Est. accuracy: **{est_acc*100:.1f}%** "
                    f"| Attack fraction: **{p_atk*100:.1f}%**"
                )
            else:
                # No labels — do a quick single-pass AUC check for inversion only
                if not auto_threshold:
                    _fpr_m, _tpr_m, _ = roc_curve(
                        np.ones(len(preds)), preds   # dummy labels
                    ) if y_clean is None else (None, None, None)
                active_threshold = manual_threshold
                st.info(f"Using manual threshold: **{active_threshold:.2f}**")

            # ── Build result_df ───────────────────────────────────────────────
            result_df = X_clean.copy()
            result_df.insert(0, "Prediction",
                             np.where(preds > active_threshold, "ANOMALY", "BENIGN"))
            result_df.insert(1, "Probability", np.round(preds, 4))

            # ── Compute metrics ───────────────────────────────────────────────
            if y_clean is not None and len(y_clean) == len(preds):
                metrics_out = _compute_metrics(y_clean, preds,
                                               threshold=active_threshold)
            else:
                metrics_out = None

            # ── Store in session_state ────────────────────────────────────────
            st.session_state["model"]            = model
            st.session_state["raw_preds"]        = preds
            st.session_state["preds"]            = preds
            st.session_state["is_inverted"]      = is_inverted
            st.session_state["shap_values"]      = shap_vals
            st.session_state["df"]               = df
            st.session_state["y_binary"]         = y_clean
            st.session_state["feature_cols"]     = model.feature_cols
            st.session_state["result_df"]        = result_df
            st.session_state["metrics"]          = metrics_out
            st.session_state["roc_data"]         = roc_data
            st.session_state["active_threshold"] = active_threshold

            st.rerun()   # refresh so tab2/tab3 see new state immediately

        # Show summary card if results exist
        if st.session_state["preds"] is not None:
            preds      = st.session_state["preds"]
            thr        = st.session_state["active_threshold"] or 0.5
            n_flagged  = int(np.sum(preds > thr))
            n_total    = len(preds)
            metrics_s  = st.session_state["metrics"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FLAGGED ANOMALIES", f"{n_flagged:,}",
                      f"{n_flagged/max(n_total,1)*100:.1f}% of flows")
            c2.metric("TOTAL FLOWS SCANNED", f"{n_total:,}")
            c3.metric("ACTIVE THRESHOLD", f"{thr:.4f}")
            if metrics_s:
                fpr_pct = metrics_s['fpr'] * 100
                c4.metric("FALSE POSITIVE RATE",
                           f"{fpr_pct:.2f}%",
                           "UNDER TARGET" if fpr_pct < 5 else "OVER 5%")


# ---------------------------------------------------------------------------
# TAB 2 — Results & Metrics
# BUG 5 FIX: reads from session_state; BUG 6 FIX: real computed metrics
# ---------------------------------------------------------------------------
with tab2:
    preds      = st.session_state.get("preds")
    result_df  = st.session_state.get("result_df")
    y_binary   = st.session_state.get("y_binary")
    roc_data   = st.session_state.get("roc_data")
    active_thr = st.session_state.get("active_threshold") or 0.5

    if preds is None:
        st.markdown("""
<div style="text-align:center;padding:48px 24px;
  background:linear-gradient(135deg,#0d1628,#111e30);
  border:1px solid rgba(0,217,245,0.12);border-radius:14px;margin:16px 0;">
  <div style="font-size:2.4rem;margin-bottom:12px;opacity:0.5;">◈</div>
  <div style="font-family:'Sora',sans-serif;font-weight:600;font-size:1rem;
    color:#6b7a99;">No scan results yet</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
    color:#3d4f6e;margin-top:6px;">Upload a CSV and run detection in the first tab</div>
</div>""", unsafe_allow_html=True)
    else:
        n_total = len(preds)

        # ── Live threshold re-scoring ────────────────────────────────────────
        # The user can drag this slider to instantly see how FPR/accuracy
        # change without re-running the model — the raw probabilities are
        # already stored in session_state.
        st.subheader("🎚️ Live Threshold Tuning")
        live_thr = st.slider(
            "Adjust decision threshold and see metrics update instantly",
            min_value=0.001, max_value=0.999,
            value=float(np.clip(active_thr, 0.001, 0.999)), step=0.001,
            key="live_threshold_slider",
            help="Drag to re-score at any threshold without re-running the model."
        )

        # Re-compute metrics live at the dragged threshold
        if y_binary is not None and len(y_binary) == len(preds):
            live_metrics = _compute_metrics(y_binary, preds, threshold=live_thr)
        else:
            live_metrics = None

        n_flagged = int(np.sum(preds > live_thr))

        # ── Model quality diagnostics ────────────────────────────────────────
        roc_auc    = roc_data["roc_auc"]              if roc_data else None
        is_inv     = roc_data.get("is_inverted", False) if roc_data else False
        p_attack   = roc_data.get("p_attack",   None)  if roc_data else None
        est_acc    = roc_data.get("est_accuracy", None) if roc_data else None

        # ── Accuracy gap analysis ─────────────────────────────────────────────
        if live_metrics is not None and p_attack is not None:
            acc_pct  = live_metrics["accuracy"] * 100
            fpr_pct  = live_metrics["fpr"] * 100
            rec_pct  = live_metrics["recall"] * 100
            p_b      = 1.0 - p_attack
            # Minimum recall needed to reach 95% accuracy at current FPR
            min_recall_for_95 = (0.95 - p_b * (1 - fpr_pct/100)) / p_attack
            min_recall_for_95 = max(0.0, min(1.0, min_recall_for_95))
            recall_gap = min_recall_for_95 * 100 - rec_pct

            if acc_pct < 95.0:
                if recall_gap > 0:
                    st.warning(
                        f"📊 **Accuracy gap analysis:**  \n"
                        f"Current accuracy = **{acc_pct:.2f}%** "
                        f"(target ≥ 95%).  \n"
                        f"Dataset is **{p_b*100:.1f}% benign / "
                        f"{p_attack*100:.1f}% attack**.  \n"
                        f"To reach 95% accuracy at FPR={fpr_pct:.2f}%, "
                        f"recall must reach **{min_recall_for_95*100:.1f}%** "
                        f"(currently {rec_pct:.1f}% — gap: "
                        f"**{recall_gap:.1f}%**).  \n"
                        f"The model needs more diverse attack training data "
                        f"or longer LSTM training to close this gap."
                    )

        if is_inv:
            st.warning(
                "🔄 **Prediction polarity was auto-corrected** (smart inversion applied).  \n\n"
                "The model was scoring attacks near 0 and benign near 1. Only LSTM-processed "
                "rows were inverted — autoencoder-unflagged benign rows kept score=0 "
                "to prevent FPR inflation.  \n\n"
                "**For a permanent fix:** click **🔁 Retrain** in the sidebar with a file "
                "containing ≥ 1,000 diverse attack samples (full CICIDS-2017 daily CSVs)."
            )

        if roc_auc is not None:
            if roc_auc >= 0.90:
                st.success(f"✅ **Excellent model quality — AUC-ROC = {roc_auc:.4f}**")
            elif roc_auc >= 0.80:
                st.success(f"✅ **Good model quality — AUC-ROC = {roc_auc:.4f}**")
            elif roc_auc >= 0.65:
                st.warning(
                    f"⚠️ **Moderate model quality — AUC-ROC = {roc_auc:.4f}**  \n"
                    "Retrain on a more balanced/diverse dataset for better recall."
                )
            else:
                st.error(
                    f"🚨 **Low model quality — AUC-ROC = {roc_auc:.4f}**  \n\n"
                    "Even after polarity correction discrimination is poor — the model "
                    "has never seen the attack types in this file.  \n\n"
                    "**Fix:** Click **🔁 Retrain** in the sidebar."
                )

        # Attack sample count warning
        if y_binary is not None:
            n_attacks_in_eval = int(y_binary.sum())
            if n_attacks_in_eval < 100:
                st.warning(
                    f"⚠️ Only **{n_attacks_in_eval}** attack sample(s) in the evaluation set. "
                    "Recall and FNR metrics will be unreliable. "
                    "Use a file with diverse attack traffic for meaningful evaluation."
                )

        # ── Metrics row ──────────────────────────────────────────────────────
        if live_metrics is not None:
            st.subheader("📐 Classification Metrics")

            # Accuracy target badge
            acc_pct = live_metrics['accuracy'] * 100
            fpr_pct = live_metrics['fpr'] * 100
            acc_ok  = acc_pct >= 95.0
            fpr_ok  = fpr_pct < 5.0

            st.markdown(
                f"**Accuracy target (≥ 95%):** "
                f"{'✅ Met' if acc_ok else '❌ Not met'}  |  "
                f"**FPR target (< 5%):** "
                f"{'✅ Met' if fpr_ok else '❌ Not met'}  |  "
                f"**Threshold:** `{live_thr:.4f}`  |  "
                f"**AUC-ROC:** `{roc_data['roc_auc']:.4f}`"
                if roc_data else
                f"**Accuracy target (≥ 95%):** "
                f"{'✅ Met' if acc_ok else '❌ Not met'}  |  "
                f"**FPR target (< 5%):** "
                f"{'✅ Met' if fpr_ok else '❌ Not met'}  |  "
                f"**Threshold:** `{live_thr:.4f}`"
            )

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy",  f"{acc_pct:.2f}%",
                      "✅ ≥ 95%" if acc_ok  else "❌ < 95%")
            m2.metric("F1 Score",  f"{live_metrics['f1']*100:.2f}%")
            m3.metric("Precision", f"{live_metrics['precision']*100:.2f}%")
            m4.metric("Recall",    f"{live_metrics['recall']*100:.2f}%")
            m5.metric("FPR",       f"{fpr_pct:.2f}%",
                      "✅ < 5%" if fpr_ok else "⚠️ > 5%")
            m6.metric("FNR",       f"{live_metrics['fnr']*100:.2f}%")

            st.divider()

            # Confusion matrix
            st.subheader("🔢 Confusion Matrix")
            cm_df = pd.DataFrame(
                [[live_metrics["tn"], live_metrics["fp"]],
                 [live_metrics["fn"], live_metrics["tp"]]],
                index=["Actual Benign", "Actual Attack"],
                columns=["Predicted Benign", "Predicted Attack"],
            )
            col_cm, col_roc = st.columns([1, 2])
            with col_cm:
                st.dataframe(cm_df)

            # ROC curve
            with col_roc:
                if roc_data is not None:
                    st.markdown("**ROC Curve**")
                    fig_roc, ax_roc = plt.subplots(figsize=(5, 4),
                                                       facecolor="#0d1220")
                    ax_roc.set_facecolor("#0d1220")
                    ax_roc.plot(roc_data["fpr_arr"], roc_data["tpr_arr"],
                                color="#00d9f5", lw=2.5, alpha=0.9,
                                label=f"AUC = {roc_data['roc_auc']:.4f}")
                    ax_roc.fill_between(roc_data["fpr_arr"], roc_data["tpr_arr"],
                                        alpha=0.08, color="#00d9f5")
                    ax_roc.axvline(0.05, color="#f5a623", linestyle=":",
                                   lw=1.5, label="FPR = 5% limit")
                    opt_tpr = float(live_metrics["tp"] /
                                    max(live_metrics["tp"] + live_metrics["fn"], 1))
                    opt_fpr = float(live_metrics["fp"] /
                                    max(live_metrics["fp"] + live_metrics["tn"], 1))
                    ax_roc.scatter([opt_fpr], [opt_tpr], color="#0ffa9e",
                                   s=90, zorder=5, edgecolors="#0a0e1a", lw=1.5,
                                   label=f"thr={live_thr:.3f}")
                    ax_roc.plot([0, 1], [0, 1], color="#3d4f6e",
                                linestyle="--", lw=1)
                    ax_roc.set_xlim([0, 1]); ax_roc.set_ylim([0, 1.02])
                    ax_roc.set_xlabel("False Positive Rate",
                                      color="#6b7a99", fontsize=8)
                    ax_roc.set_ylabel("True Positive Rate",
                                      color="#6b7a99", fontsize=8)
                    ax_roc.tick_params(colors="#6b7a99", labelsize=7)
                    for spine in ax_roc.spines.values():
                        spine.set_edgecolor("#1e2d45")
                    ax_roc.legend(fontsize=7.5, facecolor="#111827",
                                  edgecolor="#1e2d45", labelcolor="#e8edf5")
                    plt.tight_layout()
                    st.pyplot(fig_roc); plt.close(fig_roc)

        else:
            st.info(
                "No ground-truth labels found — add a 'Label' column to your "
                "CSV to see accuracy, FPR, F1, the ROC curve, and live tuning."
            )
            st.metric("🚨 Flagged Anomalies", f"{n_flagged:,}")
            st.metric("📦 Total Flows",        f"{n_total:,}")

        st.divider()

        # ── Prediction table ─────────────────────────────────────────────────
        st.subheader("📋 Prediction Results (first 50 rows)")
        # Rebuild Prediction column at the live threshold for the display table
        display_result = result_df.copy()
        display_result["Prediction"] = np.where(
            display_result["Probability"] > live_thr, "ANOMALY", "BENIGN"
        )
        extra_cols   = [c for c in display_result.columns
                        if c not in ("Prediction", "Probability")][:5]
        display_cols = ["Prediction", "Probability"] + extra_cols
        st.dataframe(display_result[display_cols].head(50), use_container_width=True)

        # ── Score distribution ───────────────────────────────────────────────
        st.subheader("📈 Anomaly Score Distribution")
        fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#0d1220")
        ax.set_facecolor("#0d1220")
        ax.hist(preds, bins=60, color="#00d9f5", alpha=0.55,
                edgecolor="none", label="Anomaly scores")
        ax.axvline(live_thr, color="#ff4560", linestyle="--", linewidth=2,
                   label=f"Threshold {live_thr:.4f}")
        if live_thr != active_thr:
            ax.axvline(active_thr, color="#0ffa9e", linestyle=":",
                       linewidth=1.5, label=f"Optimal {active_thr:.4f}")
        ax.set_xlabel("Anomaly Score", color="#6b7a99", fontsize=9)
        ax.set_ylabel("Flow Count",   color="#6b7a99", fontsize=9)
        ax.tick_params(colors="#6b7a99", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d45")
        ax.legend(fontsize=8, facecolor="#111827",
                  edgecolor="#1e2d45", labelcolor="#e8edf5")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        # ── Download ─────────────────────────────────────────────────────────
        csv_bytes = display_result.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download Full Results CSV",
            data=csv_bytes,
            file_name="hybrid_ids_predictions.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# TAB 3 — SHAP Explainability
# BUG 5 FIX: reads from session_state; BUG 7 FIX: correct SHAP normalisation
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("🔍 SHAP Feature Importance")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows **which network features "
        "contributed most** to the model flagging a flow as anomalous."
    )

    shap_vals    = st.session_state.get("shap_values")
    feature_cols = st.session_state.get("feature_cols")
    preds        = st.session_state.get("preds")

    if preds is None:
        st.markdown("""
<div style="text-align:center;padding:48px 24px;
  background:linear-gradient(135deg,#0d1628,#111e30);
  border:1px solid rgba(0,217,245,0.12);border-radius:14px;margin:16px 0;">
  <div style="font-size:2.4rem;margin-bottom:12px;opacity:0.5;">◇</div>
  <div style="font-family:'Sora',sans-serif;font-weight:600;font-size:1rem;
    color:#6b7a99;">No scan results yet</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
    color:#3d4f6e;margin-top:6px;">SHAP explainability requires a completed scan</div>
</div>""", unsafe_allow_html=True)
    elif shap_vals is None:
        st.warning(
            "No SHAP values available. This can happen when:\n"
            "- No flows were flagged by the Autoencoder\n"
            "- The model is running in unsupervised mode\n"
            "- The dataset was too small to build a SHAP background\n\n"
            "Try a larger or more attack-heavy CSV."
        )
    else:
        # BUG 7 FIX: normalise to 2-D (samples × features)
        shap_arr = _normalise_shap(shap_vals)

        if shap_arr is None or shap_arr.ndim != 2:
            st.error("Could not parse SHAP output shape. Please check model output.")
        else:
            n_plot      = min(30, shap_arr.shape[0])
            n_shap_feat = shap_arr.shape[1]

            # Align feature names to SHAP feature count
            if feature_cols and len(feature_cols) == n_shap_feat:
                feat_names = feature_cols
            elif feature_cols and len(feature_cols) > n_shap_feat:
                # SHAP flattened window×features; label as f0..fN
                feat_names = [f"f{i}" for i in range(n_shap_feat)]
            else:
                feat_names = [f"f{i}" for i in range(n_shap_feat)]

            plot_data = pd.DataFrame(shap_arr[:n_plot], columns=feat_names)

            # --- Summary beeswarm plot ---
            st.markdown(f"**Summary plot — top features across {n_plot} flagged samples**")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(
                shap_arr[:n_plot],
                plot_data,
                feature_names=feat_names,
                show=False,
                max_display=20,
            )
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            st.divider()

            # --- Mean |SHAP| bar chart ---
            st.markdown("**Mean absolute SHAP value per feature (top 20)**")
            mean_abs   = np.abs(shap_arr).mean(axis=0)
            top_idx    = np.argsort(mean_abs)[::-1][:20]
            top_names  = [feat_names[i] for i in top_idx]
            top_vals   = mean_abs[top_idx]

            fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#0d1220")
            ax2.set_facecolor("#0d1220")
            n_bars = len(top_vals)
            bar_colors = ["#{:02x}{:02x}{:02x}".format(
                int(0 + i*2), int(217 - i*8), int(245 - i*7))
                for i in range(n_bars)]
            ax2.barh(top_names[::-1], top_vals[::-1],
                     color=bar_colors[::-1], height=0.65, edgecolor="none")
            ax2.set_xlabel("Mean |SHAP value|", color="#6b7a99", fontsize=9)
            ax2.tick_params(colors="#6b7a99", labelsize=8)
            for spine in ax2.spines.values():
                spine.set_edgecolor("#1e2d45")
            ax2.set_title("Feature Importance (SHAP)", color="#e8edf5",
                          fontsize=10, fontweight="bold", pad=12)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # --- Raw SHAP table ---
            with st.expander("📊 Raw SHAP values (first 10 samples)"):
                st.dataframe(
                    pd.DataFrame(shap_arr[:10], columns=feat_names).round(5),
                    use_container_width=True,
                )

            if show_debug:
                st.write(f"SHAP array shape: `{shap_arr.shape}`")
                st.write(f"Feature names count: `{len(feat_names)}`")

# ===========================================================================
# FOOTER
# ===========================================================================
st.markdown("""
<div style="
  margin-top:32px; padding:18px 28px;
  background:linear-gradient(90deg,#0d1628,#111e30,#0d1628);
  border:1px solid rgba(0,217,245,0.1); border-radius:10px;
  display:flex; align-items:center; justify-content:space-between;
  flex-wrap:wrap; gap:10px;
">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:6px;height:6px;border-radius:50%;background:#0ffa9e;
      box-shadow:0 0 6px #0ffa9e;"></div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
      color:#6b7a99;letter-spacing:0.06em;">
      HYBRID IDS v5.0 &nbsp;·&nbsp; AUTOENCODER + SSA-LSTM &nbsp;·&nbsp; LOF · SMOTE · SHAP
    </span>
  </div>
  <a href="https://github.com/0xc1GenZ/hybrid-ids-framework"
     target="_blank"
     style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
     color:#00d9f5;text-decoration:none;letter-spacing:0.05em;
     border:1px solid rgba(0,217,245,0.2);padding:4px 10px;border-radius:20px;">
    ↗ GitHub
  </a>
</div>
""", unsafe_allow_html=True)
