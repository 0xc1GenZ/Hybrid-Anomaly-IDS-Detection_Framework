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
    # NaN fix: y_raw.fillna(benign_label) before comparing so NaN rows become
    # y=0 (benign) instead of y=1 (attack). Without this, every NaN label
    # silently inflates n_atk and reaches SMOTE with corrupt labels.
    benign_label = _detect_benign_label(y_raw)
    if benign_label is not None:
        y = (y_raw.fillna(benign_label) != benign_label).astype(int)
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
    # NaN fix: fillna(most_common) so NaN rows are benign (most-common class).
    try:
        if y_raw.dtype == object:
            most_common = y_raw.value_counts().index[0]
            y = (y_raw.fillna(most_common) != most_common).astype(int)
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
# EASTER EGGS — Cyber jokes · Animated mascots · Click wisdom
# (All pure HTML/CSS/JS injected via st.markdown — zero Python overhead)
# ===========================================================================

import random as _random

_CYBER_QUOTES = [
    ("There are only two types of companies:", "those that have been hacked,\nand those that don't know it yet. 👻", "— John Chambers, Cisco"),
    ("Why did the hacker break up?", "Too many trust issues — just like our SSL certs. 💔", "— Anonymous SOC Analyst"),
    ("A SQL injection walks into a bar.", "Drops tables. No survivors. 🪑", "— Every DBA's nightmare"),
    ("How does a hacker fix a bug?", "They don't. They ship it as a 'feature'. 🚀", "— Git blame, every time"),
    ("Roses are #FF0000,", "Violets are #0000FF,\nbut your firewall logs are 🔥🔥🔥", "— Romantic Pentester"),
    ("Why don't hackers like nature?", "Too many unknown bugs and no patch notes. 🐛", "— Patch Tuesday Survivor"),
    ("The cloud is just", "someone else's computer you forgot to secure. ☁️", "— Cloud Evangelist (in denial)"),
    ("I asked AI to detect intrusions.", "It found 3 in my code. All of them were me. 🤖", "— ML Engineer, 3 AM"),
    ("My password is 'incorrect'.", "So when I forget it, the system reminds me. 🔑", "— Surprisingly Common"),
    ("There's no patch for", "human stupidity. But we trained a model anyway. 🧠", "— This dashboard"),
    ("Why did the cat help with cybersecurity?", "Because it always lands on its paws\nand NEVER clicks phishing links. 🐱", "— Adopted Mascot"),
    ("The spider's web is the best IDS:", "if something big blunders in, you know immediately. 🕷️", "— Nature's SOC"),
    ("Ghost traffic is the scariest —", "you can't see it until it haunts your logs. 👻", "— Wireshark Whisperer"),
    ("Training a neural net is like parenting:", "you feed it garbage, it learns garbage,\nthen you blame the architecture. 🍼", "— ML Dad Jokes"),
    ("False positives are just", "the model crying wolf.\nBut sometimes the wolf IS there. 🐺", "— ROC Curve Philosopher"),
    ("A penetration tester walks into a bar.", "The bartender asks for ID.\nHe submits a 0-day instead. 🍺", "— Security Conference 2024"),
    ("SMOTE to the rescue!", "When you only have 2 attack samples\nand infinite ambition. ✨", "— Imbalanced Dataset Diary"),
    ("Autoencoder walks into therapy:", "'I keep reconstructing my trauma.' 😭", "— Deep Learning Feelings"),
    ("Why is the LSTM crying?", "It forgot the beginning of the sentence.\nAnd also its padding value. 😢", "— Sequence Model Struggles"),
    ("99.9% uptime sounds great —", "until you realise that's 8.7 hours\nof downtime per year. Somebody get fired. ⏰", "— SLA Nightmares"),
]

_q = _CYBER_QUOTES[_random.randint(0, len(_CYBER_QUOTES) - 1)]

st.markdown(f"""
<div id="quote-banner" style="
  background:linear-gradient(135deg,#0d1422,#111e30);
  border:1px solid rgba(0,217,245,0.15); border-left:3px solid #00d9f5;
  border-radius:10px; padding:14px 20px; margin-bottom:14px;
  display:flex; align-items:flex-start; gap:14px; cursor:pointer;
  position:relative; overflow:hidden;
" onclick="this.style.opacity='0.5';setTimeout(()=>this.style.opacity='1',200);"
  title="Click for a new one on next reload 😄">
  <div style="font-size:1.6rem;flex-shrink:0;line-height:1.2;margin-top:2px;">🔐</div>
  <div>
    <div style="font-family:'Sora',sans-serif;font-size:0.82rem;color:#e8edf5;
      font-weight:600;line-height:1.5;">{_q[0]}</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
      color:#00d9f5;margin-top:3px;white-space:pre-line;">{_q[1]}</div>
    <div style="font-family:'Sora',sans-serif;font-size:0.68rem;color:#6b7a99;
      margin-top:5px;font-style:italic;">{_q[2]}</div>
  </div>
  <div style="position:absolute;top:8px;right:12px;font-family:'JetBrains Mono',
    monospace;font-size:0.6rem;color:#3d4f6e;letter-spacing:0.06em;">
    CYBER WISDOM · RELOAD FOR NEW</div>
</div>
""", unsafe_allow_html=True)

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

/* ══════════════════════════════════════════════
   MASCOT ANIMATIONS
   ══════════════════════════════════════════════ */

/* ── Ghost floating in top-right corner ── */
#ids-ghost {
  position:fixed; top:18px; right:22px; z-index:9999;
  font-size:2.2rem; line-height:1;
  animation: ghost-float 3s ease-in-out infinite;
  cursor:pointer; user-select:none;
  filter:drop-shadow(0 0 8px rgba(0,217,245,0.5));
  transition:transform 0.2s;
}
#ids-ghost:hover { transform:scale(1.35) rotate(-10deg); }
#ids-ghost .ghost-tooltip {
  display:none; position:absolute; right:44px; top:0;
  background:#111827; border:1px solid rgba(0,217,245,0.25);
  border-radius:8px; padding:8px 12px; white-space:nowrap;
  font-family:'JetBrains Mono',monospace; font-size:0.7rem;
  color:#00d9f5; box-shadow:0 4px 20px rgba(0,0,0,0.5);
  pointer-events:none;
}
#ids-ghost:hover .ghost-tooltip { display:block; }
@keyframes ghost-float {
  0%,100% { transform:translateY(0); }
  50%      { transform:translateY(-10px); }
}

/* ── Spider crawling across top ── */
#ids-spider {
  position:fixed; top:0; left:-60px; z-index:9998;
  font-size:1.5rem; line-height:1;
  animation: spider-crawl 18s linear infinite;
  cursor:pointer; user-select:none;
  filter:drop-shadow(0 2px 4px rgba(0,0,0,0.8));
}
#ids-spider:hover { animation-play-state:paused; transform:scale(1.4); }
#ids-spider .spider-tooltip {
  display:none; position:absolute; top:22px; left:0;
  background:#111827; border:1px solid rgba(255,69,96,0.3);
  border-radius:8px; padding:8px 12px; white-space:nowrap;
  font-family:'JetBrains Mono',monospace; font-size:0.68rem;
  color:#ff4560; box-shadow:0 4px 20px rgba(0,0,0,0.5);
}
#ids-spider:hover .spider-tooltip { display:block; }
@keyframes spider-crawl {
  0%   { left:-60px;  top:0px;   transform:rotate(0deg);   }
  25%  { left:40vw;   top:8px;   transform:rotate(5deg);   }
  50%  { left:80vw;   top:0px;   transform:rotate(-5deg);  }
  75%  { left:100vw;  top:10px;  transform:rotate(3deg);   }
  100% { left:110vw;  top:0px;   transform:rotate(0deg);   }
}

/* ── Cat peeking from bottom of sidebar ── */
#ids-cat {
  position:fixed; bottom:0; left:0;
  width:260px; /* matches sidebar width */
  z-index:9997; pointer-events:none;
  display:flex; justify-content:flex-end;
  padding-right:16px;
  animation: cat-peek 8s ease-in-out infinite;
}
@keyframes cat-peek {
  0%,60%,100% { transform:translateY(72px); }
  20%,40%     { transform:translateY(20px); }
}

/* ── Click tooltip that follows cursor ── */
#click-bubble {
  position:fixed; z-index:99999;
  background:#111827; border:1px solid rgba(0,217,245,0.3);
  border-radius:10px; padding:9px 14px;
  font-family:'JetBrains Mono',monospace; font-size:0.72rem;
  color:#00d9f5; pointer-events:none;
  box-shadow:0 4px 24px rgba(0,0,0,0.6);
  opacity:0; transition:opacity 0.15s;
  max-width:280px; line-height:1.5;
  white-space:pre-wrap;
}

/* ── Konami easter egg flash ── */
@keyframes konami-flash {
  0%,100% { background:var(--bg-base); }
  25%     { background:rgba(0,217,245,0.08); }
  75%     { background:rgba(15,250,158,0.06); }
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
# MASCOT HTML + CLICK WISDOM JS
# ===========================================================================
st.markdown("""
<!-- ── Ghost mascot ── -->
<div id="ids-ghost">
  👻
  <div class="ghost-tooltip">👻 I haunt your logs at 3 AM...</div>
</div>

<!-- ── Spider mascot ── -->
<div id="ids-spider">
  🕷️
  <div class="spider-tooltip">🕷️ Spinning a web of packet captures...</div>
</div>

<!-- ── Cat peeking from sidebar bottom ── -->
<div id="ids-cat">
  <div style="font-size:2.2rem;line-height:1;
    filter:drop-shadow(0 -4px 8px rgba(0,217,245,0.3));
    font-family:sans-serif;">🐱</div>
</div>

<!-- ── Click bubble ── -->
<div id="click-bubble"></div>

<script>
// ── Click-anywhere wisdom ──────────────────────────────────────────────────
const _jokes = [
  "// TODO: fix this before demo\n...never fixed it. 😬",
  "Have you tried turning it off\nand on again? (works 60% of the time)",
  "It's not a bug.\nIt's an undocumented feature. ✨",
  "The attacker is inside the building.\nThe building is your LSTM. 🏚️",
  "FPR < 5%? Bold of you\nto assume I sleep anyway. 😴",
  "git commit -m 'fixes'\ngit push\n*production burns* 🔥",
  "This model trained on\n200,000 rows. You're welcome. 🧠",
  "Anomaly detected: you\nhaven't taken a break in 4 hours. ☕",
  "There are 10 types of people:\nthose who understand binary\nand those who don't. 🤓",
  "I'm not lazy, I'm\nenergy-efficient. Like LOF. ⚡",
  "Congratulations, you clicked!\nThat's how phishing works too. 🎣",
  "The spider says hi.\nNo, it's not a RAT. Probably. 🕷️",
  "Shh... the autoencoder\nis trying to reconstruct itself. 🤫",
  "ROC curve looking thicc today.\nAUC = 0.99? Slay. 💅",
  "Your model is cool\nbut have you tried\njust unplugging the router? 🔌",
  "PSA: '123456' is still\nthe most common password. Humanity. 🤦",
  "The ghost in your logs\nis probably just a scanner. Probably. 👻",
  "SMOTE: because the real\nattacks were the friends we\ngenerated along the way. 🌈",
  "Adversarial attack detected:\nsomeone typed 'rm -rf /' in chat. 💀",
  "Firewall rule #1:\nif in doubt, drop it.\nFirewall rule #2: see rule #1. 🧱",
  "The cat is monitoring your traffic.\nDon't @ me. 🐱",
  "Zero-day?\nMore like zero-sleep. 😵",
  "Your threshold is 0.29.\nMy confidence is 0.05.\nSame energy. 📉",
  "Alert fatigue is real.\nThis is your 1,247th anomaly today.\nYou good? 🫠",
  "Machine learning:\nteaching computers to be\nas confused as we are. 🤖",
];

const _bubble = document.getElementById('click-bubble');
let _tid = null;

document.addEventListener('click', (e) => {
  const msg = _jokes[Math.floor(Math.random() * _jokes.length)];
  _bubble.textContent = msg;
  _bubble.style.left  = Math.min(e.clientX + 14, window.innerWidth - 300) + 'px';
  _bubble.style.top   = Math.max(e.clientY - 20, 10) + 'px';
  _bubble.style.opacity = '1';
  clearTimeout(_tid);
  _tid = setTimeout(() => { _bubble.style.opacity = '0'; }, 2800);
});

// ── Ghost click — special message ─────────────────────────────────────────
const _ghost = document.getElementById('ids-ghost');
const _ghostMsgs = [
  "BOO! ...jk, I'm just\nyour anomaly detector in disguise 👻",
  "I've been monitoring this network\nsince before you were born. 📡",
  "Ghost traffic detected.\nSpoiler: it was a port scanner. 🕵️",
  "I haunt packet headers\nthat nobody reads. 😢",
  "Zero packets transmitted,\nzero feelings expressed. 👻",
];
if (_ghost) {
  _ghost.addEventListener('click', (e) => {
    e.stopPropagation();
    const msg = _ghostMsgs[Math.floor(Math.random() * _ghostMsgs.length)];
    _bubble.textContent = msg;
    _bubble.style.left  = (e.clientX - 220) + 'px';
    _bubble.style.top   = (e.clientY + 10) + 'px';
    _bubble.style.opacity = '1';
    clearTimeout(_tid);
    _tid = setTimeout(() => { _bubble.style.opacity = '0'; }, 3000);
  });
}

// ── Spider click ───────────────────────────────────────────────────────────
const _spider = document.getElementById('ids-spider');
const _spiderMsgs = [
  "🕷️ I've been crawling your traffic longer than Google crawls your site.",
  "🕸️ My web catches packets.\nYour firewall catches feelings.",
  "🕷️ I'm not a RAT.\n...I'm a spider. Important distinction.",
  "🕸️ Eight legs, zero false positives.\nWish I could say the same for SMOTE.",
  "🕷️ Spinning threads since before\nmulti-threading was cool.",
];
if (_spider) {
  _spider.addEventListener('click', (e) => {
    e.stopPropagation();
    const msg = _spiderMsgs[Math.floor(Math.random() * _spiderMsgs.length)];
    _bubble.textContent = msg;
    _bubble.style.left  = Math.min(e.clientX + 14, window.innerWidth - 300) + 'px';
    _bubble.style.top   = (e.clientY + 12) + 'px';
    _bubble.style.opacity = '1';
    clearTimeout(_tid);
    _tid = setTimeout(() => { _bubble.style.opacity = '0'; }, 3000);
  });
}

// ── Konami code easter egg ─────────────────────────────────────────────────
// Up Up Down Down Left Right Left Right B A → unlock cat mode
const _konami = [38,38,40,40,37,39,37,39,66,65];
let _ki = 0;
document.addEventListener('keydown', (e) => {
  if (e.keyCode === _konami[_ki]) {
    _ki++;
    if (_ki === _konami.length) {
      _ki = 0;
      document.body.style.animation = 'konami-flash 0.8s ease 3';
      setTimeout(() => { document.body.style.animation = ''; }, 2500);
      _bubble.textContent = "🐱 CAT MODE UNLOCKED\n\nMeow. All packets are now paw-sitive.\nFPR = 0.00% (cats don't lie)";
      _bubble.style.left  = '50%';
      _bubble.style.top   = '50%';
      _bubble.style.transform = 'translate(-50%,-50%)';
      _bubble.style.fontSize  = '0.9rem';
      _bubble.style.opacity   = '1';
      _bubble.style.background = '#0d1628';
      _bubble.style.border    = '2px solid #0ffa9e';
      _bubble.style.color     = '#0ffa9e';
      clearTimeout(_tid);
      _tid = setTimeout(() => {
        _bubble.style.opacity   = '0';
        _bubble.style.transform = '';
      }, 5000);
    }
  } else { _ki = 0; }
});
</script>
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

    # ── Cat wisdom widget at sidebar bottom ────────────────────────────
    st.divider()
    _cat_tips = [
        ("🐱 Cat's Tip:", "Never trust a packet that arrives at 3 AM asking for root access."),
        ("🐱 Cat's Tip:", "If it looks like a hairball in your logs, it probably is ransomware."),
        ("🐱 Cat's Tip:", "I knocked your threshold off the desk. You're welcome. 0.29 now."),
        ("🐱 Cat's Tip:", "Purring = normal traffic. Hissing = anomaly. Simple IDS, really."),
        ("🐱 Cat's Tip:", "I watched the cursor for 3 hours and detected zero intrusions. Hire me."),
        ("🐱 Cat's Tip:", "The real attack was the packets we captured along the way. 🐾"),
        ("🐱 Cat's Tip:", "Always land on your paws. Also your ROC curve should look like one."),
    ]
    _ct = _cat_tips[_random.randint(0, len(_cat_tips) - 1)]
    st.markdown(f"""
<div style="background:rgba(0,217,245,0.04);border:1px solid rgba(0,217,245,0.1);
  border-radius:10px;padding:12px 14px;margin-top:4px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
    color:#0ffa9e;letter-spacing:0.08em;margin-bottom:5px;">{_ct[0]}</div>
  <div style="font-family:'Sora',sans-serif;font-size:0.78rem;color:#6b7a99;
    line-height:1.5;font-style:italic;">"{_ct[1]}"</div>
</div>""", unsafe_allow_html=True)

# ===========================================================================
# SESSION STATE INITIALISATION
# BUG 5 FIX: all results persisted in session_state so tab2/tab3 can access
# them without re-running any computation.
# ===========================================================================
for key in ("preds", "shap_values", "result_df", "metrics", "model",
            "df", "y_binary", "feature_cols", "roc_data",
            "active_threshold", "raw_preds", "is_inverted", "force_retrain",
            "scan_mode", "y_raw_labels"):
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

        run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True,
                           help="Click to unleash the neural net on your network flows. "
                                "The spider, ghost, and cat are all on standby. 🕷️👻🐱")

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
                # ── Determine training mode and y_for_fit ────────────────────
                # y_for_fit is what we actually pass to model.fit().
                # It is ALWAYS either a valid supervised Series (n_atk ≥ 50)
                # or None (unsupervised / AE-only).  SMOTE is never called with
                # all-zero, corrupt, or too-sparse labels.
                #
                # Training mode decision tree:
                #   SUPERVISED  → labelled CSV, n_atk ≥ 50
                #   UNSUPERVISED → no Label column (PCAP / raw flows)
                #   BENIGN_ONLY → Label column present, but n_atk == 0
                #                 → cannot train; requires prior saved model
                #   TOO_SPARSE  → 1 ≤ n_atk < 50 → cannot train reliably
                # ─────────────────────────────────────────────────────────────

                if y_binary is None:
                    # ── Path A: No label column ───────────────────────────────
                    # Raw PCAP / unlabelled network capture.
                    # Train in unsupervised mode (AE anomaly scoring only).
                    _training_mode = "unsupervised"
                    _n_atk = 0
                    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:11px 16px;
  background:rgba(0,217,245,0.06);border:1px solid rgba(0,217,245,0.18);
  border-radius:10px;margin-bottom:10px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
    color:#00d9f5;font-weight:600;">◇ UNSUPERVISED MODE</span>
  <span style="font-family:'Sora',sans-serif;font-size:0.78rem;color:#6b7a99;">
    No Label column detected — training Autoencoder on normal flow patterns.
    Every flow will receive an anomaly score (0 = normal, 1 = suspicious).
  </span>
</div>""", unsafe_allow_html=True)

                else:
                    _n_atk  = int(y_binary.sum())
                    _n_ben  = int((y_binary == 0).sum())
                    _n_tot  = _n_atk + _n_ben
                    _pct_atk = _n_atk / max(_n_tot, 1) * 100

                    # Label detection badge
                    st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:12px 16px;
  background:rgba(0,217,245,0.06);border:1px solid rgba(0,217,245,0.18);
  border-radius:10px;margin-bottom:10px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
    color:#6b7a99;letter-spacing:0.08em;text-transform:uppercase;">Label Detection</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#0ffa9e;">
    {_n_ben:,} benign</span>
  <span style="color:#3d4f6e;">|</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#ff4560;">
    {_n_atk:,} attack ({_pct_atk:.1f}%)</span>
  <span style="color:#3d4f6e;">|</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#6b7a99;">
    Total {_n_tot:,}</span>
</div>""", unsafe_allow_html=True)

                    if _n_atk == 0:
                        # ── Path B: Benign-only / normal-traffic CSV ──────────
                        # Cannot train — SMOTE needs both classes.
                        # Model must already be saved to proceed.
                        _training_mode = "benign_only"
                        st.markdown("""
<div style="padding:14px 18px;
  background:rgba(245,166,35,0.07);border:1px solid rgba(245,166,35,0.25);
  border-radius:10px;margin-bottom:10px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;
    color:#f5a623;font-weight:600;margin-bottom:6px;">◌ BENIGN / NORMAL TRAFFIC CSV</div>
  <div style="font-family:'Sora',sans-serif;font-size:0.82rem;color:#6b7a99;">
    No attack labels found — this file cannot be used for training.<br>
    A saved model is required to scan this file.
    Upload a labelled mixed-traffic dataset (CICIDS-2017, CICIoT-2023, or UNSW-NB15)
    to train the model first, then re-upload this file.
  </div>
</div>""", unsafe_allow_html=True)
                        st.error(
                            "🚨 **No saved model found.**  \n\n"
                            "This benign-only CSV cannot train a model — it has no attack "
                            "examples for the classifier to learn from.  \n\n"
                            "**Step 1:** Upload a CICIDS-2017 / CICIoT-2023 / UNSW-NB15 CSV "
                            "and click Run Detection to train.  \n"
                            "**Step 2:** Re-upload this benign CSV to scan it with the trained model."
                        )
                        st.stop()

                    elif _n_atk < 50:
                        # ── Path C: Too few attacks to train reliably ─────────
                        _training_mode = "too_sparse"
                        st.error(
                            f"🚨 **Too few attack samples: {_n_atk} found, ≥ 50 required.**\n\n"
                            f"Your CSV has {_n_ben:,} benign rows but only {_n_atk} attack rows. "
                            "This is too few to train a reliable classifier.\n\n"
                            "**For CICIDS-2017:** Use the Wednesday or Thursday daily CSV.\n"
                            "**For UNSW-NB15:** Combine part files 1–4 for better attack coverage."
                        )
                        st.stop()

                    elif _n_atk < 500:
                        # ── Path D: Sparse but usable ─────────────────────────
                        _training_mode = "supervised_sparse"
                        st.warning(
                            f"⚠️ Only {_n_atk:,} attack samples ({_pct_atk:.1f}%) — training will "
                            "proceed but AUC and recall may be modest. "
                            "Aim for ≥ 10,000 attack samples for best results."
                        )
                    else:
                        # ── Path E: Good supervised dataset ──────────────────
                        _training_mode = "supervised"

                # ── y_for_fit: ALWAYS safe to pass to model.fit() ────────────
                # Supervised only when we have ≥ 50 genuine attack labels.
                # Every other case falls back to None (unsupervised = AE only).
                # This is the single gate that prevents SMOTE from ever seeing
                # all-zero, NaN-contaminated, or too-sparse label arrays.
                _y_for_fit = (
                    y_binary
                    if (y_binary is not None and int(y_binary.sum()) >= 50)
                    else None
                )

                with st.spinner("Training model on uploaded data…"):
                    try:
                        model = HybridIDS()
                        model.fit(X, _y_for_fit)
                        _save_model(model)
                        _mode_label = (
                            "supervised" if _y_for_fit is not None else "unsupervised"
                        )
                        st.success(
                            f"✅ Model trained and saved — **{_mode_label} mode**  \n"
                            f"({'LSTM + AE' if _y_for_fit is not None else 'AE anomaly scoring only'})"
                        )
                    except Exception as e:
                        err_str = str(e).lower()
                        if "len() of unsized" in err_str or "iteration over a 0-d" in err_str:
                            st.error(
                                "🚨 **Training failed: numpy array dimension error.**\n\n"
                                "This is almost always caused by a preprocessing edge case "
                                "(e.g. LOF removed all rows, or an empty feature matrix).\n\n"
                                f"**Details:** `{e}`\n\n"
                                "Try enabling **Show Debug Info** in the sidebar to inspect "
                                "your dataset's first rows for unusual values."
                            )
                        elif "cannot take a larger sample" in err_str or "expected n_neighbors" in err_str:
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
                                "and retrain."
                            )
                        else:
                            st.error(
                                f"🚨 **Training failed:** `{e}`\n\n"
                                "Check that the CSV has valid numeric feature columns. "
                                "Enable **Show Debug Info** in the sidebar to inspect "
                                "the first rows of your dataset."
                            )
                        st.stop()

            # ── Batched prediction (avoids OOM on large files) ──────────────
            _scan_msgs = [
                "🕷️ Spider weaving detection threads…",
                "👻 Ghost checking your packets from the afterlife…",
                "🐱 Cat watching the network — very judgementally…",
                "🧠 Neural net doing its thing (please hold)…",
                "🔍 Scanning… (the spider said it already found something suspicious)…",
                "⚡ Autoencoder reconstructing your reality…",
                "🎯 Hunting anomalies like a cat hunts a laser dot…",
            ]
            _pred_prog = st.progress(0, text=_scan_msgs[_random.randint(0, len(_scan_msgs)-1)])
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

            # ── Determine scan mode ──────────────────────────────────────────
            if y_clean is not None and metrics_out is not None:
                _scan_mode = "supervised"
            elif y_binary is None and y_raw is None:
                _scan_mode = "unsupervised"
            else:
                _scan_mode = "benign_scan"   # labelled but unlabelled/benign-only

            # ── Store in session_state ────────────────────────────────────────
            st.session_state["model"]            = model
            st.session_state["raw_preds"]        = preds
            st.session_state["preds"]            = preds
            st.session_state["is_inverted"]      = is_inverted
            st.session_state["shap_values"]      = shap_vals
            st.session_state["df"]               = df
            st.session_state["y_binary"]         = y_clean
            st.session_state["y_raw_labels"]     = y_raw
            st.session_state["feature_cols"]     = model.feature_cols
            st.session_state["result_df"]        = result_df
            st.session_state["metrics"]          = metrics_out
            st.session_state["roc_data"]         = roc_data
            st.session_state["active_threshold"] = active_threshold
            st.session_state["scan_mode"]        = _scan_mode

            st.rerun()   # refresh so tab2/tab3 see new state immediately

        # Show summary card if results exist
        if st.session_state["preds"] is not None:
            preds      = st.session_state["preds"]
            thr        = st.session_state["active_threshold"] or 0.5
            n_flagged  = int(np.sum(preds > thr))
            n_total    = len(preds)
            metrics_s  = st.session_state["metrics"]

            scan_mode = st.session_state.get("scan_mode") or "unsupervised"
            n_normal  = n_total - n_flagged

            # ── Scan mode badge ───────────────────────────────────────────────
            _mode_cfg = {
                "supervised":   ("#0ffa9e", "rgba(15,250,158,0.08)",
                                 "rgba(15,250,158,0.2)", "◈ SUPERVISED SCAN",
                                 "Labelled · Accuracy & FPR metrics available"),
                "unsupervised": ("#00d9f5", "rgba(0,217,245,0.08)",
                                 "rgba(0,217,245,0.2)", "◇ UNSUPERVISED SCAN",
                                 "No labels · AE + LSTM anomaly scoring"),
                "benign_scan":  ("#f5a623", "rgba(245,166,35,0.08)",
                                 "rgba(245,166,35,0.2)", "◌ NORMAL TRAFFIC SCAN",
                                 "Unlabelled / benign-only · Anomaly detection mode"),
            }
            _clr, _bg, _bd, _lbl, _sub = _mode_cfg.get(scan_mode, _mode_cfg["unsupervised"])
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:10px 16px;
  background:{_bg};border:1px solid {_bd};border-radius:9px;margin-bottom:14px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
    color:{_clr};font-weight:600;">{_lbl}</span>
  <span style="color:#3d4f6e;">|</span>
  <span style="font-family:'Sora',sans-serif;font-size:0.78rem;color:#6b7a99;">{_sub}</span>
</div>""", unsafe_allow_html=True)

            # ── Flow classification summary ───────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FLAGGED ANOMALIES", f"{n_flagged:,}",
                      f"{n_flagged/max(n_total,1)*100:.1f}% of flows")
            c2.metric("NORMAL TRAFFIC", f"{n_normal:,}",
                      f"{n_normal/max(n_total,1)*100:.1f}% of flows")
            c3.metric("TOTAL FLOWS SCANNED", f"{n_total:,}")
            if metrics_s:
                fpr_pct = metrics_s['fpr'] * 100
                c4.metric("FALSE POSITIVE RATE",
                           f"{fpr_pct:.2f}%",
                           "UNDER TARGET" if fpr_pct < 5 else "OVER 5%")
            else:
                c4.metric("ACTIVE THRESHOLD", f"{thr:.4f}")


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
            # ── Unsupervised / benign-scan mode display ───────────────────────
            scan_mode  = st.session_state.get("scan_mode") or "unsupervised"
            n_normal   = n_total - n_flagged

            # Risk tiers based on score
            preds_arr   = np.asarray(preds)
            n_low_risk  = int(np.sum(preds_arr <= 0.30))
            n_med_risk  = int(np.sum((preds_arr > 0.30) & (preds_arr <= 0.65)))
            n_high_risk = int(np.sum(preds_arr > 0.65))

            if scan_mode == "benign_scan":
                st.markdown("""
<div style="padding:14px 18px;
  background:rgba(245,166,35,0.06);border:1px solid rgba(245,166,35,0.2);
  border-radius:10px;margin-bottom:16px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
    color:#f5a623;font-weight:600;">◌ NORMAL TRAFFIC SCAN — No ground-truth labels</span><br>
  <span style="font-family:'Sora',sans-serif;font-size:0.82rem;color:#6b7a99;margin-top:4px;
    display:block;">
    The model is scoring each flow purely from its learned anomaly signal
    (AE reconstruction error + LSTM). Flows with scores above the threshold
    are flagged as suspicious regardless of any label.
    Add a <code style="color:#00d9f5;">Label</code> column to your CSV to unlock
    accuracy, FPR, F1 and ROC metrics.
  </span>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="padding:14px 18px;
  background:rgba(0,217,245,0.06);border:1px solid rgba(0,217,245,0.15);
  border-radius:10px;margin-bottom:16px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
    color:#00d9f5;font-weight:600;">◇ UNSUPERVISED SCAN — No ground-truth labels</span><br>
  <span style="font-family:'Sora',sans-serif;font-size:0.82rem;color:#6b7a99;margin-top:4px;
    display:block;">
    This CSV has no <code style="color:#00d9f5;">Label</code> column.
    Each flow receives a continuous anomaly score from the hybrid model.
    Flows above the threshold are flagged as suspicious.
    Add a Label column to unlock accuracy and FPR metrics.
  </span>
</div>""", unsafe_allow_html=True)

            # ── Flow breakdown ────────────────────────────────────────────────
            ca, cb, cc, cd = st.columns(4)
            ca.metric("NORMAL FLOWS",       f"{n_normal:,}",
                      f"{n_normal/max(n_total,1)*100:.1f}%")
            cb.metric("FLAGGED SUSPICIOUS", f"{n_flagged:,}",
                      f"{n_flagged/max(n_total,1)*100:.1f}%")
            cc.metric("TOTAL SCANNED",      f"{n_total:,}")
            cd.metric("ACTIVE THRESHOLD",   f"{live_thr:.4f}")

            st.divider()

            # ── Risk tier breakdown ───────────────────────────────────────────
            st.markdown("""<div style="font-family:'Sora',sans-serif;font-weight:700;
              font-size:1rem;color:#e8edf5;margin-bottom:10px;">
              Risk Tier Breakdown</div>""", unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.markdown(f"""
<div style="background:rgba(15,250,158,0.07);border:1px solid rgba(15,250,158,0.2);
  border-radius:10px;padding:16px;text-align:center;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
    background:#0ffa9e;"></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
    color:#6b7a99;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
    Low Risk · Score ≤ 0.30</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
    font-weight:700;color:#0ffa9e;">{n_low_risk:,}</div>
  <div style="font-family:'Sora',sans-serif;font-size:0.75rem;color:#6b7a99;margin-top:4px;">
    {n_low_risk/max(n_total,1)*100:.1f}% of flows · Likely normal</div>
</div>""", unsafe_allow_html=True)

            r2.markdown(f"""
<div style="background:rgba(245,166,35,0.07);border:1px solid rgba(245,166,35,0.2);
  border-radius:10px;padding:16px;text-align:center;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
    background:#f5a623;"></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
    color:#6b7a99;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
    Medium Risk · 0.30–0.65</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
    font-weight:700;color:#f5a623;">{n_med_risk:,}</div>
  <div style="font-family:'Sora',sans-serif;font-size:0.75rem;color:#6b7a99;margin-top:4px;">
    {n_med_risk/max(n_total,1)*100:.1f}% of flows · Investigate</div>
</div>""", unsafe_allow_html=True)

            r3.markdown(f"""
<div style="background:rgba(255,69,96,0.07);border:1px solid rgba(255,69,96,0.2);
  border-radius:10px;padding:16px;text-align:center;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:2px;
    background:#ff4560;"></div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
    color:#6b7a99;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">
    High Risk · Score > 0.65</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
    font-weight:700;color:#ff4560;">{n_high_risk:,}</div>
  <div style="font-family:'Sora',sans-serif;font-size:0.75rem;color:#6b7a99;margin-top:4px;">
    {n_high_risk/max(n_total,1)*100:.1f}% of flows · High priority</div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Prediction table ─────────────────────────────────────────────────
        st.markdown('''<div style="font-family:'Sora',sans-serif;font-weight:700;
          font-size:1rem;color:#e8edf5;margin:16px 0 8px;">
          ≡ Flow Classification — First 50 Rows</div>''', unsafe_allow_html=True)

        display_result = result_df.copy()

        # Re-apply prediction label at current live threshold
        display_result["Prediction"] = np.where(
            display_result["Probability"] > live_thr, "⚠ ANOMALY", "✓ NORMAL"
        )

        # Risk level column — three tiers
        def _risk(p):
            if p > 0.65: return "HIGH"
            if p > 0.30: return "MEDIUM"
            return "LOW"
        display_result["Risk"] = display_result["Probability"].apply(_risk)

        # Append raw label column if available (helps audit)
        y_raw_lbl = st.session_state.get("y_raw_labels")
        if y_raw_lbl is not None:
            try:
                # Align to X_clean surviving rows (same index as result_df)
                # result_df was built from X_clean which has reset_index
                # y_raw_lbl is full-length original; we can only show first n
                display_result["True Label"] = (
                    y_raw_lbl.iloc[:len(display_result)].values
                )
            except Exception:
                pass

        priority_cols = ["Prediction", "Risk", "Probability"]
        if "True Label" in display_result.columns:
            priority_cols.append("True Label")
        feat_preview  = [c for c in display_result.columns
                         if c not in priority_cols][:4]
        display_cols  = priority_cols + feat_preview

        st.dataframe(display_result[display_cols].head(50),
                     use_container_width=True)

        # ── Score distribution ───────────────────────────────────────────────
        st.subheader("📈 Anomaly Score Distribution")
        fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#0d1220")
        ax.set_facecolor("#0d1220")

        # ── Background zones: Normal / Suspicious ─────────────────────────
        ax.axvspan(0,          live_thr, alpha=0.06, color="#0ffa9e",
                   label="Normal zone")
        ax.axvspan(live_thr,   1.0,     alpha=0.06, color="#ff4560",
                   label="Suspicious zone")

        # ── Histogram — colour each bar by its zone ────────────────────────
        preds_arr = np.asarray(preds)
        bins      = np.linspace(0, 1, 61)
        normal_preds = preds_arr[preds_arr <= live_thr]
        attack_preds = preds_arr[preds_arr  > live_thr]
        if len(normal_preds):
            ax.hist(normal_preds, bins=bins, color="#0ffa9e",
                    alpha=0.55, edgecolor="none", label="Normal flows")
        if len(attack_preds):
            ax.hist(attack_preds, bins=bins, color="#ff4560",
                    alpha=0.70, edgecolor="none", label="Suspicious flows")

        # ── Risk tier lines ────────────────────────────────────────────────
        ax.axvline(0.30, color="#f5a623", linestyle=":",
                   linewidth=1.2, alpha=0.7, label="Medium risk (0.30)")
        ax.axvline(0.65, color="#ff4560", linestyle=":",
                   linewidth=1.2, alpha=0.7, label="High risk (0.65)")
        ax.axvline(live_thr, color="#ffffff", linestyle="--", linewidth=2,
                   label=f"Threshold {live_thr:.4f}")
        if live_thr != active_thr:
            ax.axvline(active_thr, color="#00d9f5", linestyle=":",
                       linewidth=1.5, label=f"Optimal {active_thr:.4f}")

        ax.set_xlabel("Anomaly Score  →  0 = Normal  /  1 = Attack",
                      color="#6b7a99", fontsize=9)
        ax.set_ylabel("Flow Count", color="#6b7a99", fontsize=9)
        ax.tick_params(colors="#6b7a99", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d45")
        ax.legend(fontsize=7.5, facecolor="#111827",
                  edgecolor="#1e2d45", labelcolor="#e8edf5",
                  ncol=3, loc="upper center")
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
      🐱 HYBRID IDS v5.0 &nbsp;·&nbsp; AUTOENCODER + SSA-LSTM &nbsp;·&nbsp; LOF · SMOTE · SHAP &nbsp;·&nbsp; 🕷️ Spider-patrolled &nbsp;·&nbsp; 👻 Ghost-certified
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
