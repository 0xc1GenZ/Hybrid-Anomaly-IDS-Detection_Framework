"""
dashboard.py — Streamlit dashboard for the Hybrid IDS Framework
================================================================
Run from the project root:
    streamlit run dashboard.py

Compatible with hybrid_ids.py, preprocessor.py, autoencoder.py,
lstm_classifier.py, shap_explainer.py (all in src/).

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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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

def _read_uploaded_csv(uploaded) -> pd.DataFrame:
    """Read a Streamlit UploadedFile into a DataFrame, stripping column spaces."""
    df = pd.read_csv(uploaded, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def _split_features_labels(df: pd.DataFrame):
    """
    Detect the label column (case-insensitive), binarise it, and return
    (X, y_binary | None, y_raw | None).
    """
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        return df, None, None

    y_raw        = df[label_col]
    benign_label = _detect_benign_label(y_raw)
    X            = df.drop(columns=[label_col])

    if benign_label is None:
        return X, None, y_raw          # no benign label → unsupervised

    y = (y_raw != benign_label).astype(int)
    return X, y, y_raw


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
    y_pred = (np.asarray(y_pred_prob) > threshold).astype(int)
    y_true = np.asarray(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred))                      , 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0))    , 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0))        , 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0))            , 4),
        "fpr":       round(float(fpr), 4),
        "fnr":       round(float(fnr), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="Hybrid IDS Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔒 Hybrid IDS Framework")
st.markdown(
    "**Anomaly-Based Intrusion Detection System** | "
    "Autoencoder + SSA-LSTM · LOF · SMOTE · SHAP"
)
st.caption("Built for IoT & Critical Infrastructure | Master's Project – Jain University")

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Model Settings")
    st.info(
        "**Model:** Hybrid Autoencoder + LSTM\n\n"
        "**Datasets:** CIC-IDS2017 · CICIoT2023\n\n"
        "**FPR Target:** < 5%"
    )
    show_debug = st.checkbox("Show Debug Info", value=False)
    use_sample = st.checkbox("Use Sample Data (quick test)", value=False)

    st.divider()
    if _model_on_disk():
        st.success("✅ Saved model found on disk")
        if st.button("🗑️ Clear saved model"):
            import shutil
            shutil.rmtree(_MODELS_DIR, ignore_errors=True)
            st.session_state.clear()
            st.rerun()
    else:
        st.warning("No saved model – will train on first upload")

# ===========================================================================
# SESSION STATE INITIALISATION
# BUG 5 FIX: all results persisted in session_state so tab2/tab3 can access
# them without re-running any computation.
# ===========================================================================
for key in ("preds", "shap_values", "result_df", "metrics", "model",
            "df", "y_binary", "feature_cols"):
    if key not in st.session_state:
        st.session_state[key] = None

# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3 = st.tabs(
    ["📤 Upload & Predict", "📊 Results & Metrics", "🔍 SHAP Explainability"]
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
        st.info("⬆️ Upload a CSV or tick **Use Sample Data** in the sidebar to begin.")
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
                # BUG 2 FIX: use our save/load helpers instead of model.load()/model.save()
                if _model_on_disk():
                    try:
                        model = _load_model()
                        st.success("✅ Saved model loaded from disk")
                    except Exception as e:
                        st.warning(f"Failed to load saved model ({e}). Re-training…")
                        model = None
                else:
                    model = None

            if model is None:
                with st.spinner("Training model on uploaded data (first run)…"):
                    try:
                        model = HybridIDS()
                        model.fit(X, y_binary)
                        _save_model(model)
                        st.success("✅ Model trained and saved")
                    except Exception as e:
                        st.error(f"Training failed: {e}")
                        st.stop()

            with st.spinner("Running predictions…"):
                try:
                    preds, shap_vals = model.predict(X)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            # --- Store everything in session_state ---
            st.session_state["model"]        = model
            st.session_state["preds"]        = preds
            st.session_state["shap_values"]  = shap_vals
            st.session_state["df"]           = df
            st.session_state["y_binary"]     = y_binary
            st.session_state["feature_cols"] = model.feature_cols

            result_df = df.copy()
            result_df["Prediction"]  = np.where(preds > 0.5, "ANOMALY", "BENIGN")
            result_df["Probability"] = preds
            st.session_state["result_df"] = result_df

            if y_binary is not None and len(y_binary) == len(preds):
                st.session_state["metrics"] = _compute_metrics(y_binary, preds)
            else:
                st.session_state["metrics"] = None

            st.rerun()   # refresh so tab2/tab3 see new state immediately

        # Show summary card if results exist
        if st.session_state["preds"] is not None:
            preds      = st.session_state["preds"]
            n_flagged  = int(np.sum(preds > 0.5))
            n_total    = len(preds)
            fpr_target = "< 5%"

            c1, c2, c3 = st.columns(3)
            c1.metric("🚨 Flagged", f"{n_flagged:,}", f"{n_flagged/max(n_total,1)*100:.1f}%")
            c2.metric("📦 Total Flows", f"{n_total:,}")
            c3.metric("🎯 FPR Target", fpr_target)


# ---------------------------------------------------------------------------
# TAB 2 — Results & Metrics
# BUG 5 FIX: reads from session_state; BUG 6 FIX: real computed metrics
# ---------------------------------------------------------------------------
with tab2:
    preds     = st.session_state.get("preds")
    result_df = st.session_state.get("result_df")
    metrics   = st.session_state.get("metrics")
    y_binary  = st.session_state.get("y_binary")

    if preds is None:
        st.info("Run detection in **Upload & Predict** first.")
    else:
        n_total   = len(preds)
        n_flagged = int(np.sum(preds > 0.5))

        # --- Metrics row ---
        if metrics is not None:
            st.subheader("📐 Classification Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
            m2.metric("F1 Score",  f"{metrics['f1']*100:.2f}%")
            m3.metric("Precision", f"{metrics['precision']*100:.2f}%")
            m4.metric("Recall",    f"{metrics['recall']*100:.2f}%")

            fpr_val   = metrics['fpr'] * 100
            fpr_delta = "✅ Below target" if fpr_val < 5.0 else "⚠️ Above 5% target"
            m5.metric("False Positive Rate", f"{fpr_val:.2f}%", fpr_delta)

            st.divider()

            # Confusion matrix
            st.subheader("🔢 Confusion Matrix")
            cm_df = pd.DataFrame(
                [[metrics["tn"], metrics["fp"]],
                 [metrics["fn"], metrics["tp"]]],
                index=["Actual Benign", "Actual Attack"],
                columns=["Predicted Benign", "Predicted Attack"],
            )
            st.dataframe(cm_df, use_container_width=False)
        else:
            st.info(
                "No ground-truth labels found — metrics unavailable. "
                "Add a 'Label' column to your CSV to see accuracy, FPR, F1, etc."
            )
            m1, m2 = st.columns(2)
            m1.metric("🚨 Flagged Anomalies", f"{n_flagged:,}")
            m2.metric("📦 Total Flows",        f"{n_total:,}")

        st.divider()

        # --- Prediction table ---
        st.subheader("📋 Prediction Results (first 50 rows)")
        display_cols = ["Prediction", "Probability"] + list(
            st.session_state["df"].columns[:5]
        )
        st.dataframe(
            result_df[display_cols].head(50),
            use_container_width=True,
        )

        # --- Score distribution ---
        st.subheader("📈 Anomaly Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(preds, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
        ax.axvline(0.5, color="#2c3e50", linestyle="--", linewidth=1.5, label="Threshold (0.5)")
        ax.set_xlabel("Anomaly Probability")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Prediction Scores")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # --- Download ---
        csv_bytes = result_df.to_csv(index=False).encode()
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
        st.info("Run detection in **Upload & Predict** first.")
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

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            bars = ax2.barh(top_names[::-1], top_vals[::-1], color="#2ecc71")
            ax2.set_xlabel("Mean |SHAP value|")
            ax2.set_title("Feature Importance (SHAP)")
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
st.divider()
st.caption(
    "Hybrid IDS Dashboard v3.0 | "
    "Autoencoder + SSA-LSTM · LOF · SMOTE · SHAP | "
    "GitHub: https://github.com/0xc1GenZ/hybrid-ids-framework"
)
