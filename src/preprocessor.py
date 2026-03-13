import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE, ADASYN

# Dataset size thresholds
_LOF_SKIP_BELOW   = 40      # skip LOF entirely if n_samples < 2 * n_neighbors
_LOF_SUBSAMPLE_AT = 50_000  # fit LOF on a random subsample above this size (O(n²) cost)
_LOF_SUBSAMPLE_N  = 10_000  # subsample size for LOF fitting on large data


class Preprocessor:
    def __init__(self):
        self.scaler           = MinMaxScaler()
        # novelty=True: allows .predict() on unseen data (required for inference path)
        self.lof              = LocalOutlierFactor(contamination=0.05, n_neighbors=20, novelty=True)
        self.smote            = None
        self.categorical_cols = None
        self.encoder_cols     = None
        self.feature_cols     = None
        self._lof_fitted      = False  # track whether LOF has been fitted at least once

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, X, y=None, fit=True):
        X = X.copy()
        if y is not None:
            y = y.copy()

        # ── Step 1: Basic cleaning ────────────────────────────────────
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
        if y is not None:
            y = y.loc[X.index]   # align labels to surviving rows

        n_samples = len(X)
        if n_samples == 0:
            print("⚠️  No samples remain after NaN/Inf removal.")
            return pd.DataFrame(), None

        # ── Step 2: Categorical encoding ──────────────────────────────
        if fit:
            self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        if self.categorical_cols:
            X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
            if fit:
                self.encoder_cols = X_encoded.columns.tolist()
            else:
                X_encoded = X_encoded.reindex(columns=self.encoder_cols, fill_value=0)
        else:
            X_encoded = X.copy()

        # ── Step 3: Type-cast, fill, clip ────────────────────────────
        X_final = X_encoded.astype(float)
        X_final = X_final.fillna(X_final.median())
        X_final = X_final.clip(lower=0, upper=X_final.quantile(0.99), axis=1)

        # ── Step 4: Scale ─────────────────────────────────────────────
        if fit:
            self.feature_cols = X_final.columns.tolist()
            X_scaled_arr      = self.scaler.fit_transform(X_final)
        else:
            X_final      = X_final.reindex(columns=self.feature_cols, fill_value=0)
            X_final      = X_final.clip(lower=0, upper=X_final.quantile(0.99), axis=1)
            X_scaled_arr = self.scaler.transform(X_final)

        X_scaled = pd.DataFrame(X_scaled_arr, columns=self.feature_cols, index=X_final.index)

        # ── Step 5: LOF outlier removal ───────────────────────────────
        X_clean, y_clean = self._apply_lof(X_scaled, y, fit, n_samples)

        # Always return a consistent 2-tuple (y_clean is None when y was not provided)
        return X_clean, y_clean

    def balance(self, X, y):
        y_arr = np.asarray(y)
        classes, counts = np.unique(y_arr, return_counts=True)

        # Guard: SMOTE requires at least 2 distinct classes
        if len(classes) < 2:
            print(f"  ⚠️  Only 1 class present in y (class={classes[0]}). "
                  "Skipping oversampling – check label binarisation.")
            return X, y

        minority_samples = int(counts.min())
        majority_samples = int(counts.max())
        print(f"  Class distribution – majority: {majority_samples:,}  minority: {minority_samples:,}")

        if minority_samples < 2:
            print("  ⚠️  Too few minority samples (<2). Skipping oversampling – relying on class_weight.")
            return X, y

        # Skip oversampling if already reasonably balanced (minority >= 20% of majority)
        if minority_samples >= 0.2 * majority_samples:
            print("  ℹ️  Classes already reasonably balanced. Skipping oversampling.")
            return X, y

        if minority_samples <= 5:
            print("  ⚠️  Low minority samples. Using ADASYN(n_neighbors=1).")
            self.smote = ADASYN(n_neighbors=1, random_state=42)
        else:
            k = min(5, minority_samples - 1)
            self.smote = SMOTE(k_neighbors=k, random_state=42)

        return self.smote.fit_resample(X, y)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_lof(self, X_scaled, y, fit, n_samples):
        """
        Size-aware LOF application:
          • tiny  (< _LOF_SKIP_BELOW rows)   → skip (too few for reliable density estimate)
          • small (< _LOF_SUBSAMPLE_AT rows)  → fit on full data
          • large (≥ _LOF_SUBSAMPLE_AT rows)  → fit on random subsample, predict on all
        During inference (fit=False), reuses the already-fitted LOF instance.
        """
        if n_samples < _LOF_SKIP_BELOW:
            print(f"  ℹ️  Skipping LOF (n_samples={n_samples} < {_LOF_SKIP_BELOW}).")
            return X_scaled.reset_index(drop=True), (
                y.reset_index(drop=True) if y is not None else None
            )

        if fit:
            if n_samples >= _LOF_SUBSAMPLE_AT:
                rng     = np.random.default_rng(42)
                sub_idx = rng.choice(n_samples, _LOF_SUBSAMPLE_N, replace=False)
                print(f"  ℹ️  Large dataset: fitting LOF on {_LOF_SUBSAMPLE_N:,} sampled rows.")
                self.lof.fit(X_scaled.iloc[sub_idx])
            else:
                self.lof.fit(X_scaled)
            self._lof_fitted = True

        if not self._lof_fitted:
            print("  ⚠️  LOF not fitted — skipping outlier removal.")
            return X_scaled.reset_index(drop=True), (
                y.reset_index(drop=True) if y is not None else None
            )

        outlier_labels = self.lof.predict(X_scaled)
        mask = outlier_labels != -1

        # Safety: never remove the only minority-class (attack) samples
        if y is not None:
            minority_mask = (y.values == 1)
            if minority_mask.sum() > 0 and (mask & minority_mask).sum() == 0:
                print("  ⚠️  LOF would erase ALL attack samples — preserving them.")
                mask = mask | minority_mask

        removed = int((~mask).sum())
        if removed:
            print(f"  ℹ️  LOF removed {removed:,} outlier rows ({removed / n_samples * 100:.1f}%).")

        X_clean = X_scaled[mask].reset_index(drop=True)
        y_clean = y[mask].reset_index(drop=True) if y is not None else None
        return X_clean, y_clean
