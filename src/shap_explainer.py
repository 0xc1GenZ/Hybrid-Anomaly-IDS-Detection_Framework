import numpy as np
import shap

# KernelExplainer scales as O(background × features) per explained sample.
# Caps keep runtime manageable on both tiny and large datasets.
_MAX_BACKGROUND = 50    # background samples fed to KernelExplainer
_MAX_EXPLAIN    = 30    # max samples explained per call


class SHAPExplainer:
    def __init__(self):
        self.explainer = None

    def build(self, model_predict, background_data):
        """
        Build a KernelExplainer with a safely capped background set.
        background_data can be 2D (tabular) or 3D (LSTM sequences — shape [n, window, features]).
        For 3D data, we flatten the time dimension so KernelExplainer sees a 2D summary.
        """
        bg = self._cap(background_data, _MAX_BACKGROUND)
        bg_2d = self._flatten_if_3d(bg)
        self.explainer = shap.KernelExplainer(
            self._wrap_predict(model_predict, background_data),
            bg_2d,
        )
        return self.explainer

    def explain(self, data, sample_size=None):
        """Return SHAP values for up to _MAX_EXPLAIN samples of data."""
        if self.explainer is None:
            return None

        cap   = sample_size if sample_size is not None else _MAX_EXPLAIN
        cap   = min(cap, len(data), _MAX_EXPLAIN)
        data  = self._cap(data, cap)
        data_2d = self._flatten_if_3d(data)

        try:
            return self.explainer.shap_values(data_2d)
        except Exception as e:
            print(f"  ⚠️  SHAP explain error: {e}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cap(data, n):
        """Uniformly subsample the first axis to at most n rows."""
        if len(data) <= n:
            return data
        idx = np.linspace(0, len(data) - 1, n, dtype=int)
        return data[idx] if isinstance(data, np.ndarray) else data.iloc[idx]

    @staticmethod
    def _flatten_if_3d(data):
        """
        LSTM sequences are 3D: (samples, window, features).
        KernelExplainer expects 2D input, so we flatten window × features into one vector.
        This preserves all information while satisfying the API contract.
        """
        arr = data.values if hasattr(data, 'values') else np.asarray(data)
        if arr.ndim == 3:
            return arr.reshape(arr.shape[0], -1)
        return arr

    @staticmethod
    def _wrap_predict(model_predict, reference_data):
        """
        Return a wrapper that accepts 2D flattened input and reshapes it back to the
        original dimensionality before calling model_predict.
        This lets KernelExplainer work with 3D LSTM models transparently.
        """
        ref = reference_data.values if hasattr(reference_data, 'values') else np.asarray(reference_data)
        original_shape = ref.shape[1:]  # e.g. (window, features) for 3D

        def predict_fn(X_flat):
            X_flat = np.asarray(X_flat)
            if len(original_shape) > 1:
                X = X_flat.reshape((X_flat.shape[0],) + original_shape)
            else:
                X = X_flat
            return model_predict(X)

        return predict_fn
