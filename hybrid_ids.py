import os; os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE, ADASYN  # Added ADASYN fallback
from sklearn.base import BaseEstimator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress LOF warning

class HybridIDS(BaseEstimator):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lof = LocalOutlierFactor(contamination=0.05, n_neighbors=20)
        self.smote = None  # Will set dynamically
        self.autoencoder = None
        self.lstm = None
        self.explainer = None
        self.threshold = None
        self.categorical_cols = ['protocol_type', 'flag']  # Define your categorical columns here
        self.encoder = None  # For one-hot encoding
        self.feature_cols = None  # Track processed columns
        self.window_size = 10  # Default timestep for LSTM sequences

    def fit(self, X, y):
        X = X.copy()  # Avoid modifying input
        y = y.copy()
        
        # Step 1: One-hot encode categorical columns
        if self.encoder is None:
            self.encoder = pd.get_dummies(X[self.categorical_cols], drop_first=True)
            X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
        else:
            X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
            # Align columns with training data
            X_encoded = X_encoded.reindex(columns=self.encoder.columns, fill_value=0)
        
        # Combine numerical and encoded categorical
        numerical_cols = [col for col in X.columns if col not in self.categorical_cols]
        X_numerical = X[numerical_cols]
        X_final = pd.concat([X_numerical, X_encoded], axis=1)
        self.feature_cols = X_final.columns.tolist()  # Save for predict
        
        # Step 2: Preprocessing (scaling)
        X_scaled = self.scaler.fit_transform(X_final)
        X_scaled = pd.DataFrame(X_scaled, columns=X_final.columns)
        
        # Step 3: Outlier removal
        outliers = self.lof.fit_predict(X_scaled)
        X_clean = X_scaled[outliers != -1]
        y_clean = y[outliers != -1]
        
        # Step 4: Dynamic SMOTE balancing
        minority_samples = np.sum(y_clean == 1)  # Count minority (attacks)
        if minority_samples <= 1:
            print("⚠️ Warning: Too few minority samples. Skipping SMOTE.")
            X_bal, y_bal = X_clean, y_clean
        else:
            k_neighbors = min(5, minority_samples - 1)  # Dynamic k
            self.smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_bal, y_bal = self.smote.fit_resample(X_clean, y_clean)
        
        # Step 5: Train Autoencoder on benign data (2D input)
        benign_idx = y_bal[y_bal == 0].index
        X_benign = X_bal.loc[benign_idx]
        self.autoencoder = self._build_autoencoder(X_bal.shape[1])
        self.autoencoder.fit(X_benign, X_benign, epochs=50, batch_size=min(64, len(X_benign)), verbose=0)
        
        # Compute threshold
        recon_errors = self.autoencoder.predict(X_benign).mean(axis=1)
        self.threshold = recon_errors.mean() + 3 * recon_errors.std()
        
        # Step 6: Dynamic window_size for small data
        self.window_size = min(10, len(X_bal) - 1)
        
        # Reshape for LSTM (3D: samples, timesteps, features)
        X_sequences, y_sequences = self._reshape_to_sequences(X_bal, y_bal, self.window_size)
        
        # Train LSTM on sequences
        self.lstm = self._build_lstm(X_sequences.shape[2])  # features as last dim
        self.lstm.fit(X_sequences, y_sequences, epochs=50, batch_size=min(64, len(X_sequences)), verbose=0)
        
        # Step 7: SHAP explainer (use a sample if small data)
        sample_size = min(100, X_sequences.shape[0])
        if sample_size > 0:
            self.explainer = shap.KernelExplainer(self.lstm.predict, X_sequences[:sample_size])
        
        return self

    def _reshape_to_sequences(self, X, y, window_size):
        """Create sequences for LSTM input from 2D data. y optional for predict."""
        if len(X) < window_size:
            # Fallback for small data: expand to (len(X), 1, features) as single timestep
            X_seq = np.expand_dims(X.values, axis=1)  # (samples, 1, features)
            if y is not None:
                y_seq = np.expand_dims(y.values, axis=1)  # (samples, 1)
                return X_seq, y_seq
            else:
                return X_seq, None
        
        X_seq = []
        y_seq = []
        for i in range(len(X) - window_size + 1):
            X_seq.append(X.iloc[i:i+window_size].values)
            if y is not None:
                y_seq.append(y.iloc[i+window_size-1])  # Label for last timestep
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        else:
            return X_seq, None

    def _build_autoencoder(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(64, activation='relu')(input_layer)
        encoder = Dense(32, activation='relu')(encoder)
        decoder = Dense(64, activation='relu')(encoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        return autoencoder

    def _build_lstm(self, input_dim):
        timesteps = self.window_size
        input_layer = Input(shape=(timesteps, input_dim))
        lstm1 = LSTM(128, return_sequences=True)(input_layer)
        lstm1 = Dropout(0.3)(lstm1)
        lstm2 = LSTM(128)(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        output = Dense(1, activation='sigmoid')(lstm2)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, X):
        X = X.copy()
        
        # One-hot encode categorical columns (same as fit)
        X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
        if self.encoder is not None:
            X_encoded = X_encoded.reindex(columns=self.encoder.columns, fill_value=0)
        
        numerical_cols = [col for col in X.columns if col not in self.categorical_cols]
        X_numerical = X[numerical_cols]
        X_final = pd.concat([X_numerical, X_encoded], axis=1)
        
        # Align with training columns
        if self.feature_cols is not None:
            X_final = X_final.reindex(columns=self.feature_cols, fill_value=0)
        
        # Scaling
        X_scaled = self.scaler.transform(X_final)
        X_scaled = pd.DataFrame(X_scaled, columns=X_final.columns)
        
        # Anomaly flagging (2D)
        recon_errors = self.autoencoder.predict(X_scaled, verbose=0).mean(axis=1)
        flagged_idx = recon_errors > self.threshold
        
        if flagged_idx.sum() == 0:
            return np.zeros(len(X)), None  # All normal
        
        flagged_X = X_scaled[flagged_idx]
        
        # Reshape flagged data for LSTM (X only, y=None for predict)
        flagged_sequences, _ = self._reshape_to_sequences(flagged_X, None, self.window_size)
        
        # Predict with small batch or verbose=0 to avoid progbar error
        if flagged_sequences.shape[0] == 0:
            flagged_preds = np.zeros(len(flagged_idx))
        else:
            flagged_preds = self.lstm.predict(flagged_sequences, batch_size=1, verbose=0).flatten()
        
        preds = np.zeros(len(X))
        preds[flagged_idx] = flagged_preds[:len(flagged_idx)]  # Match length
        
        # SHAP explanations (skip if no flagged)
        shap_values = None
        if flagged_sequences.shape[0] > 0:
            try:
                sample_size = min(100, flagged_sequences.shape[0])
                shap_values = self.explainer.shap_values(flagged_sequences[:sample_size])
            except Exception as e:
                print(f"SHAP warning: {e} – Skipping for small data")
        
        return preds, shap_values  # Return predictions and explanations

# ====================== EXAMPLE USAGE ======================
if __name__ == "__main__":
    from pathlib import Path

    # Automatic path detection
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "sample_flows.csv"

    print(f"Loading data from: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"❌ Sample file not found at: {data_path}\n"
                                f"Please make sure 'data/sample_flows.csv' exists in the project root.")

    # Load sample data
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    y = (df['label'] == 'attack').astype(int)

    # Train the model
    print("Training HybridIDS model...")
    model = HybridIDS()
    model.fit(X, y)

    # Predict
    print("Making predictions...")
    preds, shap_vals = model.predict(X)
    print("Predictions:", preds[:10])  # Show first 10 for demo
    print("✅ Done! Model is ready.")