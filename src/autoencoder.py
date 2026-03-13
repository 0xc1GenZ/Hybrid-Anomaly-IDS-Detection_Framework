from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping


class Autoencoder:
    def __init__(self):
        self.model = None

    def build(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoder     = Dense(64, activation='relu')(input_layer)
        encoder     = Dense(32, activation='relu')(encoder)
        decoder     = Dense(64, activation='relu')(encoder)
        decoder     = Dense(input_dim, activation='sigmoid')(decoder)
        self.model  = Model(inputs=input_layer, outputs=decoder)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        return self.model

    def fit(self, X, epochs=None, batch_size=None, verbose=0):
        """
        Adaptive training parameters based on dataset size:
          • small  (< 5k)   → more epochs (100), tiny batch (32)   — squeeze signal from sparse data
          • medium (< 100k) → standard (50 epochs, batch 64)
          • large  (≥ 100k) → fewer epochs (20), large batch (512)  — speed & memory
        Early stopping prevents waste in all cases.
        """
        n = len(X)
        if epochs is None:
            epochs     = 100 if n < 5_000 else (50 if n < 100_000 else 20)
        if batch_size is None:
            batch_size = 32  if n < 5_000 else (64 if n < 100_000 else 512)

        # Ensure batch_size never exceeds dataset size
        batch_size = min(batch_size, n)

        early_stop = EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True, verbose=0
        )
        self.model.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop],
        )

    def predict(self, X, batch_size=None, verbose=0):
        """Adaptive batch_size for predict — large batches are safe for inference."""
        if batch_size is None:
            batch_size = min(512, len(X))
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)
