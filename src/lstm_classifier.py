from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class LSTMClassifier:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.model       = None

    def build(self, input_dim):
        input_layer = Input(shape=(self.window_size, input_dim))
        x           = LSTM(128, return_sequences=True)(input_layer)
        x           = Dropout(0.3)(x)
        x           = LSTM(64)(x)        # reduced from 128→64 to shrink memory footprint
        x           = Dropout(0.3)(x)
        output      = Dense(1, activation='sigmoid')(x)
        self.model  = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return self.model

    def fit(self, X_sequences, y_sequences, epochs=None, batch_size=None,
            verbose=0, class_weight=None):
        """
        Adaptive training parameters:
          • small  (< 1k sequences)   → 100 epochs, batch 32
          • medium (< 50k sequences)  → 50  epochs, batch 64
          • large  (≥ 50k sequences)  → 20  epochs, batch 256
        EarlyStopping (patience=5) prevents unnecessary training in all regimes.
        batch_size is always capped to n_sequences to avoid Keras errors on tiny data.
        """
        n = len(X_sequences)

        if epochs is None:
            epochs     = 100 if n < 1_000 else (50 if n < 50_000 else 20)
        if batch_size is None:
            batch_size = 32  if n < 1_000 else (64 if n < 50_000 else 256)

        batch_size = max(1, min(batch_size, n))   # never exceed dataset size, never < 1

        early_stop = EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True, verbose=0
        )
        self.model.fit(
            X_sequences, y_sequences,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            class_weight=class_weight,
            callbacks=[early_stop],
        )

    def predict(self, X_sequences, batch_size=None, verbose=0):
        """
        Adaptive batch_size for inference.
        The original code used batch_size=1 which is extremely slow on large data.
        """
        if batch_size is None:
            batch_size = min(256, len(X_sequences))
        batch_size = max(1, batch_size)
        return self.model.predict(X_sequences, batch_size=batch_size, verbose=verbose).flatten()
