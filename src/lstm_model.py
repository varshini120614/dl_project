"""LSTM model for IMDb movie-review sentiment classification."""

from __future__ import annotations

from tensorflow import keras


def build_lstm(vocab_size: int, max_length: int) -> keras.Model:
    """Single-layer LSTM classifier for sequential dependencies."""
    inputs = keras.Input(shape=(max_length,), dtype="int32", name="token_ids")

    x = keras.layers.Embedding(
        vocab_size, 64, mask_zero=True, name="embedding"
    )(inputs)

    x = keras.layers.LSTM(64, dropout=0.2, name="lstm")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)

    outputs = keras.layers.Dense(
        1, activation="sigmoid", name="sentiment"
    )(x)

    return keras.Model(inputs, outputs, name="lstm_text_classifier")


def compile_binary_classifier(model: keras.Model) -> None:
    """Compile the binary sentiment classifier."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )