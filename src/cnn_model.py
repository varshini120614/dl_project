"""CNN text-classification architecture."""

from __future__ import annotations

from tensorflow import keras


def build_cnn(vocab_size: int, max_length: int) -> keras.Model:
    """Embedding plus Conv1D model for local phrase-pattern detection."""
    inputs = keras.Input(shape=(max_length,), dtype="int32", name="token_ids")
    x = keras.layers.Embedding(vocab_size, 64, name="embedding")(inputs)
    x = keras.layers.Conv1D(
        128, kernel_size=5, activation="relu", name="conv1d"
    )(x)
    x = keras.layers.GlobalMaxPooling1D(name="max_pooling")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="sentiment")(x)
    return keras.Model(inputs, outputs, name="cnn_text_classifier")