"""MLP text-classification architecture."""

from __future__ import annotations

from tensorflow import keras


def build_mlp(vocab_size: int, max_length: int) -> keras.Model:
    """Embedding plus global-average pooling feed-forward baseline."""
    inputs = keras.Input(shape=(max_length,), dtype="int32", name="token_ids")
    x = keras.layers.Embedding(
        vocab_size, 64, mask_zero=True, name="embedding"
    )(inputs)
    x = keras.layers.GlobalAveragePooling1D(name="average_pooling")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="sentiment")(x)
    return keras.Model(inputs, outputs, name="mlp_text_classifier")