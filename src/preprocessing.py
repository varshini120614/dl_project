"""Dataset loading and leakage-safe preprocessing."""

from __future__ import annotations

import html
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    tokenizer: Tokenizer
    metadata: dict


def set_random_seeds(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def clean_review(text: str) -> str:
    """Remove HTML markup, decode entities, lowercase, and normalize whitespace."""
    text = html.unescape(str(text))
    text = HTML_TAG_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text.lower()).strip()


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().items()}


def prepare_data(
    csv_path: Path,
    results_dir: Path,
    vocab_size: int = 20_000,
    max_length: int = 300,
    seed: int = 42,
) -> PreparedData:
    """Load, deduplicate, clean, split, tokenize, and pad the IMDb reviews.

    The tokenizer is deliberately fitted on training text only to prevent leakage.
    """
    set_random_seeds(seed)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain {sorted(required_columns)}; found {df.columns.tolist()}"
        )

    original_rows = len(df)
    missing_values = {column: int(value) for column, value in df.isna().sum().items()}
    duplicate_rows = int(df.duplicated().sum())
    original_class_distribution = _counts(df["sentiment"])

    df = df.drop_duplicates().copy()
    df["review"] = df["review"].map(clean_review)

    label_map = {"negative": 0, "positive": 1}
    unexpected_labels = sorted(set(df["sentiment"]) - set(label_map))
    if unexpected_labels:
        raise ValueError(f"Unexpected sentiment labels: {unexpected_labels}")
    df["label"] = df["sentiment"].map(label_map).astype("int32")

    train_text, temp_text, y_train, y_temp = train_test_split(
        df["review"], df["label"], test_size=0.20, random_state=seed,
        stratify=df["label"],
    )
    val_text, test_text, y_val, y_test = train_test_split(
        temp_text, y_temp, test_size=0.50, random_state=seed, stratify=y_temp,
    )

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text.tolist())  # Training text only.

    def vectorize(texts: pd.Series) -> np.ndarray:
        sequences = tokenizer.texts_to_sequences(texts.tolist())
        return pad_sequences(
            sequences, maxlen=max_length, padding="post", truncating="post"
        ).astype("int32")

    x_train = vectorize(train_text)
    x_val = vectorize(val_text)
    x_test = vectorize(test_text)
    y_train_array = y_train.to_numpy(dtype="int32")
    y_val_array = y_val.to_numpy(dtype="int32")
    y_test_array = y_test.to_numpy(dtype="int32")

    effective_vocabulary_size = min(vocab_size, len(tokenizer.word_index) + 1)
    metadata = {
        "dataset_file": csv_path.name,
        "original_rows": int(original_rows),
        "original_columns": [str(column) for column in df.columns if column != "label"],
        "missing_values_before_cleaning": missing_values,
        "duplicate_rows_removed": duplicate_rows,
        "rows_after_duplicate_removal": int(len(df)),
        "class_distribution_before_duplicate_removal": original_class_distribution,
        "class_distribution_after_duplicate_removal": _counts(df["sentiment"]),
        "label_mapping": label_map,
        "random_seed": seed,
        "split": {"train": 0.80, "validation": 0.10, "test": 0.10},
        "split_sizes": {
            "train": int(len(x_train)), "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "split_positive_counts": {
            "train": int(y_train_array.sum()), "validation": int(y_val_array.sum()),
            "test": int(y_test_array.sum()),
        },
        "requested_vocabulary_size": vocab_size,
        "effective_vocabulary_size": effective_vocabulary_size,
        "full_training_word_index_size": int(len(tokenizer.word_index) + 1),
        "maximum_sequence_length": max_length,
        "tokenizer_fit_scope": "training split only",
        "cleaning_operations": [
            "decode HTML entities", "remove HTML tags", "convert to lowercase",
            "collapse repeated whitespace",
        ],
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "preprocessing_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (results_dir / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")

    return PreparedData(
        x_train=x_train, x_val=x_val, x_test=x_test,
        y_train=y_train_array, y_val=y_val_array, y_test=y_test_array,
        tokenizer=tokenizer, metadata=metadata,
    )


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a tokenizer previously written by ``prepare_data``."""
    return tokenizer_from_json(path.read_text(encoding="utf-8"))
