"""Evaluation and plot helpers. All values come from model predictions/history."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras


RESULT_FIELDS = [
    "model", "parameters", "epochs_trained", "test_loss", "accuracy",
    "precision", "recall", "f1_score",
]


def save_history_plot(history: keras.callbacks.History, model_name: str, path: Path) -> None:
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history.history["accuracy"], label="Training")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set(title=f"{model_name} accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, history.history["loss"], label="Training")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set(title=f"{model_name} loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_and_save(
    model: keras.Model, history: keras.callbacks.History, x_test: np.ndarray,
    y_test: np.ndarray, model_name: str, results_dir: Path,
) -> dict:
    """Measure test performance and save its history and confusion matrix."""
    probabilities = model.predict(x_test, batch_size=256, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype("int32")
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    test_loss = float(model.evaluate(x_test, y_test, batch_size=256, verbose=0)[0])

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], labels=["Negative", "Positive"])
    axis.set_yticks([0, 1], labels=["Negative", "Positive"])
    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column, row, str(matrix[row, column]), ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    axis.set(title=f"{model_name} confusion matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(results_dir / f"{model_name.lower()}_confusion_matrix.png", dpi=160)
    plt.close(figure)
    save_history_plot(history, model_name, results_dir / f"{model_name.lower()}_history.png")
    return {
        "model": model_name, "parameters": int(model.count_params()),
        "epochs_trained": int(len(history.history["loss"])), "test_loss": test_loss,
        "accuracy": float(accuracy), "precision": float(precision),
        "recall": float(recall), "f1_score": float(f1),
    }


def write_results(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
