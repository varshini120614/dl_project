"""Train/evaluate the Transformer using the main project's shared pipeline.

Expected layout:
  parent/
    dl_project/                 # main project
    TRANSFORMER_CONTRIBUTION/   # this contribution folder
Or place this folder inside dl_project.
"""
from pathlib import Path
import json
import random
import sys
import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "dl_project", HERE]
PROJECT_ROOT = next((p for p in CANDIDATES if (p / "src" / "preprocessing.py").exists()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Cannot locate the main project. Keep TRANSFORMER_CONTRIBUTION beside "
        "dl_project, or place it inside dl_project."
    )

sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import load_and_preprocess_data
from src.evaluation import evaluate_model
from transformer_model import build_transformer, compile_transformer

SEED = 42
MAXLEN = 300
VOCAB_SIZE = 20000
BATCH_SIZE = 64
EPOCHS = 8

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    results_dir = HERE / "transformer_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_data(PROJECT_ROOT)
    # Verify preprocessing reference artifacts exist before training.
    summary_path = PROJECT_ROOT / "results" / "preprocessing_summary.json"
    tokenizer_path = PROJECT_ROOT / "results" / "tokenizer.json"
    if not summary_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("Expected preprocessing_summary.json and tokenizer.json in main project results/.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("random_seed") != SEED:
        raise ValueError("Saved preprocessing seed does not match expected seed 42.")
    if summary.get("vocab_size") != VOCAB_SIZE or summary.get("max_sequence_length") != MAXLEN:
        raise ValueError("Saved preprocessing settings do not match vocab_size=20000 and max_sequence_length=300.")

    model = compile_transformer(build_transformer(VOCAB_SIZE, MAXLEN))
    model.summary()
    summary_file = results_dir / "transformer_model_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]
    history = model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1
    )
    model.save(results_dir / "transformer_model.keras")
    (results_dir / "transformer_history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2),
        encoding="utf-8"
    )

    # Evaluation helper API can vary across project versions; use Keras prediction
    # and sklearn metrics for a self-contained, transparent test evaluation.
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    probabilities = model.predict(data.X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(data.y_test, predictions)),
        "precision": float(precision_score(data.y_test, predictions, zero_division=0)),
        "recall": float(recall_score(data.y_test, predictions, zero_division=0)),
        "f1": float(f1_score(data.y_test, predictions, zero_division=0)),
    }
    (results_dir / "transformer_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(data.y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
