"""Train the LSTM model using the main project's verified preprocessing pipeline.

This contribution is intended to be placed at:
    <repository-root>/LSTM_CONTRIBUTION/

It can also locate a sibling folder named ``dl_project`` while the contribution
is still outside the repository. It does not include or duplicate the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from tensorflow import keras


CONTRIBUTION_DIR = Path(__file__).resolve().parent


def locate_project_root() -> Path:
    """Find the project root either after merge or while this folder is a sibling."""
    candidates = [
        CONTRIBUTION_DIR.parent,
        CONTRIBUTION_DIR.parent / "dl_project",
    ]
    for candidate in candidates:
        if (
            (candidate / "src" / "preprocessing.py").exists()
            and (candidate / "src" / "evaluation.py").exists()
            and (candidate / "IMDB Dataset.csv").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the main project. Place LSTM_CONTRIBUTION inside "
        "the repository root, or keep it beside the dl_project folder. The "
        "project root must contain src/preprocessing.py, src/evaluation.py, "
        "and IMDB Dataset.csv."
    )


PROJECT_DIR = locate_project_root()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation import evaluate_and_save, write_results  # noqa: E402
from src.preprocessing import prepare_data, set_random_seeds  # noqa: E402
from lstm_model import build_lstm, compile_binary_classifier  # noqa: E402

DATASET_PATH = PROJECT_DIR / "IMDB Dataset.csv"
REFERENCE_RESULTS_DIR = PROJECT_DIR / "results"
REFERENCE_METADATA = REFERENCE_RESULTS_DIR / "preprocessing_summary.json"
REFERENCE_TOKENIZER = REFERENCE_RESULTS_DIR / "tokenizer.json"
OUTPUT_DIR = CONTRIBUTION_DIR / "lstm_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_tokenizer_json(path: Path) -> dict:
    """Compare tokenizer state while ignoring JSON key ordering."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    for key in ("word_counts", "word_docs", "index_docs", "index_word", "word_index"):
        config[key] = json.loads(config[key])
    return payload


def prepare_and_verify_same_data():
    """Recreate the data and verify it matches the project's saved preprocessing."""
    if not REFERENCE_METADATA.exists() or not REFERENCE_TOKENIZER.exists():
        raise FileNotFoundError(
            "The main project's results/preprocessing_summary.json and "
            "results/tokenizer.json are required. Keep these original files "
            "in the repository; this script will not silently create replacements."
        )

    reference_metadata = json.loads(
        REFERENCE_METADATA.read_text(encoding="utf-8")
    )
    vocab_size = int(reference_metadata["requested_vocabulary_size"])
    max_length = int(reference_metadata["maximum_sequence_length"])
    seed = int(reference_metadata["random_seed"])

    with tempfile.TemporaryDirectory(prefix="imdb_lstm_verification_") as temp:
        temp_dir = Path(temp)
        data = prepare_data(
            DATASET_PATH,
            temp_dir,
            vocab_size=vocab_size,
            max_length=max_length,
            seed=seed,
        )

        if data.metadata != reference_metadata:
            raise RuntimeError(
                "Regenerated preprocessing metadata differs from the saved "
                "reference. Training stopped to protect fair model comparison."
            )

        generated_tokenizer = normalized_tokenizer_json(temp_dir / "tokenizer.json")
        saved_tokenizer = normalized_tokenizer_json(REFERENCE_TOKENIZER)
        if generated_tokenizer != saved_tokenizer:
            raise RuntimeError(
                "The regenerated tokenizer differs from the saved reference. "
                "Training stopped to protect fair model comparison."
            )

    return data


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive integers")
    if args.seed != 42:
        raise ValueError("Use --seed 42 to retain the verified preprocessing split")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_random_seeds(args.seed)

    print("Verifying the original preprocessing and tokenizer...")
    data = prepare_and_verify_same_data()
    print("Verified split sizes:", data.metadata["split_sizes"])

    model = build_lstm(
        int(data.metadata["effective_vocabulary_size"]),
        int(data.metadata["maximum_sequence_length"]),
    )
    compile_binary_classifier(model)

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (OUTPUT_DIR / "lstm_model_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
                verbose=1,
            )
        ],
        verbose=2,
    )

    result = evaluate_and_save(
        model, history, data.x_test, data.y_test, "LSTM", OUTPUT_DIR
    )
    model.save(OUTPUT_DIR / "lstm_model.keras")
    (OUTPUT_DIR / "lstm_history.json").write_text(
        json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_results([result], OUTPUT_DIR / "lstm_results.csv")

    manifest = {
        "reference_preprocessing_summary_sha256": sha256(REFERENCE_METADATA),
        "reference_tokenizer_sha256": sha256(REFERENCE_TOKENIZER),
        "preprocessing_verification": "metadata and tokenizer matched",
        "split_sizes": data.metadata["split_sizes"],
        "vocabulary_size": data.metadata["effective_vocabulary_size"],
        "maximum_sequence_length": data.metadata["maximum_sequence_length"],
        "random_seed": args.seed,
        "batch_size": args.batch_size,
        "maximum_epochs": args.epochs,
        "epochs_trained": result["epochs_trained"],
        "metrics": result,
    }
    (OUTPUT_DIR / "lstm_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("LSTM training complete. Results saved to:", OUTPUT_DIR)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
