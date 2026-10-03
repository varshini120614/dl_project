"""Train and compare preliminary MLP and 1D CNN IMDb classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorflow import keras

from src.evaluation import evaluate_and_save, write_results
from src.cnn_model import build_cnn
from src.mlp_model import build_mlp
from src.model_utils import compile_binary_classifier
from src.preprocessing import prepare_data, set_random_seeds


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "IMDB Dataset.csv"
RESULTS_DIR = PROJECT_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=("mlp", "cnn"), default=("mlp", "cnn"),
        help="Models to train (default: both).",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_model(name: str, data, args: argparse.Namespace) -> dict:
    set_random_seeds(args.seed)

    builders = {
        "mlp": build_mlp,
        "cnn": build_cnn,
    }

    model = builders[name](
        data.metadata["effective_vocabulary_size"],
        args.max_length,
    )

    compile_binary_classifier(model)

    model_results_dir = RESULTS_DIR / f"{name}_results"
    model_results_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)

    (model_results_dir / f"{name}_model_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
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
        model,
        history,
        data.x_test,
        data.y_test,
        name.upper(),
        model_results_dir,
    )

    model.save(model_results_dir / f"{name}_model.keras")

    (model_results_dir / f"{name}_history.json").write_text(
        json.dumps(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive integers")
    print("Preparing data (tokenizer is fitted on training reviews only)...")
    data = prepare_data(
        DATASET_PATH, RESULTS_DIR, vocab_size=args.vocab_size,
        max_length=args.max_length, seed=args.seed,
    )
    print(json.dumps(data.metadata, indent=2))
    results = []
    for model_name in args.models:
        print(f"\nTraining {model_name.upper()}...")
        results.append(train_model(model_name, data, args))
        # Preserve genuine results if a later model run fails.
        write_results(results, RESULTS_DIR / "preliminary_results.csv")
        print(json.dumps(results[-1], indent=2))
    print(f"\nActual measured results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
