# IMDb Sentiment Classification — Interim Implementation

This project compares an MLP baseline and a 1D CNN on the same IMDb review split. LSTM and a non-pretrained Transformer are planned for the final comparison; they are not presented as completed interim models.

## Reproducible protocol

- Remove exact duplicate rows before splitting.
- Decode HTML entities, remove HTML tags, lowercase, and normalize whitespace.
- Encode `negative = 0` and `positive = 1`.
- Make a stratified 80%/10%/10% train/validation/test split with seed 42.
- Fit the tokenizer **only on training reviews**.
- Use at most 20,000 vocabulary entries and pad/truncate reviews to 300 tokens.
- Train both models with binary cross-entropy, Adam, batch size 128, at most 8 epochs, and early stopping on validation loss.

The vocabulary and sequence length keep this preliminary experiment practical on a student laptop. No stopword list or stemming is used, so useful sentiment-bearing words and negations are retained.

## Run

From PowerShell in the project directory:

```powershell
.\.venv\Scripts\python.exe main.py
```

To run one model or shorten a trial:

```powershell
.\.venv\Scripts\python.exe main.py --models mlp --epochs 2
```

The program creates `results/` and writes measured preprocessing metadata, model summaries, saved models, learning curves, confusion matrices, history JSON, and `preliminary_results.csv`. Results are written after each completed model so a later failure does not erase earlier evidence.

Do not copy example or invented metric values into the report. Use only the values generated in `results/preliminary_results.csv` by an actual completed run, and disclose library/AI assistance as required by the course policy.
