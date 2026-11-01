# Comparative Analysis of Deep Learning Architectures for Sentiment Classification of IMDb Movie Reviews

This project compares four deep learning architectures for binary sentiment classification of IMDb movie reviews: Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and Transformer. The objective is to evaluate their performance using a consistent dataset split and common classification metrics.

## Objectives

- Implement and compare four deep learning architectures for sentiment classification.
- Apply a consistent preprocessing pipeline and evaluation procedure.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Analyse the strengths and limitations of each architecture based on the results.

## Dataset and preprocessing

The project uses the IMDb movie-review dataset containing 50,000 reviews labelled as positive or negative.

- Remove 418 exact duplicate rows, leaving 49,582 reviews.
- Encode negative sentiment as `0` and positive sentiment as `1`.
- Create a stratified 80%/10%/10% training, validation, and test split using seed 42.
- Fit the tokenizer only on training reviews.
- Use a maximum vocabulary size of 20,000 tokens.
- Pad or truncate review sequences to 300 tokens.
- Use the same held-out test set to compare all four models.

### Dataset split

| Subset | Number of reviews |
| --- | ---: |
| Training | 39,665 |
| Validation | 4,958 |
| Testing | 4,959 |
| Total | 49,582 |

The preprocessing pipeline prepares textual reviews for numerical input to the neural networks. The tokenizer is fitted only on training reviews to avoid using validation or test text to construct the vocabulary.

## Model architectures

- **MLP (Multilayer Perceptron):** A feed-forward neural network that learns patterns from the numerical representation of reviews.
- **CNN (Convolutional Neural Network):** Uses convolutional operations to identify local patterns in the text representation.
- **LSTM (Long Short-Term Memory):** A recurrent neural network designed to process sequences and learn relationships across tokens.
- **Transformer:** Uses self-attention to model relationships between tokens in a review.

## Evaluation metrics

All four models are compared using the following metrics:

- **Accuracy:** Proportion of predictions classified correctly.
- **Precision:** Proportion of predicted positive reviews that are actually positive.
- **Recall:** Proportion of actual positive reviews correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

## Experimental results

The following table presents the reported results for the four architectures.

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
| --- | ---: | ---: | ---: | ---: |
| MLP | 88.89 | 90.58 | 86.90 | 88.70 |
| CNN | 88.79 | 92.71 | 84.29 | 88.30 |
| LSTM | 88.16 | 88.35 | 88.03 | 88.19 |
| Transformer | 88.61 | 85.52 | 93.05 | 89.13 |

### Results analysis

- MLP achieved the highest accuracy of 88.89%.
- CNN achieved the highest precision of 92.71%.
- LSTM produced relatively balanced precision and recall.
- Transformer achieved the highest recall of 93.05% and the highest F1-score of 89.13%.

The results demonstrate that the architecture with the highest accuracy does not necessarily achieve the highest F1-score. Multiple metrics are therefore useful when comparing classification models.

## Repository structure

```text
dl_project/
├── src/
│   ├── __init__.py
│   ├── cnn_model.py
│   ├── evaluation.py
│   ├── mlp_model.py
│   ├── model_utils.py
│   ├── preprocessing.py
│   ├── transformer_model.py
│   └── train_transformer.py
├── results/
├── README.md
└── requirements.txt
```

The `src/` directory contains model implementations, preprocessing utilities, evaluation functionality, and Transformer training code. The `results/` directory stores project result artefacts.

## Technologies used

- Python
- Deep Learning
- Natural Language Processing (NLP)
- MLP, CNN, LSTM, and Transformer architectures
- Text preprocessing and tokenisation
- Classification evaluation metrics

The required Python packages are listed in `requirements.txt`.

## Setup

### Prerequisites

- Python
- Git
- Dependencies listed in `requirements.txt`

### 1. Clone the repository

```powershell
git clone https://github.com/varshini120614/dl_project.git
cd dl_project
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the project

The repository contains model-specific implementation and training files. Before running a model, verify the corresponding training script, dataset location, required dependencies, and configured file paths.

The intended experimental workflow is to preprocess the reviews, train the selected model, evaluate its predictions on the held-out test set, and save the resulting evaluation artefacts.

The exact execution commands should be confirmed against the current source code before running a training experiment.

## Conclusion

This project compares MLP, CNN, LSTM, and Transformer architectures for binary sentiment classification of IMDb movie reviews. MLP achieved the highest accuracy, CNN achieved the highest precision, and Transformer achieved the highest recall and F1-score among the reported results. These findings highlight the importance of evaluating multiple metrics when comparing deep learning models.

## Contributors

- **Varshini Vidya Shankar:** Transformer implementation, preprocessing pipeline, dataset splitting, tokenisation and evaluation workflow, and GitHub integration.
- **Anvita Bonny Warjri:** CNN implementation.
- **Adhavann Dharmaprakash:** LSTM implementation.
- **Mayaank Kothari:** MLP implementation.
