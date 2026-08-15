# 📊 Financial Sentiment Analysis Using NLP and FinBERT

A financial sentiment analysis research project using the Financial PhraseBank dataset. The project investigates data cleaning, template-based leakage, classical machine-learning baselines, BERT, and FinBERT under both standard and leakage-controlled grouped splits.

## Project Overview

The objective is to classify financial sentences into three sentiment categories:

- 🔴 Negative
- ⚪ Neutral
- 🟢 Positive

The methodology progresses from dataset preprocessing and leakage analysis to classical ML and transformer-based models, with a final evaluation performed on completely held-out test sets.

## Dataset

The project uses the Financial PhraseBank dataset, specifically the 50Agree subset.

**After cleaning:**

| Metric | Count |
|---|---|
| Raw sentences | 4,846 |
| Cleaned sentences | 4,836 |
| Negative | 604 |
| Neutral | 2,871 |
| Positive | 1,361 |

The original dataset's agreement-level information is also retained.

## Methodology

### Stage 1 — Data Cleaning

The dataset is cleaned by:

- Removing conflicting duplicate sentences
- Collapsing consistent duplicates
- Preserving agreement-level information
- Validating the final dataset

**Final dataset size:** 4,836 sentences

### Stage 2 — Template Normalization & Leakage Analysis

Sentences are normalized into templates by replacing values such as:

- Currency values
- Percentages
- Dates
- Numerical values

**Results:**

| Metric | Count |
|---|---|
| Unique normalized templates | 4,802 |
| Multi-member template groups | 24 |
| Sentences in multi-member groups | 58 |
| Same-label groups | 22 |
| Different-label groups | 2 |

A `group_id` is assigned to each normalized template to enable leakage-controlled evaluation.

### Stage 3 — Fixed Dataset Splits

Two evaluation settings are created using `RANDOM_SEED = 42`.

**Standard split**

| Split | Samples |
|---|---|
| Train | 3,385 |
| Validation | 725 |
| Test | 726 |

**Grouped split**

| Split | Samples |
|---|---|
| Train | 3,381 |
| Validation | 728 |
| Test | 727 |

The grouped split ensures that sentences belonging to the same normalized template are kept within the same partition.

### Stage 4–6 — Classical Machine Learning

Classical ML baselines are evaluated before transformer models:

- Weighted Logistic Regression
- Weighted Linear SVM

**Grouped validation Macro-F1:**

| Model | Macro-F1 |
|---|---|
| Weighted Logistic Regression | 0.6649 |
| Weighted Linear SVM | 0.6428 |

### Stage 7 — BERT Baseline

`bert-base-uncased` is fine-tuned for three-class financial sentiment classification.

**Validation Macro-F1:**

| Model | Standard | Grouped |
|---|---|---|
| BERT | 0.8439 | 0.8465 |

### Stage 8 — FinBERT

A financial-domain pretrained BERT model is fine-tuned for the same three-class classification task.

**Validation results:**

| Model | Macro-F1 | Accuracy |
|---|---|---|
| FinBERT — Standard | 0.8234 | 0.8510 |
| FinBERT — Grouped | 0.8462 | 0.8599 |

## 🏁 Final Test Evaluation

The test sets were used only for final evaluation. No training or validation-based tuning was performed using the test data.

| Model | Test Macro-F1 | Test Accuracy |
|---|---|---|
| FinBERT — Standard Split | **0.8489** | 0.8609 |
| FinBERT — Grouped Split | 0.8363 | 0.8583 |

The standard split achieved the highest final Test Macro-F1 of **0.8489**, while the grouped split provides a stricter evaluation against template-level leakage.

## Repository Structure
Financial-Sentiment-Analysis/
│
├── Financial_Sentiment_Analysis.ipynb
├── FinancialPhraseBank-v1.0.zip
├── README.md
└── License.txt


Generated datasets, model checkpoints, and experiment outputs are created by the notebook during execution.

## Reproducibility

The notebook is designed to run sequentially from a fresh environment.

Stage 3 generates the required split artifacts automatically, including:

- `data/phrasebank_with_groups.csv`
- `data/split_assignments.csv`
- `data/standard_split.csv`
- `data/grouped_split.csv`

The notebook also verifies the expected dataset and split dimensions before model evaluation.

## Technologies

Python · Pandas · NumPy · Scikit-learn · PyTorch · Hugging Face Transformers · BERT · FinBERT · Google Colab

## ⚠️ Important Note

The Financial PhraseBank dataset is a third-party dataset. Its original licensing terms are included with the dataset and should be followed when using or redistributing the data.

## Author

**Amrutha Javvadi**

*Financial Sentiment Analysis — NLP / Machine Learning Research Project*
