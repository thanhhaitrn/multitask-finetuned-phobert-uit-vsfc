# UIT-VSFC PhoBERT Multi-task Classifier

This project studies Vietnamese student feedback classification on the UIT-VSFC dataset. It compares traditional TF-IDF baselines with single-task and multi-task PhoBERT models for two supervised tasks:

- Sentiment classification: negative, neutral, or positive feedback.
- Topic classification: lecturer, training program, facility, or others.

The main notebook is [`phobert_vietnamese_student_feedback_classification.ipynb`](phobert_vietnamese_student_feedback_classification.ipynb). It contains exploratory data analysis, baseline cross-validation, PhoBERT fine-tuning, test evaluation, confusion matrices, error analysis, and prediction demos.

## Project Structure

| Path | Purpose |
| --- | --- |
| `train/`, `dev/`, `test/` | UIT-VSFC split files with aligned sentences, sentiment labels, and topic labels |
| `phobert_vietnamese_student_feedback_classification.ipynb` | End-to-end analysis and modeling notebook |
| `results/baseline_5fold_cv_results.csv` | 5-fold cross-validation results for TF-IDF baselines |
| `results/final_results.csv` | Held-out test results for baselines and PhoBERT models |
| `requirements.txt` | Pinned Python package versions used by the notebook |
| `nguyen20183.pdf` | Dataset/reference paper |

Model checkpoints, confusion matrices, prediction dumps, and error-analysis CSV files are intentionally ignored by `.gitignore`.

## Environment

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The notebook uses `underthesea` for Vietnamese word segmentation and `vinai/phobert-base` from Hugging Face Transformers. PhoBERT was pretrained on word-segmented Vietnamese text, so the notebook creates a separate `phobert_sentence` column for all PhoBERT dataloaders and prediction helpers while keeping raw sentences for display and error analysis.

On Apple Silicon, the notebook defaults to CPU instead of MPS because PhoBERT can run out of shared-memory GPU allocation. You can set `USE_MPS = True` in the notebook if your machine has enough memory.

## Dataset Files

Each split directory contains three aligned text files. Line `n` in each file belongs to the same feedback example.

| File | Description |
| --- | --- |
| `sents.txt` | Raw Vietnamese feedback sentence |
| `sentiments.txt` | Sentiment label |
| `topics.txt` | Topic label |

Sentiment labels:

| Label | Meaning |
| --- | --- |
| `0` | Negative |
| `1` | Neutral |
| `2` | Positive |

Topic labels:

| Label | Meaning |
| --- | --- |
| `0` | Lecturer |
| `1` | Training program |
| `2` | Facility |
| `3` | Others |

Split sizes:

| Split | Rows |
| --- | ---: |
| Train | 11,426 |
| Dev | 1,583 |
| Test | 3,166 |
| Total | 16,175 |

## Modeling

The notebook evaluates:

- TF-IDF + Naive Bayes
- TF-IDF + Maximum Entropy, implemented with balanced logistic regression
- Single-task PhoBERT for sentiment
- Single-task PhoBERT for topic
- Multi-task PhoBERT with shared encoder and separate sentiment/topic heads

Class imbalance is handled with macro metrics and weighted cross-entropy for PhoBERT. Accuracy is reported, but macro precision, macro recall, and macro F1 are the main class-balanced metrics.

## Results

5-fold cross-validation on the training split:

| Task | Model | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: |
| Sentiment | TF-IDF + Naive Bayes | 0.88 | 0.60 | 0.87 |
| Sentiment | TF-IDF + Maximum Entropy | 0.89 | 0.72 | 0.89 |
| Topic | TF-IDF + Naive Bayes | 0.77 | 0.42 | 0.70 |
| Topic | TF-IDF + Maximum Entropy | 0.83 | 0.73 | 0.84 |

Held-out test results from `results/final_results.csv`:

| Task | Model | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: |
| Sentiment | TF-IDF + Naive Bayes | 0.87 | 0.59 | 0.84 |
| Sentiment | TF-IDF + Maximum Entropy | 0.87 | 0.71 | 0.87 |
| Sentiment | Single-task PhoBERT | 0.93 | 0.84 | 0.93 |
| Sentiment | Multitask PhoBERT | 0.93 | 0.83 | 0.93 |
| Topic | TF-IDF + Naive Bayes | 0.79 | 0.47 | 0.74 |
| Topic | TF-IDF + Maximum Entropy | 0.81 | 0.71 | 0.82 |
| Topic | Single-task PhoBERT | 0.85 | 0.77 | 0.86 |
| Topic | Multitask PhoBERT | 0.84 | 0.75 | 0.85 |

## Notes

The sentiment task is dominated by negative and positive feedback, while neutral examples are rare. The topic task is also imbalanced, with lecturer feedback as the majority class and facility/others as minority classes. Because of this, weighted metrics can look strong even when minority-class recall is weak; macro F1 should be checked before drawing conclusions.

Maximum Entropy is a strong traditional baseline, especially compared with Naive Bayes. PhoBERT improves both tasks on held-out testing. In the current run, single-task PhoBERT gives the best topic macro F1, while single-task and multitask PhoBERT are very close on sentiment.
