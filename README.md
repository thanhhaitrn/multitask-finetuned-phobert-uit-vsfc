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
| `baseline_5fold_cv_results.csv` | 5-fold cross-validation results for TF-IDF baselines |
| `final_results.csv` | Held-out test results for baselines and PhoBERT models |
| `requirements.txt` | Pinned Python package versions used by the notebook |
| `nguyen20183.pdf` | Dataset/reference paper |

Model checkpoints and large saved artifacts are intentionally ignored by `.gitignore`.

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
| Sentiment | TF-IDF + Naive Bayes | 0.8835 | 0.6011 | 0.8657 |
| Sentiment | TF-IDF + Maximum Entropy | 0.8852 | 0.7238 | 0.8911 |
| Topic | TF-IDF + Naive Bayes | 0.7663 | 0.4165 | 0.7010 |
| Topic | TF-IDF + Maximum Entropy | 0.8248 | 0.7311 | 0.8359 |

Held-out test results from `final_results.csv`:

| Task | Model | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: |
| Sentiment | TF-IDF + Naive Bayes | 0.8680 | 0.5942 | 0.8448 |
| Sentiment | TF-IDF + Maximum Entropy | 0.8680 | 0.7147 | 0.8731 |
| Sentiment | Single-task PhoBERT | 0.9226 | 0.8218 | 0.9245 |
| Sentiment | Multi-task PhoBERT | 0.9078 | 0.7970 | 0.9155 |
| Topic | TF-IDF + Naive Bayes | 0.7919 | 0.4706 | 0.7403 |
| Topic | TF-IDF + Maximum Entropy | 0.8089 | 0.7121 | 0.8231 |
| Topic | Single-task PhoBERT | 0.8468 | 0.7558 | 0.8565 |
| Topic | Multi-task PhoBERT | 0.8421 | 0.7617 | 0.8563 |

## Notes

The sentiment task is dominated by negative and positive feedback, while neutral examples are rare. The topic task is also imbalanced, with lecturer feedback as the majority class and facility/others as minority classes. Because of this, weighted metrics can look strong even when minority-class recall is weak; macro F1 should be checked before drawing conclusions.

Maximum Entropy is a strong traditional baseline, especially compared with Naive Bayes. PhoBERT improves both tasks on held-out testing, and the multi-task model slightly improves topic macro F1 while single-task PhoBERT remains stronger for sentiment in the current run.
