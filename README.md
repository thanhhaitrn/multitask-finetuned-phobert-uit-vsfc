# UIT-VSFC PhoBERT Multi-task Classifier

This project studies Vietnamese student feedback classification on the UIT-VSFC dataset. The target tasks are:

- Sentiment classification: predict whether a feedback sentence is negative, neutral, or positive.
- Topic classification: predict whether the feedback is about the lecturer, training program, facility, or another topic.

The current notebook, [analysis.ipynb](/Users/thanhhai/UIT-VSFC/analysis.ipynb), focuses on exploratory data analysis and traditional machine-learning baselines. The project title mentions PhoBERT multi-task classification, so the baseline results should be treated as a reference point before adding or comparing transformer-based models.

## Dataset Files

Each split directory (`train/`, `dev/`, and `test/`) contains three aligned text files. Line `n` in each file belongs to the same feedback example.

### `sents.txt`

Each line contains one original Vietnamese student feedback sentence.

### `sentiments.txt`

Each line contains one sentiment label:

| Label | Meaning |
| --- | --- |
| `0` | Negative |
| `1` | Neutral |
| `2` | Positive |

### `topics.txt`

Each line contains one topic label:

| Label | Meaning |
| --- | --- |
| `0` | Lecturer |
| `1` | Training program |
| `2` | Facility |
| `3` | Others |

## Split Sizes

| Split | Rows |
| --- | ---: |
| Train | 11,426 |
| Dev | 1,583 |
| Test | 3,166 |
| Total | 16,175 |

## EDA Notes

The sentiment task is mostly split between negative and positive labels, but neutral feedback is strongly underrepresented. This means accuracy can look high even when neutral examples are poorly detected, so macro precision, macro recall, and macro F1 should be checked.

The topic task is more imbalanced. Lecturer feedback is the majority class, while facility and others are minority classes. This can bias models toward predicting lecturer unless class weighting, sampling, or stronger contextual models are used.

Sentence lengths are short overall. Across all splits, most feedback comments are brief, with a right-skewed distribution caused by a small number of longer comments.

## Baseline Results

The baseline in `analysis.ipynb` uses:

- Vietnamese word segmentation with `underthesea`
- TF-IDF features with unigram and bigram terms
- Naive Bayes
- Maximum Entropy, implemented as logistic regression with balanced class weights
- 5-fold stratified cross-validation on the training split

The saved output is [baseline_5fold_cv_results.csv](/Users/thanhhai/UIT-VSFC/baseline_5fold_cv_results.csv).

### Baseline Analysis

Maximum Entropy performs better than Naive Bayes on both tasks, especially in macro metrics. This suggests that the balanced logistic regression baseline handles minority classes better than Naive Bayes.

For sentiment classification, weighted F1 is much higher than macro F1 because the neutral class is rare. For topic classification, the gap between weighted F1 and macro F1 is also important: strong weighted scores can still hide weak performance on facility and others.

## Acronyms and Normalized Tokens

Some informal strings in the dataset are replaced by text-only tokens:

| Original | Replacement |
| --- | --- |
| `:)` | `colonsmile` |
| `:(` | `colonsad` |
| `@@` | `colonsurprise` |
| `<3` | `colonlove` |
| `:d` | `colonsmilesmile` |
| `:3` | `coloncontemn` |
| `:v` | `colonbigsmile` |
| `:_` | `coloncc` |
| `:p` | `colonsmallsmile` |
| `>>` | `coloncolon` |
| `:">` | `colonlovelove` |
| `^^` | `colonhihi` |
| `:` | `doubledot` |
| `:'(` | `colonsadcolon` |
| `:’(` | `colonsadcolon` |
| `:@` | `colondoublesurprise` |
| `v.v` | `vdotv` |
| `...` | `dotdotdot` |
| `/` | `fraction` |
| `c#` | `cshrap` |
