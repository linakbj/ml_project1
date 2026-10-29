# ML Project 1 — Predicting coronary heart disease (MICHD) from BRFSS data

EPFL CS-433 Machine Learning, Fall 2026. Binary classification of MICHD
(heart attack / angina) from the Behavioral Risk Factor Surveillance System
survey (~330'000 respondents, 321 features). Only Python's standard library
and NumPy are used (matplotlib only for plots in the exploration notebook).

## Repository structure

| File | Content |
|------|---------|
| `implementations.py` | The 6 required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`. All return `(w, loss)`; regularized methods return the loss **without** the penalty term. |
| `preprocessing.py` | Feature cleaning: `fit_preprocessing` (learned on training data only) and `apply_preprocessing`. |
| `train.py` | Model selection on a stratified validation split; writes `validation_results.csv`. |
| `run.py` | Reads the best configuration from `validation_results.csv`, retrains it on the full training set and writes `submission.csv` (the file submitted to AIcrowd). |
| `validation_results.csv` | Every configuration tested and its validation metrics, sorted by F1. |
| `helpers.py` | Course-provided helpers for loading the data and writing the submission. |
| `analyze_missing.py` | Statistics on missing values (used for the exploratory analysis). |
| `exploration.ipynb` | Exploratory data analysis. |
| `grading_tests/` | Official public tests for `implementations.py`. |

## Reproducing the submission

1. Download the data from the
   [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
   and put the three files (names unchanged) into `data/dataset/`:

   ```
   data/dataset/x_train.csv
   data/dataset/y_train.csv
   data/dataset/x_test.csv
   ```

2. (Optional, re-runs the model selection, a few minutes)

   ```bash
   python train.py
   ```

   This overwrites `validation_results.csv`. The committed version is the
   one used for our final submission.

3. Generate the predictions:

   ```bash
   python run.py
   ```

   This writes `submission.csv` with labels in {-1, 1}.

Requirements: Python ≥ 3.9 and NumPy. All randomness is seeded
(`seed=42` for the validation split), so the output is deterministic.

To check the required implementations:

```bash
pip install pytest gitpython
pytest grading_tests --github_link <path-or-url-of-this-repo>
```

## Pipeline

### 1. Preprocessing (`preprocessing.py`)
All parameters below are computed on the training data only and then applied
unchanged to validation and test data (no data leakage).

1. **Column filtering**: drop features with more than 90% missing values, and
   empty or constant columns.
2. **Imputation**: replace remaining missing values with the column median.
3. **Standardization**: subtract the mean and divide by the standard
   deviation of each column.
4. **Bias**: add a column of ones for the intercept.

### 2. Labels
Labels are converted from {-1, 1} to {0, 1} for training. Predictions are
converted back to {-1, 1} for the submission.

### 3. Model selection (`train.py`)
- **Validation**: a single stratified 80/20 train/validation split
  (`seed=42`), which keeps the ~9% positive rate in both parts.
- **Models**: ridge regression (normal equations) and regularized logistic
  regression (gradient descent, `max_iters=300`, `gamma=0.01`, `w0 = 0`).
- **Hyper-parameters**: λ ∈ {0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2}.
- **Decision threshold**: since classes are imbalanced, predicting with a
  0.5 threshold almost never predicts the positive class. We tune the
  threshold on the validation set over {0.10, 0.11, …, 0.60}.
- **Metric**: F1-score of the positive class (the AIcrowd metric). An
  "always negative" baseline is reported for comparison (F1 = 0).

### 4. Final model (`run.py`)
The configuration with the best validation F1 is retrained on the full
training set, with preprocessing refitted on the full training set, and
applied to the test set.

## Results

Validation set (20% stratified split, best threshold for each model).
218 of the 321 features are kept after preprocessing.

| Model | Best λ | Threshold | Accuracy | Precision | Recall | F1 |
|-------|--------|-----------|----------|-----------|--------|----|
| Always negative (baseline) | – | – | 0.912 | – | 0.000 | 0.000 |
| Regularized logistic regression | 5e-4 | 0.39 | 0.841 | 0.300 | 0.597 | 0.399 |
| **Ridge regression (final)** | **1e-6** | **0.21** | **0.859** | **0.327** | **0.563** | **0.413** |

Observations:
- Tuning the decision threshold matters much more than λ: ridge F1 only
  varies between 0.410 and 0.413 over all tested λ values.
- With 300 iterations and `gamma=0.01`, logistic regression is not fully
  converged, which likely explains why it does slightly worse than ridge.

All configurations and metrics are listed in `validation_results.csv`.

## Authors
Team for EPFL CS-433, Fall 2026.
