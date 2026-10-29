import argparse
import csv

import numpy as np

from helpers import load_csv_data
from implementations import ridge_regression, reg_logistic_regression
from preprocessing import fit_preprocessing, apply_preprocessing


def split_data(y, validation_ratio=0.2, seed=42):
    """
    Create a reproducible stratified train/validation split.

    Arguments:
        y: Labels, shape (N,)
        validation_ratio: Fraction of data used for validation
        seed: Random seed

    Returns:
        train_indices: Indices of training observations
        validation_indices: Indices of validation observations
    """

    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    # Split each class separately to preserve class proportions
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_validation = int(len(indices) * validation_ratio)

        validation_indices.extend(indices[:n_validation])
        train_indices.extend(indices[n_validation:])

    return (
        rng.permutation(train_indices),
        rng.permutation(validation_indices),
    )


def compute_metrics(y, predictions):
    """
    Evaluate binary classification predictions.

    Arguments:
        y: True labels in {0, 1}, shape (N,)
        predictions: Predicted labels in {0, 1}, shape (N,)

    Returns:
        metrics: Dictionary containing accuracy, precision, recall and F1
    """

    # Count positive classification outcomes
    tp = np.sum((y == 1) & (predictions == 1))
    fp = np.sum((y == 0) & (predictions == 1))
    fn = np.sum((y == 1) & (predictions == 0))

    # Compute metrics while avoiding division by zero
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    return {
        "accuracy": float(np.mean(y == predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    """
    Train and compare models on a fixed validation split.

    Command-line arguments:
        --data: Folder containing the dataset CSV files
        --max-iters: Number of logistic gradient descent iterations
        --gamma: Logistic gradient descent learning rate

    Output:
        validation_results.csv: Configurations and evaluation metrics
    """

    # Read command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=0.01)
    args = parser.parse_args()

    # Load data and discard test features during model selection
    print("Loading data...", flush=True)
    x, x_test, y, _, _ = load_csv_data(args.data)
    del x_test

    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Expected labels in {-1, 1}.")

    # Convert labels to {0, 1} for both models
    y = (y == 1).astype(int)

    train_indices, validation_indices = split_data(y)
    y_train = y[train_indices]
    y_validation = y[validation_indices]

    # Learn preprocessing on training data and apply it to both subsets
    print("Preprocessing data...", flush=True)
    parameters = fit_preprocessing(x[train_indices])
    tx_train = apply_preprocessing(x[train_indices], parameters)
    tx_validation = apply_preprocessing(x[validation_indices], parameters)
    del x

    # Evaluate a baseline that always predicts the negative class
    baseline = compute_metrics(
        y_validation, np.zeros_like(y_validation)
    )
    print(
        f"Always-negative baseline: "
        f"accuracy={baseline['accuracy']:.4f}, "
        f"F1={baseline['f1']:.4f}",
        flush=True,
    )

    results = []

    # Compare models and regularization parameters
    for model in ("ridge", "logistic"):
        for lambda_ in (0.0, 0.001, 0.01):
            print(
                f"Training {model}, lambda={lambda_}...",
                flush=True,
            )

            if model == "ridge":
                w, loss = ridge_regression(
                    y_train, tx_train, lambda_
                )
                scores = tx_validation @ w

            else:
                initial_w = np.zeros(tx_train.shape[1])
                w, loss = reg_logistic_regression(
                    y_train,
                    tx_train,
                    lambda_,
                    initial_w,
                    args.max_iters,
                    args.gamma,
                )

                # Convert linear outputs into probabilities
                z = tx_validation @ w
                scores = np.exp(-np.logaddexp(0, -z))

            if not np.isfinite(loss) or not np.all(np.isfinite(scores)):
                raise ValueError(
                    f"Non-finite results for {model}, lambda={lambda_}."
                )

            # Evaluate classification thresholds on validation data
            current_results = []

            for threshold in np.linspace(0.05, 0.95, 19):
                predictions = (scores >= threshold).astype(int)
                metrics = compute_metrics(y_validation, predictions)

                result = {
                    "model": model,
                    "lambda": lambda_,
                    "threshold": float(threshold),
                    "max_iters": (
                        args.max_iters if model == "logistic" else ""
                    ),
                    "gamma": (
                        args.gamma if model == "logistic" else ""
                    ),
                    "training_loss": float(loss),
                    **metrics,
                }

                current_results.append(result)

            results.extend(current_results)
            best = max(current_results, key=lambda r: r["f1"])

            print(
                f"  F1={best['f1']:.4f}, "
                f"accuracy={best['accuracy']:.4f}, "
                f"threshold={best['threshold']:.2f}",
                flush=True,
            )

    # Rank configurations by validation F1 and save all results
    results.sort(key=lambda r: r["f1"], reverse=True)

    with open("validation_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nBest configuration by validation F1:")

    for key, value in results[0].items():
        print(f"{key}: {value}")

    print("\nAll results saved to validation_results.csv")


if __name__ == "__main__":
    main()