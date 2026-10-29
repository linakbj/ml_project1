
import argparse
import csv

import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import ridge_regression, reg_logistic_regression
from preprocessing import fit_preprocessing, apply_preprocessing


def load_best_configuration(filename):
    """Load the best configuration from train.py results."""
    with open(filename, newline="") as file:
        reader = csv.DictReader(file)
        results = list(reader)

    if not results:
        raise ValueError("No validation results found.")

    # Select the configuration with the highest validation F1
    return max(results, key=lambda row: float(row["f1"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset")
    parser.add_argument(
        "--results", default="validation_results.csv"
    )
    parser.add_argument(
        "--output", default="submission.csv"
    )
    args = parser.parse_args()

    # Load the best configuration selected during validation
    best = load_best_configuration(args.results)

    model = best["model"]
    lambda_ = float(best["lambda"])
    threshold = float(best["threshold"])

    print("Selected model:", model)
    print("Lambda:", lambda_)
    print("Threshold:", threshold)

    # Load full training and test datasets
    print("Loading datasets...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(
        args.data
    )

    # Logistic regression and ridge use labels in {0, 1}
    y_train = (y_train == 1).astype(float)

    # Fit preprocessing on full training data only
    print("Preprocessing...")
    parameters = fit_preprocessing(x_train)

    tx_train = apply_preprocessing(x_train, parameters)
    tx_test = apply_preprocessing(x_test, parameters)

    # Train the best model using all available training data
    print("Training final model...")

    if model == "ridge":
        w, loss = ridge_regression(
            y_train, tx_train, lambda_
        )
        scores = tx_test @ w

    elif model == "logistic":
        initial_w = np.zeros(tx_train.shape[1])

        max_iters = int(best["max_iters"])
        gamma = float(best["gamma"])

        w, loss = reg_logistic_regression(
            y_train,
            tx_train,
            lambda_,
            initial_w,
            max_iters,
            gamma,
        )

        z = tx_test @ w
        scores = np.exp(-np.logaddexp(0, -z))

    else:
        raise ValueError(f"Unknown model: {model}")

    # Convert scores to {-1, +1} predictions
    predictions = np.where(scores >= threshold, 1, -1)

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced non-finite scores.")

    print("Final training loss:", loss)
    print("Positive predictions:", np.sum(predictions == 1))
    print("Negative predictions:", np.sum(predictions == -1))

    # Generate submission file
    create_csv_submission(
        test_ids,
        predictions,
        args.output,
    )

    print("Submission saved to:", args.output)


if __name__ == "__main__":
    main()
