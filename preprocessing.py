import numpy as np


def fit_preprocessing(x):
    """
    Compute preprocessing parameters from training data.

    Arguments:
        x: Training features, shape (N, D)

    Returns:
        parameters: Column mask, medians, means and standard deviations
    """

    minimum = np.nanmin(x, axis=0)
    maximum = np.nanmax(x, axis=0)

    # Keep only non-empty, non-constant columns
    keep = np.isfinite(minimum) & np.isfinite(maximum)
    keep &= maximum > minimum

    x = x[:, keep].copy()

    medians = np.nanmedian(x, axis=0)

    # Replace missing values with the corresponding column median
    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = medians[cols]

    # Compute scaling parameters after filling missing values
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)

    stds[stds == 0] = 1

    return {
        "keep": keep,
        "medians": medians,
        "means": means,
        "stds": stds,
    }


def apply_preprocessing(x, parameters):
    """
    Transform data using parameters computed on training data.

    Arguments:
        x: Features to transform, shape (N, D)
        parameters: Dictionary returned by fit_preprocessing

    Returns:
        tx: Processed features with an intercept column
    """

    # Keep the same columns as in training
    x = x[:, parameters["keep"]].copy()


    rows, cols = np.where(np.isnan(x))
    x[rows, cols] = parameters["medians"][cols]

    # Standardize using training means and standard deviations
    x = (x - parameters["means"]) / parameters["stds"]

    # Add a column of ones for the intercept
    tx = np.column_stack((np.ones(x.shape[0]), x))

    return tx