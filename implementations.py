import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Arguments:
        y: Labels, shape (N,)
        tx: Feature matrix, shape (N, D)
        initial_w: Initial weights, shape (D,)
        max_iters: Number of gradient descent iterations
        gamma: Learning rate

    Returns:
        w: Final weights
        loss: MSE loss evaluated at the final weights
    """

    # Initialize weights
    w = initial_w.copy()
    n = len(y)

    # Gradient descent
    for i in range(max_iters):

        error = y - tx @ w
        gradient = -(tx.T @ error) / n

        w = w - gamma * gradient

    error = y - tx @ w
    loss = np.sum(error**2) / (2 * n)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using SGD.

    Arguments:
        y: Labels, shape (N,)
        tx: Feature matrix, shape (N, D)
        initial_w: Initial weights, shape (D,)
        max_iters: Number of stochastic gradient updates
        gamma: Learning rate

    Returns:
        w: Final weights
        loss: Final MSE computed over the entire dataset
    """

    w = initial_w.copy()
    n = len(y)

    for _ in range(max_iters):

        # Randomly select one observation (mini-batch size 1)
        idx = np.random.randint(0, n)

        # Extract the selected observation
        x_i = tx[idx]
        y_i = y[idx]

        error = y_i - x_i @ w
        gradient = -x_i * error

        # Update weights
        w = w - gamma * gradient

    error = y - tx @ w
    loss = np.sum(error**2) / (2 * n)

    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    Arguments:
        y: Labels, shape (N,)
        tx: Feature matrix, shape (N, D)

    Returns:
        w: Optimal weights, shape (D,)
        loss: MSE evaluated at the optimal weights
    """

    # Compute normal equation components
    xtx = tx.T @ tx
    xty = tx.T @ y

    # Solve the normal equations
    w = np.linalg.solve(xtx, xty)

    # Compute final loss
    error = y - tx @ w
    loss = np.sum(error**2) / (2 * len(y))

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    Arguments:
        y: Labels, shape (N,)
        tx: Feature matrix, shape (N, D)
        lambda_: Regularization parameter

    Returns:
        w: Optimal weights, shape (D,)
        loss: MSE evaluated at the optimal weights, without regularization
    """

    n = len(y)
    d = tx.shape[1]

    # Compute regularized normal equation components
    xtx = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    xty = tx.T @ y

    # Solve the normal equations
    w = np.linalg.solve(xtx, xty)

    # Compute final loss without regularization
    error = y - tx @ w
    loss = np.sum(error**2) / (2 * n)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Arguments:
        y: Labels in {0, 1}, shape (N,)
        tx: Feature matrix, shape (N, D)
        initial_w: Initial weights, shape (D,)
        max_iters: Number of gradient descent iterations
        gamma: Learning rate

    Returns:
        w: Final weights
        loss: Mean logistic loss evaluated at the final weights
    """

    w = initial_w.copy()
    n = len(y)

    # Gradient descent
    for i in range(max_iters):

        z = tx @ w

        # Compute sigmoid predictions without overflow
        predictions = np.exp(-np.logaddexp(0, -z))
        gradient = tx.T @ (predictions - y) / n

        w = w - gamma * gradient

    # Compute final loss
    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.

    Arguments:
        y: Labels in {0, 1}, shape (N,)
        tx: Feature matrix, shape (N, D)
        lambda_: Regularization parameter
        initial_w: Initial weights, shape (D,)
        max_iters: Number of gradient descent iterations
        gamma: Learning rate

    Returns:
        w: Final weights
        loss: Mean logistic loss at the final weights, without regularization
    """

    w = initial_w.copy()
    n = len(y)

    # Gradient descent
    for i in range(max_iters):

        z = tx @ w

        # Compute sigmoid predictions without overflow
        predictions = np.exp(-np.logaddexp(0, -z))
        gradient = tx.T @ (predictions - y) / n
        gradient = gradient + 2 * lambda_ * w

        w = w - gamma * gradient

    # Compute final loss without regularization
    z = tx @ w
    loss = np.mean(np.logaddexp(0, z) - y * z)

    return w, loss
