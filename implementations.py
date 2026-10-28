
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
    loss = np.sum(error ** 2) / (2 * n)

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

    for i in range(max_iters):

        # Randomly select one observation
        i = np.random.randint(0, n)

        # Extract the selected observation
        x_i = tx[i]
        y_i = y[i]

        error = y_i - x_i @ w
        gradient = -x_i * error

        # Update weights
        w = w - gamma * gradient

    error = y - tx @ w
    loss = np.sum(error ** 2) / (2 * n)

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
    loss = np.sum(error ** 2) / (2 * len(y))

    return w, loss


def ridge_regression(y, tx, lambda_):
    pass


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass
