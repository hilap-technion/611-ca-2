import numpy as np
import numpy.typing as npt


def linear_regression(x, y):
    """Ordinary least squares fit of y ~ X @ w + b.

    x has shape (n_samples, n_features) and y shape (n_samples,).
    Returns (weights, intercept), solved with np.linalg.lstsq.
    """
    design = np.column_stack((x, np.ones(x.shape[0])))
    coef = np.asarray(np.linalg.lstsq(design, y, rcond=None)[0], dtype=np.float64)
    return coef[:-1], float(coef[-1])
