import numpy as np


def scale_matrix(matrix, factor):
    """Multiply a matrix by a regular Python float."""
    return np.asarray(matrix, dtype=np.float64) * factor
