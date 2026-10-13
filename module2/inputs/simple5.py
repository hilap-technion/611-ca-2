import numpy as np


def determinant(matrix):
    """Determinant of a square matrix as a regular float."""
    return float(np.linalg.det(np.asarray(matrix, dtype=np.float64)))
