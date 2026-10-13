import numpy as np


def trace(matrix):
    """Trace of a square matrix, returned as a regular float."""
    return float(np.trace(np.asarray(matrix, dtype=np.float64)))
