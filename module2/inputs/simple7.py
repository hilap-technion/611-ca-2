import numpy as np


def count_positive(matrix):
    """Number of strictly positive entries, returned as a regular int."""
    return int(np.sum(np.asarray(matrix) > 0))
