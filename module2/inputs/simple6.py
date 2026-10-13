import numpy as np


def identity_plus(n, value):
    """Identity matrix of size n with `value` added to every entry."""
    return np.eye(n, dtype=np.float64) + value
