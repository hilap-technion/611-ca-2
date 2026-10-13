import numpy as np
import numpy.typing as npt


def power_iteration(matrix, iterations = 200, seed = 0):
    """Dominant eigenvalue and eigenvector of a square matrix."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((matrix.shape[0],))
    v = np.asarray(v / np.linalg.norm(v), dtype=np.float64)
    for _ in range(iterations):
        w = matrix @ v
        v = np.asarray(w / np.linalg.norm(w), dtype=np.float64)
    eigenvalue = float(v @ matrix @ v)
    return eigenvalue, v
