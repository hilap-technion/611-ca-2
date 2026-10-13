import numpy as np
import numpy.typing as npt


def pairwise_distances(points):
    """All-pairs Euclidean distance matrix via broadcasting (no loops)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.asarray(np.sqrt((diff**2).sum(axis=-1)), dtype=np.float64)
