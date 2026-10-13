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


def mandelbrot(width = 80, height = 40, max_iter = 50):
    """Escape-time iteration counts for the Mandelbrot set."""
    x = np.linspace(-2.0, 0.6, width)
    y = np.linspace(-1.2, 1.2, height)
    c = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    z = np.zeros_like(c)
    counts = np.zeros(c.shape, dtype=np.int64)
    for _ in range(max_iter):
        alive = np.abs(z) <= 2.0
        z = np.where(alive, z * z + c, z)
        counts += alive
    return counts


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


def pairwise_distances(points):
    """All-pairs Euclidean distance matrix via broadcasting (no loops)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.asarray(np.sqrt((diff**2).sum(axis=-1)), dtype=np.float64)


def fft_dominant_frequency(signal, sample_rate):
    """Dominant frequency (Hz) of a real signal using the real FFT."""
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


