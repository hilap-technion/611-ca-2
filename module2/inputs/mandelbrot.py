import numpy as np
import numpy.typing as npt


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
