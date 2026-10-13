"""Tests for module1 (run with: python -m pytest module1_tests.py)."""


import numpy as np
import numpy.typing as npt

from module1 import (
    fft_dominant_frequency,
    linear_regression,
    mandelbrot,
    pairwise_distances,
    power_iteration,
)


def test_linear_regression_exact_fit() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = 2.0 * x[:, 0] + 5.0
    w, b = linear_regression(x, y)
    assert w.shape == (1,)
    assert abs(w[0] - 2.0) < 1e-9
    assert abs(b - 5.0) < 1e-9


def test_linear_regression_multiple_features() -> None:
    rng = np.random.default_rng(7)
    x = rng.random((200, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    y = x @ true_w - 3.0
    w, b = linear_regression(x, y)
    assert np.allclose(w, true_w, atol=1e-9)
    assert abs(b + 3.0) < 1e-9


def test_linear_regression_noisy_data_is_close() -> None:
    rng = np.random.default_rng(1)
    x = rng.random((500, 2))
    noise = 0.01 * rng.standard_normal((500,))
    y = x @ np.array([3.0, -2.0]) + 1.0 + noise
    w, b = linear_regression(x, y)
    assert np.allclose(w, [3.0, -2.0], atol=0.05)
    assert abs(b - 1.0) < 0.05


def test_linear_regression_residuals_orthogonal_to_features() -> None:
    rng = np.random.default_rng(3)
    x = rng.random((50, 2))
    y = rng.random((50,))
    w, b = linear_regression(x, y)
    resid = y - (x @ w + b)
    assert abs(resid.sum()) < 1e-9
    assert np.allclose(x.T @ resid, 0.0, atol=1e-9)


def test_mandelbrot_shape_and_bounds() -> None:
    m = mandelbrot(width=30, height=20, max_iter=25)
    assert m.shape == (20, 30)
    assert m.min() >= 0
    assert m.max() <= 25


def test_mandelbrot_origin_in_set_and_far_point_escapes() -> None:
    m = mandelbrot(width=261, height=241, max_iter=50)
    # x = linspace(-2, 0.6), y = linspace(-1.2, 1.2): origin is at column 200, row 120
    assert m[120, 200] == 50
    assert m[0, 0] < 5  # corner (-2, -1.2) escapes quickly


def test_power_iteration_symmetric_matrix() -> None:
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    val, vec = power_iteration(a)
    assert abs(val - (5 + np.sqrt(5)) / 2) < 1e-9
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-9
    assert np.allclose(a @ vec, val * vec, atol=1e-8)


def test_power_iteration_diagonal() -> None:
    a = np.diag(np.array([1.0, 7.0, 3.0]))
    val, vec = power_iteration(a)
    assert abs(val - 7.0) < 1e-9
    assert abs(abs(vec[1]) - 1.0) < 1e-9


def test_pairwise_distances_known_values() -> None:
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    d = pairwise_distances(pts)
    expected = np.array(
        [[0.0, 5.0, 10.0], [5.0, 0.0, 5.0], [10.0, 5.0, 0.0]]
    )
    assert np.allclose(d, expected)


def test_pairwise_distances_properties() -> None:
    pts = np.random.default_rng(42).random((10, 4))
    d = pairwise_distances(pts)
    assert d.shape == (10, 10)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert (d >= 0).all()


def test_fft_dominant_frequency_single_tone() -> None:
    t = np.arange(0, 1, 1 / 1000)
    sig = np.sin(2 * np.pi * 50 * t)
    assert fft_dominant_frequency(sig, 1000.0) == 50.0


def test_fft_dominant_frequency_ignores_dc_offset_and_weaker_tone() -> None:
    t = np.arange(0, 1, 1 / 1000)
    sig = 10.0 + np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    assert fft_dominant_frequency(sig, 1000.0) == 50.0
    sig2 = 0.3 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    assert fft_dominant_frequency(sig2, 1000.0) == 120.0
