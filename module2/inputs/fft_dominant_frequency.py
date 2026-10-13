import numpy as np
import numpy.typing as npt


def fft_dominant_frequency(signal, sample_rate):
    """Dominant frequency (Hz) of a real signal using the real FFT."""
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])
