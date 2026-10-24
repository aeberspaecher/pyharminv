"""Synthetic signals and line shapes for testing and examples."""

import numpy as np

__all__ = ["lorentzian", "random_modes", "synthetic_signal"]


def synthetic_signal(t, omega, amplitude, real=False, noise=0.0, rng=None):
    """Sum of damped exponentials, optionally with white Gaussian noise.

    Parameters
    ----------
    t : array_like
        Sample times.
    omega, amplitude : array_like
        Complex angular frequencies (``Im < 0`` for decay) and amplitudes.
    real : bool, optional
        If true, return ``Re sum_k d_k exp(-i omega_k t)``, i.e. damped
        cosines ``|d_k| exp(-Gamma_k t / 2) cos(Omega_k t - arg d_k)``.
        Otherwise return the complex sum.
    noise : float, optional
        Standard deviation of additive noise (per real and imaginary part).
    rng : numpy.random.Generator or int, optional
        Random generator or seed for the noise.

    Returns
    -------
    numpy.ndarray
    """
    t = np.asarray(t, dtype=float)
    omega = np.atleast_1d(omega)
    amplitude = np.atleast_1d(amplitude)
    s = (amplitude * np.exp(-1j * np.outer(t, omega))).sum(axis=1)
    rng = np.random.default_rng(rng)
    if real:
        s = s.real
        if noise:
            s = s + noise * rng.standard_normal(len(t))
    elif noise:
        s = s + noise * (rng.standard_normal(len(t))
                         + 1j * rng.standard_normal(len(t)))
    return s


def random_modes(n, omega_min, omega_max, q_min=20, q_max=200,
                 min_spacing=0.0, rng=None):
    """Draw random resonances for tests.

    Parameters
    ----------
    n : int
        Number of modes.
    omega_min, omega_max : float
        Range of line centres.
    q_min, q_max : float
        Range of quality factors ``Q = Omega / Gamma``.
    min_spacing : float, optional
        Minimal distance between line centres.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    omega, amplitude : numpy.ndarray
        Complex frequencies and unit-magnitude amplitudes with random phases.
    """
    rng = np.random.default_rng(rng)
    free = (omega_max - omega_min) - (n - 1) * min_spacing
    if free < 0:
        raise ValueError("cannot place modes with the requested spacing")
    # uniform positions in the reduced interval, then spread apart
    centers = (omega_min + np.sort(rng.uniform(0, free, n))
               + min_spacing * np.arange(n))
    q = rng.uniform(q_min, q_max, n)
    omega = centers - 0.5j * centers / q
    amplitude = rng.uniform(0.5, 1.5, n) * np.exp(2j * np.pi * rng.random(n))
    return omega, amplitude


def lorentzian(omega, center, fwhm, height=1.0):
    """Lorentzian line ``height / (1 + ((omega - center) / (fwhm/2))**2)``.

    Several lines can be given as arrays; their sum is returned.
    """
    w = np.asarray(omega, dtype=float)[..., None]
    center = np.atleast_1d(center)
    fwhm = np.atleast_1d(fwhm)
    height = np.atleast_1d(height)
    return (height / (1 + ((w - center) / (fwhm / 2)) ** 2)).sum(axis=-1)
