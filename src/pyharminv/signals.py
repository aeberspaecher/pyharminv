"""Synthetic signals and line shapes for testing and examples."""

import numpy as np

__all__ = ["lorentzian", "random_modes", "random_vibration", "synthetic_signal"]


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


def random_vibration(M, C, K, dt, n_samples, input_dofs=None, output="displacement",
                     force_std=1.0, noise=0.0, rng=None):
    """Response of a mechanical system ``M x'' + C x' + K x = f`` to random forces.

    The forces are white Gaussian noise, held constant over each sampling
    interval, and the system is discretised exactly for such inputs (zero
    order hold), so the result has no discretisation error.

    Parameters
    ----------
    M, C, K : array_like
        Mass, damping and stiffness matrices, shape ``(m, m)``, or scalars
        for a single oscillator.
    dt : float
        Sampling interval.
    n_samples : int
        Number of samples.
    input_dofs : array_like of int, optional
        Degrees of freedom that are driven, each by an independent force.
        Default: all.
    output : {"displacement", "velocity", "acceleration"}, optional
        Measured quantity at every degree of freedom.
    force_std : float, optional
        Standard deviation of each force sample.
    noise : float, optional
        Standard deviation of white measurement noise, relative to the
        standard deviation of each output channel.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    y : numpy.ndarray
        Shape ``(n_samples, m)``.
    omega : numpy.ndarray
        Complex angular frequencies ``Omega - 1j Gamma / 2`` of the modes
        with ``Re(omega) >= 0``, in the convention of
        :func:`~pyharminv.harminv`, sorted by real part.
    """
    from scipy.linalg import expm
    from scipy.signal import lfilter

    M, C, K = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (M, C, K))
    m = len(M)
    rng = np.random.default_rng(rng)
    dofs = np.arange(m) if input_dofs is None else np.atleast_1d(input_dofs)
    B = np.zeros((m, len(dofs)))
    B[dofs, np.arange(len(dofs))] = 1.0
    Minv = np.linalg.inv(M)
    Ac = np.block([[np.zeros((m, m)), np.eye(m)], [-Minv @ K, -Minv @ C]])
    Bc = np.vstack([np.zeros((m, len(dofs))), Minv @ B])
    # zero-order hold: [[Ad, Bd], [0, I]] = expm([[Ac, Bc], [0, 0]] dt)
    n, p = Ac.shape[0], Bc.shape[1]
    E = expm(np.block([[Ac, Bc], [np.zeros((p, n)), np.zeros((p, p))]]) * dt)
    Ad, Bd = E[:n, :n], E[:n, n:]

    f = force_std * rng.standard_normal((n_samples, p))
    # diagonalise Ad for a fast simulation with lfilter, one mode at a time
    lam, V = np.linalg.eig(Ad)
    u = np.linalg.solve(V, Bd @ f.T)              # modal inputs, n x N
    z = np.empty_like(u)
    for k in range(n):
        # z_{t+1} = lam z_t + u_t
        z[k] = lfilter([0, 1], [1, -lam[k]], u[k])
    x = (V @ z).real.T                            # states, N x n
    if output == "displacement":
        y = x[:, :m]
    elif output == "velocity":
        y = x[:, m:]
    elif output == "acceleration":
        y = x @ (Ac[m:]).T + f @ (Minv @ B).T
    else:
        raise ValueError(f"unknown output {output!r}")
    if noise:
        y = y + noise * y.std(axis=0) * rng.standard_normal(y.shape)

    s = np.linalg.eigvals(Ac)
    omega = 1j * s
    omega = omega[omega.real >= 0]
    return y, omega[np.argsort(omega.real)]
