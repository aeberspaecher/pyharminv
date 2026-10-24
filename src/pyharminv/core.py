"""Harmonic inversion of time signals.

Harmonic inversion decomposes a signal sampled at equidistant times
``t_n = t_0 + n dt`` into a finite sum of damped complex exponentials

.. math::

    c(t) = \\sum_k d_k \\, e^{-i \\omega_k (t - t_0)},
    \\qquad \\omega_k = \\Omega_k - \\tfrac{i}{2} \\Gamma_k ,

with complex frequencies :math:`\\omega_k` and complex amplitudes
:math:`d_k`. Unlike the discrete Fourier transform, the frequencies are not
tied to a grid, so closely spaced and strongly overlapping resonances can be
resolved from short signals. Each term corresponds to a Lorentzian line of
centre :math:`\\Omega_k` and full width at half maximum :math:`\\Gamma_k` in
the spectrum.

Two solvers are provided. The default splits the frequency range into
windows and solves each with the filter diagonalization method of
Mandelshtam and Taylor [3]_, which keeps the local problems small without
discarding any part of the signal and copes well with noise. The alternative
fits the whole band at once with the Padé variant of harmonic inversion
described by Wiersig and Main [1]_, using the robust Padé algorithm of
Gonnet, Güttel and Trefethen [2]_.

References
----------
.. [1] J. Wiersig and J. Main, "Fractal Weyl law for chaotic microcavities:
   Fresnel's laws imply multifractal scattering", Phys. Rev. E 77, 036205
   (2008), section II.
.. [2] P. Gonnet, S. Güttel and L. N. Trefethen, "Robust Padé approximation
   via SVD", SIAM Review 55, 101-117 (2013).
.. [3] V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time
   signals and its applications", J. Chem. Phys. 107, 6756 (1997).
"""

from dataclasses import dataclass, field

import numpy as np

from .fdm import _FDMGrid, _fdm_solve
from .pade import padeapprox

__all__ = ["HarmInvResult", "harminv", "harminv_window"]


@dataclass
class HarmInvResult:
    """Modes found by :func:`harminv`.

    The signal is represented as

    * ``c(t) = sum_k amplitude[k] * exp(-1j * omega[k] * (t - t0))`` for a
      complex input signal, and
    * ``c(t) = Re sum_k amplitude[k] * exp(-1j * omega[k] * (t - t0))`` for a
      real input signal. Only modes with ``Re(omega) >= 0`` are returned in
      that case; their mirror images at ``-conj(omega)`` are implied.

    Attributes
    ----------
    omega : numpy.ndarray
        Complex angular frequencies, sorted by real part. Decaying modes have
        ``Im(omega) < 0``.
    amplitude : numpy.ndarray
        Complex amplitudes.
    error : numpy.ndarray
        Error estimate for each frequency: the distance to the closest pole
        of a second, independent Padé fit of the same window. Genuine modes
        have small errors; spurious ones have errors comparable to their
        spacing. ``nan`` if no estimate could be made.
    real : bool
        Whether the input signal was real.
    t0 : float
        Time of the first sample, the reference time of the amplitudes.
    """

    omega: np.ndarray
    amplitude: np.ndarray
    error: np.ndarray
    real: bool = False
    t0: float = 0.0
    info: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return len(self.omega)

    @property
    def center(self):
        """Line centres ``Re(omega)`` (angular frequency)."""
        return self.omega.real

    @property
    def frequency(self):
        """Line centres in cycles per unit time, ``Re(omega) / (2 pi)``."""
        return self.omega.real / (2 * np.pi)

    @property
    def decay_rate(self):
        """Amplitude decay rates ``-Im(omega)``."""
        return -self.omega.imag

    @property
    def width(self):
        """Full widths at half maximum ``Gamma = -2 Im(omega)``."""
        return -2 * self.omega.imag

    @property
    def Q(self):
        """Quality factors ``Re(omega) / Gamma``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.omega.real / self.width

    @property
    def magnitude(self):
        """Absolute values of the amplitudes."""
        return np.abs(self.amplitude)

    @property
    def phase(self):
        """Phases of the amplitudes."""
        return np.angle(self.amplitude)

    def __getitem__(self, idx):
        idx = np.atleast_1d(np.arange(len(self))[idx])
        return HarmInvResult(self.omega[idx], self.amplitude[idx],
                             self.error[idx], self.real, self.t0,
                             dict(self.info))

    def select(self, min_magnitude=0.0, max_error=np.inf, max_rel_error=np.inf,
               max_width=np.inf, omega_min=-np.inf, omega_max=np.inf):
        """Return the subset of modes that satisfy all given criteria.

        Parameters
        ----------
        min_magnitude : float
            Smallest accepted ``abs(amplitude)``.
        max_error : float
            Largest accepted absolute error estimate.
        max_rel_error : float
            Largest accepted ``error / abs(omega)``.
        max_width : float
            Largest accepted line width ``Gamma``.
        omega_min, omega_max : float
            Accepted range of ``Re(omega)``.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = self.error / np.abs(self.omega)
        err = np.nan_to_num(self.error, nan=0.0)
        rel = np.nan_to_num(rel, nan=0.0)
        mask = ((self.magnitude >= min_magnitude) & (err <= max_error)
                & (rel <= max_rel_error) & (self.width <= max_width)
                & (self.center >= omega_min) & (self.center <= omega_max))
        return self[mask]

    def modes(self, t):
        """Contribution of each mode at times ``t``, shape ``(len(t), K)``."""
        t = np.asarray(t, dtype=float)
        return self.amplitude * np.exp(-1j * np.outer(t - self.t0, self.omega))

    def reconstruct(self, t):
        """Evaluate the fitted signal at times ``t``.

        Returns a real array for real input signals and a complex array
        otherwise.
        """
        s = self.modes(t).sum(axis=1)
        return s.real if self.real else s

    def refine(self, signal, dt, max_modes=100, **kwargs):
        """Polish all modes by nonlinear least squares on the full signal.

        Harmonic inversion gives frequencies and amplitudes close to the
        best fit. Using them as a starting point, this method minimises
        ``sum_n |c_n - model(t_n)|^2`` over all frequencies and amplitudes
        jointly. For white Gaussian noise this is the maximum likelihood
        estimate, whose errors reach the Cramér-Rao bound. Spurious modes
        should be removed with :meth:`select` first.

        Parameters
        ----------
        signal : array_like
            The signal the result was computed from.
        dt : float
            Sampling interval.
        max_modes : int, optional
            Refuse larger problems; the Jacobian has ``4 K`` columns.
        **kwargs
            Passed to :func:`scipy.optimize.least_squares`.

        Returns
        -------
        HarmInvResult
            The refined modes. ``error`` holds the standard deviation of each
            complex frequency estimated from the residual and the Jacobian;
            ``info`` also contains ``omega_std``, ``amplitude_std``, the
            noise estimate ``sigma`` from the residual (standard deviation
            per real component, like the ``noise`` argument of
            :func:`~pyharminv.synthetic_signal`) and the optimizer's result.
        """
        from scipy.optimize import least_squares

        K = len(self)
        if K == 0:
            return self[:]
        if K > max_modes:
            raise ValueError(f"{K} modes exceed max_modes={max_modes}; "
                             "select fewer modes first")
        signal = np.asarray(signal)
        tau = dt * np.arange(len(signal))
        # purely damped modes of a real signal keep Re(omega) = 0, Im(d) = 0
        fixed = (self.omega.real == 0) if self.real else np.zeros(K, bool)

        def unpack(p):
            p = p.reshape(4, K)
            return p[0] + 1j * p[1], p[2] + 1j * p[3]

        def model(p):
            w, d = unpack(p)
            E = np.exp(-1j * np.outer(tau, w))
            return E, w, d

        def residual(p):
            E, w, d = model(p)
            r = E @ d - signal
            return r.real if self.real else np.concatenate([r.real, r.imag])

        def jacobian(p):
            E, w, d = model(p)
            dE = E * d
            cols = [-1j * tau[:, None] * dE, tau[:, None] * dE, E, 1j * E]
            J = np.hstack(cols)
            free = np.concatenate([~fixed, np.ones(K, bool), np.ones(K, bool),
                                   ~fixed])
            J[:, ~free] = 0
            return J.real if self.real else np.vstack([J.real, J.imag])

        p0 = np.concatenate([self.omega.real, self.omega.imag,
                             self.amplitude.real, self.amplitude.imag])
        kwargs.setdefault("x_scale", "jac")
        sol = least_squares(residual, p0, jac=jacobian, **kwargs)

        w, d = unpack(sol.x)
        J = sol.jac
        n_obs, n_par = J.shape
        n_free = n_par - 2 * int(fixed.sum())
        dof = max(1, n_obs - n_free)
        sigma2 = 2 * sol.cost / dof
        keep = np.abs(J).sum(axis=0) > 0
        cov = np.zeros((n_par, n_par))
        cov[np.ix_(keep, keep)] = sigma2 * np.linalg.pinv(J[:, keep].T @ J[:, keep])
        var = np.diag(cov).reshape(4, K)
        omega_std = np.sqrt(var[0] + var[1])
        amp_std = np.sqrt(var[2] + var[3])
        noise = np.sqrt(sigma2)

        order = np.argsort(w.real)
        info = dict(self.info)
        info.update(omega_std=omega_std[order], amplitude_std=amp_std[order],
                    sigma=noise, least_squares=sol)
        return HarmInvResult(omega=w[order], amplitude=d[order],
                             error=omega_std[order], real=self.real,
                             t0=self.t0, info=info)

    def spectrum(self, omega):
        """Complex spectrum of the fitted signal.

        This is the one-sided Fourier transform
        ``integral_{t0}^{inf} c(t) exp(i omega (t - t0)) dt`` of the fitted
        signal, a sum of complex Lorentzians ``1j * d_k / (omega - omega_k)``.
        For a real signal the mirror modes are included. The real part is the
        absorption-like spectrum, ``abs(...)**2`` the power spectrum.
        """
        w = np.asarray(omega, dtype=float)[..., None]
        if self.real:
            s = 0.5j * self.amplitude / (w - self.omega)
            s = s + 0.5j * np.conj(self.amplitude) / (w + np.conj(self.omega))
        else:
            s = 1j * self.amplitude / (w - self.omega)
        return s.sum(axis=-1)

    def lorentzians(self):
        """Lorentzian line parameters of the modes.

        Returns
        -------
        dict of numpy.ndarray
            ``center`` (Omega), ``fwhm`` (Gamma), ``height`` and ``area``.
            For a mode with real, positive amplitude ``d`` the real part of
            :meth:`spectrum` near the line is
            ``height / (1 + ((omega - center) / (fwhm / 2))**2)``, with
            ``height = 2 d / fwhm`` (``d / fwhm`` for a real signal, whose
            amplitude is split between the line and its mirror) and
            ``area = pi * height * fwhm / 2``. A complex amplitude adds a
            dispersive part; ``height`` then refers to ``abs(d)``.
        """
        gamma = self.width
        mag = self.magnitude / (2 if self.real else 1)
        height = 2 * mag / gamma
        return {"center": self.center, "fwhm": gamma, "height": height,
                "area": np.pi * height * gamma / 2}

    def __repr__(self):
        lines = [f"HarmInvResult with {len(self)} modes ({'real' if self.real else 'complex'} signal)",
                 f"{'omega.real':>14} {'omega.imag':>14} {'Q':>10} {'|d|':>11} {'arg d':>8} {'error':>9}"]
        for w, d, e, q in zip(self.omega, self.amplitude, self.error, self.Q):
            lines.append(f"{w.real:14.7g} {w.imag:14.7g} {q:10.4g} {abs(d):11.4g} "
                         f"{np.angle(d):8.4f} {e:9.2g}")
        return "\n".join(lines)


def _pade_modes(c, dt, K, tol):
    """Fit ``c_n = sum_k d_k z_k^n`` with a type (K-1, K) Padé approximant.

    The samples are the Taylor coefficients of
    ``g(x) = sum_n c_n x^n = sum_k d_k / (1 - z_k x)``, so the poles of the
    approximant are ``x_k = 1/z_k`` and the residues are ``-d_k / z_k``.
    """
    pade = padeapprox(c[:2 * K], K - 1, K, tol=tol)
    x = pade.poles
    if x.size == 0:
        return np.zeros(0, complex), np.zeros(0, complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = -1j * np.log(x.astype(complex)) / dt
        d = -pade.residues / x
    ok = np.isfinite(omega) & np.isfinite(d)
    return omega[ok], d[ok]


def _match_error(omega, omega_ref):
    """Distance of each ``omega`` to the closest frequency in ``omega_ref``."""
    if omega_ref.size == 0:
        return np.full(omega.shape, np.inf)
    return np.min(np.abs(omega[:, None] - omega_ref[None, :]), axis=1)


def _lsq_pade_modes(c, dt, L, tol):
    """Least-squares Padé fit of ``c_n = sum_k d_k z_k^n`` from all samples.

    The denominator of a type (nu-1, nu) approximant of ``sum_n c_n x^n``
    is ``q(x) = prod_k (1 - z_k x)``. Here the ``z_k`` are determined from
    all samples: the Hankel matrix of order ``L`` is truncated to its
    numerical rank ``nu`` (singular values below ``tol`` times the largest
    are noise), and the ``z_k`` are the eigenvalues of the shift operator
    on its dominant subspace (matrix pencil method). The amplitudes follow
    from a linear least squares fit to all samples.
    """
    from scipy.linalg import hankel, svd

    N = len(c)
    Y = hankel(c[:N - L], c[N - L - 1:])
    _, s, Vh = svd(Y, full_matrices=False)
    if s[0] == 0:
        return np.zeros(0, complex), np.zeros(0, complex)
    nu = int(np.sum(s > tol * s[0]))
    if nu == 0:
        return np.zeros(0, complex), np.zeros(0, complex)
    # the rows of Vh span the vectors (z_k^j)_j; shifting j by one
    # multiplies by z_k
    V = Vh[:nu].T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:]).astype(complex)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        logz = np.log(z)
        omega = 1j * logz / dt
        ok = np.isfinite(omega)
        logz, omega = logz[ok], omega[ok]
        # amplitudes by least squares, columns scaled to at most 1
        log_scale = np.maximum(0.0, logz.real * (N - 1))
        E = np.exp(np.outer(np.arange(N), logz) - log_scale)
        d, *_ = np.linalg.lstsq(E, c, rcond=None)
        d = d * np.exp(-log_scale)
    ok = np.isfinite(d)
    return omega[ok], d[ok]


def harminv_window(c, dt, K=None, tol=1e-10, error_estimate=True,
                   variant="least-squares"):
    """Harmonic inversion of a signal with Padé approximants, without windows.

    Fits ``c_n = sum_k d_k exp(-1j * omega_k * n * dt)`` by reading the
    samples as Taylor coefficients of ``sum_k d_k / (1 - z_k x)``.

    Parameters
    ----------
    c : array_like
        Signal samples.
    dt : float
        Sampling interval.
    K : int, optional
        ``"robust"``: maximum number of modes; the first ``2K`` samples are
        used (default ``len(c) // 2``). ``"least-squares"``: order of the
        Toeplitz matrix used for the rank decision, i.e. the maximum number
        of modes (default ``min(len(c) // 3, 200)``); all samples are used.
    tol : float, optional
        ``"least-squares"``: relative noise level of the signal (singular
        values below ``tol`` times the largest are discarded).
        ``"robust"``: tolerance of :func:`pyharminv.padeapprox`, relative to
        the norm of the samples; about a hundredth of the noise level.
    error_estimate : bool, optional
        Whether to compute error estimates by a second fit of the signal
        with the first sample dropped.
    variant : {"least-squares", "robust"}, optional
        ``"robust"`` is the robust Padé approximant of Gonnet, Güttel and
        Trefethen of type (K-1, K). Its rank reduction lowers the type to
        ``(nu-1, nu)``, which is then determined by the first ``2 nu``
        samples only, so noise is not averaged out. ``"least-squares"``
        (default) determines the denominator of degree ``nu``, i.e. the
        poles, from all samples via a rank-``nu`` truncated SVD of their
        Hankel matrix (the matrix pencil method of Hua and Sarkar) and the
        amplitudes by linear least squares; it is much more accurate for
        noisy data. Both are exact for noise-free signals.

    Returns
    -------
    omega, amplitude, error : numpy.ndarray
        All poles of the approximant, unfiltered.
    """
    c = np.asarray(c)
    if variant == "robust":
        if K is None:
            K = len(c) // 2
        K = int(min(K, len(c) // 2))
        if K < 1:
            raise ValueError("need at least two samples")
        fit = lambda c, K: _pade_modes(c, dt, K, tol)
        K2 = K if len(c) > 2 * K else K - 1
    elif variant == "least-squares":
        if K is None:
            K = min(len(c) // 3, 200)
        K = int(min(K, (len(c) - 1) // 2))
        if K < 1:
            raise ValueError("need at least three samples")
        fit = lambda c, K: _lsq_pade_modes(c, dt, K, tol)
        K2 = min(K, (len(c) - 2) // 2)
    else:
        raise ValueError(f"unknown Padé variant {variant!r}")

    omega, d = fit(c, K)
    if error_estimate and K2 >= 1:
        # second, independent fit: the signal starting one sample later has
        # the same frequencies but different rounding and noise patterns
        omega2, _ = fit(c[1:], K2)
        error = _match_error(omega, omega2)
    else:
        error = np.full(omega.shape, np.nan)
    return omega, d, error


def _geometric_sum(L, N):
    """``sum_{n=0}^{N-1} exp(n L)``, accurate for small ``L``."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        g = np.expm1(N * L) / np.expm1(L)
    return np.where(L == 0, N, g)


def _fit_amplitudes(signal, dt, omega, real, selfmirror, max_direct=4_000_000):
    """Least-squares amplitudes for fixed frequencies on the full signal.

    For a real signal the model is ``Re sum_k a_k z_k^n``, which is linear in
    ``Re(a_k)`` and ``Im(a_k)``; purely damped modes only have a real part.
    Small problems are solved directly; for large ones the normal equations
    are assembled from closed-form geometric sums to save memory.
    """
    N = len(signal)
    K = len(omega)
    lam = -1j * omega * dt          # z_k = exp(lam_k)
    # column scaling: the largest |z_k^n| over the signal
    log_scale = np.maximum(0.0, lam.real * (N - 1))
    osc = ~selfmirror if real else np.ones(K, bool)

    if N * K * (2 if real else 1) <= max_direct:
        n = np.arange(N)
        with np.errstate(over="ignore", invalid="ignore"):
            V = np.exp(np.outer(n, lam) - log_scale)
        if not real:
            sol, *_ = np.linalg.lstsq(V, signal, rcond=None)
            return sol * np.exp(-log_scale)
        A = np.hstack([V.real, -V.imag[:, osc]])
        sol, *_ = np.linalg.lstsq(A, signal.real, rcond=None)
    else:
        # P = V^H V and R = V^T V in closed form
        sc = log_scale[:, None] + log_scale[None, :]
        P = _geometric_sum(np.conj(lam)[:, None] + lam[None, :], N) * np.exp(-sc)
        R = _geometric_sum(lam[:, None] + lam[None, :], N) * np.exp(-sc)
        # V^T c and V^H c, accumulated in blocks of rows
        vt = np.zeros(K, complex)
        vh = np.zeros(K, complex)
        block = max(1, max_direct // max(K, 1))
        for start in range(0, N, block):
            n = np.arange(start, min(N, start + block))
            with np.errstate(over="ignore", invalid="ignore"):
                V = np.exp(np.outer(n, lam) - log_scale)
            vt += V.T @ signal[n]
            vh += V.conj().T @ signal[n]
        if not real:
            sol = _solve_normal(P, vh)
            return sol * np.exp(-log_scale)
        RR = 0.5 * (P + R).real
        II = 0.5 * (P - R).real
        # RI[k, l] = sum_n Re V_nk Im V_nl
        RI = 0.5 * (R + P).imag
        AtA = np.block([[RR, -RI[:, osc]], [-RI[:, osc].T, II[np.ix_(osc, osc)]]])
        Atc = np.concatenate([vt.real, -vt.imag[osc]])
        sol = _solve_normal(AtA, Atc)

    amp = sol[:K].astype(complex)
    amp[osc] += 1j * sol[K:]
    return amp * np.exp(-log_scale)


def _solve_normal(G, b):
    """Solve Hermitian normal equations, ignoring tiny eigenvalues."""
    w, U = np.linalg.eigh(G)
    cutoff = np.max(np.abs(w)) * 1e-13
    inv = np.where(np.abs(w) > cutoff, 1 / np.where(w == 0, 1, w), 0)
    return U @ (inv * (U.conj().T @ b))


def _remove_duplicates(omega, source, offcentre, distance):
    """Mask that drops modes found again by a neighbouring window.

    Of two modes from different windows that are closer than ``distance``,
    the one nearer to the centre of its window is kept.
    """
    keep = np.ones(len(omega), bool)
    order = np.argsort(offcentre)
    for i in order:
        if not keep[i]:
            continue
        dup = ((np.abs(omega - omega[i]) < distance) & (source != source[i])
               & keep)
        dup[i] = False
        keep[dup] = False
    return keep


def harminv(signal, dt, omega_min=None, omega_max=None, method="fdm",
            n_windows=None, window_width=None, K=None, k_max=150, tol=1e-10,
            guard=0.25, density=1.1, min_magnitude=None, allow_growing=False,
            refine_amplitudes=True, t0=0.0, pade_variant="least-squares"):
    """Harmonic inversion of an equidistantly sampled signal.

    Parameters
    ----------
    signal : array_like
        Real or complex samples ``c(t0 + n dt)``, ``n = 0, ..., N-1``.
    dt : float
        Sampling interval.
    omega_min, omega_max : float, optional
        Range of angular frequencies ``Re(omega)`` to search. Defaults to
        ``[0, pi/dt]`` for real and ``[-pi/dt, pi/dt]`` for complex signals.
    method : {"fdm", "pade"}, optional
        ``"fdm"`` (default) splits the range into frequency windows and
        solves each with filter diagonalization [3]_, which uses the whole
        signal in every window, scales to long signals with many lines and
        is the more robust choice for noisy data. ``"pade"`` fits the whole
        band at once with a Padé approximant as in [1]_, [2]_ (see
        ``pade_variant``); its cost grows like the cube of the number of
        samples, so it suits signals of up to a few hundred or thousand
        samples, and it ignores the frequency range and window options
        except for discarding modes outside the range.
    n_windows : int, optional
        Number of frequency windows for FDM.
    window_width : float, optional
        Width of each window, alternative to ``n_windows``. By default the
        windows are chosen so that each needs about ``k_max`` basis functions.
    K : int, optional
        FDM: number of basis functions per window. Padé: maximum number of
        modes, see :func:`harminv_window`.
    k_max : int, optional
        Size of the local problems when choosing defaults.
    tol : float, optional
        Relative noise level of the data, which sets the rank decisions:
        ``1e-10`` (default) suits clean synthetic data; for measured data use
        the ratio of the noise standard deviation to the rms of the signal,
        or somewhat less. Too large a value loses weak or broad lines; too
        small a value adds spurious weak modes, which can be removed with
        :meth:`HarmInvResult.select`. (The robust Padé variant is more
        sensitive and internally uses ``tol / 100``.)
    guard : float, optional
        FDM: each window's basis extends by this fraction of the window width
        on both sides; only modes inside the window proper are kept.
    density : float, optional
        FDM: basis density relative to the Fourier resolution.
    min_magnitude : float, optional
        Discard modes with ``abs(amplitude)`` below this value. Defaults to
        ``1e-8`` times the largest sample magnitude.
    allow_growing : bool, optional
        Keep modes with ``Im(omega) > 0``. They are usually artefacts.
    refine_amplitudes : bool, optional
        Recompute all amplitudes jointly by linear least squares on the full
        signal with the frequencies held fixed. Recommended; it makes the
        amplitudes consistent across windows.
    t0 : float, optional
        Time of the first sample; amplitudes refer to it.
    pade_variant : {"least-squares", "robust"}, optional
        Padé solver variant, see :func:`harminv_window`. ``"least-squares"``
        uses all samples and is much more accurate for noisy data;
        ``"robust"`` is the classical (robust) Padé approximant from the
        first samples, as in [1]_.

    Returns
    -------
    HarmInvResult

    References
    ----------
    .. [1] J. Wiersig and J. Main, Phys. Rev. E 77, 036205 (2008).
    .. [2] P. Gonnet, S. Güttel and L. N. Trefethen, SIAM Review 55, 101
       (2013).
    .. [3] V. A. Mandelshtam and H. S. Taylor, J. Chem. Phys. 107, 6756
       (1997).
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    N = len(signal)
    if N < 4:
        raise ValueError("signal needs at least four samples")
    if dt <= 0:
        raise ValueError("dt must be positive")
    real = not np.iscomplexobj(signal)
    signal_c = signal.astype(complex)

    nyq = np.pi / dt
    band_lo = 0.0 if real else -nyq
    lo = band_lo if omega_min is None else float(omega_min)
    hi = nyq if omega_max is None else float(omega_max)
    if not hi > lo:
        raise ValueError("omega_max must be larger than omega_min")

    if method not in ("pade", "fdm"):
        raise ValueError(f"unknown method {method!r}")

    if min_magnitude is None:
        min_magnitude = 1e-8 * np.max(np.abs(signal_c))

    windows = []
    if method == "pade":
        pade_tol = tol / 100 if pade_variant == "robust" else tol
        omega, amp, err = harminv_window(signal_c, dt, K=K, tol=pade_tol,
                                         variant=pade_variant)
        windows.append((lo, hi))
    else:
        if window_width is not None and n_windows is not None:
            raise ValueError("give either n_windows or window_width, not both")
        M = (N - 3) // 2
        resolution = 2 * np.pi / ((M + 1) * dt)
        if window_width is not None:
            n_windows = int(np.ceil((hi - lo) / window_width))
        if n_windows is None:
            n_basis = K if K is not None else k_max
            width_max = n_basis * resolution / (density * (1 + 2 * guard))
            n_windows = int(np.ceil((hi - lo) / width_max))
        n_windows = max(1, int(n_windows))
        width = (hi - lo) / n_windows

        # one equidistant basis grid for all windows; each window uses the
        # grid points within its range extended by the guard bands
        if K is not None:
            step = width * (1 + 2 * guard) / K
        else:
            step = resolution / density
        ext = guard * width
        n_grid = int(np.ceil((hi - lo + 2 * ext) / step))
        grid = _FDMGrid(signal_c, dt, lo - ext + step / 2, step, n_grid, M)

        preps = []
        for j in range(n_windows):
            w_lo = lo + j * width
            w_hi = w_lo + width
            i0 = int(np.floor((w_lo - ext - grid.phi[0]) / step + 0.5))
            i1 = int(np.ceil((w_hi + ext - grid.phi[0]) / step + 0.5))
            sl = slice(max(0, i0), min(n_grid, max(i1, i0 + 2)))
            preps.append(grid.prepare(sl))
            windows.append((w_lo, w_hi))

        # the singular value threshold refers to the whole signal, so that
        # windows without strong lines do not fit noise
        s_ref = max(p["s"][0] for p in preps)

        # accept modes slightly beyond each window, so that a line on a
        # window boundary is never missed, and resolve duplicates below
        margin = 0.5 * ext
        omegas, amps, errors, offcentre = [], [], [], []
        for prep, (w_lo, w_hi) in zip(preps, windows):
            w, d, e = _fdm_solve(prep, tol * s_ref)
            inside = (w.real >= w_lo - margin) & (w.real <= w_hi + margin)
            omegas.append(w[inside])
            amps.append(d[inside])
            errors.append(e[inside])
            offcentre.append(np.abs(w[inside].real - 0.5 * (w_lo + w_hi)) / width)
        omega = np.concatenate(omegas)
        amp = np.concatenate(amps)
        err = np.concatenate(errors)
        offcentre = np.concatenate(offcentre)
        source = np.concatenate([np.full(len(o), j) for j, o in enumerate(omegas)])
        keep = _remove_duplicates(omega, source, offcentre, resolution)
        omega, amp, err = omega[keep], amp[keep], err[keep]

    # a real signal yields pairs (omega, -conj(omega)); keep one of each.
    # Purely damped modes (Re(omega) = 0) are their own mirror images; they
    # are found with a tiny real part of either sign, which is set to zero.
    zero_tol = 1e-8 * nyq
    if real:
        selfmirror = np.abs(omega.real) < zero_tol
        omega = np.where(selfmirror, 1j * omega.imag, omega)
        lo = max(lo, 0.0)

    # keep the requested band and physically sensible modes
    keep = (omega.real >= lo) & (omega.real <= hi) & np.isfinite(amp)
    if not allow_growing:
        keep &= omega.imag <= 0
    omega, amp, err = omega[keep], amp[keep], err[keep]

    # a real signal is c_n = sum_k d_k z_k^n + conj(d_k) conj(z_k)^n
    # = Re sum_k 2 d_k z_k^n, with purely damped modes counted once
    selfmirror = (omega.real == 0) if real else np.zeros(len(omega), bool)
    if refine_amplitudes and omega.size:
        amp = _fit_amplitudes(signal_c, dt, omega, real, selfmirror)
    elif real:
        amp = np.where(selfmirror, amp, 2 * amp)

    keep = np.abs(amp) >= min_magnitude
    omega, amp, err = omega[keep], amp[keep], err[keep]

    order = np.argsort(omega.real)
    return HarmInvResult(omega=omega[order], amplitude=amp[order],
                         error=err[order], real=real, t0=t0,
                         info={"method": method, "windows": windows,
                               "tol": tol})
