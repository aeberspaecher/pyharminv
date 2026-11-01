"""Stochastic subspace identification (SSI) of noise-driven systems.

Harmonic inversion needs a signal that *is* a sum of damped exponentials: a
free decay or an impulse response. Many systems can only be observed while
they are driven, for example a bridge in the wind, a machine in operation
or a resonator excited by thermal noise. The measured response is then
noise filtered by the system, not a decay, and the excitation is unknown.
Stochastic subspace identification finds the modes of such a system from
the output alone (*operational* or *output-only* modal analysis).

The model is a linear, time-invariant system driven by unmeasured white
noise, in discrete time ``t_k = k dt``,

.. math::

    x_{k+1} = A x_k + w_k, \\qquad y_k = C x_k + v_k,

with state :math:`x_k` of dimension ``n`` (the *model order*, twice the
number of oscillating modes), ``l`` measured channels :math:`y_k` and
white noise :math:`w_k` (process noise, the excitation) and :math:`v_k`
(measurement noise). The eigenvalues :math:`\\lambda_j` of :math:`A` are the
discrete poles, ``lambda_j = exp(-1j * omega_j * dt)`` in the convention of
:func:`~pyharminv.harminv`, and the columns of :math:`C \\Psi`, with
:math:`\\Psi` the eigenvectors of :math:`A`, are the mode shapes.

The output correlations :math:`R_k = E[y_{m+k} y_m^T]` of such a system
are, for ``k >= 1``,

.. math::

    R_k = C A^{k-1} G, \\qquad G = E[x_{k+1} y_k^T],

a sum of damped exponentials with the same poles as the free decay. Both
algorithms below exploit this:

* :func:`ssi_cov` (covariance-driven SSI) estimates :math:`R_k`, arranges
  them in a block Toeplitz matrix ``O_i Gamma_i`` and reads :math:`A` and
  :math:`C` from its dominant singular subspace.
* :func:`ssi_data` (data-driven SSI) projects the future outputs onto the
  past outputs, which yields the same observability matrix ``O_i`` and in
  addition the Kalman filter states; :math:`A` and :math:`C` follow from
  the states by least squares. No correlations are formed, which is more
  accurate for short records and more robust numerically, at a higher
  cost.

Both return an :class:`SSIResult`, which is a
:class:`~pyharminv.HarmInvResult` of the output correlation function:
``amplitude`` holds the coefficients of the auto-correlation of a reference
channel, so that :meth:`~pyharminv.HarmInvResult.reconstruct` gives the
model correlation function and :meth:`SSIResult.psd` its power spectrum. On
top of that it carries mode shapes and damping ratios.

References
----------
.. [1] P. Van Overschee and B. De Moor, *Subspace Identification for Linear
   Systems*, Kluwer (1996), chapter 3.
.. [2] B. Peeters and G. De Roeck, "Reference-based stochastic subspace
   identification for output-only modal analysis", Mech. Syst. Signal
   Process. 13, 855-878 (1999).
.. [3] B. Peeters and G. De Roeck, "Stochastic system identification for
   operational modal analysis: a review", J. Dyn. Syst. Meas. Control 123,
   659-667 (2001).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import qr, svd

from .core import HarmInvResult

__all__ = ["SSIResult", "Stabilization", "mac", "output_correlation",
           "ssi", "ssi_cov", "ssi_data", "ssi_stabilization"]


# --- results -------------------------------------------------------------------

@dataclass(repr=False)
class SSIResult(HarmInvResult):
    """Modes found by stochastic subspace identification.

    The result describes the output correlation function of the identified
    model, for lags ``tau >= dt``,

    * ``R_ab(tau) = Re sum_k residue[k, a, b] * exp(-1j * omega[k] * tau)``
      between channels ``a`` and ``b``, and
    * ``R_rr(tau) = Re sum_k amplitude[k] * exp(-1j * omega[k] * tau)`` for
      the reference channel ``r = info["ref_channel"]``,

    with the same conventions as :class:`~pyharminv.HarmInvResult` for a
    real signal: only modes with ``Re(omega) >= 0`` are listed, their
    mirror images are implied. All methods of
    :class:`~pyharminv.HarmInvResult` (``select``, ``spectrum``,
    ``lorentzians``, ``reconstruct``, ...) therefore refer to the
    auto-correlation of the reference channel. (At lag zero the measured
    correlation contains the variance of the measurement noise in addition.)

    Attributes
    ----------
    omega : numpy.ndarray
        Complex angular frequencies ``Omega - 1j * Gamma / 2``, sorted by
        real part.
    amplitude : numpy.ndarray
        Coefficients of the auto-correlation of the reference channel.
        Their magnitude measures how strongly each mode contributes to the
        response of that channel; spurious modes usually have small ones.
    error : numpy.ndarray
        Distance of each frequency to the closest pole identified with
        model order ``order + 2`` (``order - 2`` if not available). Poles of
        physical modes are stable under a change of the order; poles that
        model noise move. This is the numerical version of a stabilization
        diagram, see :func:`ssi_stabilization`.
    mode_shape : numpy.ndarray
        Complex mode shapes, shape ``(K, n_channels)``, normalised to a
        largest component of 1.
    residue : numpy.ndarray
        Correlation coefficients between all channels, shape
        ``(K, n_channels, n_channels)``.
    order : int
        Model order (state dimension) of the identification.
    """

    mode_shape: np.ndarray = None
    residue: np.ndarray = None
    order: int = 0

    def __post_init__(self):
        K = len(self.omega)
        if self.mode_shape is None:
            self.mode_shape = np.ones((K, 1), complex)
        if self.residue is None:
            self.residue = self.amplitude.reshape(K, 1, 1).astype(complex)

    @property
    def natural_frequency(self):
        """Undamped natural angular frequencies ``abs(omega)``."""
        return np.abs(self.omega)

    @property
    def damping_ratio(self):
        """Damping ratios ``zeta = -Im(omega) / abs(omega)``.

        For a mode of a damped oscillator ``x'' + 2 zeta w0 x' + w0^2 x``
        this is ``zeta``; ``Q = 1 / (2 zeta)`` for weak damping.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return -self.omega.imag / np.abs(self.omega)

    @property
    def n_channels(self):
        """Number of measured channels."""
        return self.mode_shape.shape[1]

    def __getitem__(self, idx):
        idx = np.atleast_1d(np.arange(len(self))[idx])
        return SSIResult(self.omega[idx], self.amplitude[idx],
                         self.error[idx], self.real, self.t0, dict(self.info),
                         mode_shape=self.mode_shape[idx],
                         residue=self.residue[idx], order=self.order)

    def select(self, min_magnitude=0.0, max_error=np.inf, max_rel_error=np.inf,
               max_width=np.inf, omega_min=-np.inf, omega_max=np.inf,
               min_damping=-np.inf, max_damping=np.inf):
        """Return the subset of modes that satisfy all given criteria.

        As :meth:`pyharminv.HarmInvResult.select`, with bounds on the
        damping ratio in addition. ``min_damping`` and ``max_damping``
        remove the typical spurious poles of SSI: nearly undamped ones
        (often harmonic components of the excitation) and very strongly
        damped ones (noise).
        """
        sub = HarmInvResult.select(self, min_magnitude, max_error,
                                   max_rel_error, max_width, omega_min,
                                   omega_max)
        z = sub.damping_ratio
        return sub[(z >= min_damping) & (z <= max_damping)]

    def correlation(self, tau):
        """Model correlation matrix ``R(tau)``, shape ``(len(tau), l, l)``.

        Valid for ``tau >= dt``; negative lags follow from
        ``R(-tau) = R(tau).T``.
        """
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        e = np.exp(-1j * np.outer(tau, self.omega))
        return np.einsum("tk,kab->tab", e, self.residue).real

    def psd(self, omega, full=False):
        """Power spectral density of the output in the identified model.

        This is the two-sided density
        ``S(omega) = integral R(tau) exp(1j omega tau) dtau`` over all lags,
        per unit angular frequency, built from the modes alone (measurement
        noise, i.e. the lag-zero spike, is not included). To compare with
        :func:`scipy.signal.welch` (one-sided, per Hz) use
        ``S_welch(f) = 2 * 2 pi * S(2 pi f) / (2 pi) = 2 * S(2 pi f)``.

        Parameters
        ----------
        omega : array_like
            Angular frequencies.
        full : bool, optional
            Return the full cross-spectral matrix, shape ``(..., l, l)``,
            instead of the auto-spectrum of the reference channel.
        """
        w = np.asarray(omega, dtype=float)[..., None, None, None]
        M = self.residue
        # one-sided transforms of R(tau) = Re sum M exp(-i omega tau)
        F = (0.5j * M / (w - self.omega[:, None, None])
             + 0.5j * np.conj(M) / (w + np.conj(self.omega)[:, None, None]))
        F = F.sum(axis=-3)
        S = F + np.conj(np.swapaxes(F, -1, -2))
        if full:
            return S
        r = self.info.get("ref_channel", 0)
        return S[..., r, r].real

    def mac(self, other=None):
        """Modal assurance criterion between the mode shapes.

        With no argument, the ``K x K`` auto-MAC matrix of this result;
        otherwise the MAC between these mode shapes and those of ``other``
        (an :class:`SSIResult` or an array of shapes ``(K2, l)``).
        """
        b = self.mode_shape if other is None else getattr(other, "mode_shape", other)
        return mac(self.mode_shape, b)

    def __repr__(self):
        lines = [f"SSIResult with {len(self)} modes, order {self.order}, "
                 f"{self.n_channels} channel(s), method {self.info.get('method', '?')}",
                 f"{'omega.real':>14} {'omega.imag':>14} {'zeta':>10} {'|d|':>11} {'error':>9}"]
        for w, d, e, z in zip(self.omega, self.amplitude, self.error, self.damping_ratio):
            lines.append(f"{w.real:14.7g} {w.imag:14.7g} {z:10.4g} {abs(d):11.4g} {e:9.2g}")
        return "\n".join(lines)


def mac(phi1, phi2=None):
    """Modal assurance criterion ``|phi1^H phi2|^2 / (|phi1|^2 |phi2|^2)``.

    Parameters
    ----------
    phi1, phi2 : array_like
        Mode shapes, one per row (shape ``(K, l)``) or a single vector.

    Returns
    -------
    numpy.ndarray
        MAC values between 0 and 1, shape ``(K1, K2)``.
    """
    a = np.atleast_2d(phi1)
    b = a if phi2 is None else np.atleast_2d(phi2)
    num = np.abs(np.conj(a) @ b.T) ** 2
    den = np.outer(np.sum(np.abs(a) ** 2, 1), np.sum(np.abs(b) ** 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


# --- correlations --------------------------------------------------------------

def _as_channels(y):
    y = np.asarray(y)
    if np.iscomplexobj(y):
        raise ValueError("SSI needs real signals")
    y = y.astype(float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError("signal must have shape (n_samples,) or "
                         "(n_samples, n_channels)")
    if y.shape[0] < y.shape[1]:
        raise ValueError("signal has more channels than samples; time runs "
                         "along the first axis")
    return y


def output_correlation(y, n_lags, unbiased=True):
    """Output correlation matrices ``R_k = E[y_{m+k} y_m^T]``.

    Parameters
    ----------
    y : array_like
        Signal, shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    n_lags : int
        Number of lags ``k = 0, ..., n_lags - 1``.
    unbiased : bool, optional
        Divide by ``N - k`` (default) instead of ``N``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_lags, n_channels, n_channels)``, ``R[k, a, b]`` the
        correlation of channel ``a`` at time ``m + k`` with channel ``b`` at
        time ``m``.
    """
    y = _as_channels(y)
    N, l = y.shape
    if not 1 <= n_lags <= N:
        raise ValueError("need 1 <= n_lags <= n_samples")
    nfft = 1 << int(np.ceil(np.log2(N + n_lags)))
    Y = np.fft.rfft(y, n=nfft, axis=0)
    R = np.empty((n_lags, l, l))
    for b in range(l):
        R[:, :, b] = np.fft.irfft(Y * np.conj(Y[:, b:b + 1]), n=nfft, axis=0)[:n_lags]
    norm = (N - np.arange(n_lags)) if unbiased else np.full(n_lags, N)
    return R / norm[:, None, None]


# --- the two realizations ------------------------------------------------------

class _CovRealization:
    """SSI-Cov: SVD of the block Toeplitz matrix of output correlations."""

    method = "cov"

    def __init__(self, y, i):
        N, l = y.shape
        R = output_correlation(y, 2 * i)
        # block (r, c) = R_{i + r - c} = (C A^r) (A^{i-1-c} G)
        T = np.empty((i * l, i * l))
        for r in range(i):
            for c in range(i):
                T[r * l:(r + 1) * l, c * l:(c + 1) * l] = R[i + r - c]
        U, s, Vh = svd(T)
        self.l, self.i = l, i
        self.U, self.s, self.Vh = U, s, Vh
        self.R0 = R[0]

    @property
    def max_order(self):
        return self.i * self.l - self.l

    def system(self, n):
        l = self.l
        sq = np.sqrt(self.s[:n])
        O = self.U[:, :n] * sq
        C = O[:l]
        A = np.linalg.lstsq(O[:-l], O[l:], rcond=None)[0]
        G = (sq[:, None] * self.Vh[:n])[:, -l:]
        return A, C, G


class _DataRealization:
    """SSI-Data: projection of future onto past outputs, via an LQ factor."""

    method = "data"

    def __init__(self, y, i, chunk=20000):
        N, l = y.shape
        j = N - 2 * i + 1
        if j < 2 * i * l:
            raise ValueError("record too short for this number of block rows")
        # R factor of H^T, H the scaled block Hankel matrix (2 i l x j);
        # accumulated over chunks of columns to bound the memory
        R = None
        for start in range(0, j, chunk):
            stop = min(j, start + chunk)
            Ht = np.hstack([y[start + r:stop + r] for r in range(2 * i)])
            Ht /= np.sqrt(j)
            block = Ht if R is None else np.vstack([R, Ht])
            R = qr(block, mode="r", overwrite_a=True, check_finite=False)[0]
            R = R[:min(R.shape)]
        L = R.T
        p, q = i * l, l
        self.l, self.i, self.p = l, i, p
        # coordinates in the orthonormal row basis Q^T of H = L Q^T
        self.Pi = L[p:, :p]                     # Y_f / Y_p
        self.Pim1 = L[p + q:, :p + q]           # Y_f^- / Y_p^+
        self.Yii = L[p:p + q, :p + q]           # outputs at time i
        U, s, Vh = svd(self.Pi, full_matrices=False)
        self.U, self.s, self.Vh = U, s, Vh

    @property
    def max_order(self):
        return self.i * self.l - self.l

    def system(self, n):
        l, p = self.l, self.p
        sq = np.sqrt(self.s[:n])
        O = self.U[:, :n] * sq
        # Kalman filter states X_i and X_{i+1} in the same coordinates
        Xi = sq[:, None] * self.Vh[:n]                         # n x p
        Xip1 = np.linalg.lstsq(O[:-l], self.Pim1, rcond=None)[0]  # n x (p + l)
        lhs = np.vstack([Xip1[:, :p], self.Yii[:, :p]])
        AC = np.linalg.lstsq(Xi.T, lhs.T, rcond=None)[0].T
        A, C = AC[:n], AC[n:]
        G = Xip1 @ self.Yii.T
        return A, C, G


def _modes(A, C, G, dt):
    """Poles, mode shapes and correlation residues of a realization."""
    lam, Psi = np.linalg.eig(A)
    keep = lam.imag <= 0          # one of each conjugate pair
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.linalg.solve(Psi, np.eye(len(lam)))
    lam, Psi, chi = lam[keep], Psi[:, keep], chi[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        omega = 1j * np.log(lam.astype(complex)) / dt
    selfmirror = lam.imag == 0
    omega = np.where(selfmirror, np.abs(omega.real) + 1j * omega.imag, omega)
    phi = C @ Psi                             # l x K
    g = chi @ G                               # K x l
    with np.errstate(divide="ignore", invalid="ignore"):
        M = phi.T[:, :, None] * g[:, None, :] / lam[:, None, None]
    M = np.where(selfmirror[:, None, None], M.real, 2 * M)
    return omega, phi.T, M


def _match_error(omega, omega_ref):
    if omega_ref.size == 0:
        return np.full(omega.shape, np.inf)
    return np.min(np.abs(omega[:, None] - omega_ref[None, :]), axis=1)


def _normalise_shapes(phi):
    if phi.size == 0:
        return phi
    k = np.argmax(np.abs(phi), axis=1)
    ref = phi[np.arange(len(phi)), k]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = phi / ref[:, None]
    return np.where(np.isfinite(out), out, 0)


def _default_block_rows(N, l, order):
    i = int(np.ceil(order / l)) + 1 if order is not None else 1
    i = max(i, int(np.ceil(64 / l)) + 1)
    return int(max(2, min(i, (N + 1) // (2 * l + 2) // 2)))


def _realization(y, method, block_rows, order):
    y = _as_channels(y)
    N, l = y.shape
    i = _default_block_rows(N, l, order) if block_rows is None else int(block_rows)
    if i < 2 or 2 * i > N:
        raise ValueError("block_rows must be at least 2 and at most n_samples / 2")
    if method in ("cov", "covariance"):
        return _CovRealization(y, i)
    if method in ("data", "direct"):
        return _DataRealization(y, i)
    raise ValueError(f"unknown SSI method {method!r}")


def _auto_order(s, max_order, ratio=2.0):
    """Heuristic model order from the singular values.

    The order is placed after the last drop of the singular values by more
    than a factor ``ratio`` among the first ``max_order``, and rounded up
    to an even number (modes come in conjugate pairs). The noise floor
    decays smoothly and is cut off this way.
    """
    s = s[:max_order + 1]
    s = s[s > s[0] * 1e-14]
    if len(s) < 3:
        return max(1, len(s) - 1)
    drops = np.flatnonzero(s[:-1] / s[1:] >= ratio)
    n = int(drops[-1]) + 1 if drops.size else len(s) - 1
    n += n % 2
    return int(min(max(n, 2), max_order))


def _identify(real_, n, dt, ref_channel, allow_growing, omega_min, omega_max):
    A, C, G = real_.system(n)
    omega, phi, M = _modes(A, C, G, dt)
    alt = n + 2 if n + 2 <= real_.max_order else n - 2
    if alt >= 1:
        omega_alt = _modes(*real_.system(alt), dt)[0]
        error = _match_error(omega, omega_alt)
    else:
        error = np.full(omega.shape, np.nan)
    keep = np.isfinite(omega)
    if not allow_growing:
        keep &= omega.imag <= 0
    if omega_min is not None:
        keep &= omega.real >= omega_min
    if omega_max is not None:
        keep &= omega.real <= omega_max
    omega, phi, M, error = omega[keep], phi[keep], M[keep], error[keep]
    idx = np.argsort(omega.real)
    omega, phi, M, error = omega[idx], phi[idx], M[idx], error[idx]
    l = real_.l
    if not 0 <= ref_channel < l:
        raise ValueError("ref_channel out of range")
    return SSIResult(omega=omega, amplitude=M[:, ref_channel, ref_channel],
                     error=error, real=True, t0=0.0,
                     info={"method": real_.method, "block_rows": real_.i,
                           "singular_values": real_.s, "ref_channel": ref_channel,
                           "A": A, "C": C, "G": G, "R0": getattr(real_, "R0", None)},
                     mode_shape=_normalise_shapes(phi), residue=M, order=n)


# --- public functions ----------------------------------------------------------

def ssi(y, dt, order=None, method="cov", block_rows=None, ref_channel=0,
        omega_min=None, omega_max=None, allow_growing=False):
    """Output-only modal identification by stochastic subspace identification.

    Parameters
    ----------
    y : array_like
        Measured response, shape ``(n_samples,)`` for one channel or
        ``(n_samples, n_channels)``. The system should be driven by
        broadband, ideally white, noise that is not measured. Remove the
        mean (:func:`~pyharminv.remove_mean`) or drifts first.
    dt : float
        Sampling interval.
    order : int, optional
        Model order, the dimension of the state: two per oscillating mode,
        plus some extra to absorb noise and colour of the excitation. A
        moderate over-estimate (e.g. twice the expected number of modes
        times two) is usually best; spurious poles are removed afterwards
        with :meth:`SSIResult.select`. By default the order is placed after
        the last drop of the singular values by more than a factor of two,
        a heuristic; :func:`ssi_stabilization` is the reliable way to
        choose it.
    method : {"cov", "data"}, optional
        ``"cov"`` (covariance-driven, default) works on the estimated output
        correlations and is fast even for very long records. ``"data"``
        (data-driven) projects the raw data and is more accurate for short
        records and closely spaced or weakly excited modes, at a cost that
        grows linearly with the record length times ``(block_rows *
        n_channels)**2``. ``"covariance"`` and ``"direct"`` are accepted as
        aliases.
    block_rows : int, optional
        Number of block rows ``i`` of the Hankel/Toeplitz matrices: lags up
        to ``2 i dt`` are used. ``i * n_channels`` must exceed the order,
        and ``i dt`` should cover at least about one period of the slowest
        mode of interest. Default: enough for ``order`` and at least 64
        rows in total.
    ref_channel : int, optional
        Channel whose auto-correlation defines ``amplitude``.
    omega_min, omega_max : float, optional
        Keep only modes with ``Re(omega)`` in this range.
    allow_growing : bool, optional
        Keep unstable poles (``Im(omega) > 0``), which are artefacts.

    Returns
    -------
    SSIResult
        See there for the conventions. ``info`` contains the singular
        values, the matrices ``A``, ``C`` and ``G`` and the number of block
        rows.

    See Also
    --------
    ssi_cov, ssi_data, ssi_stabilization
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    real_ = _realization(y, method, block_rows, order)
    if order is None:
        order = _auto_order(real_.s, real_.max_order)
    order = int(order)
    if not 1 <= order <= real_.max_order:
        raise ValueError(f"order must be between 1 and {real_.max_order} "
                         "(block_rows * n_channels - n_channels); increase "
                         "block_rows")
    return _identify(real_, order, dt, ref_channel, allow_growing,
                     omega_min, omega_max)


def ssi_cov(y, dt, order=None, **kwargs):
    """Covariance-driven SSI, ``ssi(y, dt, order, method="cov", ...)``.

    The output correlations ``R_1 ... R_{2i-1}`` are estimated (by FFT) and
    arranged in a block Toeplitz matrix ``T = O_i Gamma_i``, the product of
    the observability and the reversed controllability matrix. Its rank-
    ``order`` truncated SVD gives ``O_i``; ``C`` is its first block row and
    ``A`` follows from the shift structure ``O_i[1:] = O_i[:-1] A``.
    This is the matrix pencil method applied to the correlation function,
    the natural excitation technique (NExT) carried out in one step.

    See :func:`ssi` for the parameters.
    """
    return ssi(y, dt, order, method="cov", **kwargs)


def ssi_data(y, dt, order=None, **kwargs):
    """Data-driven SSI, ``ssi(y, dt, order, method="data", ...)``.

    The block Hankel matrix of the outputs is split into past ``Y_p`` and
    future ``Y_f``. The orthogonal projection ``Y_f / Y_p``, computed from
    an LQ factorisation, equals ``O_i X_i`` with ``X_i`` the states of a
    bank of Kalman filters. Its truncated SVD gives ``O_i`` and ``X_i``;
    the same is done for ``X_{i+1}``, and ``[A; C]`` is the least-squares
    solution of ``[X_{i+1}; Y_i] = [A; C] X_i`` (the unweighted principal
    component algorithm of Van Overschee and De Moor).

    See :func:`ssi` for the parameters.
    """
    return ssi(y, dt, order, method="data", **kwargs)


@dataclass
class Stabilization:
    """Poles over a range of model orders, as in a stabilization diagram.

    A pole at order ``n`` is *stable* in frequency, damping or shape if a
    pole at the next lower order in the list lies within the tolerances.
    Physical modes appear as vertical lines of stable poles, spurious poles
    scatter.

    Attributes
    ----------
    orders : list of int
    results : list of SSIResult
        One identification per order (no filtering by damping).
    stable_frequency, stable_damping, stable_shape : list of numpy.ndarray
        Boolean flags per pole of each result.
    """

    orders: list
    results: list
    stable_frequency: list
    stable_damping: list
    stable_shape: list
    info: dict = field(default_factory=dict, repr=False)

    def stable(self, k):
        """Flags of the poles at ``orders[k]`` stable in all respects."""
        return self.stable_frequency[k] & self.stable_damping[k] & self.stable_shape[k]

    def poles(self, stable_only=False):
        """All poles as arrays ``(order, omega, damping_ratio, stable)``."""
        o, w, z, s = [], [], [], []
        for k, (n, r) in enumerate(zip(self.orders, self.results)):
            st = self.stable(k)
            if stable_only:
                r = r[st]
                st = st[st]
            o.append(np.full(len(r), n))
            w.append(r.omega)
            z.append(r.damping_ratio)
            s.append(st)
        cat = np.concatenate
        return cat(o), cat(w), cat(z), cat(s)

    def plot(self, ax=None, frequency_unit="angular", **kwargs):
        """Draw the stabilization diagram with matplotlib.

        Stable poles are drawn as filled circles, poles stable in frequency
        only as small crosses, the rest as dots. ``frequency_unit`` is
        ``"angular"`` (``Re omega``) or ``"Hz"`` (``Re omega / 2 pi``).
        Returns the axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        scale = 1 / (2 * np.pi) if frequency_unit == "Hz" else 1.0
        for k, (n, r) in enumerate(zip(self.orders, self.results)):
            f = r.omega.real * scale
            st = self.stable(k)
            sf = self.stable_frequency[k] & ~st
            rest = ~(st | sf)
            ax.plot(f[rest], np.full(rest.sum(), n), ".", color="0.75", ms=3)
            ax.plot(f[sf], np.full(sf.sum(), n), "x", color="C1", ms=4)
            ax.plot(f[st], np.full(st.sum(), n), "o", color="C0", ms=4, **kwargs)
        ax.set_xlabel("frequency / Hz" if frequency_unit == "Hz" else "Re ω")
        ax.set_ylabel("model order")
        return ax


def ssi_stabilization(y, dt, orders, method="cov", block_rows=None,
                      freq_tol=0.01, damping_tol=0.05, mac_tol=0.98,
                      ref_channel=0, omega_min=None, omega_max=None):
    """Identify the modes for a range of model orders.

    All orders share one SVD, so the whole diagram costs little more than a
    single identification.

    Parameters
    ----------
    y, dt, method, block_rows, ref_channel, omega_min, omega_max
        As in :func:`ssi`. ``block_rows`` defaults to enough rows for the
        largest order.
    orders : iterable of int
        Model orders, e.g. ``range(2, 61, 2)``.
    freq_tol : float, optional
        Relative tolerance of ``abs(omega)`` for a stable pole.
    damping_tol : float, optional
        Relative tolerance of the damping ratio.
    mac_tol : float, optional
        Smallest MAC between the mode shapes (ignored for one channel).

    Returns
    -------
    Stabilization
    """
    orders = sorted(int(n) for n in orders)
    real_ = _realization(y, method, block_rows, max(orders))
    if orders[-1] > real_.max_order or orders[0] < 1:
        raise ValueError(f"orders must lie between 1 and {real_.max_order}; "
                         "increase block_rows")
    results, sf, sd, ss = [], [], [], []
    prev = None
    for n in orders:
        r = _identify(real_, n, dt, ref_channel, False, omega_min, omega_max)
        K = len(r)
        f_ok = np.zeros(K, bool)
        d_ok = np.zeros(K, bool)
        m_ok = np.zeros(K, bool)
        if prev is not None and len(prev) and K:
            wn, wp = r.natural_frequency, prev.natural_frequency
            rel = np.abs(wn[:, None] - wp[None, :]) / wn[:, None]
            j = np.argmin(rel, axis=1)
            f_ok = rel[np.arange(K), j] <= freq_tol
            zn, zp = r.damping_ratio, prev.damping_ratio[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                d_ok = np.abs(zn - zp) <= damping_tol * np.abs(zn)
            if r.n_channels > 1:
                m_ok = np.diag(mac(r.mode_shape, prev.mode_shape[j])) >= mac_tol
            else:
                m_ok = np.ones(K, bool)
        results.append(r)
        sf.append(f_ok)
        sd.append(d_ok & f_ok)
        ss.append(m_ok & f_ok)
        prev = r
    return Stabilization(orders, results, sf, sd, ss,
                         info={"method": real_.method, "block_rows": real_.i,
                               "singular_values": real_.s})
