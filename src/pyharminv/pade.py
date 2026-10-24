"""Robust Padé approximation via the SVD.

This module is a Python translation of ``padeapprox`` from the MATLAB package
Chebfun (http://www.chebfun.org), which implements the algorithm of
Gonnet, Güttel and Trefethen [1]_. The algorithm computes a type (m, n) Padé
approximant ``r = p/q`` of a function given by its Taylor coefficients. By
reducing the numerical rank of the underlying Toeplitz system it removes
spurious pole-zero pairs ("Froissart doublets"), which is what makes it
suitable for harmonic inversion of noisy signals.

Chebfun is Copyright (c) 2017 by The University of Oxford and The Chebfun
Developers and distributed under the BSD 3-clause license, see
``LICENSES/chebfun-BSD-3-Clause.txt``.

References
----------
.. [1] P. Gonnet, S. Güttel and L. N. Trefethen, "Robust Padé approximation
   via SVD", SIAM Review 55, 101-117 (2013).
.. [2] L. N. Trefethen, "Approximation Theory and Approximation Practice",
   SIAM (2013), chapter 27.
"""

from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm, qr, svd
from scipy.linalg import toeplitz

__all__ = ["PadeResult", "padeapprox", "taylor_coeffs"]


def taylor_coeffs(f, n=2048):
    """Compute Taylor coefficients of ``f`` about ``x = 0`` via the FFT.

    ``f`` is sampled at ``n`` roots of unity, so it must be analytic in a
    neighbourhood of the closed unit disc.

    Parameters
    ----------
    f : callable
        Vectorised function of a complex argument.
    n : int, optional
        Number of sample points, and thus of returned coefficients.

    Returns
    -------
    numpy.ndarray
        Coefficients ``c_0, ..., c_{n-1}``. Real-valued if ``f`` is real on
        the real axis up to rounding errors.
    """
    z = np.exp(2j * np.pi * np.arange(n) / n)
    coeffs = np.fft.fft(f(z)) / n

    # discard near-zero coefficients
    tc = 1e-15 * norm(coeffs)
    coeffs[np.abs(coeffs) < tc] = 0.0

    # remove imaginary rounding errors (make real functions real)
    if norm(coeffs.imag, np.inf) < tc:
        coeffs = coeffs.real

    return coeffs


@dataclass
class PadeResult:
    """Result of :func:`padeapprox`.

    Attributes
    ----------
    a, b : numpy.ndarray
        Coefficients of numerator ``p`` and denominator ``q`` in *ascending*
        powers of ``x``, normalised to ``b[0] = 1``.
    mu, nu : int
        Exact degrees of ``p`` and ``q``. They can be smaller than the
        requested ``m, n`` because of the rank reduction. ``mu`` is ``-inf``
        if ``r = 0``.
    poles, residues, zeros : numpy.ndarray
        Roots of ``q``, residues of ``r`` at those roots, and roots of ``p``.
    """

    a: np.ndarray
    b: np.ndarray
    mu: int
    nu: int
    poles: np.ndarray
    residues: np.ndarray
    zeros: np.ndarray

    def __call__(self, x):
        """Evaluate the approximant ``r(x) = p(x)/q(x)``."""
        x = np.asarray(x)
        return np.polyval(self.a[::-1], x) / np.polyval(self.b[::-1], x)


def padeapprox(f, m, n, tol=1e-14, residues="analytic"):
    """Compute a robust type (m, n) Padé approximant.

    Parameters
    ----------
    f : array_like or callable
        Taylor coefficients ``f_0, f_1, ...`` of the function about ``x = 0``,
        or a callable whose coefficients are then computed with
        :func:`taylor_coeffs`. Missing coefficients are taken as zero.
    m, n : int
        Requested degrees of numerator and denominator.
    tol : float, optional
        Relative tolerance for the rank decision. Singular values below
        ``tol * norm(c)`` are treated as zero, which lowers the degrees and
        removes spurious poles. ``tol = 0`` switches the robustness off. The
        default suits data that is exact up to rounding; noisy data needs a
        tolerance near its relative noise level.
    residues : {"analytic", "finite-difference"}, optional
        How to compute residues. ``"analytic"`` uses ``p(x_k) / q'(x_k)``,
        which assumes simple poles. ``"finite-difference"`` reproduces the
        estimate of the Chebfun original.

    Returns
    -------
    PadeResult
    """
    if callable(f):
        f = taylor_coeffs(f)

    f = np.asarray(f).ravel()
    if f.dtype == object:
        # e.g. Python integers too large for int64
        f = f.astype(complex if any(isinstance(v, complex) for v in f) else float)
    c = np.zeros(m + n + 1, dtype=np.result_type(f, float))
    k = min(len(f), m + n + 1)
    c[:k] = f[:k]

    ts = tol * norm(c)

    if norm(c[:m + 1], np.inf) <= tol * norm(c, np.inf):
        # special case r = 0
        a = np.zeros(1, dtype=c.dtype)
        b = np.ones(1, dtype=c.dtype)
        mu, nu = -np.inf, 0
    else:
        row = np.concatenate([c[:1], np.zeros(n, dtype=c.dtype)])
        col = c

        # diagonal hopping across a block of the Padé table
        while True:
            if n == 0:
                a = c[:m + 1].copy()
                b = np.ones(1, dtype=c.dtype)
                break

            Z = toeplitz(col[:m + n + 1], row[:n + 1])
            C = Z[m + 1:m + n + 1, :]
            rho = np.sum(svd(C, compute_uv=False) > ts)
            if rho == n:
                break

            # rank deficient: reduce degrees
            m -= n - rho
            n = rho

        if n > 0:
            # the right singular vector of the smallest singular value spans
            # the null space of C and gives b
            _, _, Vh = svd(C)
            b = np.conj(Vh[-1, :])

            # reweighted QR for better zero preservation
            eps = np.finfo(float).eps
            D = np.diag(np.abs(b) + np.sqrt(eps))
            Q, _ = qr(np.conj((C @ D).T), mode="complete")
            b = D @ Q[:, n]
            b = b / norm(b)

            a = Z[:m + 1, :n + 1] @ b

            # discard leading zeros of b (and a)
            lam = np.flatnonzero(np.abs(b) > tol)[0]
            b = b[lam:]
            a = a[lam:]

            # discard trailing zeros of b
            b = b[:np.flatnonzero(np.abs(b) > tol)[-1] + 1]

        # discard trailing zeros of a
        nonzero = np.flatnonzero(np.abs(a) > ts)
        a = a[:nonzero[-1] + 1] if nonzero.size else a[:1] * 0

        a = a / b[0]
        b = b / b[0]

        # remove imaginary rounding errors
        if np.iscomplexobj(a) and norm(a.imag, np.inf) <= ts:
            a = a.real
        if np.iscomplexobj(b) and norm(b.imag, np.inf) <= ts:
            b = b.real

        mu = len(a) - 1
        nu = len(b) - 1

    p = np.poly1d(a[::-1])
    q = np.poly1d(b[::-1])
    poles = q.roots if nu > 0 else np.zeros(0, dtype=complex)
    zeros = p.roots if len(a) > 1 else np.zeros(0, dtype=complex)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        res = _residues(p, q, poles, nu, tol, residues)

    return PadeResult(a=np.asarray(a), b=np.asarray(b), mu=mu, nu=nu,
                      poles=np.asarray(poles), residues=np.asarray(res),
                      zeros=np.asarray(zeros))


def _residues(p, q, poles, nu, tol, residues):
    if residues == "analytic":
        res = p(poles) / np.polyder(q)(poles) if nu > 0 else poles.copy()
    elif residues == "finite-difference":
        t = max(tol, 1e-7)
        r = lambda x: p(x) / q(x)
        res = t * (r(poles + t) - r(poles - t)) / 2
    else:
        raise ValueError(f"unknown residue method {residues!r}")
    return res
