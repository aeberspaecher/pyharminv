"""Filter diagonalization for harmonic inversion in a frequency window.

The filter diagonalization method (FDM) of Wall and Neuhauser [1]_ in the
form of Mandelshtam and Taylor [2]_ solves the harmonic inversion problem
locally: the signal operator is projected onto a small basis of Fourier
functions with frequencies inside a window. The projected matrices can be
computed in closed form from the *whole* signal, so the frequency window does
not cost any of the information at the start of the signal, which matters for
broad resonances.

References
----------
.. [1] M. R. Wall and D. Neuhauser, "Extraction, through filter-
   diagonalization, of general quantum eigenvalues or classical normal mode
   frequencies from a small number of residues or a short-time segment of a
   signal", J. Chem. Phys. 102, 8011 (1995).
.. [2] V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time
   signals and its applications", J. Chem. Phys. 107, 6756 (1997).
"""

import numpy as np
from scipy.linalg import eig, svd
from scipy.signal import czt

__all__ = ["fdm_matrices", "fdm_window"]


class _FDMGrid:
    """Signal transforms on an equidistant grid of basis frequencies.

    Holds, for ``p = 0, 1, 2``, the polynomials needed by the closed form of
    the FDM matrices on the whole grid, so that the matrices of any window
    (a contiguous slice of the grid) are cheap to assemble.
    """

    def __init__(self, c, dt, phi0, dphi, J, M):
        c = np.asarray(c, dtype=complex)
        if len(c) < 2 * M + 3:
            raise ValueError("need at least 2M + 3 samples")
        self.dt = dt
        self.M = M
        self.phi = phi0 + dphi * np.arange(J)
        self.a = np.exp(1j * self.phi * dt)
        self.aM1 = self.a ** (M + 1)
        n_s = np.arange(2 * M + 1)
        # multiplicity of the index s = n + m in the double sum
        count = np.minimum(n_s, 2 * M - n_s) + 1
        # chirp z-transform: X_j = sum_s x_s z_j^(-s), z_j = A W^(-j), here
        # with z_j = 1/a_j
        w = np.exp(1j * dphi * dt)
        A = np.exp(-1j * phi0 * dt)
        transform = lambda x: czt(x, m=J, w=w, a=A)
        self.F, self.G, self.diag = {}, {}, {}
        for p in (0, 1, 2):
            cp = c[p:p + 2 * M + 1]
            self.F[p] = transform(cp[:M + 1])
            # G(x) = sum_{s=M+1}^{2M} c_s x^(s-M)
            self.G[p] = self.a * transform(cp[M + 1:])
            self.diag[p] = transform(cp * count)

    def matrices(self, sl=slice(None)):
        """FDM matrices ``U^(p)`` for the grid points ``sl``."""
        a = self.a[sl]
        aM1 = self.aM1[sl]
        A = a[:, None]
        B = a[None, :]
        U = {}
        for p in (0, 1, 2):
            F = self.F[p][sl]
            G = self.G[p][sl]
            with np.errstate(divide="ignore", invalid="ignore"):
                num = (B * F[None, :] - A * F[:, None]
                       + aM1[None, :] * G[:, None] - aM1[:, None] * G[None, :])
                Up = num / (B - A)
            Up[np.diag_indices(len(a))] = self.diag[p][sl]
            U[p] = Up
        return U, self.F[0][sl]

    def prepare(self, sl=slice(None)):
        """Matrices and SVD of ``U^(0)`` for the grid points ``sl``."""
        U, F0 = self.matrices(sl)
        W, s, Vh = svd(U[0])
        return {"U": U, "F0": F0, "W": W, "s": s, "Vh": Vh, "dt": self.dt}


def fdm_matrices(c, dt, phi, M):
    """Matrices ``U^(p)``, ``p = 0, 1, 2``, of the filter diagonalization method.

    ``U^(p)_{jl} = sum_{n,m=0}^{M} a_j^n a_l^m c_{n+m+p}`` with
    ``a_j = exp(i phi_j dt)``, evaluated in closed form.

    Parameters
    ----------
    c : array_like
        Signal samples; at least ``2M + 3`` are needed.
    dt : float
        Sampling interval.
    phi : numpy.ndarray
        Equidistant basis frequencies.
    M : int
        Basis length parameter.

    Returns
    -------
    U : dict
        ``U[p]`` for ``p = 0, 1, 2``.
    F0 : numpy.ndarray
        ``sum_{n=0}^{M} c_n a_j^n``, needed for the amplitudes.
    """
    phi = np.asarray(phi, dtype=float)
    dphi = phi[1] - phi[0] if len(phi) > 1 else 1.0
    return _FDMGrid(c, dt, phi[0], dphi, len(phi), M).matrices()


def _fdm_prepare(c, dt, omega_lo, omega_hi, n_basis, density):
    M = (len(c) - 3) // 2
    if M < 1:
        raise ValueError("signal too short for filter diagonalization")
    resolution = 2 * np.pi / ((M + 1) * dt)
    if n_basis is None:
        n_basis = int(np.ceil(density * (omega_hi - omega_lo) / resolution))
    J = max(2, int(n_basis))
    step = (omega_hi - omega_lo) / J
    return _FDMGrid(c, dt, omega_lo + step / 2, step, J, M).prepare()


def _fdm_solve(prep, threshold):
    U, F0, W, s, Vh, dt = (prep[k] for k in ("U", "F0", "W", "s", "Vh", "dt"))
    r = int(np.sum(s > threshold))
    if r == 0:
        return np.zeros(0, complex), np.zeros(0, complex), np.zeros(0)
    # restrict the generalized eigenproblem U1 B = u U0 B to the
    # numerically significant subspace of U0
    P = Vh[:r].conj().T
    Q = W[:, :r]
    A1 = (Q.conj().T @ U[1] @ P) / s[:r, None]
    u, y = eig(A1)
    B = P @ y

    # complex-symmetric normalisation B^T U0 B = 1
    norm = np.einsum("jk,jl,lk->k", B, U[0], B)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (B.T @ F0) ** 2 / norm
        omega = 1j * np.log(u) / dt
        # error estimate: U2 must have the eigenvalues u^2 on the same vectors
        u2 = np.einsum("jk,jl,lk->k", B, U[2], B) / norm
        error = np.abs(u2 - u ** 2) / (2 * np.abs(u) * dt)
    ok = np.isfinite(omega) & np.isfinite(d)
    return omega[ok], d[ok], error[ok]


def fdm_window(c, dt, omega_lo, omega_hi, n_basis=None, density=1.1,
               tol=1e-10, threshold=None):
    """Harmonic inversion in a frequency window with filter diagonalization.

    Parameters
    ----------
    c : array_like
        Complex signal samples ``c_n = sum_k d_k exp(-i omega_k n dt)``.
    dt : float
        Sampling interval.
    omega_lo, omega_hi : float
        Frequency range covered by the basis. Modes are returned for the
        whole range (and occasionally outside it) and should be filtered by
        the caller; those near the edges are less accurate.
    n_basis : int, optional
        Number of basis functions. Defaults to ``density`` times the number
        of Fourier resolution cells ``2 pi / (M dt)`` in the range, with
        ``M = (len(c) - 3) // 2``.
    density : float, optional
        Basis density relative to the Fourier resolution.
    tol : float, optional
        Singular values of ``U^(0)`` below ``tol`` times the largest one are
        discarded. Choose it near the relative noise level of the signal.
    threshold : float, optional
        Absolute singular value threshold, overrides ``tol``.

    Returns
    -------
    omega, amplitude, error : numpy.ndarray
        Complex frequencies, amplitudes and error estimates (the mismatch
        between the eigenvalues of ``U^(1)`` and ``U^(2)``).
    """
    prep = _fdm_prepare(c, dt, omega_lo, omega_hi, n_basis, density)
    if threshold is None:
        threshold = tol * prep["s"][0]
    return _fdm_solve(prep, threshold)
