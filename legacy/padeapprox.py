#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Compute Pade approximant using a robust SVD based algorithm.

This is an almost literal translation to Python from the MATLAB routine
padeapprox from the MATLAB package chebfun. Check http://www.chebfun.org for
more information, also compare the original paper [1] and the great book [2].

References

[1] P. Gonnet, S. Guettel, and L. N. Trefethen, "ROBUST PADE APPROXIMATION
       VIA SVD", SIAM Rev., 55:101-117, 2013.
[2] L. N. Trefethen, "Approximation Theory and Approximation Practice", SIAM
"""

import numpy as np
from numpy.fft import fft
from numpy.linalg import norm, qr, svd
from scipy.linalg import toeplitz, svdvals


def taylor_coeffs(f, N):
    """Compute N Taylor series coefficients for function f.
    """

    # sample at many root of unity and use FFT to obtain coefficients
    z = np.exp(2j*np.pi*np.arange(0, N)/N)
    coeffs = fft(f(z))/N  # there aren't any fftshifts/ifftshifts because the
                          # roots of unity are chosen in the angle interval
                          # [0; 2pi). this is clever!

    # discard near-zero coefficients:
    tc = 1E-15*norm(coeffs)
    coeffs[np.abs(coeffs) < tc] = 0.0

    # remove imaginary rounding errors:
    if norm(np.imag(coeffs), np.inf) < tc:
        coeffs = np.real(coeffs)

    return coeffs


def padeapprox(f, M, N, tol=1E-14, assume_isolated_poles=True):
    """Compute Pade approximant for function.

    Parameters
    ----------
    f : array or callable
        If f is callable, f(x) shall be able to be called for vector-valued x.
        If f is an array, it is considered to be the Taylor expansion about x=0 of a function f.
    M, N : ints
        Allowed degrees of the p, q polynomial of the approximant r(x) = p(x)/q(x).
    tol : number, otional
        Discard expansion coefficients smaller in magnitude than tol. This
        prevents spurious resonances to some extent. The default value of 1E-14
        is suitable for data plagued only by roundoff noise. Noisy data may
        work better with smaller tol.
    assume_isolated_poles : boolean, optional
        If isolated / first order poles can be assumed for r the residuals of r
        can be computed from an analytic formula, res f(z_n) = p(z_n)/q'(z_n).
        Defaults to False, where a finite difference calculation is made (same
        as chebfun's padeapprox()).

    Returns
    -------
    a, b : arrays
        Coefficients of p, q polynomials. Ordered by descending powers.
    mu, nu : ints
        Polynomial degrees finally used. These numbers may be smaller than M, N
        due to the robust algorithm.
    r : np.poly1d object
       The (M, N) Pade approximant to function f.
    zeros, poles : arrays
       Zeros and poles of the Pade approximant, i.e. roots of the p, q polynomials.
    residues : numbers
       Residues at the poles.
    """

    eps = np.finfo(float).eps

    if callable(f):
        f = taylor_coeffs(f, N=2048)

    # make sure coefficient vector is long enough but not longer than
    # necessary:
    zeros_length = M+N+1-len(f)
    zeros_length = max(zeros_length, 0)
    coeffs = list(f) + list(np.zeros(zeros_length)) + [1]
    coeffs = coeffs[0:M+N+1]

    tc = 1E-15*norm(coeffs)

    # compute absolute tolerance
    ts = tol*norm(coeffs)

    # compute Pade approximant:
    if(norm(coeffs[:M+1], np.inf) <= tol*norm(coeffs, np.inf)):  # special case
                                                                 # r=0
        a = np.array([0.])
        b = np.array([1.])
        mu  = -np.inf
        nu = 0
    else:  # general case
        # first row/column of Toeplitz matrix
        row = np.hstack([coeffs[0], np.zeros(N)])
        col = np.array(coeffs)

        # do diagonal hopping across block
        while(True):
            # special case N == 0:
            if N == 0:
                a = coeffs[0:M+1]
                b = np.array([1.])
                break

            # form Toeplitz matrix
            Z = toeplitz(col[:M+N+1], row[:N+1])

            # compute numerical rank
            C = Z[M+1:M+N+1, :]
            rho = np.sum(svdvals(C) > ts)

            if rho == N:
                break

            # decrease m, n if rank-deficient
            M = M - (N - rho)
            N = rho

        # hopping finished; now compute b and a
        if N > 0:
            U, S, V = svd(C, full_matrices=True)  # Matlab: svd(C, 0)

            # null vector gives b
            b = V[:,  N]

            # do final computation via re-weighted QR for better zero
            # preservation
            D = np.diag(np.abs(b) + np.sqrt(eps))
            Q, R = qr(np.conj(np.dot(C, D).T), "complete")
            # compensate for re-weighting
            b = np.dot(D, Q[:, N])
            b /= norm(b)

            # multiplying gives a
            a = np.dot(Z[:M+1, :N+1], b)

            # count leading zeros of b
            lam = np.nonzero(np.abs(b) > tol)[0][0]  # note double [0][0]!

            # discard leading zeros of b and a, keep array data type (thus np.take):
            b = np.take(b, np.arange(lam, len(b)))
            a = np.take(a, np.arange(lam, len(a)))

            # discard trailing zeros of b
            b = b[:np.nonzero(np.abs(b) > tol)[0][-1] + 1]

        # discard trailing zeros coefficients in a:
        a = a[:np.nonzero(np.abs(a) > ts)[0][-1] +1]

        # normalize
        a = a/b[0]
        b = b/b[0]

        # remove imaginary rounding error:
        if(norm(np.imag(a), np.inf) < ts):  # TODO: which norm to use here and below?
            a = np.real(a)

        if(norm(np.imag(b), np.inf) < ts):
            b = np.real(b)

        # exact numerator, denominator degrees:
        mu = len(a) - 1
        nu = len(b) - 1

        p = np.poly1d(a[::-1])
        q = np.poly1d(b[::-1])

        r = lambda x: p(x)/q(x)

        poles = q.roots
        zeros = p.roots
        t = max(tol, 1E-7)
        # print("Res paper:", residues)

        # assume isolated poles?
        if assume_isolated_poles:
            q_prime = np.polyder(q)
            residues = p(poles)/q_prime(poles)
        else:
            residues = t*(r(poles+t) - r(poles-t))/2.  # TODO: improve calculation using the formula in my Master thesis
            # print(residues)

        return a[::-1], b[::-1], mu, nu, r, zeros, poles, residues


if __name__ == "__main__":
    #f = lambda x: 3*x**3 + 2*x**2 + x + 0.1
    #f = lambda x: np.sin(x**2)
    # f = np.hstack([[1.0, 1], np.arange(1, 51)])
    #f = lambda x: np.tan(x)
    # f = lambda x: np.tan(x**4)
    #f = lambda x: np.sin(x)/x

    # print(taylor_coeffs(f, 7))
    M, N = 12, 12
    a, b, mu, nu, r, zeros, poles, res = padeapprox(f, M, N, tol=1E-12)
    print(a)
    print(b)
    print(mu)
    print(nu)
    print(r)
    print(zeros)
    print(poles)
    print(res)

    x = np.linspace(-15, +15 , 1024)
    y_orig = f(x)
    y_pade = r(x)

    import matplotlib.pyplot as plt

    plt.plot(x, y_orig, label="orig")
    plt.plot(x, y_pade, label=r"({}, {}) Pade; $(\mu, \nu) = ({}, {})$".format(M, N, mu, nu))
    # plt.plot(x, np.polyval(taylor_coeffs(f, M+N+1), x), label="Taylor order {}".format(M+N+1))
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()

    plt.scatter(np.real(zeros), np.imag(zeros), marker="o", color="blue", facecolors="none", label="zeros")
    plt.scatter(np.real(poles), np.imag(poles), marker="x", color="red" , label="poles")
    plt.title(r"({}, {}) Pade; $(\mu, \nu) = ({}, {})$".format(M, N, mu, nu))
    plt.legend(loc="best")
    plt.show()
