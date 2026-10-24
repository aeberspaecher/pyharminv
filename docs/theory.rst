Theory
======

The harmonic inversion problem
------------------------------

Given samples :math:`c_n = c(t_0 + n\,\Delta t)`, :math:`n = 0, \dots, N-1`,
find :math:`K` complex frequencies :math:`\omega_k` and amplitudes
:math:`d_k` with

.. math::

   c_n = \sum_{k=1}^{K} d_k z_k^n, \qquad z_k = e^{-i\omega_k \Delta t}.

This nonlinear problem is equivalent to fitting the spectrum with a sum of
complex Lorentzians :math:`i d_k / (\omega - \omega_k)`, because that is the
Fourier transform of each term. Unlike the discrete Fourier transform,
the :math:`\omega_k` are not bound to a grid, so resonances closer than the
Fourier resolution :math:`2\pi / (N \Delta t)` can be separated
[MandelshtamTaylor1997]_.

Padé approximants
-----------------

Following [WiersigMain2008]_, interpret the samples as Taylor coefficients
of

.. math::

   g(x) = \sum_{n=0}^{\infty} c_n x^n
        = \sum_{k} \frac{d_k}{1 - z_k x},

a rational function of type :math:`(K-1, K)`. Its Padé approximant from the
first :math:`2K` samples therefore reproduces it exactly for noise-free data.
The poles are :math:`x_k = 1/z_k`, so

.. math::

   \omega_k = -\frac{i}{\Delta t} \ln x_k, \qquad
   d_k = -\frac{\operatorname{Res}_{x_k} g}{x_k}.

(Wiersig and Main expand in :math:`z^{-1}` instead of :math:`x`, which
flips the signs in their formulas; the earlier version of this code mixed the
two conventions.)

Plain Padé approximation is unstable: noise and rounding create spurious
pole-zero pairs ("Froissart doublets"). The algorithm of Gonnet, Güttel and
Trefethen [GGT2013]_ computes the denominator from the null space of a
Toeplitz matrix and lowers the degree when singular values fall below the
tolerance ``tol``, which removes these doublets.

The classical approximant uses only the first 2K samples. The least-squares
variant (the matrix pencil method [HuaSarkar1990]_) instead builds a Hankel
matrix from all N samples, truncates its SVD to the numerical rank
:math:`\nu` (singular values above ``tol`` times the largest), and obtains
the :math:`z_k` as eigenvalues of the shifted, truncated singular vectors.
The amplitudes then follow from a linear least-squares fit to all samples.
This averages the noise over the whole record.

Nonlinear refinement
--------------------

Both solvers give frequencies near the best fit, but not the best fit
itself. :meth:`~pyharminv.HarmInvResult.refine` minimises
:math:`\sum_n |c_n - \sum_k d_k e^{-i\omega_k t_n}|^2` over all
:math:`\omega_k, d_k` with :func:`scipy.optimize.least_squares`, starting from
the harmonic inversion result. For white Gaussian noise this is the
maximum-likelihood estimator, which attains the Cramér-Rao bound; the
covariance follows from the Jacobian and the residual.

Filter diagonalization
----------------------

For long signals the number of poles, and thus the size of the Padé problem,
becomes large. Filter diagonalization [WallNeuhauser1995]_,
[MandelshtamTaylor1997]_ solves the problem locally in frequency. Write
:math:`c_n = (\Phi_0, \hat U^n \Phi_0)` with an evolution operator
:math:`\hat U` whose eigenvalues are the :math:`z_k`, and project
:math:`\hat U` onto Fourier vectors with frequencies :math:`\varphi_j` in a
window. The matrix elements

.. math::

   U^{(p)}_{jl} = \sum_{n,m=0}^{M} a_j^{n} a_l^{m} c_{n+m+p}, \qquad
   a_j = e^{i\varphi_j \Delta t}, \quad M = \lfloor (N-3)/2 \rfloor,

follow in closed form from sums over the signal, evaluated for all basis
frequencies at once with a chirp z-transform. The generalized eigenvalue
problem

.. math::

   U^{(1)} B_k = z_k U^{(0)} B_k, \qquad
   d_k = \frac{\left(\sum_j B_{jk} \sum_{n=0}^{M} c_n a_j^n\right)^2}
              {B_k^T U^{(0)} B_k},

yields the modes in the window. Small singular values of :math:`U^{(0)}`
(below ``tol`` times the largest over all windows) are projected out, which
plays the same role as the rank reduction of the robust Padé algorithm. The
error estimate compares :math:`z_k^2` with the Rayleigh quotient of
:math:`U^{(2)}`.

Each window's basis extends into guard bands on both sides; modes are kept
only in the window proper, and duplicates found by two windows are merged.
Finally all amplitudes are refitted jointly by linear least squares on the
full signal with the frequencies held fixed.

References
----------

.. [WiersigMain2008] J. Wiersig and J. Main, "Fractal Weyl law for chaotic
   microcavities: Fresnel's laws imply multifractal scattering",
   Phys. Rev. E 77, 036205 (2008).
.. [GGT2013] P. Gonnet, S. Güttel and L. N. Trefethen, "Robust Padé
   approximation via SVD", SIAM Review 55, 101-117 (2013).
.. [HuaSarkar1990] Y. Hua and T. K. Sarkar, "Matrix pencil method for
   estimating parameters of exponentially damped/undamped sinusoids in
   noise", IEEE Trans. Acoust. Speech Signal Process. 38, 814-824 (1990).
.. [MandelshtamTaylor1997] V. A. Mandelshtam and H. S. Taylor, "Harmonic
   inversion of time signals and its applications", J. Chem. Phys. 107,
   6756 (1997).
.. [WallNeuhauser1995] M. R. Wall and D. Neuhauser, "Extraction, through
   filter-diagonalization, of general quantum eigenvalues or classical normal
   mode frequencies from a small number of residues or a short-time segment
   of a signal", J. Chem. Phys. 102, 8011 (1995).
