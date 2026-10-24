Usage
=====

Installation
------------

.. code-block:: bash

   pip install .              # library only
   pip install -e .[test,docs,examples]

Basic use
---------

.. code-block:: python

   import numpy as np
   from pyharminv import harminv

   dt = 0.1
   t = dt * np.arange(300)
   x = (np.exp(-0.05 * t) * np.cos(3.0 * t)
        + 0.8 * np.exp(-0.02 * t) * np.cos(7.0 * t + 1.2))

   result = harminv(x, dt)
   print(result)

:func:`~pyharminv.harminv` returns a :class:`~pyharminv.HarmInvResult`
with the complex frequencies ``omega``, amplitudes ``amplitude`` and error
estimates ``error`` of all modes, sorted by frequency, and derived
quantities:

=====================  =====================================================
``center``             line centres :math:`\Omega_k = \mathrm{Re}\,\omega_k`
``frequency``          :math:`\Omega_k / 2\pi`
``width``              full widths at half maximum
                       :math:`\Gamma_k = -2\,\mathrm{Im}\,\omega_k`
``decay_rate``         amplitude decay rates :math:`\Gamma_k / 2`
``Q``                  quality factors :math:`\Omega_k / \Gamma_k`
``magnitude, phase``   :math:`|d_k|`, :math:`\arg d_k`
``lorentzians()``      centre, FWHM, peak height and area of each line
``reconstruct(t)``     the fitted signal
``spectrum(omega)``    the complex spectrum of the fitted signal
``select(...)``        subset of modes by amplitude, error, width or range
=====================  =====================================================

Conventions
-----------

* Time dependence :math:`e^{-i\omega t}`; decaying modes have
  :math:`\mathrm{Im}\,\omega < 0`.
* Complex signals: :math:`c(t) = \sum_k d_k e^{-i\omega_k (t - t_0)}`.
* Real signals: :math:`x(t) = \mathrm{Re} \sum_k d_k e^{-i\omega_k (t-t_0)}`
  with :math:`\mathrm{Re}\,\omega_k \ge 0`. A term is the damped cosine
  :math:`|d_k| e^{-\Gamma_k t/2} \cos(\Omega_k t - \arg d_k)`.
* Amplitudes refer to the first sample, at time ``t0`` (default 0).
* All frequencies are angular frequencies in units of ``1/dt``.

Lorentzian lines
----------------

A damped cosine :math:`A e^{-\Gamma t/2}\cos\Omega t` has the one-sided
Fourier transform
:math:`\int_0^\infty x(t) e^{i\omega t}\,dt`, whose real part near
:math:`\Omega` is the Lorentzian

.. math::

   \frac{A}{\Gamma} \, \frac{1}{1 + \left(\frac{\omega - \Omega}{\Gamma/2}\right)^2} .

:meth:`~pyharminv.HarmInvResult.lorentzians` returns these parameters for
every mode, and :func:`~pyharminv.lorentzian` evaluates them. A complex
amplitude adds a dispersive part to the line, see
:meth:`~pyharminv.HarmInvResult.spectrum`.

Noisy data
----------

Set ``tol`` to the relative noise level of the data, i.e. the standard
deviation of the noise divided by the rms of the signal. Smaller values are
safe but return more spurious weak modes; larger values can lose weak or
broad lines. Spurious modes have small amplitudes and large error estimates
and are removed with :meth:`~pyharminv.HarmInvResult.select`:

.. code-block:: python

   result = harminv(x, dt, tol=rel_noise)
   lines = result.select(min_magnitude=0.05, max_rel_error=1e-3)

Only record as long as the signal is above the noise: a long tail of pure
noise adds no information. The accuracy of a line depends on its signal to
noise ratio; weak broad lines are the hardest.

Long signals and frequency windows
----------------------------------

With the default solver the frequency range (``omega_min`` to ``omega_max``,
by default :math:`[0, \pi/dt]` for real and :math:`[-\pi/dt, \pi/dt]` for
complex signals) is split into windows, each requiring about ``k_max``
basis functions. Set ``n_windows``, ``window_width`` or ``K`` (basis
functions per window) to override this. Lines on window boundaries are found
by both neighbouring windows; the duplicate is removed automatically.

Padé solver
-----------

``method="pade"`` fits the whole band with a type (K-1, K) Padé
approximant, as in Wiersig and Main. The cost grows like :math:`K^3`, so it
is meant for signals of up to a few hundred samples. Two variants exist:

* ``pade_variant="least-squares"`` (default) finds the poles from the
  Hankel matrix of *all* samples (matrix pencil method) and fits the
  amplitudes by least squares. Under noise it is as accurate as FDM.
* ``pade_variant="robust"`` is the classical approximant to the first 2K
  samples, computed with the robust SVD algorithm of
  :func:`~pyharminv.padeapprox`. It is exact for clean data, but with noise
  it uses only part of the record and its rank decision is fragile, so it
  internally uses ``tol / 100``.

Refinement
----------

:meth:`HarmInvResult.refine <pyharminv.HarmInvResult.refine>` takes the
modes as a starting point and minimises the squared residual over all
frequencies and amplitudes jointly. For white Gaussian noise this is the
maximum-likelihood estimate; its errors reach the Cramér-Rao bound, and
``info["omega_std"]`` gives matching error bars. Remove spurious modes with
:meth:`~pyharminv.HarmInvResult.select` first::

    fit = harminv(x, dt, tol=0.1 * rel_noise).select(min_magnitude=0.05).refine(x, dt)

A ``tol`` below the noise level keeps more modes, which helps to resolve
close lines; ``select`` then drops the spurious extra modes.
The low-level functions :func:`~pyharminv.harminv_window` and
:func:`~pyharminv.padeapprox` are available for direct use.

Example
-------

``examples/lorentz_fit.py`` fits twelve lines, two of them closer than the
Fourier resolution, to a noisy sum of damped cosines and produces the figure
on the front page.
