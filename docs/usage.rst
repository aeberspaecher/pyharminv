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
``zeros()``            zeros of the spectrum (anti-resonances)
=====================  =====================================================

Signals can be prepared with :func:`~pyharminv.detrend` and the FIR filters
:func:`~pyharminv.lowpass`, :func:`~pyharminv.highpass` and
:func:`~pyharminv.bandpass`, see `Preprocessing: offsets, trends and
filters`_.

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

Anti-resonances
---------------

If the signal is the impulse response of a linear system, the spectrum
returned by :meth:`~pyharminv.HarmInvResult.spectrum` is (the complex
conjugate of) its frequency response function, a sum of poles
:math:`r_k / (\omega - \omega_k)`. Besides the resonances (the poles) it
can have sharp dips where neighbouring modes cancel, the anti-resonances.
They are zeros of the rational function and do not appear as modes, but
:meth:`~pyharminv.HarmInvResult.zeros` computes them from the fitted
modes:

.. code-block:: python

   result = harminv(impulse_response, dt)
   z = result.zeros()      # Re(z): positions of the dips, -2 Im(z): widths

For a driving-point response (force and response at the same point) there
is one anti-resonance between each pair of neighbouring resonances, at the
resonances of the system clamped at that point; a transfer response can
have none. Unlike a pole, a zero depends on *all* modes, including those
outside the analysed range or removed by ``select``: keep the modes around
the zeros of interest, or approximate the missing ones near the zeros by a
constant ``residual``. See ``notebooks/04_anti_resonances.ipynb``.

Preprocessing: offsets, trends and filters
------------------------------------------

A constant offset is just a mode at zero frequency and needs no treatment.
A drift, however, is a double pole at zero that a sum of simple exponentials
cannot represent; it biases the fitted lines. Remove it with
:func:`~pyharminv.detrend` (least-squares polynomial, ``order=1`` by
default):

.. code-block:: python

   from pyharminv import detrend, harminv

   result = harminv(detrend(x), dt)

A polynomial fit also absorbs part of lines whose period is comparable to
the record length, so keep the order low. What is left of the drift appears
as a weak mode very close to zero frequency, which
``select(omega_min=...)`` removes.

Strong lines outside the band of interest, or a drift close to slow lines,
are better removed with a filter. The usual recursive (IIR) filters are
unsuited to harmonic inversion: their start-up transient decays with the
filter's own poles, which then show up as additional modes, and zero-phase
filtering adds a transient at the end that looks like a growing mode.
:func:`~pyharminv.lowpass`, :func:`~pyharminv.highpass` and
:func:`~pyharminv.bandpass` return linear-phase FIR filters
(:class:`~pyharminv.FIRFilter`) that only keep the output samples for which
the filter fully overlaps the data. The result is *exactly* a sum of the
same damped exponentials, with frequencies and widths unchanged and
amplitudes multiplied by the frequency response :math:`H(\omega_k)`, which
:meth:`~pyharminv.FIRFilter.correct` divides out again:

.. code-block:: python

   from pyharminv import harminv, highpass

   hp = highpass(dt, omega_c=1.0)         # transition band 0.75 ... 1.25
   y, t0 = hp.apply(x)                    # len(y) == len(x) - len(hp) + 1
   result = hp.correct(harminv(y, dt, t0=t0)).select(omega_min=1.25)

The filter costs ``len(hp) - 1`` samples, half of them at the start where
a decaying signal is strongest. Its length is set by the width of the
transition band (``transition``, by default half the cut-off frequency) and
the stop-band attenuation (``attenuation``, 60 dB by default), roughly
``(attenuation - 8) / (2.285 * transition * dt)``: make the transition as
wide as the spacing of the lines allows. Amplitudes refer to the returned
``t0``, the time of the first output sample. Modes in the stop band are
attenuated, not removed; after ``correct`` they carry large, meaningless
amplitudes, so select the pass band by frequency. The high-pass and
two-sided band-pass filters remove constants and linear trends exactly.
For complex signals, ``bandpass(..., one_sided=True)`` passes only the given
band and not its mirror image at negative frequencies.

Filtering is not needed just to restrict the analysis to a band:
``omega_min`` and ``omega_max`` of :func:`~pyharminv.harminv` do that
without losing samples.

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

Noise-driven systems: stochastic subspace identification
--------------------------------------------------------

Harmonic inversion needs a free decay. If a system can only be measured
while it is driven by an unknown, broadband excitation (wind, traffic,
operation, thermal noise), the response is not a sum of damped exponentials
and :func:`~pyharminv.harminv` fails on it. Stochastic subspace
identification (SSI) identifies the modes from such output-only data. The
theory is explained in :doc:`theory_ssi`.

.. code-block:: python

   from pyharminv import remove_mean, ssi_cov, ssi_data, ssi_stabilization

   y = remove_mean(y, axis=0)          # shape (n_samples, n_channels)
   r = ssi_cov(y, dt, order=16, block_rows=40)
   modes = r.select(max_rel_error=0.01, max_damping=0.2)
   print(modes)
   modes.omega, modes.damping_ratio, modes.mode_shape

The signal has time along the first axis and one column per channel; a
one-dimensional array is a single channel. Two variants are available:

* :func:`~pyharminv.ssi_cov` (covariance-driven) estimates the output
  correlations by FFT and realizes a state-space model from them. After the
  FFT its cost does not depend on the record length, so it suits long
  records, many channels and exploring the order.
* :func:`~pyharminv.ssi_data` (data-driven) projects future onto past
  outputs with an LQ factorisation of the data. It is numerically more
  robust and slightly more accurate for short records, at a cost that grows
  linearly with the record length.

:func:`~pyharminv.ssi` selects the variant with ``method="cov"`` or
``method="data"`` (aliases ``"covariance"`` and ``"direct"``).

Results
~~~~~~~

Both return an :class:`~pyharminv.SSIResult`. It is a
:class:`~pyharminv.HarmInvResult` of the **output correlation function**,
which for a system driven by white noise decays like a free response with
the same poles. ``omega``, ``error``, ``center``, ``width``, ``Q``,
``select``, ``spectrum`` and ``reconstruct`` therefore mean what they mean
for harmonic inversion; ``amplitude`` holds the coefficients of the
auto-correlation of the reference channel (``ref_channel``, default 0).
In addition:

==========================  ================================================
``natural_frequency``       :math:`|\omega_k|`, the undamped frequency
``damping_ratio``           :math:`\zeta_k = -\mathrm{Im}\,\omega_k / |\omega_k|`
``mode_shape``              complex mode shapes, shape ``(K, n_channels)``,
                            largest component 1
``residue``                 coefficients of all correlations,
                            shape ``(K, n_channels, n_channels)``
``correlation(tau)``        model correlation matrices for lags ``tau``
``psd(omega, full=False)``  model power spectral density (two-sided, per
                            unit angular frequency); ``2 * psd(2 pi f)``
                            compares with :func:`scipy.signal.welch`
``mac(other=None)``         modal assurance criterion between mode shapes
``select(...)``             as for harmonic inversion, plus bounds
                            ``min_damping``, ``max_damping`` on the
                            damping ratio
``order``, ``info``         model order; singular values, ``A``, ``C``,
                            ``G``, number of block rows
==========================  ================================================

The conventions are those of a real signal in harmonic inversion: only
poles with :math:`\mathrm{Re}\,\omega \ge 0` are listed, and
:math:`R(\tau) = \mathrm{Re} \sum_k M_k e^{-i\omega_k\tau}` for
:math:`\tau \ge \Delta t` with the residue matrices :math:`M_k`. At lag
zero the measured correlation also contains the variance of the measurement
noise, which the model leaves out.

Model order and spurious poles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The order is the dimension of the state: two per oscillating mode. Choose
it generously, about twice what the expected modes need, and remove the
spurious poles afterwards:

* ``error`` is the distance of each pole to the closest pole at order
  ``order + 2``. Physical poles barely move, noise poles jump; keep poles
  with ``max_rel_error`` of about ``0.01``.
* Noise poles are often strongly damped (``max_damping``); harmonic
  components of the excitation have almost no damping (``min_damping``).
* Poles that hardly contribute to the reference correlation have a small
  ``magnitude`` (``min_magnitude``).

Without ``order``, a heuristic places it after the last large drop of the
singular values (``info["singular_values"]``). The reliable tool is the
stabilization diagram, which identifies the poles for a range of orders
from a single SVD:

.. code-block:: python

   st = ssi_stabilization(y, dt, range(2, 61, 2), block_rows=40)
   st.plot()                                  # needs matplotlib
   order, omega, zeta, stable = st.poles(stable_only=True)

A pole is stable if a pole at the previous order matches its natural
frequency (``freq_tol``, default 1 %), damping ratio (``damping_tol``, 5 %)
and, for several channels, mode shape (``mac_tol``, MAC 0.98). Physical
modes form vertical lines of stable poles.

Block rows and sampling
~~~~~~~~~~~~~~~~~~~~~~~

``block_rows`` (:math:`i`) sets the longest lag, :math:`(2i-1)\,\Delta t`,
and the largest order, :math:`(i-1)` times the number of channels.
:math:`i\,\Delta t` should cover at least one period of the slowest mode
of interest; 20 to 100 rows are typical. Sample, or decimate, to a few
times the highest frequency of interest: strong oversampling crowds the
poles near :math:`\lambda = 1` and requires many block rows.

Excitation and noise
~~~~~~~~~~~~~~~~~~~~

* Remove the mean and drifts first (:func:`~pyharminv.remove_mean`,
  :func:`~pyharminv.detrend`); an offset is a pole at zero frequency.
* White measurement noise does not bias SSI, since it only enters the
  correlation at lag zero, but it inflates the statistical error; at very
  low signal-to-noise ratios the required record length grows steeply.
  Do not band-pass filter narrowly, which colours the noise.
* A coloured excitation adds poles of its own, usually strongly damped. A
  harmonic component appears as an undamped pole.
* Several channels separate closely spaced modes by their shapes, which a
  single channel cannot.
* The damping ratio of a randomly excited system is known only to a
  statistical error that decreases like :math:`1/\sqrt{T}`. If a free
  decay can be measured, harmonic inversion of it is far more accurate.

Notebooks 05 to 07 illustrate all of this: the method step by step, the
choice between SSI-Cov, SSI-Data and harmonic inversion, and the limits
under extreme measurement noise.
