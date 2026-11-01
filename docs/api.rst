API reference
=============

.. currentmodule:: pyharminv

Harmonic inversion
------------------

.. autofunction:: harminv

.. autoclass:: HarmInvResult
   :members:

.. autofunction:: harminv_window

.. autofunction:: fdm_window

Padé approximation
------------------

.. autofunction:: padeapprox

.. autoclass:: PadeResult
   :members:

.. autofunction:: taylor_coeffs

Preprocessing
-------------

.. automodule:: pyharminv.preprocessing
   :no-members:

.. currentmodule:: pyharminv

.. autofunction:: remove_mean

.. autofunction:: detrend

.. autofunction:: lowpass

.. autofunction:: highpass

.. autofunction:: bandpass

.. autoclass:: FIRFilter
   :members:

Stochastic subspace identification
----------------------------------

.. automodule:: pyharminv.subspace
   :no-members:

.. currentmodule:: pyharminv

.. autofunction:: ssi

.. autofunction:: ssi_cov

.. autofunction:: ssi_data

.. autoclass:: SSIResult
   :members:

.. autofunction:: ssi_stabilization

.. autoclass:: Stabilization
   :members:

.. autofunction:: output_correlation

.. autofunction:: mac

Test signals
------------

.. autofunction:: synthetic_signal

.. autofunction:: lorentzian

.. autofunction:: pyharminv.signals.random_modes

.. autofunction:: random_vibration
