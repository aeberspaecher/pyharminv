"""pyharminv: harmonic inversion of signals with robust Padé approximants.

Harmonic inversion decomposes a signal into damped complex exponentials
``c(t) = sum_k d_k exp(-i omega_k t)`` and thereby fits Lorentzian lines to
its spectrum. See :func:`harminv` for the main entry point.

For systems that are driven by unmeasured broadband noise rather than
decaying freely, :func:`ssi` (stochastic subspace identification, in the
covariance-driven and the data-driven variant) identifies the modes from
the response alone.
"""

from .fdm import fdm_window
from .core import HarmInvResult, harminv, harminv_window
from .pade import PadeResult, padeapprox, taylor_coeffs
from .preprocessing import (FIRFilter, bandpass, detrend, highpass, lowpass,
                            remove_mean)
from .signals import lorentzian, random_vibration, synthetic_signal
from .ssi import (SSIResult, Stabilization, mac, output_correlation, ssi, ssi_cov,
                  ssi_data, ssi_stabilization)

__version__ = "1.0.0"

__all__ = ["FIRFilter", "HarmInvResult", "PadeResult", "SSIResult",
           "Stabilization", "bandpass", "detrend", "fdm_window", "harminv",
           "harminv_window", "highpass", "lorentzian", "lowpass", "mac",
           "output_correlation", "padeapprox", "random_vibration",
           "remove_mean", "ssi", "ssi_cov", "ssi_data", "ssi_stabilization",
           "synthetic_signal", "taylor_coeffs", "__version__"]
