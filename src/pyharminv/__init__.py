"""pyharminv: harmonic inversion of signals with robust Padé approximants.

Harmonic inversion decomposes a signal into damped complex exponentials
``c(t) = sum_k d_k exp(-i omega_k t)`` and thereby fits Lorentzian lines to
its spectrum. See :func:`harminv` for the main entry point.
"""

from .fdm import fdm_window
from .core import HarmInvResult, harminv, harminv_window
from .pade import PadeResult, padeapprox, taylor_coeffs
from .signals import lorentzian, synthetic_signal

__version__ = "1.0.0"

__all__ = ["HarmInvResult", "PadeResult", "fdm_window", "harminv", "harminv_window",
           "lorentzian", "padeapprox", "synthetic_signal", "taylor_coeffs",
           "__version__"]
