"""Preprocessing of signals before harmonic inversion.

Harmonic inversion models a signal as a finite sum of damped exponentials.
Offsets, slow drifts and strong lines outside the band of interest are part
of that model too, and they can cost resolution or produce spurious modes.
This module provides

* :func:`remove_mean` and :func:`detrend` to remove offsets and polynomial
  drifts, and
* linear-phase FIR filters (:func:`lowpass`, :func:`highpass`,
  :func:`bandpass`) that are *exact* for sums of exponentials.

Why FIR filters and not the usual IIR filters
---------------------------------------------

A linear, time-invariant filter maps every exponential ``exp(-i w t)`` of the
input to ``H(w) exp(-i w t)``: frequencies and decay rates are unchanged,
only the amplitudes are multiplied by the frequency response ``H``, evaluated
at the complex frequency. This holds only for the steady state, though:

* a recursive (IIR, e.g. Butterworth) filter starts with a transient that
  decays with the filter's own poles; harmonic inversion finds those poles
  as additional modes and the start of the signal is distorted;
* zero-phase filtering (``filtfilt``) also produces a transient at the end of
  the record, which looks like a *growing* mode.

A finite impulse response filter of length ``M`` has no such transient if
only the ``N - M + 1`` output samples are kept for which the filter fully
overlaps the data (``mode="valid"`` in a convolution). The filtered signal is
then exactly a sum of the same damped exponentials with amplitudes
``H(w_k) d_k``, and :meth:`FIRFilter.correct` divides that factor out again.
The price is the loss of ``M - 1`` samples, ``(M - 1) / 2`` at either end.
The filters here are symmetric (linear phase), so the output samples sit on
the original time grid, shifted by ``(M - 1) / 2`` samples, and ``H`` is
real for real frequencies.

Filtering is worthwhile when strong lines outside the band of interest
dominate the signal, or when a drift has to be removed without biasing the
lines. It is not needed to restrict the analysis to a band, which
:func:`~pyharminv.harminv` does by itself (``omega_min``, ``omega_max``).
Keep in mind that the lost samples at the start carry the strongest part of
a decaying signal: choose the transition band as wide as the problem allows.
"""

from dataclasses import replace

import numpy as np
from scipy.signal import firwin, get_window, kaiserord

__all__ = ["FIRFilter", "bandpass", "detrend", "highpass", "lowpass", "remove_mean"]


# --- offsets and drifts --------------------------------------------------------

def remove_mean(x, axis=-1):
    """Subtract the mean value.

    Parameters
    ----------
    x : array_like
        Real or complex samples.
    axis : int, optional
        Axis along which the mean is taken.

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    A constant offset is itself a mode at zero frequency, which harmonic
    inversion fits like any other line, so removing it is optional. Note that
    the mean of a short record of slowly decaying lines is not zero:
    subtracting it does not remove those lines but adds an offset of opposite
    sign, i.e. a zero-frequency mode.
    """
    x = np.asarray(x)
    return x - x.mean(axis=axis, keepdims=True)


def detrend(x, order=1, axis=-1, return_trend=False):
    """Subtract a least-squares polynomial fit.

    Parameters
    ----------
    x : array_like
        Real or complex samples, equidistant along ``axis``.
    order : int, optional
        Degree of the polynomial: ``0`` removes the mean, ``1`` (default) a
        linear trend, higher orders slow drifts.
    axis : int, optional
        Time axis.
    return_trend : bool, optional
        Also return the subtracted polynomial.

    Returns
    -------
    y : numpy.ndarray
        ``x`` minus the fitted polynomial.
    trend : numpy.ndarray
        The fitted polynomial, if ``return_trend`` is true.

    Notes
    -----
    A drift ``a + b t`` is a double pole at zero frequency, which a sum of
    simple exponentials cannot represent; harmonic inversion then returns
    a cluster of spurious broad modes near zero. Removing the trend first
    avoids this. The polynomial is fitted in a Legendre basis on the record,
    which stays well conditioned for high orders.

    The fit cannot distinguish a drift from lines whose period is comparable
    to the record length or longer: such lines are partly absorbed into the
    trend, and their fitted amplitudes and widths are biased. Keep ``order``
    low, or remove drifts with :func:`highpass`, whose effect on the lines is
    exactly known and can be undone.
    """
    x = np.asarray(x)
    if order < 0:
        raise ValueError("order must be non-negative")
    xm = np.moveaxis(x, axis, -1)
    N = xm.shape[-1]
    if N <= order:
        raise ValueError("need more samples than the polynomial order")
    s = np.linspace(-1.0, 1.0, N)
    V = np.polynomial.legendre.legvander(s, order)         # (N, order+1)
    flat = xm.reshape(-1, N).T                             # (N, rest)
    coef = np.linalg.lstsq(V, flat, rcond=None)[0]
    trend = np.moveaxis((V @ coef).T.reshape(xm.shape), -1, axis)
    y = x - trend
    return (y, trend) if return_trend else y


# --- FIR filters ---------------------------------------------------------------

class FIRFilter:
    """Linear-phase FIR filter that is exact for sums of damped exponentials.

    Create instances with :func:`lowpass`, :func:`highpass` or
    :func:`bandpass`, or directly from symmetric taps (or, for a one-sided
    pass band, taps with ``taps[::-1] == conj(taps)`` times a phase).

    Parameters
    ----------
    taps : array_like
        Filter coefficients ``h[0], ..., h[M-1]``, ``M`` odd.
    dt : float
        Sampling interval of the signals the filter is meant for.

    Attributes
    ----------
    taps : numpy.ndarray
    dt : float
    """

    def __init__(self, taps, dt):
        taps = np.asarray(taps)
        if taps.ndim != 1 or len(taps) % 2 != 1:
            raise ValueError("taps must be one-dimensional with odd length")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.taps = taps
        self.dt = float(dt)

    def __len__(self):
        return len(self.taps)

    @property
    def half_length(self):
        """Samples lost at each end, ``(M - 1) // 2``."""
        return (len(self.taps) - 1) // 2

    @property
    def delay(self):
        """Time shift of the first output sample, ``half_length * dt``."""
        return self.half_length * self.dt

    def apply(self, x, t0=0.0):
        """Filter a signal, keeping only the transient-free samples.

        Parameters
        ----------
        x : array_like
            Real or complex samples ``x(t0 + n dt)``, ``n = 0, ..., N-1``.
        t0 : float, optional
            Time of the first input sample.

        Returns
        -------
        y : numpy.ndarray
            The ``N - M + 1`` filtered samples. ``y[n]`` belongs to the time
            ``t0 + (n + half_length) dt``, i.e. ``x[half_length:-half_length]``
            filtered.
        t0 : float
            Time of the first output sample, ``t0 + delay``. Pass it to
            :func:`~pyharminv.harminv` so that the amplitudes refer to it.
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if len(x) < len(self.taps) + 3:
            raise ValueError(
                f"signal of {len(x)} samples too short for a filter of length "
                f"{len(self.taps)}; use a wider transition band")
        # y[n] = sum_m h[m] x[n + m]
        y = np.convolve(x, self.taps[::-1], mode="valid")
        return y, t0 + self.delay

    def response(self, omega):
        """Frequency response ``H(omega)`` for real or complex ``omega``.

        A component ``d exp(-i omega (t - t0))`` of the input becomes
        ``H(omega) d exp(-i omega (t - t0))`` in the output, in the same time
        frame ``t``. For symmetric taps ``H`` is real on the real axis.
        """
        w = np.asarray(omega, dtype=complex)
        m = np.arange(len(self.taps)) - self.half_length
        return np.exp(-1j * np.multiply.outer(w, m) * self.dt) @ self.taps

    def correct(self, result):
        """Undo the filter's effect on the amplitudes of a harmonic inversion.

        Parameters
        ----------
        result : HarmInvResult
            Modes found in the output of :meth:`apply`, with ``t0`` as
            returned by :meth:`apply`.

        Returns
        -------
        HarmInvResult
            The same modes with amplitudes divided by ``H(omega)``: the
            amplitudes of the unfiltered signal, referred to ``result.t0``.
            Modes in the stop band get large, unreliable amplitudes; discard
            them with :meth:`~pyharminv.HarmInvResult.select` by frequency.

        Notes
        -----
        For a real signal filtered with a one-sided (complex) band pass, the
        corrected amplitudes are half those of the real-signal convention,
        because only the positive-frequency half of each cosine passes.
        """
        H = self.response(result.omega)
        return replace(result, amplitude=result.amplitude / H,
                       info=dict(result.info))

    def __repr__(self):
        return f"FIRFilter(length={len(self.taps)}, dt={self.dt:g})"


def _num_taps(dt, transition, attenuation):
    nyq = np.pi / dt
    if not 0 < transition < nyq:
        raise ValueError("transition must be between 0 and pi/dt")
    M, beta = kaiserord(attenuation, transition / nyq)
    return M + (1 - M % 2), beta                       # odd length


def _block_dc(taps, beta):
    """Make the response vanish exactly at zero frequency.

    Subtracting a multiple of the (smooth) window changes the response only
    near zero frequency, by at most the stop-band ripple. With symmetric
    taps the derivative at zero vanishes too, so constants and linear
    trends are removed exactly.
    """
    w = get_window(("kaiser", beta), len(taps), fftbins=False)
    return taps - taps.sum() * w / w.sum()


def _check_cutoff(dt, *cutoffs):
    nyq = np.pi / dt
    for c in cutoffs:
        if not 0 < c < nyq:
            raise ValueError("cut-off frequencies must lie between 0 and pi/dt")


def lowpass(dt, omega_c, transition=None, attenuation=60.0):
    """Low-pass FIR filter.

    Parameters
    ----------
    dt : float
        Sampling interval.
    omega_c : float
        Cut-off angular frequency, the centre of the transition band.
    transition : float, optional
        Width of the transition band (angular frequency). It sets the filter
        length, roughly ``M = (attenuation - 8) / (2.285 transition dt)``,
        and thus the number of samples lost. Defaults to ``omega_c / 2``,
        limited by the distance to the Nyquist frequency ``pi / dt``.
    attenuation : float, optional
        Stop-band attenuation in dB (Kaiser window design). The pass-band
        ripple is of the same relative size, ``10**(-attenuation/20)``.

    Returns
    -------
    FIRFilter
    """
    _check_cutoff(dt, omega_c)
    if transition is None:
        transition = min(0.5 * omega_c, 2 * (np.pi / dt - omega_c))
    M, beta = _num_taps(dt, transition, attenuation)
    taps = firwin(M, omega_c, window=("kaiser", beta), fs=2 * np.pi / dt)
    return FIRFilter(taps, dt)


def highpass(dt, omega_c, transition=None, attenuation=60.0):
    """High-pass FIR filter.

    Parameters are as for :func:`lowpass`; ``transition`` defaults to
    ``omega_c / 2``, limited by the distance to the Nyquist frequency.

    Notes
    -----
    The response and its derivative vanish exactly at zero frequency, so
    the filter removes a constant offset and a linear trend exactly (the
    same holds for the two-sided :func:`bandpass`): it is an alternative to :func:`detrend` whose effect on
    the lines is known and undone by :meth:`FIRFilter.correct`.
    """
    _check_cutoff(dt, omega_c)
    if transition is None:
        transition = min(0.5 * omega_c, 2 * (np.pi / dt - omega_c))
    M, beta = _num_taps(dt, transition, attenuation)
    taps = firwin(M, omega_c, window=("kaiser", beta), pass_zero=False,
                  fs=2 * np.pi / dt)
    return FIRFilter(_block_dc(taps, beta), dt)


def bandpass(dt, omega_low, omega_high, transition=None, attenuation=60.0,
             one_sided=False):
    """Band-pass FIR filter.

    Parameters
    ----------
    dt : float
        Sampling interval.
    omega_low, omega_high : float
        Edges of the pass band (centres of the two transition bands).
    transition : float, optional
        Width of each transition band. Defaults to half the smaller of
        ``omega_low`` and the band width (for a one-sided filter: half the
        band width), limited by the distance to the
        Nyquist frequency.
    attenuation : float, optional
        Stop-band attenuation in dB.
    one_sided : bool, optional
        If false (default), the filter has real taps and passes both
        ``[omega_low, omega_high]`` and its mirror image at negative
        frequencies, as needed for real signals. If true, it has complex taps
        and passes only the positive band; the output of a real signal is
        then complex and holds only the positive-frequency half of each line.
        With ``one_sided=True``, ``omega_low`` may be zero or negative, which
        selects an arbitrary band of a complex signal.

    Returns
    -------
    FIRFilter
    """
    nyq = np.pi / dt
    if not omega_high > omega_low:
        raise ValueError("omega_high must be larger than omega_low")
    if one_sided:
        if not (-nyq < omega_low and omega_high < nyq):
            raise ValueError("band edges must lie between -pi/dt and pi/dt")
        half = 0.5 * (omega_high - omega_low)
        center = 0.5 * (omega_high + omega_low)
        if transition is None:
            transition = min(half, 2 * (nyq - omega_high),
                             2 * (omega_low + nyq))
        M, beta = _num_taps(dt, transition, attenuation)
        base = firwin(M, half, window=("kaiser", beta), fs=2 * nyq)
        m = np.arange(M) - (M - 1) // 2
        return FIRFilter(base * np.exp(1j * center * m * dt), dt)
    _check_cutoff(dt, omega_low, omega_high)
    if transition is None:
        transition = min(0.5 * omega_low, 0.5 * (omega_high - omega_low),
                         2 * (nyq - omega_high))
    M, beta = _num_taps(dt, transition, attenuation)
    taps = firwin(M, [omega_low, omega_high], window=("kaiser", beta),
                  pass_zero=False, fs=2 * nyq)
    return FIRFilter(_block_dc(taps, beta), dt)
