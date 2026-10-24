
"""Compute harmonic inversions of signals.

Harmonic inversion is the task of decomposing a signal as a superposition of
sinusoidals $\sum_n d_n \exp(i \omega_n t)$, where both the amplitudes $d_n$ as
well as the frequencies $\omega_n$ are potentially complex-valued. The
corresponds to a decomosition in potentially exponentially decaying
sinusoidals. This problem is encountered in the analysis of resonances in
acoustics, mechanics, electromagnetics or quantum mechanics - essentially in
the study of phenomena that exhibit modes or resonances with (potentially
decaying) time-harmonic solutions.

Harmonic inversion differs from the discrete Fourier transform in such as that
the DFT computes the amplitudes at equidistant frequencies given by the
sampling rate and the number of samples, whereas harmonic inversion computes
the relevant frequencies by itself.

This implementation follows the algorithm described in [1] or [2].

"""

import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
from scipy.signal import detrend, hilbert

from padeapprox import padeapprox


FT = lambda x: fftshift(ifft(ifftshift(x)))
iFT = lambda x: fftshift(fft(ifftshift(x)))

# TODO: check norm here!
FT_unitary = lambda x: FT(x)*np.sqrt(np.product(np.shape(x)))
iFT_unitary = lambda x: iFT(x)/np.sqrt(np.product(np.shape(x)))


def suggest_num_windows(N_signal, N_pade=10, window_overlap=1.0):
    """Suggest number of frequency windows from length of signal and desired Pade
    order.
    """

    # to determine a (N, N) Pade approximant, 2N+1 samples are needed
    samples_per_window = (2*N_pade+1)*window_overlap

    return int(np.floor(N_signal/samples_per_window))


def suggest_N_pade(N_signal, N_windows, window_overlap):
    """Suggest degree (N, N) of Pade approximant from number of samples in signal
    and desired window overlap.
    """

    samples_in_window = N_signal/(N_windows*window_overlap)
    N_pade = int(np.floor((samples_in_window-1.)/2.))

    return N_pade


def freq_shift_signal(signal, t, omega0):
    """Shift signal in frequency space by w0.
    """

    # shifts in frequency space are multiplications with exp(i*omega0*t) in time
    # domain
    fac = np.exp(+1j*omega0*t)   # FIXME: sign?

    return fac*signal


def time_shift_signal(signal, t, t0):
    """Shift signal in time domain by t0.
    """

    # shifts in time domain are multiplications with exp(-i*w*t0) in frequency
    # domain

    N_signal = len(signal)
    dt = (t[-1] - t[0])/(len(t) - 1)
    omega_vals = 2*np.pi*fftshift(fftfreq(N_signal, dt))
    fac = np.exp(-1j*omega_vals*t0)

    Signal = FT_unitary(signal)
    shifted_signal = iFT_unitary(fac*Signal)

    return shifted_signal


def center_frequencies(t, N_windows, N_signal):
    dt = (t[-1] - t[0])/(len(t) - 1)
    omega_vals = 2*np.pi*fftshift(fftfreq(N_signal, dt))
    omega_range = 2*np.abs(omega_vals[0]) # - omega_vals[0]  # TODO: use t_range and
                                                  # len(t) instead?
    window_range = omega_range/N_windows
    omega_window_min = omega_vals[0] + window_range/2.
    center_freqs = np.arange(0, N_windows)*window_range + omega_window_min

    return center_freqs, window_range


def band_pass(signal, t, omega_min, omega_max):
    """Perform hard band-pass filtering of a signal by cutting all Fourier
    coefficients that correspond to frequencies outside a given window.
    """

    import matplotlib.pyplot as plt

    N_signal = len(signal)
    dt = (t[-1] - t[0])/(N_signal - 1)
    omega_vals = 2*np.pi*fftshift(fftfreq(N_signal, dt))
    Signal = FT_unitary(signal)
    plt.plot(omega_vals, Signal, c="b", label="unfiltered")
    Signal[(omega_vals < omega_min) | (omega_vals > omega_max)] = 0.0

    plt.plot(omega_vals, Signal, c="r", ls="--", label="filtered")
    plt.axvline(omega_min, ls="--", color="black")
    plt.axvline(omega_max, ls="--", color="black")
    plt.xlabel("$\omega$")
    plt.legend()
    plt.show()

    return iFT_unitary(Signal)


def harmonic_inversion(signal, t, N_windows=None, N_pade=None, window_overlap=1.0):
    # The algorithm as described in the reference starts from a signal in
    # frequency domain. The signal is shifted and windowed in frequency in
    # order to reduce phase oscillations and keep the order of Pade
    # approximants low.
    # Although this implementation starts with a signal in time domain, similar
    # functionality is offered:
    # - using the shift theorem, the time signal is frequency-shifted by
    #   multiplication with the phasor \exp(i \omega_0 t)
    # - frequency space windowing is realised by the convolution theorem: the
    #   time signal is convolved with the Fourier transform of the window
    #   function (e.g. a rectangular window maps to convolution with the sinc
    #   function)

    import matplotlib.pyplot as plt

    signal = make_analytic(signal)

    N_signal = len(signal)
    if((N_windows is None) and (N_pade is None)):
        raise ValueError("N_windows and N_pade cannot both be None, set one of them")
    elif((N_windows is None) and (N_pade is not None)):
        N_windows = suggest_num_windows(N_signal, N_pade, window_overlap)
    elif((N_windows is not None) and (N_pade is None)):
        N_pade = suggest_N_pade(N_signal, N_windows, window_overlap)
    else:
        raise ValueError("Cannot set N_windows and N_pade simultaneously")

    print("Using Pade of order ({}, {}) with {} windows".format(N_pade, N_pade, N_windows))
    print("len(signal) = {}".format(len(signal)))
    dt = (t[-1] - t[0])/(len(t) - 1.)  # assume equidistant time samples

    # compute the center frequencies of the frequency windows to consider:
    center_freqs, window_width = center_frequencies(t, N_windows, N_signal)
    print("Center frequencies: {}".format(center_freqs))

    # compute number of samples in each frequency window
    N_samples_in_window = np.ceil(float(N_signal)/N_windows)

    omega_vals, ampl_vals = [], []
    print("Width of single freq window = {}".format(window_width))
    for center_freq in center_freqs:
        print("Evaluating frequency window around omega0 = {}".format(center_freq))
        signal_shifted = freq_shift_signal(signal, t, center_freq)
        signal_filtered = band_pass(signal_shifted, t, -(window_width*window_overlap)/2, +(window_width*window_overlap)/2)


        plt.plot(t, np.real(signal), c="b", ls="-", label="signal")
        plt.plot(t, np.real(signal_filtered), c="r", ls="--", label="signal filtered")
        plt.xlabel("t")
        plt.legend()
        plt.show()

        # signal_filtered = time_shift_signal(signal_filtered, t, np.mean(t))  # TODO: experimental time shift

        # plt.ioff()
        # plt.figure()
        # plt.plot(t, np.real(signal))
        # plt.plot(t, np.real(signal_filtered))
        # plt.show()

        a, b, mu, nu, r, zeros, poles, res = padeapprox(signal_filtered, N_pade, N_pade,
                                                        tol=0,  # FIXME: 0.0?
                                                        assume_isolated_poles=True)

        omega = 1j/dt*np.log(poles)  # FIXME: minus for testing!
        ampls = res/poles

        mask = np.abs(omega) <= window_width/2
        omega = omega[mask]
        ampls = ampls[mask]

        print("Found {} valid resonances".format(len(omega)))

        omega += center_freq  # shift frequencies back
        # print(omega)

        omega_vals.extend(list(omega))
        ampl_vals.extend(list(ampls))
        # print(omega_vals)


    # filter negative frequencies and positive imaginary parts:
    print("In total {} resonances found by harminv".format(len(omega_vals)))
    omega_vals = np.array(omega_vals)
    ampl_vals = np.array(ampl_vals)
    mask = (np.real(omega_vals) >= 0.0)  & (np.imag(omega_vals) < 0.0)
    omega_vals = omega_vals[mask]
    ampl_vals = ampl_vals[mask]
    print("After filtering: {} resonances".format(len(omega_vals)))

    print("omega = {}".format(omega_vals))

    return omega_vals, ampl_vals


def time_series(omega_n, ampl_n, t):

    lorentzian = lambda time, omega, ampl : (ampl*np.exp(-1j*omega*time))

    series_n = [lorentzian(t, omega, ampl) for omega, ampl in zip(omega_n,
                                                                  ampl_n)]
    # TODO: imag? real?
    return np.imag(sum(series_n))  # TODO: order of real and sum? doesn't matter because of linearity?


def make_analytic(signal):
    """Return complexified analytic signal with only positive frequencies in the
    sense of exp(-i \omega t) sense.
    """

    return np.conj(hilbert(np.conj(signal)))


def main():
    import matplotlib as mpl
    mpl.rcParams["font.size"] = 16
    # mpl.rcParams["font.family"] = "serif"
    import matplotlib.pyplot as plt

    # generate resonances frequencies
    dt = 1/10.  # sampling frequency

    N_res_test = 2
    N_samples = 200  # number of samples in time signal

    oversampling_factor = 3.1
    omega_min = 1
    omega_max = np.pi/dt/oversampling_factor # sample resonances up to fraction
                                             # of Nyquist frequency
    Q_min = 20
    Q_max = 200

    ampl_min = 1.0
    ampl_max = 1.0

    Q_values = (Q_max - Q_min)*np.random.rand(N_res_test) + Q_min

    omega_test_real = ((omega_max - omega_min)*np.random.rand(N_res_test)
                       + omega_min)
    omega_test_imag = -omega_test_real/(2*Q_values)
    omega_test = omega_test_real + 1j*omega_test_imag

    # plt.plot(omega_test_real, omega_test_imag, ls="", marker="x")
    # plt.show()

    # random amplitudes and random phases
    ampl_test = (((ampl_max - ampl_min)*np.random.rand(N_res_test) + ampl_min)
                   *np.exp(2j*np.pi*np.random.rand(N_res_test)))

    print(ampl_test)
    print(np.abs(ampl_test))

    t = dt*np.arange(N_samples)

    signal = time_series(omega_test, ampl_test, t)
    # signal -= np.mean(signal)

    signal_clean = detrend(signal)
    # signal_clean = signal - np.mean(signal)
    # signal_clean = signal

    f = 2*np.pi*fftshift(fftfreq(len(t), (t[-1]-t[0])/(len(t) - 1)))
    print("Frequencies: {}".format(f))

    N_pade = 20
    window_overlap = 1.2
    #N_windows = suggest_num_windows(len(signal), N_pade, window_overlap)

    N_windows = 1
    N_pade = suggest_N_pade(len(signal), N_windows, window_overlap)

    omega_n, amplitudes_n = harmonic_inversion(signal_clean, t, N_pade=N_pade,
                                               window_overlap=window_overlap)
    signal_reconstructed = time_series(omega_n, amplitudes_n, t)

    plt.scatter(np.real(omega_test), np.imag(omega_test), marker="o",
                c=np.abs(ampl_test), facecolors="none", label="input")
    plt.scatter(np.real(omega_n), np.imag(omega_n), marker="x", c=np.abs(amplitudes_n),
                label="reconstructed")
    plt.legend(loc="best")
    plt.colorbar()
    plt.show()

    plt.plot(t, signal, ls="-", marker="x", label="input")
    plt.plot(t, signal_reconstructed, marker="o", ls="--",
             label="reconstructed")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
