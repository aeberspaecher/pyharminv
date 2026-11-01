import numpy as np
import pytest

from pyharminv import (FIRFilter, bandpass, detrend, harminv, highpass, lowpass,
                       remove_mean, synthetic_signal)

DT = 0.1
T = DT * np.arange(800)
OMEGA = np.array([1.0 - 0.01j, 3.0 - 0.02j, 6.0 - 0.05j])
AMP = np.array([1.0, 0.5 * np.exp(0.4j), 2.0])


def real_signal():
    return synthetic_signal(T, OMEGA, AMP, real=True)


def amplitudes_at_zero(result):
    """Amplitudes referred to t = 0 instead of result.t0."""
    return result.amplitude * np.exp(1j * result.omega * result.t0)


# --- mean and trend ------------------------------------------------------------

def test_remove_mean():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 50)) + 1j * rng.standard_normal((3, 50)) + 2.0
    np.testing.assert_allclose(remove_mean(x).mean(axis=-1), 0, atol=1e-14)
    np.testing.assert_allclose(remove_mean(x, axis=0).mean(axis=0), 0, atol=1e-14)


@pytest.mark.parametrize("order", [0, 1, 2, 5])
def test_detrend_removes_polynomials(order):
    s = np.linspace(-3, 7, 200)
    poly = sum((0.3 + 0.1j * k) * s ** k for k in range(order + 1))
    y, trend = detrend(poly, order=order, return_trend=True)
    np.testing.assert_allclose(y, 0, atol=1e-9 * np.abs(poly).max())
    np.testing.assert_allclose(trend, poly, atol=1e-9 * np.abs(poly).max())


def test_detrend_axis():
    s = np.arange(100.0)
    x = np.stack([1 + 2 * s, 3 - s, np.sin(s)])
    y = detrend(x.T, axis=0).T
    np.testing.assert_allclose(y[:2], 0, atol=1e-10)
    np.testing.assert_allclose(y[2], detrend(np.sin(s)))


def test_detrend_before_harminv():
    x = real_signal() + 0.3 + 0.05 * T
    r = harminv(detrend(x), DT).select(min_magnitude=0.1)
    assert len(r) == 3
    np.testing.assert_allclose(r.omega, OMEGA, atol=1e-6)
    np.testing.assert_allclose(r.amplitude, AMP, atol=5e-3)


# --- FIR filters ---------------------------------------------------------------

FILTERS = {
    "lowpass": lambda: lowpass(DT, 4.5),
    "highpass": lambda: highpass(DT, 2.0),
    "bandpass": lambda: bandpass(DT, 2.0, 4.5),
    "one-sided": lambda: bandpass(DT, 2.0, 4.5, one_sided=True),
}


@pytest.mark.parametrize("name", FILTERS)
def test_filter_is_exact_for_exponentials(name):
    """Output = same exponentials with amplitudes H(omega) d, no transient."""
    f = FILTERS[name]()
    omega = np.array([1.3 - 0.2j, -4.0 - 0.05j, 3.2 - 0.01j, 8.0 + 0.0j])
    amp = np.array([1.0, 0.3j, 0.5 - 0.5j, 2.0])
    x = synthetic_signal(T, omega, amp)
    y, t0 = f.apply(x, t0=0.0)
    assert len(y) == len(x) - len(f) + 1
    assert t0 == pytest.approx(f.half_length * DT)
    t_out = t0 + DT * np.arange(len(y))
    expected = synthetic_signal(t_out, omega, f.response(omega) * amp)
    np.testing.assert_allclose(y, expected, atol=1e-12)


@pytest.mark.parametrize("name", FILTERS)
def test_response_shape(name):
    f = FILTERS[name]()
    w = np.array([0.0, 1.0, 3.2, 6.0, 9.0])
    H = f.response(w)
    pas = {"lowpass": [1, 1, 1, 0, 0], "highpass": [0, 0, 1, 1, 1],
           "bandpass": [0, 0, 1, 0, 0], "one-sided": [0, 0, 1, 0, 0]}[name]
    np.testing.assert_allclose(np.abs(H), pas, atol=2e-3)
    if name != "one-sided":
        assert np.isrealobj(f.taps)
        np.testing.assert_allclose(H.imag, 0, atol=1e-12)
    else:
        assert abs(f.response(-3.2)) < 2e-3


@pytest.mark.parametrize("name", ["lowpass", "highpass", "bandpass"])
def test_filter_then_harminv_then_correct(name):
    f = FILTERS[name]()
    y, t0 = f.apply(real_signal())
    r = f.correct(harminv(y, DT, t0=t0))
    passed = np.abs(f.response(OMEGA)) > 0.5
    lines = r.select(min_magnitude=0.1)
    lines = lines[np.abs(f.response(lines.omega)) > 0.01]
    assert len(lines) == passed.sum()
    np.testing.assert_allclose(lines.omega, OMEGA[passed], atol=1e-8)
    np.testing.assert_allclose(amplitudes_at_zero(lines), AMP[passed], atol=1e-8)


def test_one_sided_bandpass_of_real_signal():
    f = bandpass(DT, 2.0, 4.5, one_sided=True)
    y, t0 = f.apply(real_signal())
    assert np.iscomplexobj(y)
    r = f.correct(harminv(y, DT, t0=t0))
    r = r[np.abs(f.response(r.omega)) > 0.01]
    assert len(r) == 1
    np.testing.assert_allclose(r.omega, OMEGA[1], atol=1e-8)
    # only the positive-frequency half of the cosine passes
    np.testing.assert_allclose(amplitudes_at_zero(r), AMP[1] / 2, atol=1e-8)


def test_highpass_removes_linear_trend():
    f = highpass(DT, 0.5, transition=0.7)
    trend = 0.3 + 0.05 * T
    y, _ = f.apply(trend)
    assert np.abs(y).max() < 1e-12
    y, t0 = f.apply(real_signal() + trend)
    r = f.correct(harminv(y, DT, t0=t0)).select(omega_min=0.7)
    np.testing.assert_allclose(r.omega, OMEGA, atol=1e-8)
    np.testing.assert_allclose(amplitudes_at_zero(r), AMP, atol=1e-6)


@pytest.mark.parametrize("f", [highpass(DT, 1.0), bandpass(DT, 1.0, 2.0)])
def test_dc_blocked_exactly(f):
    assert abs(f.response(0.0)) < 1e-14
    assert abs(f.taps.sum()) < 1e-14


def test_transition_sets_length():
    assert len(lowpass(DT, 4.5, transition=0.5)) > len(lowpass(DT, 4.5, transition=2.0))
    assert len(lowpass(DT, 4.5, attenuation=80)) > len(lowpass(DT, 4.5, attenuation=40))
    assert len(lowpass(DT, 4.5)) % 2 == 1


def test_errors():
    with pytest.raises(ValueError):
        lowpass(DT, 40.0)                       # above Nyquist
    with pytest.raises(ValueError):
        bandpass(DT, 3.0, 2.0)
    with pytest.raises(ValueError):
        FIRFilter([1.0, 1.0], DT)               # even length
    with pytest.raises(ValueError):
        highpass(DT, 0.05).apply(real_signal())  # too short for the filter
    with pytest.raises(ValueError):
        detrend(np.ones(3), order=3)
