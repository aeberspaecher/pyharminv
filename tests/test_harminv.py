import numpy as np
import pytest
from scipy.optimize import curve_fit

from pyharminv import (HarmInvResult, fdm_window, harminv, harminv_window, lorentzian,
                       synthetic_signal)
from pyharminv.fdm import fdm_matrices
from pyharminv.signals import random_modes

METHODS = ["fdm", "pade"]
PADE_VARIANTS = ["least-squares", "robust"]


def match(result, omega_true):
    """Index of the closest found mode for each true frequency."""
    return np.array([np.argmin(np.abs(result.omega - w)) for w in omega_true])


def assert_modes(result, omega, amplitude, atol_omega, atol_amp):
    idx = match(result, omega)
    np.testing.assert_allclose(result.omega[idx], omega, rtol=0, atol=atol_omega)
    np.testing.assert_allclose(result.amplitude[idx], amplitude, rtol=0,
                               atol=atol_amp)


# --- the conventions -------------------------------------------------------

def test_single_complex_mode_exact():
    dt = 0.1
    t = dt * np.arange(64)
    omega = 2.0 - 0.1j
    s = 0.7 * np.exp(0.3j) * np.exp(-1j * omega * t)
    r = harminv_window(s, dt, tol=1e-12)
    k = np.argmax(np.abs(r[1]))
    assert abs(r[0][k] - omega) < 1e-12
    assert abs(r[1][k] - 0.7 * np.exp(0.3j)) < 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_damped_cosines(method):
    # x(t) = sum_k A_k exp(-Gamma_k t / 2) cos(Omega_k t - phi_k)
    dt = 0.1
    t = dt * np.arange(200)
    omega = np.array([3.0 - 0.05j, 7.0 - 0.02j])
    amp = np.array([np.exp(0.7j), 0.8 * np.exp(-1.2j)])
    x = (np.abs(amp)[None, :] * np.exp(np.outer(t, omega.imag))
         * np.cos(np.outer(t, omega.real) - np.angle(amp))).sum(axis=1)
    r = harminv(x, dt, method=method)
    assert r.real and len(r) == 2
    assert_modes(r, omega, amp, 1e-9, 1e-8)
    np.testing.assert_allclose(r.reconstruct(t), x, atol=1e-9)
    np.testing.assert_allclose(r.Q, [30, 175], rtol=1e-8)
    np.testing.assert_allclose(r.width, [0.1, 0.04], rtol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_complex_signal_negative_frequencies(method):
    dt = 0.05
    t = dt * np.arange(300)
    omega = np.array([-20.0 - 0.3j, -4.0 - 0.01j, 5.5 - 0.2j, 30.0 - 1.0j])
    amp = np.array([1.0, 0.5j, 2.0 * np.exp(1j), 0.3])
    s = synthetic_signal(t, omega, amp)
    r = harminv(s, dt, method=method)
    assert not r.real and len(r) == 4
    assert_modes(r, omega, amp, 1e-8, 1e-8)


def test_purely_damped_mode_in_real_signal():
    dt = 0.1
    t = dt * np.arange(150)
    x = 0.5 * np.exp(-0.3 * t) + np.exp(-0.02 * t) * np.cos(4 * t)
    r = harminv(x, dt)
    assert_modes(r, [-0.3j, 4 - 0.02j], [0.5, 1.0], 1e-9, 1e-8)
    np.testing.assert_allclose(r.reconstruct(t), x, atol=1e-9)


def test_t0_reference():
    dt = 0.1
    t = 5.0 + dt * np.arange(100)
    omega = 2.0 - 0.1j
    s = np.exp(-1j * omega * t)
    r = harminv(s, dt, t0=t[0])
    # amplitude refers to the first sample
    np.testing.assert_allclose(r.amplitude, np.exp(-1j * omega * t[0]), rtol=1e-9)
    np.testing.assert_allclose(r.reconstruct(t), s, atol=1e-9)


# --- accuracy on synthetic data --------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_many_modes_long_signal(seed):
    omega, amp = random_modes(20, 1, 25, q_min=50, q_max=500, min_spacing=0.3,
                              rng=seed)
    dt = 0.1
    t = dt * np.arange(5000)
    x = synthetic_signal(t, omega, amp, real=True)
    r = harminv(x, dt)
    assert len(r.info["windows"]) > 1
    strong = r.select(min_magnitude=1e-3)
    assert len(strong) == 20
    assert_modes(strong, omega, amp, 1e-7, 1e-6)


def test_closely_spaced_overlapping_lines():
    # two lines closer than their widths and than the Fourier resolution
    dt = 0.1
    t = dt * np.arange(400)
    omega = np.array([5.0 - 0.1j, 5.05 - 0.08j])
    amp = np.array([1.0, 0.7j])
    assert omega[1].real - omega[0].real < 2 * np.pi / (t[-1] - t[0])
    s = synthetic_signal(t, omega, amp)
    for method in METHODS:
        r = harminv(s, dt, method=method).select(min_magnitude=1e-3)
        assert len(r) == 2
        assert_modes(r, omega, amp, 1e-8, 1e-7)


def test_line_on_window_boundary():
    dt = 0.1
    t = dt * np.arange(3000)
    # with 10 windows over [0, pi/dt] the boundaries are multiples of pi
    omega = np.array([3 * np.pi - 0.01j, 5.0 - 0.02j])
    s = synthetic_signal(t, omega, [1.0, 1.0], real=True)
    r = harminv(s, dt, n_windows=10).select(min_magnitude=1e-3)
    assert len(r) == 2
    assert_modes(r, omega, [1.0, 1.0], 1e-8, 1e-7)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("noise", [1e-3, 1e-2])
def test_noisy_signal(method, noise):
    omega, amp = random_modes(5, 1, 25, q_min=50, q_max=500, min_spacing=1,
                              rng=2)
    dt = 0.1
    t = dt * np.arange(300)
    x = synthetic_signal(t, omega, amp, real=True, noise=noise, rng=7)
    rel_noise = noise / np.sqrt(np.mean(x ** 2))
    r = harminv(x, dt, method=method, tol=rel_noise)
    strong = r.select(min_magnitude=0.1)
    assert len(strong) == 5
    # errors scale with the noise
    assert_modes(strong, omega, amp, 0.2 * noise, 2 * noise)


def test_error_estimates_flag_spurious_modes():
    omega, amp = random_modes(5, 1, 25, q_min=50, q_max=500, min_spacing=1,
                              rng=1)
    dt = 0.1
    t = dt * np.arange(300)
    x = synthetic_signal(t, omega, amp, real=True, noise=1e-3, rng=3)
    r = harminv(x, dt, tol=1e-6)
    idx = match(r, omega)
    spurious = np.setdiff1d(np.arange(len(r)), idx)
    assert len(spurious) > 0
    assert np.max(r.error[idx]) < np.median(r.error[spurious])


def test_frequency_range():
    dt = 0.1
    t = dt * np.arange(2000)
    omega = np.array([2.0 - 0.01j, 6.0 - 0.01j, 12.0 - 0.02j])
    x = synthetic_signal(t, omega, [1, 1, 1], real=True)
    r = harminv(x, dt, omega_min=4, omega_max=8).select(min_magnitude=1e-3)
    assert len(r) == 1
    assert abs(r.omega[0] - omega[1]) < 1e-8


def test_window_options_agree():
    omega, amp = random_modes(10, 1, 20, q_min=50, q_max=300, min_spacing=0.5,
                              rng=4)
    dt = 0.1
    t = dt * np.arange(2000)
    x = synthetic_signal(t, omega, amp, real=True)
    for kw in [{"n_windows": 3}, {"window_width": 2.0}, {"K": 60}]:
        r = harminv(x, dt, **kw).select(min_magnitude=1e-3)
        assert len(r) == 10, kw
        assert_modes(r, omega, amp, 1e-7, 1e-6)


def test_fdm_matrices_closed_form():
    rng = np.random.default_rng(0)
    c = rng.standard_normal(30) + 1j * rng.standard_normal(30)
    M, dt = 12, 0.3
    phi = np.linspace(-2, 3, 7)
    U, F0 = fdm_matrices(c, dt, phi, M)
    a = np.exp(1j * phi * dt)
    n = np.arange(M + 1)
    for p in (0, 1, 2):
        H = c[n[:, None] + n[None, :] + p]
        brute = (a[:, None] ** n[None, :]) @ H @ (a[:, None] ** n[None, :]).T
        np.testing.assert_allclose(U[p], brute, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(F0, (a[:, None] ** n[None, :]) @ c[:M + 1],
                               rtol=1e-12)


def test_fdm_window_direct():
    dt = 0.1
    t = dt * np.arange(1000)
    omega = np.array([3.0 - 0.05j, 3.2 - 0.3j, 7.0 - 0.02j])
    amp = np.array([1.0, 0.5j, 0.8])
    s = synthetic_signal(t, omega, amp)
    w, d, e = fdm_window(s, dt, 2, 4)
    inside = (w.real > 2) & (w.real < 4)
    order = np.argsort(w[inside].real)
    np.testing.assert_allclose(w[inside][order], omega[:2], atol=1e-10)
    np.testing.assert_allclose(d[inside][order], amp[:2], atol=1e-10)
    assert np.all(e[inside] < 1e-10)


def test_amplitude_fit_large_problem_uses_normal_equations():
    from pyharminv.core import _fit_amplitudes
    rng = np.random.default_rng(3)
    omega = np.array([3 - 0.05j, 7 - 0.02j, -0.1j, 5 - 0.001j])
    amp = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    selfmirror = np.isclose(omega.real, 0)
    amp[selfmirror] = amp[selfmirror].real
    dt = 0.1
    t = dt * np.arange(3000)
    x = synthetic_signal(t, omega, amp, real=True, noise=1e-3, rng=1)
    direct = _fit_amplitudes(x.astype(complex), dt, omega, True, selfmirror)
    normal = _fit_amplitudes(x.astype(complex), dt, omega, True, selfmirror,
                             max_direct=100)
    np.testing.assert_allclose(normal, direct, atol=1e-12)


# --- Lorentzian line parameters --------------------------------------------

def test_lorentzian_parameters_match_fit_of_spectrum():
    # damped cosine: its spectrum near Omega is a Lorentzian of width Gamma
    dt = 0.05
    t = dt * np.arange(40000)
    Omega, Gamma, A = 6.0, 0.2, 1.5
    x = A * np.exp(-Gamma * t / 2) * np.cos(Omega * t)
    r = harminv(x[:600], dt)
    lines = r.lorentzians()
    np.testing.assert_allclose(lines["center"], [Omega], rtol=1e-10)
    np.testing.assert_allclose(lines["fwhm"], [Gamma], rtol=1e-8)
    np.testing.assert_allclose(lines["height"], [A / Gamma], rtol=1e-8)

    # numerical one-sided Fourier transform of the long signal
    w = np.linspace(Omega - 2, Omega + 2, 801)
    spec = (np.exp(1j * np.outer(w, t)) @ x) * dt - 0.5 * x[0] * dt
    np.testing.assert_allclose(r.spectrum(w), spec, atol=1e-3 * np.abs(spec).max())

    # least-squares Lorentzian fit to the real part of the spectrum
    popt, _ = curve_fit(lorentzian, w, spec.real, p0=[Omega + 0.1, 0.3, 5])
    np.testing.assert_allclose(popt, [Omega, Gamma, A / Gamma], rtol=2e-3)


def test_lorentzian_helper():
    w = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(lorentzian(w, 1.0, 2.0, 3.0), [1.5, 3.0, 1.5])
    two = lorentzian(w, [0.0, 2.0], [1.0, 1.0], [1.0, 1.0])
    np.testing.assert_allclose(two, [1 + 1 / 17, 0.4, 1 + 1 / 17])


# --- zeros (anti-resonances) -----------------------------------------------

def spring_chain(masses, springs, alpha=0.004, beta=0.004):
    """Mass, damping and stiffness matrices of a fixed-free spring chain."""
    n = len(masses)
    K = np.zeros((n, n))
    for i, k in enumerate(springs):
        K[i, i] += k
        if i > 0:
            K[i - 1, i - 1] += k
            K[i - 1, i] -= k
            K[i, i - 1] -= k
    M = np.diag(masses)
    return M, alpha * M + beta * K, K


def eigenfrequencies(M, C, K):
    """Complex frequencies omega = i s of M s^2 + C s + K, Re(omega) > 0."""
    n = len(M)
    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
    w = 1j * np.linalg.eigvals(A)
    return np.sort_complex(w[w.real > 0])


def impulse_response(M, C, K, i, j, t):
    """Displacement at DOF i after a unit impulse force at DOF j."""
    n = len(M)
    A = np.block([[np.zeros((n, n)), np.eye(n)],
                  [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
    lam, V = np.linalg.eig(A)
    b = np.zeros(2 * n)
    b[n:] = np.linalg.solve(M, np.eye(n)[j])
    r = V[i] * np.linalg.solve(V, b)
    return (np.exp(np.outer(t, lam)) @ r).real


@pytest.mark.parametrize("method", METHODS)
def test_zeros_are_antiresonances_of_driving_point_response(method):
    # the anti-resonances of H_jj are the resonances with DOF j clamped
    M, C, K = spring_chain([1.0, 1.0, 1.0], [1.0, 1.5, 2.0])
    dt = 0.25
    t = dt * np.arange(4000 if method == "fdm" else 600)
    for j in range(3):
        x = impulse_response(M, C, K, j, j, t)
        r = harminv(x, dt, omega_min=0.05, omega_max=4, method=method)
        assert len(r) == 3
        keep = [k for k in range(3) if k != j]
        clamped = eigenfrequencies(*(X[np.ix_(keep, keep)] for X in (M, C, K)))
        np.testing.assert_allclose(r.zeros(), clamped, rtol=0, atol=1e-7)
        # the spectrum has a dip at each anti-resonance
        dips = r.zeros().real
        for shift in (-0.02, 0.02):
            assert np.all(abs(r.spectrum(dips)) < abs(r.spectrum(dips + shift)))


def test_zeros_of_transfer_response():
    # the far end of a chain has no anti-resonance
    M, C, K = spring_chain([1.0, 1.0, 1.0], [1.0, 1.5, 2.0])
    dt = 0.25
    t = dt * np.arange(4000)
    r = harminv(impulse_response(M, C, K, 2, 0, t), dt, omega_min=0.05,
                omega_max=4)
    assert len(r) == 3 and len(r.zeros()) == 0


def test_zeros_complex_signal():
    # i/(w - p1) + i/(w - p2) vanishes halfway between the poles
    r = HarmInvResult(np.array([1 - 0.01j, 2 - 0.03j]), np.ones(2),
                      np.zeros(2))
    np.testing.assert_allclose(r.zeros(), [1.5 - 0.02j], atol=1e-13)
    # a constant background moves the zero: i/(w-1) + i/(w-2) + R = 0
    R = 0.3j
    z = r.zeros(residual=R)
    assert len(z) == 2
    np.testing.assert_allclose(1j / (z - r.omega[0]) + 1j / (z - r.omega[1])
                               + R, 0, atol=1e-12)


def test_zeros_residual_compensates_truncation():
    M, C, K = spring_chain([1.0, 1.0, 1.0], [1.0, 1.5, 2.0])
    dt = 0.25
    t = dt * np.arange(4000)
    full = harminv(impulse_response(M, C, K, 0, 0, t), dt, omega_min=0.05,
                   omega_max=4)
    low = full.select(omega_max=2)
    exact = full.zeros()[0]
    assert abs(low.zeros()[0] - exact) > 1e-2
    w0 = 1.0
    residual = full.spectrum(w0) - low.spectrum(w0)
    assert abs(low.zeros(residual=residual)[0] - exact) < 3e-3


def test_zeros_edge_cases():
    empty = HarmInvResult(np.zeros(0, complex), np.zeros(0, complex),
                          np.zeros(0))
    assert len(empty.zeros()) == 0
    one = HarmInvResult(np.array([2 - 0.1j]), np.array([1.0 + 0j]),
                        np.zeros(1))
    assert len(one.zeros()) == 0


# --- the result object and input validation ---------------------------------

def test_result_helpers():
    dt = 0.1
    t = dt * np.arange(200)
    x = synthetic_signal(t, [3 - 0.05j, 7 - 0.02j], [1.0, 0.1], real=True)
    r = harminv(x, dt)
    assert len(r[1:]) == 1
    assert len(r.select(min_magnitude=0.5)) == 1
    assert len(r.select(omega_min=5)) == 1
    assert len(r.select(max_width=0.05)) == 1
    np.testing.assert_allclose(r.frequency, r.center / (2 * np.pi))
    np.testing.assert_allclose(r.decay_rate, r.width / 2)
    assert r.modes(t).shape == (len(t), 2)
    assert "2 modes" in repr(r)


@pytest.mark.parametrize("kwargs, message", [
    ({"method": "magic"}, "unknown method"),
    ({"omega_min": 3, "omega_max": 2}, "larger"),
    ({"n_windows": 2, "window_width": 1.0}, "either"),
])
def test_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        harminv(np.ones(50), 0.1, **kwargs)


def test_invalid_signals():
    with pytest.raises(ValueError):
        harminv(np.ones((5, 5)), 0.1)
    with pytest.raises(ValueError):
        harminv(np.ones(3), 0.1)
    with pytest.raises(ValueError):
        harminv(np.ones(10), -0.1)


# --- nonlinear refinement ---------------------------------------------------

@pytest.mark.parametrize("real", [True, False])
def test_refine_reaches_least_squares_optimum(real):
    rng = np.random.default_rng(5)
    dt = 0.1
    t = dt * np.arange(400)
    omega = np.array([5.0 - 0.05j, 5.3 - 0.08j])
    amp = np.array([1.0, 0.7 * np.exp(1j)])
    clean = synthetic_signal(t, omega, amp, real=real)
    x = synthetic_signal(t, omega, amp, real=real, noise=0.1, rng=rng)
    r = harminv(x, dt, tol=0.05).select(min_magnitude=0.3)
    assert len(r) == 2
    refined = r.refine(x, dt)
    # the refined fit has a smaller residual than the harmonic inversion fit
    res0 = np.linalg.norm(r.reconstruct(t) - x)
    res1 = np.linalg.norm(refined.reconstruct(t) - x)
    assert res1 <= res0
    # residual noise estimate and error bars are consistent with the truth
    assert abs(refined.info["sigma"] - 0.1) < 0.02
    dev = np.abs(refined.omega - omega)
    assert np.all(dev < 5 * refined.error)
    assert np.linalg.norm(refined.reconstruct(t) - clean) < np.linalg.norm(x - clean)


def test_refine_noise_free_is_stable():
    dt = 0.1
    t = dt * np.arange(200)
    x = synthetic_signal(t, [3 - 0.05j, 7 - 0.02j, -0.3j], [1.0, 0.8j, 0.5],
                         real=True)
    r = harminv(x, dt)
    refined = r.refine(x, dt)
    np.testing.assert_allclose(refined.omega, r.omega, atol=1e-9)
    np.testing.assert_allclose(refined.amplitude, r.amplitude, atol=1e-8)
    assert refined.omega[0].real == 0


def test_refine_limits():
    dt = 0.1
    x = synthetic_signal(dt * np.arange(100), [3 - 0.05j], [1.0], real=True)
    r = harminv(x, dt)
    with pytest.raises(ValueError, match="max_modes"):
        r.refine(x, dt, max_modes=0)


@pytest.mark.parametrize("variant", PADE_VARIANTS)
def test_pade_variants_exact_on_clean_data(variant):
    dt = 0.1
    t = dt * np.arange(200)
    omega = np.array([3.0 - 0.05j, 7.0 - 0.02j, 7.3 - 0.1j])
    amp = np.array([1.0, 0.8j, 0.5])
    x = synthetic_signal(t, omega, amp, real=True)
    r = harminv(x, dt, method="pade", pade_variant=variant)
    assert len(r) == 3
    assert_modes(r, omega, amp, 1e-7, 1e-6)


def test_least_squares_pade_beats_robust_pade_with_noise():
    omega, amp = random_modes(4, 1, 25, q_min=50, q_max=300, min_spacing=2,
                              rng=3)
    dt = 0.1
    t = dt * np.arange(300)
    errors = {}
    for variant in PADE_VARIANTS:
        worst = []
        for seed in range(5):
            x = synthetic_signal(t, omega, amp, real=True, noise=1e-2, rng=seed)
            rel = 1e-2 / np.sqrt(np.mean(x ** 2))
            r = harminv(x, dt, method="pade", pade_variant=variant, tol=rel)
            r = r.select(min_magnitude=0.15)
            worst.append(max(np.min(np.abs(r.omega - w)) if len(r) else np.inf
                             for w in omega))
        errors[variant] = np.median(worst)
    assert errors["least-squares"] < 1e-2
    assert errors["least-squares"] < errors["robust"]


def test_unknown_pade_variant():
    with pytest.raises(ValueError, match="variant"):
        harminv_window(np.ones(20), 0.1, variant="magic")
