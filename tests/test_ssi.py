import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.signal import welch

from pyharminv import (HarmInvResult, SSIResult, mac, output_correlation,
                       random_vibration, ssi, ssi_cov, ssi_data,
                       ssi_stabilization)
from pyharminv.subspace import _DataRealization

METHODS = ["cov", "data"]

# three masses in a chain, light Rayleigh damping
MASS = np.eye(3)
_k = np.array([1.0, 1.5, 2.0])
STIFF = np.diag(_k + np.append(_k[1:], 0)) - np.diag(_k[1:], 1) - np.diag(_k[1:], -1)
DAMP = 0.02 * MASS + 0.02 * STIFF
DT = 0.3


@pytest.fixture(scope="module")
def chain():
    y, omega = random_vibration(MASS, DAMP, STIFF, DT, 100_000, rng=1)
    return y, omega


def true_shapes():
    _, V = eigh(STIFF, MASS)
    return V.T


def physical(r):
    r = r.select(max_rel_error=0.01, max_damping=0.2)
    return r.select(min_magnitude=1e-3 * r.magnitude.max())


def test_random_vibration_single_oscillator():
    w0, zeta = 2.0, 0.05
    y, omega = random_vibration(1.0, 2 * zeta * w0, w0**2, 0.1, 1000, rng=0)
    assert y.shape == (1000, 1)
    assert omega.shape == (1,)
    assert omega[0] == pytest.approx(w0 * np.sqrt(1 - zeta**2) - 1j * zeta * w0)


def test_output_correlation_direct():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((50, 2))
    R = output_correlation(y, 5)
    for k in range(5):
        direct = y[k:].T @ y[:50 - k] / (50 - k)
        np.testing.assert_allclose(R[k], direct, atol=1e-12)
    Rb = output_correlation(y[:, 0], 3, unbiased=False)
    assert Rb.shape == (3, 1, 1)
    assert Rb[2, 0, 0] == pytest.approx(y[2:, 0] @ y[:-2, 0] / 50)


@pytest.mark.parametrize("method", METHODS)
def test_modes_of_chain(chain, method):
    y, omega = chain
    r = physical(ssi(y, DT, order=12, method=method, block_rows=30))
    assert isinstance(r, SSIResult) and isinstance(r, HarmInvResult)
    assert len(r) == 3
    np.testing.assert_allclose(r.omega.real, omega.real, rtol=3e-3)
    zeta_true = -omega.imag / np.abs(omega)
    np.testing.assert_allclose(r.damping_ratio, zeta_true, rtol=0.15)
    assert np.all(np.diag(mac(r.mode_shape, true_shapes())) > 0.999)
    np.testing.assert_allclose(np.abs(r.mode_shape).max(axis=1), 1)


@pytest.mark.parametrize("method", METHODS)
def test_single_channel(chain, method):
    y, omega = chain
    r = physical(ssi(y[:, 0], DT, order=12, method=method, block_rows=40))
    assert len(r) == 3
    assert r.n_channels == 1
    np.testing.assert_allclose(r.omega.real, omega.real, rtol=5e-3)


@pytest.mark.parametrize("method", METHODS)
def test_correlation_and_psd(chain, method):
    y, _ = chain
    r = physical(ssi(y, DT, order=12, method=method, block_rows=30))
    R = output_correlation(y, 100)
    tau = DT * np.arange(100)
    scale = np.abs(R).max()
    assert np.abs(r.correlation(tau)[1:] - R[1:]).max() < 0.03 * scale
    # amplitude refers to the auto-correlation of the reference channel
    np.testing.assert_allclose(r.reconstruct(tau), r.correlation(tau)[:, 0, 0])
    f, P = welch(y[:, 0], fs=1 / DT, nperseg=4096)
    band = (f > 0.05) & (f < 0.4)
    ratio = 2 * r.psd(2 * np.pi * f[band]) / P[band]
    assert np.median(ratio) == pytest.approx(1, abs=0.05)
    S = r.psd(2 * np.pi * f[band], full=True)
    assert S.shape == (band.sum(), 3, 3)
    np.testing.assert_allclose(S, np.conj(np.swapaxes(S, -1, -2)), atol=1e-12)


def test_ref_channel(chain):
    y, _ = chain
    r = ssi(y, DT, order=12, block_rows=30, ref_channel=2)
    np.testing.assert_allclose(r.amplitude, r.residue[:, 2, 2])
    with pytest.raises(ValueError):
        ssi(y, DT, order=12, ref_channel=3)


def test_cov_and_data_agree(chain):
    y, _ = chain
    a = physical(ssi_cov(y, DT, order=12, block_rows=30))
    b = physical(ssi_data(y, DT, order=12, block_rows=30))
    np.testing.assert_allclose(a.omega.real, b.omega.real, rtol=4e-3)
    assert a.info["method"] == "cov" and b.info["method"] == "data"


def test_aliases(chain):
    y, _ = chain
    y = y[:5000]
    np.testing.assert_allclose(ssi(y, DT, 8, method="direct").omega,
                               ssi(y, DT, 8, method="data").omega)
    np.testing.assert_allclose(ssi(y, DT, 8, method="covariance").omega,
                               ssi(y, DT, 8, method="cov").omega)
    with pytest.raises(ValueError):
        ssi(y, DT, 8, method="nope")


def test_data_chunks_give_same_result(chain):
    y = chain[0][:3000]
    a = _DataRealization(y, 10, chunk=100_000)
    b = _DataRealization(y, 10, chunk=257)
    np.testing.assert_allclose(a.s, b.s, rtol=1e-8, atol=1e-12 * a.s[0])
    # the realizations agree up to a similarity transform
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(a.system(6)[0])),
                               np.sort_complex(np.linalg.eigvals(b.system(6)[0])),
                               atol=1e-10)


def test_measurement_noise_and_acceleration():
    y, omega = random_vibration(MASS, DAMP, STIFF, DT, 60_000,
                                output="acceleration", noise=0.3, rng=3)
    for method in METHODS:
        r = physical(ssi(y, DT, order=20, method=method, block_rows=30))
        found = [np.min(np.abs(r.omega - w)) for w in omega]
        assert max(found) < 0.01


def test_error_marks_spurious_modes(chain):
    y, omega = chain
    r = ssi(y[:20000], DT, order=30, block_rows=30)
    assert len(r) > 3
    idx = [np.argmin(np.abs(r.omega - w)) for w in omega]
    genuine = r.error[idx]
    others = np.delete(r.error, idx)
    assert genuine.max() < np.median(others)


def test_select_and_getitem(chain):
    y, _ = chain
    r = ssi(y, DT, order=20, block_rows=30)
    s = r.select(min_damping=0.01, max_damping=0.05)
    assert isinstance(s, SSIResult)
    assert np.all((s.damping_ratio >= 0.01) & (s.damping_ratio <= 0.05))
    sub = r[:2]
    assert len(sub) == 2 and sub.mode_shape.shape == (2, 3)
    assert sub.residue.shape == (2, 3, 3) and sub.order == 20
    assert "SSIResult" in repr(sub)
    np.testing.assert_allclose(r.natural_frequency, np.abs(r.omega))


def test_auto_order(chain):
    y, _ = chain
    r = ssi(y, DT, block_rows=30)
    assert r.order % 2 == 0 and r.order >= 6


def test_stabilization(chain):
    y, omega = chain
    st = ssi_stabilization(y[:30000], DT, range(4, 41, 2))
    assert st.orders[0] == 4 and len(st.results) == len(st.orders)
    assert not st.stable(0).any()
    o, w, z, s = st.poles(stable_only=True)
    assert s.all()
    # every physical mode is stable at most orders
    for wk in omega:
        hits = np.abs(w - wk) < 0.01 * abs(wk)
        assert len(np.unique(o[hits])) > 0.7 * len(st.orders)
    with pytest.raises(ValueError):
        ssi_stabilization(y, DT, [2, 400], block_rows=10)


def test_mac():
    a = np.array([[1, 2, 3], [1, 0, 0]], float)
    m = mac(a)
    np.testing.assert_allclose(np.diag(m), 1)
    assert m[0, 1] == pytest.approx(1 / 14)
    np.testing.assert_allclose(mac(a[0], 2j * a[0]), [[1.0]])


def test_bad_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ssi(rng.standard_normal(100) + 1j, 1.0, 4)
    with pytest.raises(ValueError):
        ssi(rng.standard_normal((3, 100)), 1.0, 2)
    with pytest.raises(ValueError):
        ssi(rng.standard_normal(1000), 1.0, 50, block_rows=10)
    with pytest.raises(ValueError):
        ssi(rng.standard_normal(1000), -1.0, 4)


def test_random_vibration_given_force():
    f = np.zeros((200, 1))
    f[0] = 1.0
    y, omega = random_vibration(1.0, 0.1, 4.0, 0.05, 200, force=f)
    # zero-order hold impulse: response of a unit force over [0, dt]
    assert y[0, 0] == 0 and y[1, 0] > 0
    y2, _ = random_vibration(1.0, 0.1, 4.0, 0.05, 200, force=2 * f)
    np.testing.assert_allclose(y2, 2 * y)
