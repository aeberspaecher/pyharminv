import numpy as np
import pytest

from pyharminv import padeapprox, taylor_coeffs


def test_exp_22_matches_known_approximant():
    # the (2, 2) Padé approximant of exp is (1 + x/2 + x^2/12) / (1 - x/2 + x^2/12)
    c = 1 / np.array([1, 1, 2, 6, 24])
    r = padeapprox(c, 2, 2)
    np.testing.assert_allclose(r.a, [1, 1 / 2, 1 / 12], rtol=1e-12)
    np.testing.assert_allclose(r.b, [1, -1 / 2, 1 / 12], rtol=1e-12)
    assert (r.mu, r.nu) == (2, 2)


def test_rational_function_is_recovered_exactly():
    # f(x) = (1 + 2x) / ((1 - 0.5x)(1 + 0.25x)) has poles at 2 and -4
    p = np.array([1.0, 2.0])
    q = np.polymul([1.0, -0.5][::-1], [1.0, 0.25][::-1])[::-1]
    f = lambda x: np.polyval(p[::-1], x) / np.polyval(q[::-1], x)
    r = padeapprox(f, 1, 2)
    np.testing.assert_allclose(np.sort(r.poles.real), [-4, 2], rtol=1e-10)
    x = np.linspace(-1, 1, 7)
    np.testing.assert_allclose(r(x), f(x), rtol=1e-10)


def test_rank_reduction_removes_excess_degrees():
    # 1/(1 - x) is exactly of type (0, 1); asking for (5, 5) must reduce
    r = padeapprox(np.ones(20), 5, 5)
    assert (r.mu, r.nu) == (0, 1)
    np.testing.assert_allclose(r.poles, [1.0])
    np.testing.assert_allclose(r.residues, [-1.0])


def test_without_robustness_spurious_poles_appear():
    # with tol = 0 the same data yields Froissart doublets
    noisy = np.ones(20) + 1e-12 * np.random.default_rng(0).standard_normal(20)
    assert padeapprox(noisy, 5, 5, tol=0).nu > 1
    assert padeapprox(noisy, 5, 5, tol=1e-10).nu == 1


def test_residues_analytic_and_finite_difference_agree():
    c = 0.5 ** np.arange(10) + (-0.25) ** np.arange(10)
    ra = padeapprox(c, 1, 2, residues="analytic")
    rf = padeapprox(c, 1, 2, residues="finite-difference")
    order_a = np.argsort(ra.poles.real)
    order_f = np.argsort(rf.poles.real)
    np.testing.assert_allclose(ra.residues[order_a], rf.residues[order_f],
                               rtol=1e-6)
    # 1/(1 - z x) has residue -1/z at x = 1/z; here z = -0.25 and 0.5
    np.testing.assert_allclose(ra.residues[order_a], [4.0, -2.0], rtol=1e-10)


def test_matlab_example_from_pade_test():
    # pade_test.m: f = [1 1 (1:50)], m = 2, n = 5, tol = 0
    f = np.concatenate([[1, 1], np.arange(1, 51)])
    r = padeapprox(f, 2, 5, tol=0)
    # the approximant must reproduce the first m + n + 1 Taylor coefficients
    x = 1e-3 * np.exp(2j * np.pi * np.arange(64) / 64)
    series = np.polyval(f[:8][::-1], x)
    np.testing.assert_allclose(r(x), series, rtol=1e-12)


def test_zero_function():
    r = padeapprox(np.zeros(5), 2, 2)
    assert r.mu == -np.inf and r.nu == 0
    np.testing.assert_array_equal(r(np.array([0.3])), [0.0])


def test_taylor_coeffs_of_polynomial():
    c = taylor_coeffs(lambda x: 3 * x ** 3 + 2 * x ** 2 + x + 0.1, n=16)
    np.testing.assert_allclose(c[:5], [0.1, 1, 2, 3, 0], atol=1e-14)
    assert not np.iscomplexobj(c)


def test_unknown_residue_method():
    with pytest.raises(ValueError):
        padeapprox(np.ones(5), 1, 1, residues="magic")


def test_object_array_of_large_integers():
    import math
    c = [1 / math.factorial(k) for k in range(5)]
    big = np.array([math.factorial(k) for k in range(25)])   # dtype object
    assert big.dtype == object
    r = padeapprox(1 / big, 2, 2)
    np.testing.assert_allclose(r.a, padeapprox(c, 2, 2).a)
