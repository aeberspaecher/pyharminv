# pyharminv

Harmonic inversion of signals in Python: decompose a sampled signal into
damped complex exponentials

    c(t) = Σ_k d_k exp(-i ω_k t),    ω_k = Ω_k − i Γ_k / 2,

and thereby fit Lorentzian lines (centre Ω_k, full width Γ_k) to its
spectrum. Unlike a Fourier transform, harmonic inversion is not limited by
the Fourier resolution `2π/T`: it resolves overlapping lines and finds
widths and amplitudes from short signals.

Two solvers are included:

* **Filter diagonalization** (default, Mandelshtam & Taylor 1997) splits the
  frequency range into windows, each solved from the *whole* signal. It
  scales to long signals with hundreds of lines and is robust to noise.
* **Padé approximants** (Wiersig & Main 2008) fit the whole band at once,
  for signals of up to a few hundred samples. The default least-squares
  variant (matrix pencil, Hua & Sarkar 1990) uses all samples and is as
  accurate as FDM under noise; `pade_variant="robust"` selects the classical
  SVD-based algorithm of Gonnet, Güttel & Trefethen (2013).

`HarmInvResult.refine` polishes either result by nonlinear least squares,
which reaches the Cramér-Rao bound and gives error bars.

## Installation

```bash
pip install .            # or: pip install -e .[test,docs,examples]
```

Requires Python ≥ 3.9, NumPy ≥ 1.22 and SciPy ≥ 1.8.

## Quick start

```python
import numpy as np
from pyharminv import harminv

dt = 0.1
t = dt * np.arange(300)
x = (np.exp(-0.05 * t) * np.cos(3.0 * t)
     + 0.8 * np.exp(-0.02 * t) * np.cos(7.0 * t + 1.2))

result = harminv(x, dt)
print(result)
#   omega.real   omega.imag      Q   |d|    arg d   error
#            3        -0.05     30     1   0.0000   ...
#            7        -0.02    175   0.8  -1.2000   ...

result.center, result.width, result.Q        # Ω_k, Γ_k, Ω_k/Γ_k
result.lorentzians()                         # centre, FWHM, height, area
result.reconstruct(t)                        # fitted signal
result.spectrum(np.linspace(0, 10, 1000))    # fitted complex spectrum
result.zeros()                               # its zeros (anti-resonances)
```

For a real signal only modes with `Re ω ≥ 0` are returned, and
`x(t) = Re Σ_k d_k exp(-i ω_k t)`.

### Noisy data

Set `tol` to the relative noise level of the data (noise standard deviation
divided by the rms of the signal), then drop spurious weak modes:

```python
result = harminv(x, dt, tol=1e-3)
lines = result.select(min_magnitude=0.05, max_rel_error=1e-3)
```

Every mode carries an error estimate (`result.error`); genuine lines have
much smaller errors than noise-induced ones. Avoid long stretches of pure
noise after the signal has decayed: they add noise but no information.

For the best accuracy, refine the selected lines:

```python
fit = lines.refine(x, dt)
fit.omega, fit.info["omega_std"]   # maximum-likelihood estimate, 1σ errors
```

A smaller `tol` (e.g. a tenth of the noise level) followed by `select` and
`refine` resolves lines closer than the Fourier resolution at moderate SNR.

### Long signals

`harminv` chooses frequency windows automatically. Restrict the search with
`omega_min`/`omega_max`, or set `n_windows`, `window_width` or the basis
size `K` explicitly. 100 000 samples with 200 lines take about 5 s.

## Example

`examples/lorentz_fit.py` fits twelve Lorentzian lines, including two closer
than the Fourier resolution, to a noisy sum of damped cosines and plots the
result:

![example](https://raw.githubusercontent.com/aeberspaecher/pyharminv/main/examples/lorentz_fit.png)

## Notebooks

`notebooks/` contains four illustrated, executed notebooks:

* `01_harmonic_inversion_basics.ipynb`: damped cosines, recovered
  parameters, Lorentzian lines versus the Fourier spectrum, and two lines
  closer than the Fourier resolution.
* `02_noise_and_long_signals.ipynb`: noisy data, the choice of `tol`,
  separating spurious modes, 100 lines in 50 000 samples, frequency
  windows; how far the signal-to-noise ratio can go (Cramér-Rao bound,
  `refine`, detection limit, super-resolution versus SNR, record length);
  and Padé in depth (pole pictures, rank decision, classical versus
  least-squares Padé, FDM versus Padé).
* `03_pade_approximation.ipynb`: Padé approximation as a tool in its own
  right: beyond the radius of convergence, branch cuts, pole location,
  divergent series, Froissart doublets and the robust algorithm, and the
  link to harmonic inversion.
* `04_anti_resonances.ipynb`: anti-resonances (zeros of a frequency
  response) from the fitted modes with `HarmInvResult.zeros()`: a damped
  spring chain, driving-point versus transfer responses, the clamped-system
  interpretation, a short noisy record where the FFT shows no dips, and
  why zeros need all modes (residual terms).

```bash
pip install -e .[examples]
jupyter notebook notebooks/
```

## Tests

```bash
pip install -e .[test]
pytest
```

The tests use synthetic signals with known frequencies, widths and
amplitudes: clean and noisy, real and complex, many lines in long signals,
overlapping lines, lines on window boundaries, and Lorentzian fits to the
Fourier spectrum. The Padé core is tested against known approximants and the
example from `pade_test.m`.

## Documentation

```bash
pip install -e .[docs]
sphinx-build docs docs/_build/html
```

The documentation covers the theory, the conventions, practical advice and
the API.

## Building a release

```bash
pip install build twine
python -m build              # dist/pyharminv-<version>.tar.gz and .whl
twine check dist/*
twine upload dist/*          # needs a PyPI account and token
```

The version is set in `src/pyharminv/__init__.py`; see `CHANGELOG.md`.

## References

1. J. Wiersig and J. Main, Phys. Rev. E **77**, 036205 (2008).
2. P. Gonnet, S. Güttel and L. N. Trefethen, SIAM Review **55**, 101 (2013).
3. Y. Hua and T. K. Sarkar, IEEE Trans. Acoust. Speech Signal Process.
   **38**, 814 (1990).
4. V. A. Mandelshtam and H. S. Taylor, J. Chem. Phys. **107**, 6756 (1997).
5. M. R. Wall and D. Neuhauser, J. Chem. Phys. **102**, 8011 (1995).

## License

BSD 3-clause, see `LICENSE`. `src/pyharminv/pade.py` is a translation of `padeapprox.m` from
[Chebfun](http://www.chebfun.org) (BSD 3-clause, see `LICENSES/`).
The earlier Python 2 code and the reference material are kept in `legacy/`
and `references/`.
