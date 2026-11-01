# Changelog

## 1.0.0 (unreleased)

First Python 3 release, a rewrite of the earlier Python 2 `pyharminv`.

* `harminv` with two solvers: filter diagonalization (default) for long
  signals with many lines, and Padé approximants for short signals.
* Least-squares Padé (matrix pencil) as the default Padé variant; the
  classical robust SVD algorithm is available as `pade_variant="robust"`.
* `HarmInvResult` with Lorentzian parameters, error estimates, `select`,
  `reconstruct`, `spectrum`, `zeros` (anti-resonances) and `refine`
  (nonlinear least squares with error bars).
* `pyharminv.preprocessing`: `remove_mean`, `detrend` (polynomial trends)
  and linear-phase FIR `lowpass`, `highpass` and `bandpass` filters that
  are exact for sums of damped exponentials (no transients) and whose
  effect on the amplitudes is undone by `FIRFilter.correct`.
* `padeapprox`: Python translation of Chebfun's robust Padé algorithm.
* Synthetic test signals, tests, Sphinx documentation, an example script
  and four notebooks.
* Requires Python 3.10 or newer.
* Fixed the sign convention and residue errors of the Python 2 version.
