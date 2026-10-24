# Changelog

## 1.0.0 (unreleased)

First Python 3 release, a rewrite of the earlier Python 2 `pyharminv`.

* `harminv` with two solvers: filter diagonalization (default) for long
  signals with many lines, and Padé approximants for short signals.
* Least-squares Padé (matrix pencil) as the default Padé variant; the
  classical robust SVD algorithm is available as `pade_variant="robust"`.
* `HarmInvResult` with Lorentzian parameters, error estimates, `select`,
  `reconstruct`, `spectrum` and `refine` (nonlinear least squares with
  error bars).
* `padeapprox`: Python translation of Chebfun's robust Padé algorithm.
* Synthetic test signals, tests, Sphinx documentation, an example script
  and three notebooks.
* Fixed the sign convention and residue errors of the Python 2 version.
