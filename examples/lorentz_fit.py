"""Fit Lorentzian lines to a noisy sum of damped cosines.

Run with ``python examples/lorentz_fit.py``; writes ``lorentz_fit.png``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyharminv import harminv, lorentzian, synthetic_signal
from pyharminv.signals import random_modes

rng = np.random.default_rng(1)

# ten damped cosines with quality factors 30 ... 150, plus a pair of lines
# closer than the Fourier resolution of the signal
omega, _ = random_modes(10, 2, 20, q_min=30, q_max=150, min_spacing=0.8,
                        rng=rng)
omega = np.append(omega, [8.3 - 0.04j, 8.36 - 0.05j])
amp = rng.uniform(0.5, 1.5, len(omega))   # real amplitudes: pure cosines

dt = 0.05
t = dt * np.arange(3000)
noise = 2e-3
x = synthetic_signal(t, omega, amp, real=True, noise=noise, rng=rng)

# harmonic inversion; tol is the relative noise level of the data
rel_noise = noise / np.sqrt(np.mean(x ** 2))
result = harminv(x, dt, omega_max=25, tol=rel_noise)
modes = result.select(min_magnitude=0.05)
print(modes)

lines = modes.lorentzians()
for c, g, h in zip(lines["center"], lines["fwhm"], lines["height"]):
    print(f"Lorentzian: center {c:8.4f}  FWHM {g:7.4f}  height {h:8.3f}")

# real part of the one-sided Fourier transform of the data
w = np.linspace(1, 21, 6000)
spectrum = (np.exp(1j * np.outer(w, t)) @ x) * dt

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 10),
                                    constrained_layout=True)
ax0.plot(t, x, lw=0.5)
ax0.set_xlim(0, 60)
ax0.set_xlabel("$t$")
ax0.set_ylabel("signal")

ax1.plot(w, spectrum.real, color="0.6", lw=2.5, label="Fourier transform of data")
ax1.plot(w, lorentzian(w, lines["center"], lines["fwhm"], lines["height"]),
         color="C0", lw=1, label="Lorentzians from harmonic inversion")
ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel("Re spectrum")
ax1.legend()
inset = ax1.inset_axes([0.55, 0.35, 0.4, 0.35])
zoom = (w > 7.9) & (w < 8.8)
inset.plot(w[zoom], spectrum.real[zoom], color="0.6", lw=2.5)
for c, g, h in zip(lines["center"], lines["fwhm"], lines["height"]):
    if 7.9 < c < 8.8:
        inset.plot(w[zoom], lorentzian(w[zoom], c, g, h), "--", lw=1)
inset.set_title("overlapping pair", fontsize=9)

ax2.scatter(omega.real, omega.imag, s=80, facecolors="none", edgecolors="k",
            label="true")
ax2.scatter(modes.omega.real, modes.omega.imag, marker="x", color="C3",
            label="found")
ax2.set_xlabel(r"Re $\omega$")
ax2.set_ylabel(r"Im $\omega$")
ax2.legend()
fig.savefig("lorentz_fit.png", dpi=110)
