Theory: stochastic subspace identification
==========================================

Harmonic inversion fits a *deterministic* sum of damped exponentials to a
signal, which suits free decays and impulse responses. Stochastic subspace
identification (SSI) solves the corresponding problem for systems driven by
an unknown, random excitation: it identifies the poles and mode shapes of
a linear system from its response alone ("output-only" or "operational"
modal analysis). This page derives the two variants implemented in
:func:`~pyharminv.ssi_cov` and :func:`~pyharminv.ssi_data`, shows how they
relate to each other and to harmonic inversion, and discusses order
selection, error estimates and the effect of noise. The standard references
are the monograph of Van Overschee and De Moor [VODM1996]_ and the papers of
Peeters and De Roeck [PeetersDeRoeck1999]_, [PeetersDeRoeck2001]_; Reynders
[Reynders2012]_ gives a thorough review, and Brincker and Ventura
[BrinckerVentura2015]_ an introduction to operational modal analysis.

The stochastic state-space model
--------------------------------

A linear mechanical system with mass, damping and stiffness matrices
:math:`M, C_d, K` and force :math:`f(t)`,
:math:`M\ddot q + C_d \dot q + K q = f`, becomes a first-order system for
the state :math:`x = (q, \dot q)`,

.. math::

   \dot x = A_c x + B_c f, \qquad
   A_c = \begin{pmatrix} 0 & I \\ -M^{-1}K & -M^{-1}C_d \end{pmatrix}.

Sampled at :math:`t_k = k\,\Delta t` with the force held over each
interval, the state evolves exactly as :math:`x_{k+1} = e^{A_c \Delta t}
x_k + \dots`. If the force is unknown and random, SSI models the sampled
system as

.. math::

   x_{k+1} = A x_k + w_k, \qquad y_k = C x_k + v_k,

with the state :math:`x_k \in \mathbb{R}^n`, the measured outputs
:math:`y_k \in \mathbb{R}^l` (``l`` channels), the *process noise*
:math:`w_k` (the excitation, mapped into the state) and the *measurement
noise* :math:`v_k`. Both are assumed zero-mean, white and jointly
stationary,

.. math::

   E\begin{bmatrix} w_p \\ v_p \end{bmatrix}
    \begin{bmatrix} w_q^T & v_q^T \end{bmatrix}
   = \begin{pmatrix} Q & S \\ S^T & R \end{pmatrix} \delta_{pq}.

The dimension :math:`n` of the state is the **model order**: two per
oscillating mode, plus whatever is needed to describe the colour of the
excitation and other non-modal effects. The system is assumed stable, so
the state covariance :math:`\Sigma = E[x_k x_k^T]` is stationary and solves
the Lyapunov equation :math:`\Sigma = A \Sigma A^T + Q`.

Poles, frequencies and damping
------------------------------

The eigenvalues :math:`\lambda_j` of :math:`A` are the discrete poles. They
are related to the continuous-time eigenvalues :math:`s_j` of
:math:`A_c` by :math:`\lambda_j = e^{s_j \Delta t}`. In the convention of
pyharminv, :math:`e^{-i\omega t}` with decaying modes having
:math:`\mathrm{Im}\,\omega < 0`, this gives

.. math::

   \lambda_j = e^{-i \omega_j \Delta t}, \qquad
   \omega_j = \frac{i}{\Delta t} \ln \lambda_j = i s_j
            = \Omega_j - \tfrac{i}{2}\Gamma_j ,

the same relation as between :math:`z_k` and :math:`\omega_k` in harmonic
inversion. For a mode of a weakly damped oscillator,
:math:`s = -\zeta\omega_0 \pm i\omega_0\sqrt{1-\zeta^2}`, so

.. math::

   |\omega_j| = \omega_0, \qquad
   \zeta_j = -\frac{\mathrm{Im}\,\omega_j}{|\omega_j|}, \qquad
   Q_j = \frac{\Omega_j}{\Gamma_j} \approx \frac{1}{2\zeta_j},

available as :attr:`~pyharminv.SSIResult.natural_frequency`,
:attr:`~pyharminv.SSIResult.damping_ratio` and
:attr:`~pyharminv.HarmInvResult.Q`. Since :math:`A` is real, the poles come
in conjugate pairs :math:`\lambda, \bar\lambda`, i.e. :math:`\omega` and
:math:`-\bar\omega`; as for real signals in harmonic inversion only the
member with :math:`\mathrm{Re}\,\omega \ge 0` is listed. A real positive
:math:`\lambda` is an overdamped (purely decaying) pole with
:math:`\mathrm{Re}\,\omega = 0`; a real negative :math:`\lambda` lies at the
Nyquist frequency :math:`\pi/\Delta t`.

With the eigendecomposition :math:`A = \Psi \Lambda \Psi^{-1}`, the
**mode shapes** are the columns of :math:`\Phi = C\Psi`, the observed parts
of the eigenvectors. For a mechanical system with displacement sensors they
are the (complex) modal displacements at the sensor positions. Mode shapes
are defined up to a complex factor; pyharminv scales each so that its
largest component is 1.

Output correlations
-------------------

The key property of the stochastic model is the form of its output
correlations :math:`R_k = E[y_{m+k} y_m^T]`. Since :math:`w_m` and
:math:`v_m` are uncorrelated with :math:`x_m` and with all later noise,

.. math::

   R_0 = C \Sigma C^T + R, \qquad
   R_k = C A^{k-1} G \quad (k \ge 1), \qquad
   G = E[x_{k+1} y_k^T] = A \Sigma C^T + S .

Two consequences carry the whole method:

* **For** :math:`k \ge 1` **the correlations decay like a free response.**
  In modal form,

  .. math::

     R_k = \sum_j \phi_j\, g_j^T \lambda_j^{k-1}, \qquad
     \phi_j = C\psi_j, \quad g_j^T = (\Psi^{-1} G)_{j,:},

  a sum of damped exponentials with the poles of the system, exactly the
  model of harmonic inversion. This is the *natural excitation technique*
  (NExT) of James, Carne and Lauffer [JamesCarneLauffer1995]_: the
  correlation function of a randomly excited system can be analysed like an
  impulse response. Each mode contributes a rank-one residue matrix
  :math:`\phi_j g_j^T`: its column space is the mode shape, its row space
  says how strongly the mode correlates with each channel.

* **White measurement noise only enters** :math:`R_0`. It is uncorrelated
  over time and with the response, so it adds :math:`R` at lag zero and
  nothing elsewhere. Methods that use :math:`R_1, R_2, \dots` only are not
  biased by it.

:class:`~pyharminv.SSIResult` stores the model correlation function in the
conventions of :class:`~pyharminv.HarmInvResult` for a real signal,

.. math::

   R(\tau) = \mathrm{Re} \sum_{j:\,\mathrm{Re}\,\omega_j \ge 0}
             M_j\, e^{-i\omega_j \tau}, \qquad
   M_j = \frac{2\,\phi_j g_j^T}{\lambda_j}
   \quad (\text{once, not twice, for real } \lambda_j),

valid for :math:`\tau \ge \Delta t`. The factor :math:`1/\lambda_j`
shifts the reference from lag one to lag zero, and the factor two collects
a pole and its conjugate. :attr:`~pyharminv.SSIResult.residue` holds the
:math:`M_j`, and :attr:`~pyharminv.HarmInvResult.amplitude` the diagonal
element of a reference channel, so that the inherited
:meth:`~pyharminv.HarmInvResult.reconstruct`,
:meth:`~pyharminv.HarmInvResult.spectrum` and
:meth:`~pyharminv.HarmInvResult.select` refer to that channel's
auto-correlation. The power spectral density follows from
:math:`R(-\tau) = R(\tau)^T`: with the one-sided transform
:math:`F(\omega) = \int_0^\infty R(\tau) e^{i\omega\tau} d\tau
= \sum_j \bigl[\tfrac{i}{2} M_j/(\omega - \omega_j)
+ \tfrac{i}{2} \bar M_j/(\omega + \bar\omega_j)\bigr]`,

.. math::

   S(\omega) = \int_{-\infty}^{\infty} R(\tau) e^{i\omega\tau} d\tau
             = F(\omega) + F(\omega)^H ,

a sum of Lorentzian-like terms, which :meth:`~pyharminv.SSIResult.psd`
evaluates. It is the two-sided density per unit angular frequency; the
one-sided density per hertz returned by :func:`scipy.signal.welch` is
:math:`2 S(2\pi f)`.

Correlations are estimated by

.. math::

   \hat R_k = \frac{1}{N-k} \sum_{m=0}^{N-k-1} y_{m+k}\, y_m^T,

computed for all lags and channel pairs at once with FFTs
(:func:`~pyharminv.output_correlation`).

Covariance-driven SSI
---------------------

Stack the correlations of lags :math:`1, \dots, 2i-1` into the block
Toeplitz matrix of :math:`i` block rows and columns. By the factorisation
above,

.. math::

   T_{1|i} =
   \begin{pmatrix}
      R_i      & R_{i-1} & \cdots & R_1 \\
      R_{i+1}  & R_i     & \cdots & R_2 \\
      \vdots   &         & \ddots & \vdots \\
      R_{2i-1} & R_{2i-2} & \cdots & R_i
   \end{pmatrix}
   =
   \underbrace{\begin{pmatrix} C \\ CA \\ \vdots \\ CA^{i-1} \end{pmatrix}}_{O_i}
   \underbrace{\begin{pmatrix} A^{i-1}G & \cdots & AG & G \end{pmatrix}}_{\Gamma_i},

the product of the extended observability matrix :math:`O_i` and the
reversed controllability matrix :math:`\Gamma_i`. Both have rank :math:`n`
as long as :math:`il \ge n`, so :math:`T_{1|i}` has rank :math:`n`. This is
the stochastic version of the Ho-Kalman realization from impulse response
samples [HoKalman1966]_; using the SVD for it goes back to Kung
[Kung1978]_ and, for the stochastic problem, to Akaike [Akaike1974]_.

The algorithm:

1. Compute the SVD :math:`T_{1|i} = U S V^T` and keep the :math:`n` largest
   singular values, :math:`T_{1|i} \approx U_1 S_1 V_1^T`.
2. Split the factors symmetrically,

   .. math::

      O_i = U_1 S_1^{1/2}, \qquad \Gamma_i = S_1^{1/2} V_1^T .

   Any factorisation :math:`O_i \mathcal T`, :math:`\mathcal T^{-1}\Gamma_i`
   with an invertible :math:`\mathcal T` is equally valid: it describes
   the same system in other state coordinates, with the same poles and
   mode shapes. The symmetric split gives a balanced realization.
3. Read off :math:`C` as the first block row of :math:`O_i` and :math:`G` as
   the last block column of :math:`\Gamma_i`.
4. Find :math:`A` from the *shift invariance* of :math:`O_i`: deleting the
   first block row gives the same matrix as deleting the last one and
   multiplying by :math:`A`,

   .. math::

      \underline{O_i} = \overline{O_i}\, A
      \quad\Longrightarrow\quad
      A = \overline{O_i}^{\,\dagger}\, \underline{O_i}

   in the least-squares sense (:math:`\overline{O_i}`: without the last,
   :math:`\underline{O_i}`: without the first block row). This needs
   :math:`(i-1)\,l \ge n`, the largest order allowed for a given number of
   block rows.

All model orders share the same SVD: the realization of order :math:`n`
uses the first :math:`n` singular triplets. That is why a full
stabilization diagram (below) costs hardly more than a single
identification. After the FFT for the correlations, the cost no longer
depends on the record length: an SVD of a matrix of size :math:`il`.

**Relation to harmonic inversion.** For a single channel,
:math:`T_{1|i}` is a Toeplitz matrix of the scalar correlation sequence;
reversing its columns makes it a Hankel matrix, and steps 1–4 are then the
least-squares Padé (matrix pencil) method of :func:`~pyharminv.harminv_window`
applied to :math:`R_1, R_2, \dots`. Covariance-driven SSI is NExT plus
harmonic inversion, carried out for all channels at once: the channels share
one set of poles, and the mode shapes come out of the realization.

Data-driven SSI
---------------

Data-driven SSI avoids estimating correlations. It builds the block Hankel
matrix of the outputs, with :math:`2i` block rows and :math:`j = N-2i+1`
columns, and splits it into a *past* and a *future* half,

.. math::

   \begin{pmatrix} Y_p \\ Y_f \end{pmatrix}
   = \frac{1}{\sqrt j}
   \begin{pmatrix}
      y_0 & y_1 & \cdots & y_{j-1} \\
      \vdots & & & \vdots \\
      y_{i-1} & y_i & \cdots & y_{i+j-2} \\ \hline
      y_i & y_{i+1} & \cdots & y_{i+j-1} \\
      \vdots & & & \vdots \\
      y_{2i-1} & y_{2i} & \cdots & y_{2i+j-2}
   \end{pmatrix}.

The scaling makes products of rows averages over time: :math:`Y_f Y_p^T` is
an estimate of :math:`T_{1|i}`.

**Projection.** The orthogonal projection of the future onto the row space
of the past,

.. math::

   P_i = Y_f / Y_p = Y_f Y_p^T \left(Y_p Y_p^T\right)^{\dagger} Y_p ,

is the best linear prediction of the future outputs from the past ones. For
:math:`j \to \infty` it factors as

.. math::

   P_i = O_i \hat X_i ,

where :math:`\hat X_i = (\hat x_i, \hat x_{i+1}, \dots)` are the state
estimates of a bank of Kalman filters, each run over the :math:`i` past
outputs of its column [VODM1993]_, [VODM1996]_. The future part of the
noise is unpredictable from the past and drops out; this is how the
data-driven variant removes measurement noise. Shifting the border between
past and future by one block row gives, in the same way,
:math:`P_{i-1} = Y_f^- / Y_p^+ = O_{i-1} \hat X_{i+1}`, where
:math:`Y_p^+` includes the block row :math:`Y_{i|i}` of outputs at time
:math:`i` and :math:`Y_f^-` excludes it.

**The algorithm** (unweighted principal components, UPC):

1. Compute the projections from an LQ factorisation of the Hankel matrix,
   :math:`\bigl(\begin{smallmatrix}Y_p\\Y_{i|i}\\Y_f^-\end{smallmatrix}\bigr)
   = L\,Q^T` with lower triangular :math:`L` and orthonormal :math:`Q`.
   In this basis :math:`P_i = L_{2:3,1}\, Q_1^T` and
   :math:`P_{i-1} = L_{3,1:2}\, Q_{1:2}^T`; all following steps work with the
   small blocks of :math:`L` only, and :math:`Q` is never formed.
2. SVD :math:`P_i = U_1 S_1 V_1^T + \dots`, truncated to order :math:`n`;
   :math:`O_i = U_1 S_1^{1/2}`.
3. States: :math:`\hat X_i = O_i^\dagger P_i` and
   :math:`\hat X_{i+1} = \overline{O_i}^{\,\dagger} P_{i-1}`.
4. Solve the least-squares problem

   .. math::

      \begin{pmatrix} \hat X_{i+1} \\ Y_{i|i} \end{pmatrix}
      = \begin{pmatrix} A \\ C \end{pmatrix} \hat X_i
      + \begin{pmatrix} \rho_w \\ \rho_v \end{pmatrix}

   for :math:`A` and :math:`C`. The residuals :math:`\rho` estimate the
   noise sequences, whose covariances give :math:`Q, R, S`.
5. :math:`G = \hat X_{i+1} Y_{i|i}^T`, the sample version of
   :math:`E[x_{k+1} y_k^T]`, gives the correlation residues as above.

The LQ factor is accumulated over blocks of columns (a QR factorisation of
:math:`H^T`, updated block by block), so the memory stays bounded by the
size of :math:`L`, :math:`(2il)^2`, even for very long records. The cost
grows like :math:`j\,(2il)^2`, linearly in the record length.

**Relation between the two variants.** Write the LQ factor of the past as
:math:`Y_p = L_{11} Q_1^T`. Then

.. math::

   P_i = Y_f Y_p^T (L_{11} L_{11}^T)^{-1} L_{11} Q_1^T
       = T_{1|i}\, L_{11}^{-T}\, Q_1^T ,

so the singular values and left singular vectors of :math:`P_i` are those
of :math:`T_{1|i} L_{11}^{-T}`: data-driven SSI is the covariance-driven
algorithm with a weighting of the columns by the inverse square root of the
past covariance (and with :math:`A, C` from the states instead of from the
shift structure). Both estimate the same model and are consistent; they
differ in how finite-sample errors are weighted. Van Overschee and De Moor
place SSI algorithms in a common framework with weightings
:math:`W_1 O_i \hat X_i W_2`; principal components, canonical variate
analysis (CVA) and others differ only in these weights [VODM1996]_. In
practice the two variants give statistically equivalent results
[PeetersDeRoeck2001]_ (see notebook 06); the data-driven one is
numerically more robust, never forms products of the data, and gives the
Kalman states, at a higher cost.

Model order, spurious poles and error estimates
-----------------------------------------------

With exact correlations, :math:`T_{1|i}` has exactly :math:`n` non-zero
singular values. With estimated ones, the remaining singular values form a
floor that decreases only like :math:`1/\sqrt N`, and weakly excited modes
are often not clearly separated from it. The realization of order
:math:`n` then contains three kinds of poles:

* **physical poles** of the structure,
* **noise poles** that fit the estimation errors of the correlations
  (usually strongly damped and erratic),
* **excitation poles**: the model describes the system *and* the filter
  that colours the excitation, so a coloured excitation adds poles of its
  own, and a harmonic component adds a pole on the unit circle
  (:math:`\zeta \approx 0`).

Practice is therefore to **overestimate** the order and sort the poles
afterwards. Physical poles are stable: they barely move when the order
changes, since the realization of order :math:`n+2` contains that of order
:math:`n` in its leading singular triplets. Noise poles jump. pyharminv
uses this in two ways.

* :attr:`~pyharminv.HarmInvResult.error` is the distance of each pole to
  the closest pole of the realization of order :math:`n + 2` (or
  :math:`n - 2` if :math:`n + 2` exceeds the maximum order), in the same
  spirit as the error estimate of harmonic inversion, which compares two
  independent fits. :meth:`~pyharminv.SSIResult.select` with
  ``max_rel_error`` keeps the stable poles; ``min_damping`` and
  ``max_damping`` remove harmonic components and noise poles, and
  ``min_magnitude`` removes poles that hardly contribute to the reference
  channel's correlation.

* :func:`~pyharminv.ssi_stabilization` computes the realizations for a
  whole list of orders and marks a pole at order :math:`n` as stable if a
  pole at the previous order in the list has a natural frequency within
  ``freq_tol`` (relative), a damping ratio within ``damping_tol``
  (relative) and, with several channels, a mode shape with a **modal
  assurance criterion**

  .. math::

     \mathrm{MAC}(\phi_a, \phi_b)
     = \frac{|\phi_a^H \phi_b|^2}{(\phi_a^H\phi_a)(\phi_b^H\phi_b)}

  of at least ``mac_tol`` [AllemangBrown1982]_. In the stabilization
  diagram, physical modes form vertical lines of stable poles.

Without an explicit order, :func:`~pyharminv.ssi` places it after the last
drop of the singular values by more than a factor of two (rounded up to an
even number). This heuristic works for clean data; the stabilization
diagram is the reliable tool.

Block rows
----------

The number of block rows :math:`i` sets the longest lag used,
:math:`(2i-1)\Delta t`, and the largest order, :math:`(i-1)\,l`.

* :math:`i\,\Delta t` should cover at least about one period of the slowest
  mode of interest, better several, so that the decay of its correlation is
  seen.
* More block rows reduce the bias of the realization (more lags, more
  equations for the shift invariance), but the correlations at large lags
  are estimated from fewer products and are noisier, and the cost grows.
* Strong oversampling crowds all poles near :math:`\lambda = 1` and demands
  many block rows to span a period; decimating to a few times the highest
  frequency of interest helps.

Typical values are 20 to 100 block rows; :func:`~pyharminv.ssi` defaults to
enough rows for the order and at least 64 rows in total.

Statistical errors and noise
----------------------------

Unlike a free decay, a randomly excited response carries the randomness of
the excitation itself. Even with perfect sensors, the correlation
estimates scatter like :math:`1/\sqrt N`, and so do the identified poles.
Damping ratios are the most sensitive quantity; their relative error
decreases roughly like :math:`1/\sqrt{T/\tau}`, with :math:`\tau` the decay
time of the mode. When a free decay can be measured, harmonic inversion of
it is therefore far more accurate per unit of measurement time
(notebook 06).

**Measurement noise.** White measurement noise does not bias either
variant, as shown above, but it inflates the variance of the correlation
estimates, which scales with the total variance of the data. With noise
standard deviation :math:`\sigma_n` and signal standard deviation
:math:`\sigma_s`,

.. math::

   \frac{\delta R_k}{R_k} \sim
   \frac{\sigma_s^2 + \sigma_n^2}{\sigma_s^2 \sqrt N},

so at low signal-to-noise ratio ten times more noise requires up to ten
thousand times more samples. In practice a mode is identified reliably as
long as its singular values stand clearly above the floor, and lost quickly
beyond (notebook 07). Measurement noise that is *not* white, for example
after narrow band-pass filtering, enters many lags and biases the result.

**Assumptions worth checking.** The theory assumes a linear,
time-invariant system, a stationary response and an excitation that is
white or at least broadband and smooth across the modes. Remove means and
drifts first (:func:`~pyharminv.remove_mean`, :func:`~pyharminv.detrend`).
Harmonic excitation, e.g. from rotating machinery, is identified as an
undamped "mode"; narrow-band excitation produces stable poles that cannot
be told from structural ones by the data alone.

References
----------

.. [VODM1993] P. Van Overschee and B. De Moor, "Subspace algorithms for the
   stochastic identification problem", Automatica 29, 649-660 (1993).
.. [VODM1996] P. Van Overschee and B. De Moor, *Subspace Identification for
   Linear Systems: Theory, Implementation, Applications*, Kluwer (1996).
.. [PeetersDeRoeck1999] B. Peeters and G. De Roeck, "Reference-based
   stochastic subspace identification for output-only modal analysis",
   Mech. Syst. Signal Process. 13, 855-878 (1999).
.. [PeetersDeRoeck2001] B. Peeters and G. De Roeck, "Stochastic system
   identification for operational modal analysis: a review", J. Dyn. Syst.
   Meas. Control 123, 659-667 (2001).
.. [Reynders2012] E. Reynders, "System identification methods for
   (operational) modal analysis: review and comparison", Arch. Comput.
   Methods Eng. 19, 51-124 (2012).
.. [JamesCarneLauffer1995] G. H. James, T. G. Carne and J. P. Lauffer, "The
   natural excitation technique (NExT) for modal parameter extraction from
   operating structures", Int. J. Anal. Exp. Modal Anal. 10, 260-277
   (1995).
.. [HoKalman1966] B. L. Ho and R. E. Kalman, "Effective construction of
   linear state-variable models from input/output functions",
   Regelungstechnik 14, 545-548 (1966).
.. [Kung1978] S. Y. Kung, "A new identification and model reduction
   algorithm via singular value decomposition", Proc. 12th Asilomar Conf.
   on Circuits, Systems and Computers, 705-714 (1978).
.. [Akaike1974] H. Akaike, "Stochastic theory of minimal realization",
   IEEE Trans. Autom. Control 19, 667-674 (1974).
.. [AllemangBrown1982] R. J. Allemang and D. L. Brown, "A correlation
   coefficient for modal vector analysis", Proc. 1st Int. Modal Analysis
   Conf., 110-116 (1982).
.. [BrinckerVentura2015] R. Brincker and C. Ventura, *Introduction to
   Operational Modal Analysis*, Wiley (2015).
