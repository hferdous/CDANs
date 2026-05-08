"""Independent-change-principle direction inference.

Implements the score function described in:

    Huang, B., Zhang, K., Sanchez-Romero, R., Ramsey, J., Glymour, C.,
    Glymour, M. (2017/2020). *Behind Distribution Shift: Mining Driving
    Forces of Changes and Causal Arrows.* (CD-NOD).

Direct port of the MATLAB ``infer_nonsta_dir.m`` from the CDANs reference
repo (which itself is the CD-NOD reference implementation by the
original authors). No external causal-discovery libraries are used.

Idea
----

Suppose ``P_t(parents)`` and ``P_t(effect | parents)`` are the two
"modules" of the joint distribution of ``(parents, effect)``, indexed
by a non-stationarity surrogate ``C`` (typically time). Under the
independent-change principle of Huang & Zhang, if ``parents -> effect``
is the correct causal direction, these two modules should change
**independently** as ``C`` varies. If we tested the wrong direction
(``effect -> parents``), regressing parents on effect would mix the
changes in ``P_t(effect)`` and ``P_t(parents | effect)``, making the
two modules' variations look correlated.

The score below quantifies the dependence between (a) a kernel
representation of ``P_t(effect | parents)`` and (b) a kernel
representation of ``P_t(parents)``, both as functions of ``C``. **Lower
score = more independent change = stronger evidence the assumed
direction is correct.**

Used in :mod:`cdans.steps.step4_orient` to break ties on undirected
edges between two changing modules — the case where v-structure
detection plus Meek's rules cannot decide direction on their own.
"""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import pinv

from cdans.ci_tests._kernels import gaussian_kernel, median_bandwidth

# Type alias: a kernel bandwidth is either a positive float, the string
# "auto", or the string "gp".
WidthSpec = Union[float, str]
AUTO = "auto"
GP = "gp"

# Empirically-derived scaling factor for the X/Y auto bandwidth.
# The independent-change algorithm requires a narrow kernel to resolve
# the local structure of P(effect | parents); the bare median heuristic
# (which gives sigma ~ 1.6 on standardized normal data) is too wide.
# Sweep over fractions on synthetic non-stationary data shows
# fractions in [0.02, 0.08] all give >= 90% direction-recovery
# accuracy across 60 trials; 0.10 drops to 72% and 0.15 fails.
_AUTO_WIDTH_XY_SCALE = 0.05


def independent_change_score(
    parents: NDArray[np.float64],
    effect: NDArray[np.float64],
    surrogate: NDArray[np.float64],
    *,
    width: WidthSpec = AUTO,
    width_t: WidthSpec = AUTO,
    lambda_reg: float = 2.0,
) -> float:
    """Score the hypothesis ``parents -> effect`` under non-stationarity.

    Returns a scalar dependence statistic between the kernel
    representation of ``P(effect | parents)`` and the kernel
    representation of ``P(parents)``, both as they vary with the
    surrogate. **Lower values support the direction.**

    Parameters
    ----------
    parents:
        Putative cause(s). Shape ``(T,)``, ``(T, 1)``, or ``(T, k)``
        for ``k`` joint parents.
    effect:
        Putative effect. Shape ``(T,)`` or ``(T, 1)``.
    surrogate:
        Distribution-shift surrogate (time index, domain id, etc.).
        Shape ``(T,)`` or ``(T, 1)``.
    width:
        Bandwidth for the Gaussian kernels on ``parents`` and ``effect``
        (after standardization). Three accepted forms:

        * a positive float — manual; used verbatim;
        * ``"auto"`` (default) — median heuristic on the standardized
          parents, scaled down by an empirical constant. Fast.
        * ``"gp"`` — fit a Gaussian process with an ARD-RBF kernel
          and use the marginal-likelihood-optimized length scales.
          Adaptive but slower (~10–50× the cost of ``"auto"`` per
          score call). When ``"gp"`` is selected, **both** ``width``
          and ``width_t`` are learned together by the same GP,
          regardless of what ``width_t`` was set to.

        The MATLAB reference uses a fixed ``0.1`` here — pass that
        explicitly to reproduce the reference exactly.
    width_t:
        Bandwidth for the Gaussian kernel on the surrogate. Accepts the
        same three forms as ``width``. Ignored when ``width="gp"``
        (the GP fit handles both).
    lambda_reg:
        Tikhonov regularization for the kernel matrix inverses.

    Returns
    -------
    float
        Dependence statistic. Lower is better evidence for the
        ``parents -> effect`` direction.
    """
    parents = np.asarray(parents, dtype=float)
    effect = np.asarray(effect, dtype=float)
    surrogate = np.asarray(surrogate, dtype=float)

    if parents.ndim == 1:
        parents = parents.reshape(-1, 1)
    if effect.ndim == 1:
        effect = effect.reshape(-1, 1)
    if surrogate.ndim == 1:
        surrogate = surrogate.reshape(-1, 1)

    if not (parents.shape[0] == effect.shape[0] == surrogate.shape[0]):
        raise ValueError(
            f"sample-size mismatch: parents={parents.shape[0]}, "
            f"effect={effect.shape[0]}, surrogate={surrogate.shape[0]}"
        )
    T = parents.shape[0]
    if T < 5:
        raise ValueError(f"need T >= 5 for kernel fits, got {T}")

    # Standardize parents and effect (mean zero, unit std), per MATLAB.
    parents = _standardize(parents)
    effect = _standardize(effect)
    # Standardize the surrogate too so that the auto bandwidth on it
    # is dataset-scale-invariant (the MATLAB reference doesn't
    # standardize the surrogate, which makes its hard-coded ``Wt=1``
    # implicitly only sensible when the surrogate already has unit
    # scale; we standardize so users can pass raw time indices).
    surrogate = _standardize(surrogate)

    # Resolve bandwidths.
    #
    # Empirically the independent-change score is sensitive to the X/Y
    # bandwidth and works best when the kernel is narrow enough to
    # resolve local structure (the MATLAB reference uses 0.1 on
    # standardized data). The median heuristic alone gives ~1.6 on
    # standardized normal data, which is too wide and causes direction
    # inference to fail. We therefore scale the X/Y auto bandwidth down
    # by ``_AUTO_WIDTH_XY_SCALE`` so it lands in the working range.
    # The surrogate bandwidth uses the standard median heuristic.
    #
    # The "gp" path is more expensive but adaptive: it fits a Gaussian
    # process with an ARD-RBF kernel on (parents, surrogate) -> effect
    # and reads off the marginal-likelihood-optimized length scales as
    # bandwidths. Useful when a single fixed bandwidth gives the wrong
    # direction on a particular DGP.
    if width == GP:
        # GP for both bandwidths from a single fit. width_t is ignored.
        bw_x, bw_t = _fit_gp_bandwidths(parents, effect, surrogate)
    elif width_t == GP:
        # User wants GP only for the surrogate side; resolve X side normally.
        _, bw_t = _fit_gp_bandwidths(parents, effect, surrogate)
        bw_x = _resolve_bandwidth(
            width, parents, name="width", auto_scale=_AUTO_WIDTH_XY_SCALE
        )
    else:
        bw_x = _resolve_bandwidth(
            width, parents, name="width", auto_scale=_AUTO_WIDTH_XY_SCALE
        )
        bw_t = _resolve_bandwidth(
            width_t, surrogate, name="width_t", auto_scale=1.0
        )

    theta = 1.0 / bw_x**2
    theta_t = 1.0 / bw_t**2

    # Kernel matrices.
    K_xx = gaussian_kernel(parents, theta)
    K_yy = gaussian_kernel(effect, theta)
    K_tt = gaussian_kernel(surrogate, theta_t)

    eye = np.eye(T)

    # --- module 1: P(effect | parents), as it varies with C ---
    #
    # Mirrors `infer_nonsta_dir.m` non-GP path:
    #     invK      = (Kxx .* Ktt + lambda*I)^{-1}
    #     prod_invK = invK * Kyy * invK
    #     Ml        = (1/T^2) * Ktt * (Kxx^3 .* prod_invK) * Ktt
    inv_K = pinv(K_xx * K_tt + lambda_reg * eye)
    K_xx_cubed = K_xx @ K_xx @ K_xx
    prod_inv_K = inv_K @ K_yy @ inv_K
    Ml = (1.0 / T**2) * (K_tt @ (K_xx_cubed * prod_inv_K) @ K_tt)
    Mg = _gram_to_kernel(Ml)

    # --- module 2: P(parents), as it varies with C ---
    #     invK2 = (Ktt + lambda*I)^{-1}
    #     Ml2   = Ktt * invK2 * Kxx * invK2 * Ktt
    inv_K2 = pinv(K_tt + lambda_reg * eye)
    Ml2 = K_tt @ inv_K2 @ K_xx @ inv_K2 @ K_tt
    Mg2 = _gram_to_kernel(Ml2)

    # --- HSIC-style dependence between Mg and Mg2 ---
    # Center both with H = I - 1/T * 1 1^T, then take (1/T^2) * sum(Mg * Mg2).
    H = eye - np.full((T, T), 1.0 / T)
    Mg_c = H @ Mg @ H
    Mg2_c = H @ Mg2 @ H
    return float((1.0 / T**2) * np.sum(Mg_c * Mg2_c))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standardize(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Center to zero mean and scale to unit std (column-wise, ddof=1).

    Constant columns are passed through (mean-subtracted but no scale).
    """
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True, ddof=1)
    safe_std = np.where(std > 1e-12, std, 1.0)
    return (X - mean) / safe_std


def _resolve_bandwidth(
    spec: WidthSpec,
    data: NDArray[np.float64],
    *,
    name: str,
    auto_scale: float = 1.0,
) -> float:
    """Turn a user-supplied bandwidth spec into a concrete positive float.

    ``spec`` may be either:

    * the string ``"auto"`` — pick by the median heuristic on ``data``,
      multiplied by ``auto_scale``;
    * a positive float — used verbatim (``auto_scale`` is ignored).

    The string ``"gp"`` is handled at the
    :func:`independent_change_score` level (it requires fitting a GP on
    the joint ``(parents, surrogate, effect)`` data, not a single
    column), so this helper rejects it with a clear error if it ever
    reaches here.
    """
    if isinstance(spec, str):
        if spec == AUTO:
            return float(auto_scale * median_bandwidth(data))
        if spec == GP:
            raise ValueError(
                f"{name}={spec!r} must be handled by the calling score "
                "function, not _resolve_bandwidth (this is a bug)"
            )
        raise ValueError(
            f"{name} must be 'auto', 'gp', or a positive float, got {spec!r}"
        )
    try:
        value = float(spec)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{name} must be 'auto', 'gp', or a positive float, got {spec!r}"
        ) from e
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _fit_gp_bandwidths(
    parents: NDArray[np.float64],
    effect: NDArray[np.float64],
    surrogate: NDArray[np.float64],
    *,
    n_restarts: int = 2,
    random_state: int = 0,
) -> tuple[float, float]:
    """Fit a GP with an ARD-RBF kernel and return ``(bw_x, bw_t)``.

    Inputs are assumed to be already standardized (mean 0, unit std).

    Regresses ``effect ~ GP([parents, surrogate])`` with kernel
    ``ConstantKernel * RBF(ARD) + WhiteKernel`` and reads off the
    learned length scales. The parent length scales are averaged into
    a single ``bw_x`` (the score function uses a scalar X/Y bandwidth);
    the surrogate length scale becomes ``bw_t``.

    Falls back to median-heuristic bandwidths with a ``RuntimeWarning``
    if the GP fit fails (degenerate data, optimizer non-convergence,
    etc.). The ``auto`` fallback uses ``_AUTO_WIDTH_XY_SCALE *
    median_bandwidth(parents)`` for ``bw_x`` and the unscaled median
    heuristic for ``bw_t``, matching the ``"auto"`` defaults.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel,
            RBF,
            WhiteKernel,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "GP-based bandwidth learning requires scikit-learn; "
            "install with: pip install cdans"
        ) from e

    n_parents = parents.shape[1]
    Z = np.column_stack([parents, surrogate])
    d = Z.shape[1]

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1.0))
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence ConvergenceWarning
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=n_restarts,
                normalize_y=False,
                random_state=random_state,
            )
            gp.fit(Z, effect.ravel())

        # Kernel structure: (ConstantKernel * RBF) + WhiteKernel
        rbf = gp.kernel_.k1.k2
        learned = np.atleast_1d(rbf.length_scale).astype(float)
        bw_x = float(np.mean(learned[:n_parents]))
        bw_t = float(learned[n_parents])

        # Clip to a sane range — pathological GP fits can return values
        # at the kernel's hyperparameter bounds, which then cause numerical
        # blow-ups downstream.
        bw_x = float(np.clip(bw_x, 1e-2, 1e2))
        bw_t = float(np.clip(bw_t, 1e-2, 1e2))
        return bw_x, bw_t
    except Exception as exc:  # pragma: no cover - hard to trigger reliably
        warnings.warn(
            f"GP-based bandwidth learning failed "
            f"({type(exc).__name__}: {exc}); falling back to 'auto' "
            f"bandwidths.",
            RuntimeWarning,
            stacklevel=2,
        )
        return (
            _AUTO_WIDTH_XY_SCALE * median_bandwidth(parents),
            median_bandwidth(surrogate),
        )


def _gram_to_kernel(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an inner-product-like matrix ``M`` to a Gaussian kernel.

    Treats ``M`` as if it were ``X X^T``: forms the implied squared
    distances ``D[i, j] = M[i, i] + M[j, j] - 2 M[i, j]`` and returns
    ``exp(-D / (2 * sigma^2))`` where ``sigma^2`` is the median of the
    strictly-lower-triangular distances (the standard median heuristic).

    Returns the identity if the median distance is non-positive
    (degenerate input).
    """
    diag = np.diag(M)
    D = diag[:, None] + diag[None, :] - 2.0 * M
    n = D.shape[0]
    mask = np.tri(n, k=-1, dtype=bool)
    distances = D[mask]
    if distances.size == 0:
        return np.eye(n)
    sigma2 = float(np.median(distances))
    if sigma2 <= 0:
        return np.eye(n)
    return np.exp(-D / (2.0 * sigma2))
