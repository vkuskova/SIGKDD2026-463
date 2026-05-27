# tvNAR.py
"""
tvNAR: Time-varying Network Autoregression (VAR(1) with nodal influence trajectories)

Model (VAR(1) form; matches tvNAR paper's parameterization):
    Y_t = (W + I) * diag(lambda(tau_t)) * Y_{t-1} + e_t

Equivalently, for each i:
    y_i,t = sum_j (w_ij + 1{i=j}) * lambda_j(tau_t) * y_j,t-1 + e_i,t

Key idea:
- W is a known network (row-normalized recommended). In our pipeline, W is inferred by NAVAR.
- lambda_j(tau) is a time-varying "nodal influence" trajectory (one curve per variable).

This module provides:
- construction of panel-safe lag pairs (no leakage across countries)
- kernel-weighted local ridge regression to estimate lambda(tau) on a grid
- optional simple bootstrap test for time-variation (practical, not fully asymptotic)

Dependencies: numpy, pandas (optional for I/O)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray]


def _as_2d_float(X: ArrayLike, name: str) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D array-like, got shape {X.shape}")
    X = X.astype(float, copy=False)
    return X


def _split_lengths_to_slices(lengths: List[int]) -> List[slice]:
    if any(l <= 1 for l in lengths):
        raise ValueError("All series lengths must be >= 2 to form (t-1, t) pairs.")
    slices = []
    start = 0
    for L in lengths:
        slices.append(slice(start, start + L))
        start += L
    return slices


def _infer_split_lengths(
    T: int,
    split_timeseries: Union[bool, int, List[int], np.ndarray, Tuple[int, ...], None],
) -> Optional[List[int]]:
    """
    Normalize split_timeseries into a list of per-series lengths or None.

    - None/False: treat as a single series of length T
    - int: treat as equal-length series of this length (must divide T exactly)
    - list/array: treat as provided per-series lengths (must sum to T)
    """
    if split_timeseries is None or split_timeseries is False:
        return None
    if isinstance(split_timeseries, bool):
        # True without a length is ambiguous; treat as error to avoid silent leakage.
        raise ValueError("split_timeseries=True is ambiguous; provide int length or list of lengths.")
    if isinstance(split_timeseries, (int, np.integer)):
        L = int(split_timeseries)
        if L <= 1:
            raise ValueError("split_timeseries length must be >= 2.")
        if T % L != 0:
            raise ValueError(f"T={T} is not divisible by split_timeseries={L}.")
        return [L] * (T // L)
    # list-like
    lengths = [int(x) for x in list(split_timeseries)]
    if sum(lengths) != T:
        raise ValueError(f"split_timeseries lengths sum to {sum(lengths)} but T={T}.")
    return lengths


@dataclass
class PanelLagPairs:
    """
    Container for panel-safe (Y_{t-1}, Y_t) pairs.

    y_prev: (M, N)
    y_curr: (M, N)
    tau:    (M,)  within-series normalized time in [0, 1]
    series_id: (M,) integer id of originating series (country)
    """
    y_prev: np.ndarray
    y_curr: np.ndarray
    tau: np.ndarray
    series_id: np.ndarray


def build_panel_lag_pairs(
    Y: ArrayLike,
    split_timeseries: Union[bool, int, List[int], np.ndarray, Tuple[int, ...], None] = None,
    tau_mode: str = "within_series",
) -> PanelLagPairs:
    """
    Build (t-1, t) lag pairs from stacked panel data without crossing series boundaries.

    Parameters
    ----------
    Y : array (T, N)
        Stacked time series (e.g., country panels concatenated).
    split_timeseries : None/False, int, or list[int]
        - None/False: single series
        - int: equal-length series
        - list[int]: per-series lengths
    tau_mode : str
        - "within_series": tau = (k / (L-1)) for k=1..L-1 within each series
          (recommended for cross-country comparability when countries are "replicates")
        - "global_index": tau = global t / (T-1) (rarely what you want for panels)

    Returns
    -------
    PanelLagPairs
    """
    Y = _as_2d_float(Y, "Y")
    T, N = Y.shape
    lengths = _infer_split_lengths(T, split_timeseries)

    y_prev_list: List[np.ndarray] = []
    y_curr_list: List[np.ndarray] = []
    tau_list: List[np.ndarray] = []
    sid_list: List[np.ndarray] = []

    if lengths is None:
        # Single series
        y_prev = Y[:-1]
        y_curr = Y[1:]
        if tau_mode == "within_series":
            tau = np.arange(1, T) / (T - 1)
        elif tau_mode == "global_index":
            tau = np.arange(1, T) / (T - 1)
        else:
            raise ValueError("tau_mode must be 'within_series' or 'global_index'.")
        series_id = np.zeros(T - 1, dtype=int)
        return PanelLagPairs(y_prev=y_prev, y_curr=y_curr, tau=tau, series_id=series_id)

    slices = _split_lengths_to_slices(lengths)
    for sid, sl in enumerate(slices):
        Ys = Y[sl]  # (L, N)
        L = Ys.shape[0]
        y_prev = Ys[:-1]
        y_curr = Ys[1:]
        if tau_mode == "within_series":
            tau = np.arange(1, L) / (L - 1)
        elif tau_mode == "global_index":
            # Map local positions to global index range
            g_idx = np.arange(sl.start + 1, sl.stop)
            tau = g_idx / (T - 1)
        else:
            raise ValueError("tau_mode must be 'within_series' or 'global_index'.")
        y_prev_list.append(y_prev)
        y_curr_list.append(y_curr)
        tau_list.append(tau)
        sid_list.append(np.full(L - 1, sid, dtype=int))

    return PanelLagPairs(
        y_prev=np.vstack(y_prev_list),
        y_curr=np.vstack(y_curr_list),
        tau=np.concatenate(tau_list),
        series_id=np.concatenate(sid_list),
    )


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Gaussian kernel (unnormalized): exp(-0.5 u^2)."""
    return np.exp(-0.5 * (u ** 2))


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel: 0.75*(1-u^2) for |u|<=1 else 0."""
    w = np.zeros_like(u, dtype=float)
    m = np.abs(u) <= 1.0
    w[m] = 0.75 * (1.0 - u[m] ** 2)
    return w


def _get_kernel(name: str):
    name = name.lower()
    if name in ("gaussian", "normal"):
        return gaussian_kernel
    if name in ("epanechnikov", "epa"):
        return epanechnikov_kernel
    raise ValueError("kernel must be 'gaussian' or 'epanechnikov'.")


def row_normalize(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize W (handles all-zero rows safely)."""
    W = np.asarray(W, dtype=float)
    rs = W.sum(axis=1, keepdims=True)
    rs = np.where(rs < eps, 1.0, rs)
    return W / rs


def fit_tvNAR_lambda_paths(
    Y: ArrayLike,
    W: ArrayLike,
    split_timeseries: Union[bool, int, List[int], np.ndarray, Tuple[int, ...], None] = None,
    tau_grid: Optional[np.ndarray] = None,
    grid_size: int = 50,
    bandwidth: float = 0.15,
    kernel: str = "gaussian",
    ridge: float = 1e-4,
    tau_mode: str = "within_series",
    center: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Estimate nodal influence trajectories lambda(tau) on a tau grid via kernel-weighted ridge regression.

    For each tau0 in tau_grid, solve:
        minimize_lambda  sum_m w_m(tau0) || y_curr[m] - (W+I) * (lambda ⊙ y_prev[m]) ||^2  + ridge*||lambda||^2
    """
    Y = _as_2d_float(Y, "Y")
    W = _as_2d_float(W, "W")
    T, N = Y.shape
    if W.shape != (N, N):
        raise ValueError(f"W must be shape (N,N) with N={N}, got {W.shape}")

    if center:
        Y = Y - Y.mean(axis=0, keepdims=True)

    pairs = build_panel_lag_pairs(Y, split_timeseries=split_timeseries, tau_mode=tau_mode)
    y_prev = pairs.y_prev  # (M,N)
    y_curr = pairs.y_curr  # (M,N)
    tau = pairs.tau        # (M,)
    M = y_prev.shape[0]
    if M == 0:
        raise ValueError("No usable (t-1,t) pairs. Check split_timeseries and data length.")

    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, int(grid_size))
    else:
        tau_grid = np.asarray(tau_grid, dtype=float).ravel()

    if not (0 < bandwidth <= 1.0):
        raise ValueError("bandwidth must be in (0, 1].")

    K = _get_kernel(kernel)
    WpI = W + np.eye(N, dtype=float)

    lambdas = np.zeros((tau_grid.size, N), dtype=float)

    # Weighted ridge at each tau0
    for g, tau0 in enumerate(tau_grid):
        u = (tau - tau0) / bandwidth
        w = K(u)

        # fallback if bandwidth too tight
        if w.sum() <= 1e-12:
            w = np.ones_like(w)

        A = np.zeros((N, N), dtype=float)
        b = np.zeros(N, dtype=float)

        for m in range(M):
            wm = w[m]
            if wm <= 0:
                continue
            # X_m = WpI * y_prev[m] (broadcast on columns)
            X = WpI * y_prev[m][None, :]  # (N,N)
            y = y_curr[m]                 # (N,)
            A += wm * (X.T @ X)
            b += wm * (X.T @ y)

        A += ridge * np.eye(N, dtype=float)

        try:
            lambdas[g] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            lambdas[g] = np.linalg.lstsq(A, b, rcond=None)[0]

    return {
        "tau_grid": tau_grid,
        "lambda": lambdas,
        "W_plus_I": WpI,
        "meta": {
            "bandwidth": float(bandwidth),
            "kernel": kernel,
            "ridge": float(ridge),
            "tau_mode": tau_mode,
            "center": bool(center),
            "n_pairs": int(M),
        },
    }


#Adding code for flexible p paths for tvNAR
def fit_tvNAR_lambda_paths_p(
    Y,
    W,
    split_timeseries=None,
    tau_grid=None,
    grid_size=50,
    bandwidth=0.2,
    kernel="gaussian",
    ridge=1e-4,
    tau_mode="within_series",
    center=False,
    tvnar_lags=2,
):
    """
    tvNAR(p): estimate nodal influence trajectories for p lags.

    Model (one-step):
      y_t ≈ sum_{ℓ=1..p} (W+I) @ ( λ^{(ℓ)}(tau_t) ⊙ y_{t-ℓ} )

    Estimation:
      For each tau0, solve a weighted ridge regression for concatenated parameter vector:
        θ(tau0) = [λ^(1)(tau0); ...; λ^(p)(tau0)]  in R^(pN)

    Returns dict:
      - tau_grid: (G,)
      - lambda: (G, p, N)  (lambda[g, ℓ-1, j] = λ_j^(ℓ)(tau_g))
      - W_plus_I: (N,N)
      - meta: dict
    """
    Y = np.asarray(Y, dtype=float)
    W = np.asarray(W, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be (T,N)")
    T, N = Y.shape
    if W.shape != (N, N):
        raise ValueError("W must be (N,N)")
    p = int(tvnar_lags)
    if p < 1:
        raise ValueError("tvnar_lags must be >=1")
    if T <= p:
        raise ValueError(f"Need T>tvnar_lags. Got T={T}, p={p}")

    if center:
        Y = Y - Y.mean(axis=0, keepdims=True)

    # --- build panel-safe (t, lags) pairs ---
    # We reuse your existing build_panel_lag_pairs logic style, but for p lags.
    # We will assemble lists of (y_t, [y_{t-1},...,y_{t-p}], tau_t)
    def _infer_lengths(T, split_timeseries):
        if split_timeseries is None or split_timeseries is False:
            return None
        if isinstance(split_timeseries, bool):
            raise ValueError("split_timeseries=True is ambiguous; provide int or list.")
        if isinstance(split_timeseries, (int, np.integer)):
            L = int(split_timeseries)
            if T % L != 0:
                raise ValueError("T not divisible by split_timeseries")
            return [L] * (T // L)
        lengths = [int(x) for x in list(split_timeseries)]
        if sum(lengths) != T:
            raise ValueError("split_timeseries list must sum to T")
        return lengths

    lengths = _infer_lengths(T, split_timeseries) or [T]

    # kernel
    def gaussian(u): return np.exp(-0.5 * u*u)
    def epan(u):
        w = np.zeros_like(u)
        m = np.abs(u) <= 1.0
        w[m] = 0.75 * (1 - u[m]**2)
        return w
    K = gaussian if kernel.lower() in ("gaussian", "normal") else epan

    if tau_grid is None:
        tau_grid = np.linspace(0.0, 1.0, int(grid_size))
    else:
        tau_grid = np.asarray(tau_grid, dtype=float).ravel()

    WpI = W + np.eye(N, dtype=float)

    # collect samples
    y_curr_list = []
    y_lags_list = []
    tau_list = []

    start = 0
    for L in lengths:
        Ys = Y[start:start+L]
        start += L
        if L <= p:
            continue

        if tau_mode == "within_series":
            taus = np.arange(L) / (L - 1)
        elif tau_mode == "global_index":
            # map to global index fraction
            # (rough; only used if you really want it)
            g0 = start - L
            taus = np.arange(g0, g0 + L) / (T - 1)
        else:
            raise ValueError("tau_mode must be 'within_series' or 'global_index'")

        for t in range(p, L):
            y_curr_list.append(Ys[t])
            # collect lags as list [t-1..t-p]
            y_lags_list.append([Ys[t - ell] for ell in range(1, p+1)])
            tau_list.append(taus[t])

    y_curr = np.asarray(y_curr_list, dtype=float)            # (M,N)
    y_lags = np.asarray(y_lags_list, dtype=float)            # (M,p,N)
    tau = np.asarray(tau_list, dtype=float)                  # (M,)
    M = y_curr.shape[0]
    if M == 0:
        raise ValueError("No usable samples after lagging; check split_timeseries and tvnar_lags.")

    # Solve for each tau0
    lambdas = np.zeros((tau_grid.size, p, N), dtype=float)

    # Design per sample m:
    # y_hat = sum_{ℓ=1..p} X_m^(ℓ) @ λ^(ℓ), where
    # X_m^(ℓ) is (N,N): X[i,j] = WpI[i,j] * y_{t-ℓ}[j]
    #
    # Stack λ^(ℓ) into θ in R^(pN).
    # Then define X_stack_m as (N, pN) = [X^(1) | X^(2) | ... | X^(p)]
    #
    # Weighted ridge normal equations:
    # A = Σ w_m X^T X, b = Σ w_m X^T y
    for g, tau0 in enumerate(tau_grid):
        u = (tau - tau0) / bandwidth
        w = K(u)
        if w.sum() <= 1e-12:
            w = np.ones_like(w)

        A = np.zeros((p*N, p*N), dtype=float)
        b = np.zeros((p*N,), dtype=float)

        for m in range(M):
            wm = w[m]
            if wm <= 0:
                continue

            # Build X_stack_m (N, pN)
            blocks = []
            for ell in range(p):
                y_prev = y_lags[m, ell]  # (N,)
                Xell = WpI * y_prev[None, :]  # (N,N)
                blocks.append(Xell)
            Xstack = np.concatenate(blocks, axis=1)  # (N, pN)

            y = y_curr[m]  # (N,)
            A += wm * (Xstack.T @ Xstack)
            b += wm * (Xstack.T @ y)

        A += ridge * np.eye(p*N, dtype=float)

        theta = np.linalg.solve(A, b)  # (pN,)
        lambdas[g] = theta.reshape(p, N)

    return {
        "tau_grid": tau_grid,
        "lambda": lambdas,      # (G,p,N)
        "W_plus_I": WpI,
        "meta": {
            "tvnar_lags": int(p),
            "bandwidth": float(bandwidth),
            "kernel": kernel,
            "ridge": float(ridge),
            "tau_mode": tau_mode,
            "center": bool(center),
            "n_samples": int(M),
        },
    }


    # Precompute X blocks per observation:
    # X_m is (N,N): X_m[i,j] = (WpI[i,j] * y_prev[m,j])
    # We'll avoid storing all X_m (could be large); compute weighted normal equations on the fly.
    for g, tau0 in enumerate(tau_grid):
        u = (tau - tau0) / bandwidth
        w = K(u)
        # If kernel yields all ~0 due to narrow bandwidth, broaden effectively
        sw = w.sum()
        if sw <= 1e-12:
            # fall back to uniform weights
            w = np.ones_like(w)
            sw = w.sum()

        # Weighted normal equations:
        # A = sum_m w_m * X_m^T X_m
        # b = sum_m w_m * X_m^T y_curr[m]
        A = np.zeros((N, N), dtype=float)
        b = np.zeros(N, dtype=float)

        for m in range(M):
            wm = w[m]
            if wm <= 0:
                continue
            # X_m = WpI * y_prev[m] (broadcast on columns)
            X = WpI * y_prev[m][None, :]  # (N,N)
            y = y_curr[m]                 # (N,)
            A += wm * (X.T @ X)
            b += wm * (X.T @ y)

        # Ridge stabilization
        A += ridge * np.eye(N, dtype=float)

        # Solve for lambda(tau0)
        try:
            lambdas[g] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # robust fallback
            lambdas[g] = np.linalg.lstsq(A, b, rcond=None)[0]

    return {
        "tau_grid": tau_grid,
        "lambda": lambdas,
        "W_plus_I": WpI,
        "meta": {
            "bandwidth": float(bandwidth),
            "kernel": kernel,
            "ridge": float(ridge),
            "tau_mode": tau_mode,
            "center": bool(center),
            "n_pairs": int(M),
        },
    }

#Add tvNAR forecasting utilities
def tvnar_p_forecast(
    series,
    W,
    lambda_vecs,   # (p,N) for fixed-lambda forecasting
    horizon,
):
    """
    Deterministic multi-horizon forecast with tvNAR(p) using fixed lambda vectors per lag.

    series: (T,N)
    W: (N,N)
    lambda_vecs: (p,N) where lambda_vecs[ell-1] is λ^(ell)
    """
    series = np.asarray(series, dtype=float)
    W = np.asarray(W, dtype=float)
    lam = np.asarray(lambda_vecs, dtype=float)
    p, N = lam.shape
    WpI = W + np.eye(N, dtype=float)

    buf = series[-p:].copy()  # (p,N)  last p observations
    out = np.zeros((horizon, N), dtype=float)

    for h in range(horizon):
        yhat = np.zeros(N, dtype=float)
        # buf[-1] is y_{t}, buf[-ell] is y_{t-ell+1}
        for ell in range(1, p+1):
            y_prev = buf[-ell]
            yhat += WpI @ (lam[ell-1] * y_prev)
        out[h] = yhat
        buf = np.vstack([buf[1:], yhat[None, :]])

    return out


def predict_tvNAR(
    y_prev: np.ndarray,
    W: np.ndarray,
    lambda_vec: np.ndarray,
) -> np.ndarray:
    """
    One-step prediction given y_{t-1}, W, and lambda(t).

    y_prev: (N,)
    W: (N,N)
    lambda_vec: (N,)

    returns y_hat: (N,)
    """
    y_prev = np.asarray(y_prev, dtype=float).ravel()
    lambda_vec = np.asarray(lambda_vec, dtype=float).ravel()
    if y_prev.ndim != 1:
        raise ValueError("y_prev must be 1D")
    N = y_prev.size
    if W.shape != (N, N):
        raise ValueError("W shape mismatch")
    if lambda_vec.size != N:
        raise ValueError("lambda_vec size mismatch")

    WpI = W + np.eye(N, dtype=float)
    return WpI @ (lambda_vec * y_prev)


def bootstrap_time_variation_test(
    Y: ArrayLike,
    W: ArrayLike,
    split_timeseries: Union[bool, int, List[int], np.ndarray, Tuple[int, ...], None] = None,
    grid_size: int = 30,
    bandwidth: float = 0.2,
    kernel: str = "gaussian",
    ridge: float = 1e-4,
    B: int = 200,
    block_len: int = 5,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Practical bootstrap test for whether lambda(tau) varies over time.

    Statistic:
        T = mean_g || lambda(tau_g) - lambda_bar ||^2

    Bootstrap:
        - Fit tvNAR, compute residuals e_t = y_t - yhat_t (using tau-matched lambdas via nearest grid)
        - Within each series, do moving-block bootstrap of residual time steps
        - Generate pseudo series: y*_t = yhat_t + e*_t (keeping y_0 fixed)
        - Refit tvNAR on pseudo data, recompute T*
        - p-value = (1 + #{T* >= T})/(B+1)

    Notes:
    - This is a pragmatic diagnostic for your 35-year setting.
    - It is not a verbatim re-implementation of the paper’s asymptotic stability test.
    """
    rng = np.random.default_rng(seed)

    Y = _as_2d_float(Y, "Y")
    W = _as_2d_float(W, "W")
    T, N = Y.shape
    lengths = _infer_split_lengths(T, split_timeseries) or [T]
    slices = _split_lengths_to_slices(lengths)

    # Fit original
    fit = fit_tvNAR_lambda_paths(
        Y=Y,
        W=W,
        split_timeseries=split_timeseries,
        grid_size=grid_size,
        bandwidth=bandwidth,
        kernel=kernel,
        ridge=ridge,
        tau_mode="within_series",
        center=False,
    )
    tau_grid = fit["tau_grid"]
    lambdas = fit["lambda"]  # (G,N)

    lam_bar = lambdas.mean(axis=0, keepdims=True)
    T_stat = float(np.mean(np.sum((lambdas - lam_bar) ** 2, axis=1)))

    # Build residuals per series for bootstrap
    # For each series, compute yhat_t using nearest lambda on grid by local tau.
    resid_series: List[np.ndarray] = []
    yhat_series: List[np.ndarray] = []
    Y_series: List[np.ndarray] = []

    for sid, sl in enumerate(slices):
        Ys = Y[sl]  # (L,N)
        L = Ys.shape[0]
        if L <= 1:
            continue
        taus = np.arange(0, L) / (L - 1)  # includes tau=0 at first obs
        # one-step preds for t>=1
        yhat = np.zeros_like(Ys)
        yhat[0] = Ys[0]
        for t in range(1, L):
            tau_t = taus[t]
            g = int(np.argmin(np.abs(tau_grid - tau_t)))
            yhat[t] = predict_tvNAR(Ys[t - 1], W, lambdas[g])
        resid = Ys - yhat
        resid_series.append(resid)
        yhat_series.append(yhat)
        Y_series.append(Ys)

    def mbb_indices(L: int, block_len: int) -> np.ndarray:
        # returns indices 1..L-1 (residuals aligned to time)
        idx = []
        t = 1
        while t < L:
            start = rng.integers(1, max(2, L - block_len + 1))
            blk = np.arange(start, min(L, start + block_len))
            idx.append(blk)
            t += blk.size
        return np.concatenate(idx)[: (L - 1)]

    def refit_stat(Y_star: np.ndarray) -> float:
        f = fit_tvNAR_lambda_paths(
            Y=Y_star,
            W=W,
            split_timeseries=split_timeseries,
            grid_size=grid_size,
            bandwidth=bandwidth,
            kernel=kernel,
            ridge=ridge,
            tau_mode="within_series",
            center=False,
        )
        lam = f["lambda"]
        lb = lam.mean(axis=0, keepdims=True)
        return float(np.mean(np.sum((lam - lb) ** 2, axis=1)))

    # Bootstrap
    T_star = np.zeros(B, dtype=float)
    for b in range(B):
        Yb_list = []
        for Ys, yhat, resid in zip(Y_series, yhat_series, resid_series):
            L = Ys.shape[0]
            idx = mbb_indices(L, block_len=block_len)
            e_star = np.zeros_like(resid)
            e_star[0] = 0.0
            e_star[1:] = resid[idx]  # bootstrap residuals for t>=1
            # reconstruct
            Y_star = yhat + e_star
            # keep initial condition fixed
            Y_star[0] = Ys[0]
            Yb_list.append(Y_star)
        Y_star_full = np.vstack(Yb_list)
        T_star[b] = refit_stat(Y_star_full)

    p_value = float((1.0 + np.sum(T_star >= T_stat)) / (B + 1.0))
    return {
        "T_stat": T_stat,
        "p_value": p_value,
        "B": float(B),
        "grid_size": float(grid_size),
        "bandwidth": float(bandwidth),
        "block_len": float(block_len),
    }