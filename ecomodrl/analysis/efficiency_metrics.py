# =============================================================================
# Learning-curve efficiency metrics (AULC, time-to-convergence)
#
# A "history" is a pair (t, U) of training-only wall-clock times and CRN-paired
# evaluation returns, as saved in eval_history.npz.
#
# Metrics are computed per training seed, then aggregated with the IQM and a
# bootstrap confidence interval over seeds.
# =============================================================================

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def last_value_carried_forward(
    t: NDArray[np.float64], 
    y: NDArray[np.float64], 
    t_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Left-continuous step interpolation of (t, y) onto t_grid.

    Used to align irregular evaluation times onto a common grid 
    for area computations.
    """
    if t.size == 0:
        return np.full_like(t_grid, np.nan, dtype=np.float64)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    idx = np.searchsorted(t, t_grid, side="left")
    return y[idx]


def normalized_regret(
    egm: ArrayLike,
    rl: ArrayLike,
) -> NDArray[np.float64]:
    """Share-deviation regret: (U_EGM - U_RL) / |U_EGM|.

    Positive when EGM outperforms RL, and robust to the sign of the utility
    level. Returns NaN wherever U_EGM == 0 or U_EGM is non-finite.
    """
    egm_arr = np.asarray(egm, dtype=np.float64)
    rl_arr = np.asarray(rl, dtype=np.float64)

    # Make egm broadcast along trailing RL dimensions, e.g. (N,) -> (N, 1)
    if egm_arr.ndim < rl_arr.ndim and egm_arr.shape == rl_arr.shape[:egm_arr.ndim]:
        egm_arr = egm_arr.reshape(
            egm_arr.shape + (1,) * (rl_arr.ndim - egm_arr.ndim)
        )

    denom = np.abs(egm_arr)
    valid = (denom != 0.0) & np.isfinite(egm_arr)

    out_shape = np.broadcast_shapes(egm_arr.shape, rl_arr.shape)
    out = np.full(out_shape, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(
            egm_arr - rl_arr,
            denom,
            out=out,
            where=valid,
        )

    return out


def aulc_per_seed(histories, egm_returns, t_grid):
    """Per-seed area under the learning curve of the normalized regret.

    AULC = (1 / T) * trapezoid(regret_norm(t), t_grid), a time-average of the
    normalized regret over the training budget. Lower is better.
    """
    t_grid = np.asarray(t_grid, dtype=np.float64)
    T = float(t_grid[-1] - t_grid[0])

    out = []
    for hist, egm in zip(histories, egm_returns):
        if not np.isfinite(egm) or egm == 0.0:
            out.append(np.nan)
            continue
        
        t = np.asarray(hist["t"], dtype=np.float64)
        u = np.asarray(hist["u"], dtype=np.float64)
        rl = last_value_carried_forward(t, u, t_grid)
        r = normalized_regret(egm, rl)

        out.append(float(np.trapz(r, t_grid) / T))

    return out


def time_to_best_per_seed(histories):
    """Per-seed training-only wall-clock time at which best_eval was stored."""
    out = []
    for hist in histories:
        t = np.asarray(hist["t"], dtype=np.float64)
        u = np.asarray(hist["u"], dtype=np.float64)
        out.append(float(t[int(np.argmax(u))]))
    return out


def time_to_converged_per_seed(histories, egm_returns, tol=0.01):
    """Per-seed time to *practical* convergence.

    For each seed:
      1. compute the normalized regret r(t) at every evaluation;
      2. take the best (smallest) regret r_best = min_t r(t);
      3. return the earliest time at which r(t) first falls within a tolerance
         band of r_best, i.e. r(t) <= r_best + tol.

    This is more stable than time-to-best: a tiny, late improvement in
    best_eval no longer pushes the reported convergence time far to the right.
    """
    out = []
    for hist, egm in zip(histories, egm_returns):
        t = np.asarray(hist["t"], dtype=np.float64)
        u = np.asarray(hist["u"], dtype=np.float64)

        mask = np.isfinite(t) & np.isfinite(u)
        if not mask.any() or not np.isfinite(egm) or egm == 0.0:
            out.append(np.nan)
            continue

        order = np.argsort(t[mask])
        tt = t[mask][order]
        uu = u[mask][order]

        r = normalized_regret(egm, uu)
        if not np.any(np.isfinite(r)):
            out.append(np.nan)
            continue

        r_best = float(np.nanmin(r))

        threshold = r_best + float(tol)
        hit = np.where(r <= threshold)[0]

        out.append(float(tt[hit[0]]) if hit.size else np.nan)

    return out


