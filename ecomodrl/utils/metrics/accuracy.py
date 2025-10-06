
import logging
import numpy as np
from typing import Union, Callable, Tuple, Optional
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

# --- Helper for inverse cubic interpolation (1D case) ---
def inverse_cubic_interpolation(V: Callable[[np.ndarray], np.ndarray],
                                k: np.ndarray,
                                extrapolate: bool = True) -> Union[CubicSpline, None]:
    """Inverse interpolation via cubic splines. Builds k = cs(v) with cs(V(k)) ≈ k on k-grid."""
    try:
        # Attempt to construct spline mapping from values back to capital
        cs = CubicSpline(V(k), k, bc_type="natural", extrapolate=extrapolate)
    except Exception:
        # If monotonicity fails, log a warning and return None
        logging.warning("Skipping inverse interpolation: Function is not monotone! "
                        "Inverse interpolation is not well-defined globally.")
        cs = None
        pass
    # Return the inverse spline
    return cs

# --- Wealth-Equivalent Variation (WEV) ---
def wev(
    s: Union[float, np.ndarray, Tuple[float, ...]],
    V_bm: Callable[[np.ndarray], np.ndarray],
    V: Callable[[np.ndarray], np.ndarray],
    m: Callable[..., np.ndarray]
) -> Union[float, np.ndarray, None]:
    
    """
    Wealth-Equivalent Variation (WEV) in basis points.

    Auto-detects problem dimension from s:
      - 1D (Cass–Koopmans): s is scalar k or array of k's. Uses inversion of V_RL(k).
      - >=2D (e.g., Brock–Mirman): s is (k,z,...) or array of shape (N,d).
        Root-solves in k holding the other state components fixed.

    Interpretation 1: The additional capital that would make the agent indifferent between the two value functions
                      If RL is worse than benchmark, WEV > 0 (need more capital to be as well off as with benchmark)
    Interpretation 2: Distance between the two value functions in terms of capital (cash-on-hand).
                      In >=2D, this “distance” is taken along the capital dimension only, with other states fixed (z, …).

    Parameters
    ----------
    s : float | np.ndarray | tuple
        State(s) at which to measure WEV.
        - 1D (Cass–Koopmans): s is a scalar k or a 1D array of capital values k_vec.
        - >=2D (Brock–Mirman etc.): s is a tuple/array like (k, z, ...) or an array of shape (N, d>=2),
          where the first component is capital k and the remaining components are other states (e.g. productivity z).
    V_bm : Callable[[np.ndarray], np.ndarray]
        Benchmark value function V_bm(s) evaluated at the same state format as s.
        In 1D, it is called as V_bm(k); in >=2D, as V_bm([k, z, ...]).
    V : Callable[[np.ndarray], np.ndarray]
        RL (or alternative) value function with the same calling convention as V_bm.
        Used either for inversion (1D) or in the root equation V([k, z, ...]) - V_bm([k0, z, ...]) = 0 (>=2D).
    m : Callable[..., np.ndarray]
        Resource mapping used to convert capital (and, in >=2D, the full state) into “cash on hand”. 
        This is the scale in which WEV is reported:
          - 1D: m(k) -> array_like.
          - >=2D: m(k, z, ...) -> array_like (the first argument is always k; additional arguments match s[1:]).
        - Usage inside this function:
          - 1D: Δm = m(k_RL) - m(k0), WEV(bp) = 10_000 * Δm / m(k0).
          - >=2D: Δm = m(k*, z, ...) - m(k0, z, ...), with k* the root-found capital level that equalizes values.

    Returns
    -------
    float | np.ndarray
        WEV in basis points (bp). Returns a scalar if the input corresponds to a single state, otherwise an array.
    """

    # Convert input state(s) into a numpy array
    s_arr = np.asarray(s, dtype=float)

    # ----- 1D path (Cass–Koopmans) -----
    # Check if input is scalar or 1D array of k values
    if s_arr.ndim == 0 or (s_arr.ndim == 1 and s_arr.size >= 1):
        # Flatten into 1D array of capital values
        k_vec = s_arr.ravel()
        # Sort k values for monotone inversion
        k_sorted = np.sort(k_vec)

        # Build inverse of V_RL using cubic spline
        V_inv = inverse_cubic_interpolation(V, k_sorted)
        if V_inv is None:
            # If inversion failed, exit early with None
            return None

        # Compute target values under benchmark at k
        v_target = V_bm(k_vec)
        # Find k_RL that gives the same values under RL policy
        k_rl = V_inv(v_target)
        # Resource shift Δm
        dm = m(np.asarray(k_rl)) - m(k_vec)
        # Normalize to basis points relative to initial resources
        bp = 10_000.0 * (dm / m(k_vec))

        # Return scalar if input was scalar, else array
        return float(bp) if np.asarray(s).ndim == 0 and np.size(bp) == 1 else bp

    # ----- >=2D path (Brock–Mirman or higher-D) -----
    # If input is a single vector (k,z,...), reshape to (1,d)
    if s_arr.ndim == 1:
        s_arr = s_arr.reshape(1, -1)
        single = True
    # If already 2D (N,d), check if it's a single row
    elif s_arr.ndim == 2:
        single = (s_arr.shape[0] == 1)
    else:
        # Invalid shape
        raise ValueError("s must be scalar/1D for Cass, or (N,d>=2) for multi-D.")

    # Ensure there is at least one shock/state dimension beyond k
    if s_arr.shape[1] < 2:
        raise ValueError("For multi-D case, provide states with at least two components (k, z, ...).")

    # Bracketing for root-finding in capital dimension
    k_min = 1e-3
    k_max = 5.0 * float(s_arr[:, 0].max())

    # Helper: evaluate benchmark value function at state
    def _Vbm_at(state_vec: np.ndarray) -> float:
        return float(np.asarray(V_bm(np.asarray(state_vec))).squeeze())

    # Helper: evaluate RL value function at state
    def _Vrl_at(state_vec: np.ndarray) -> float:
        return float(np.asarray(V(np.asarray(state_vec))).squeeze())

    # Collect WEV results in basis points
    bps = []
    for state in s_arr:
        # Extract initial capital and other states (z, ...)
        k0 = float(state[0])
        others = state[1:]
        # Target benchmark value at this state
        v_target = _Vbm_at(state)

        # Define function g(k) = V_RL(k,·) - v_target
        def g(k: float) -> float:
            st = np.concatenate(([k], others))
            return _Vrl_at(st) - v_target

        # Check sign change for root existence
        ga, gb = g(k_min), g(k_max)
        if np.sign(ga) == np.sign(gb):
            # If no sign change, warn and use closest endpoint
            logging.warning(
                f"No sign change for V_RL(k,·)-V_bm(k0,·) on k∈[{k_min:.6g},{k_max:.6g}] at state={state}. "
                f"Returning closest endpoint."
            )
            k_star = k_min if abs(ga) < abs(gb) else k_max
        else:
            # Find root with Brent’s method
            k_star = brentq(g, k_min, k_max)

        # Compute Δm = m(k_star,·) - m(k0,·)
        dm = float(m(np.asarray(k_star), *[np.asarray(x) for x in others]) -
                   m(np.asarray(k0), *[np.asarray(x) for x in others]))
        # Denominator = initial resources m(k0,·)
        denom = float(m(np.asarray(k0), *[np.asarray(x) for x in others]))
        # Normalize to basis points
        bp = 10_000.0 * (dm / denom)
        bps.append(bp)

    # Convert results to array
    bps = np.asarray(bps)
    # Return scalar if input was a single state
    return bps[0] if single else bps


# --- Consumption-Equivalent Variation (CEV) ---
def cev_crra(
    states: Union[np.ndarray, float],
    V_bm: Callable[[np.ndarray], np.ndarray],
    V: Callable[[np.ndarray], np.ndarray],
    gamma: float,
    beta: Optional[float] = None,
) -> Union[np.ndarray, float]:
    """
    Consumption-Equivalent Variation (CEV), statewise, for CRRA utility.

    For gamma != 1:
        CEV(s) = (V_bm(s) / V(s))**(1/(1-gamma)) - 1
    For gamma == 1 (log utility):
        CEV(s) = exp((1 - beta) * (V_bm(s) - V(s))) - 1

    Works for any state dimensionality as long as V_bm and V are defined on states.
    """
    # Ensure input state(s) are a NumPy array
    s = np.asarray(states)

    # Evaluate benchmark and RL value functions at the state(s)
    Vb = V_bm(s)
    Vr = V(s)

    # CRRA case (gamma != 1): closed-form CEV formula
    if gamma != 1.0:
        return (Vb / Vr)**(1.0 / (1.0 - gamma)) - 1.0
    else:
        # Log utility case (gamma == 1): requires discount factor beta
        if beta is None:
            raise ValueError("For gamma == 1 (log utility), provide beta.")
        # CEV formula for log utility
        return np.exp((1.0 - beta) * (Vb - Vr)) - 1.0