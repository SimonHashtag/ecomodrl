from typing import Sequence
from scipy.interpolate import RegularGridInterpolator
import numpy as np

class VPostExtrapolator:
    """
    Wrapper around an interior RegularGridInterpolator for V⁺(a,z) that
    provides a stable, CRRA-consistent tail for a > a_max:

        V⁺(a,z) ≈ C(z) + B(z) * a^(1-γ),    γ ≠ 1
        V⁺(a,z) ≈ v_max(z) + s(z) * (a - a_max),  γ = 1  (log case: linear tail)

    - Inside the grid (a in [a_min, a_max]): delegate to interior interpolator.
    - Along z: we clamp to the z-grids (no multidim extrapolation).
    """
    def __init__(
        self,
        v_post_grid: np.ndarray,
        v_post_interior: RegularGridInterpolator,
        rs_grid: np.ndarray,
        z_grids: Sequence[np.ndarray],
        gamma: float,
    ):  
        # Fast interior interpolator (uniform-grid multilinear); falls back to RGI
        self._f = v_post_interior
        self.rs_grid = rs_grid
        self.rs_min, self.rs_max = float(self.rs_grid[0]), float(self.rs_grid[-1])
        self.z_grids = z_grids
        self.gamma = float(gamma)
        # expose grid for compatibility with any helper that inspects it
        self.grid = (rs_grid, *z_grids)
        self.C_itp, self.B_itp, self.vmax_itp, self.slope_itp = self._fit_vpost_tail_coeffs_CRRA(v_post_grid)
        self._zmins = np.array([zg[0] for zg in self.z_grids], dtype=float)
        self._zmaxs = np.array([zg[-1] for zg in self.z_grids], dtype=float)
        self._pow = 1.0 - self.gamma  # used when gamma != 1

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        a = pts[:, 0]
        z = pts[:, 1:]
        # clamp z to grid; leave 'a' for tail logic
        zc = np.minimum(np.maximum(z, self._zmins), self._zmaxs)
        out = np.empty(a.shape, dtype=float)
        # in-bound mask along a
        m_in = a <= self.rs_max
        if np.any(m_in):
            a_c = np.clip(a[m_in], self.rs_min, self.rs_max)
            pts_in = np.column_stack([a_c, zc[m_in]])
            out[m_in] = self._f(pts_in)
        m_out = ~m_in
        if np.any(m_out):
            ao = a[m_out]
            zo = zc[m_out]
            if abs(self.gamma - 1.0) > 1e-12:
                C = self.C_itp(zo)
                B = self.B_itp(zo)
                out[m_out] = C + B * (ao ** self._pow)
            else:
                v_max = self.vmax_itp(zo)
                s = self.slope_itp(zo)
                out[m_out] = v_max + s * (ao - self.rs_max)
        return out
    
    def _fit_vpost_tail_coeffs_CRRA(
        self,
        v_post_grid: np.ndarray,
    ) -> tuple[RegularGridInterpolator | None,
            RegularGridInterpolator | None,
            RegularGridInterpolator | None,
            RegularGridInterpolator | None]:
        """
        Fit per-z tail parameters at the upper rs boundary using the last two slices.
        For γ≠1: V⁺(a,z) ≈ C(z) + B(z)*a^(1-γ).  For γ=1: linear tail via endpoint slope.
        Returns (C_itp, B_itp, vmax_itp, slope_itp). Only the relevant pair is non-None.
        """
        a2, a1 = float(self.rs_grid[-1]), float(self.rs_grid[-2])
        h = max(1e-12, a2 - a1)
        v2 = v_post_grid[-1, ...]          # shape: (*z_dims,)
        v1 = v_post_grid[-2, ...]
        s = np.maximum((v2 - v1) / h, 0.0)  # enforce monotone endpoint slope
        if abs(self.gamma - 1.0) > 1e-12:
            # CRRA tail
            B = s * (a2 ** self.gamma) / (1.0 - self.gamma)       # ≤ 0 for γ>1
            C = v2 - B * (a2 ** (1.0 - self.gamma))
            C = np.minimum(C, 0.0)                       # CRRA γ≥1 ⇒ V ≤ 0 (normalized)
            C_itp = RegularGridInterpolator(self.z_grids, C, bounds_error=False, fill_value=None)
            B_itp = RegularGridInterpolator(self.z_grids, B, bounds_error=False, fill_value=None)
            return C_itp, B_itp, None, None
        else:
            # log utility: linear tail with endpoint slope
            vmax_itp  = RegularGridInterpolator(self.z_grids, v2, bounds_error=False, fill_value=None)
            slope_itp = RegularGridInterpolator(self.z_grids, s,  bounds_error=False, fill_value=None)
            return None, None, vmax_itp, slope_itp