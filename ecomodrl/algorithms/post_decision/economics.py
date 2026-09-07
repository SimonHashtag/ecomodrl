# -----------------------------------------------------------------------------
# Torch replicas of the economic pieces the post-decision critics need.
#
# Every model family implements, from the plain-float spec returned by
# `EconomicModel.post_decision_spec()`:
#
#     controls(obs, action) -> (c, pd_features)
#         consumption implied by the Gym action and the post-decision features
#         the continuation network W is evaluated on
#
#     next_obs(obs, action) -> (next_obs, weights)
#         next observations for every quadrature node, shape (B, K, obs_dim),
#         and the node weights, shape (K,)
#
#     utility(c)
#         CRRA utility clamped at the environment's reward floor
#
# Observation convention (see ecomodrl/envs/envs.py):
#     obs = (log(m / m_ref), log(x1), ..., log(xd))   growth models, buffer stock
#     obs = (log(m / m_ref), z1, ..., zN)              portfolio choice
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch as th

EPS_CONSUMPTION = 1e-8  # the environment floors consumption at this value


class PDEconomics:
    """Base class. Subclasses implement `_core` and `next_obs`."""

    def __init__(self, spec: dict):
        if spec is None:
            raise ValueError("A post-decision spec is required (EconomicModel.post_decision_spec).")
        self.spec = dict(spec)

        s = self.spec
        self.family = str(s["family"])
        self.m_ref = float(s["m_ref"])
        self.smin = float(s["smin"])
        self.smax = float(s["smax"])
        self.crra = float(s["crra"])
        self.reward_floor = float(s["reward_floor"])
        self.beta_eff = float(s["beta_eff"])
        self.obs_dim = int(s["obs_dim"])
        self.exo_dim = int(s["exo_dim"])
        self.action_dim = int(s["action_dim"])
        self.pd_dim = int(s["pd_dim"])

        self._cache: dict[tuple[str, th.device], th.Tensor] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def tensor(self, key: str, ref: th.Tensor) -> th.Tensor:
        """Float32 tensor of `spec[key]` on the device of `ref`, cached."""
        k = (key, ref.device)
        t = self._cache.get(k)
        if t is None:
            t = th.as_tensor(np.asarray(self.spec[key], dtype=np.float32), device=ref.device)
            self._cache[k] = t
        return t

    @staticmethod
    def _prep(obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return obs.float(), actions.float().clamp(-1.0, 1.0)

    def cash_on_hand(self, obs: th.Tensor) -> th.Tensor:
        return self.m_ref * th.exp(obs[:, 0:1])

    def share(self, actions: th.Tensor) -> th.Tensor:
        return self.smin + 0.5 * (self.smax - self.smin) * (actions[:, 0:1] + 1.0)

    def utility(self, c: th.Tensor) -> th.Tensor:
        if self.crra == 1.0:
            u = th.log(c)
        else:
            u = c ** (1.0 - self.crra) / (1.0 - self.crra)
        return th.clamp(u, min=-self.reward_floor)  # mirrors TransformReward in setup_envs

    # ------------------------------------------------------------------
    # interface
    # ------------------------------------------------------------------

    def _core(self, obs: th.Tensor, actions: th.Tensor) -> dict:
        raise NotImplementedError

    def controls(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        core = self._core(obs, actions)
        return core["c"], core["feats"]

    def next_obs(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        raise NotImplementedError


# =============================================================================
# Growth models: Cass-Koopmans (one node) and Brock-Mirman (AR(1) technology)
# =============================================================================
class GrowthPD(PDEconomics):
    """Post-decision state a = m - c (and z); m' = z' A a^alpha + (1 - delta) a."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.A = float(spec["A"])
        self.alpha = float(spec["alpha"])
        self.delta = float(spec["delta"])
        self.rho = float(spec["rho"])

    def _core(self, obs, actions):
        obs, actions = self._prep(obs, actions)
        m = self.cash_on_hand(obs)
        c = th.clamp(self.share(actions) * m, min=EPS_CONSUMPTION)
        a = th.clamp(m - c, min=1e-12)
        if self.exo_dim == 0:
            z_log = th.zeros_like(a)
            feats = th.log(a)
        else:
            z_log = obs[:, 1:2]
            feats = th.cat([th.log(a), z_log], dim=1)
        return {"m": m, "c": c, "a": a, "z_log": z_log, "feats": feats}

    def next_obs(self, obs, actions):
        core = self._core(obs, actions)
        a, z_log = core["a"], core["z_log"]

        eps = self.tensor("eps_nodes", a)  # (K,)
        weights = self.tensor("weights", a)  # (K,)

        z_next_log = self.rho * z_log + eps[None, :]  # (B, K)
        z_next = th.exp(z_next_log)
        m_next = z_next * self.A * a ** self.alpha + (1.0 - self.delta) * a  # (B, K)

        m_obs = th.log(m_next / self.m_ref)
        if self.exo_dim == 0:
            return m_obs[..., None], weights
        return th.stack([m_obs, z_next_log], dim=-1), weights


# =============================================================================
# Buffer stock
# =============================================================================
class BufferStockPD(PDEconomics):
    """Post-decision state a = m - c; m'_k = R / (G psi_k) a + theta_k."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.R = float(spec["R"])
        self.G = float(spec["G"])

    def _core(self, obs, actions):
        obs, actions = self._prep(obs, actions)
        m = self.cash_on_hand(obs)
        c = th.clamp(self.share(actions) * m, min=EPS_CONSUMPTION)
        a = th.clamp(m - c, min=0.0)
        return {"m": m, "c": c, "a": a, "feats": th.log1p(a)}

    def next_obs(self, obs, actions):
        a = self._core(obs, actions)["a"]

        psi = self.tensor("psi_nodes", a)
        theta = self.tensor("theta_nodes", a)
        weights = self.tensor("weights", a)

        m_next = (self.R / (self.G * psi))[None, :] * a + theta[None, :]  # (B, K)
        return th.log(m_next / self.m_ref)[..., None], weights


# =============================================================================
# Portfolio choice
# =============================================================================
class PortfolioPD(PDEconomics):
    """Post-decision state (rs, w, z); W takes (log1p(rs), w_risky, z).

    Node k of the spec carries (eps_k, eta_k, psi_k, theta_emp_k). Each node is
    expanded into an employed branch (theta_emp_k, weight (1 - p) w_k) and an
    unemployed branch (theta_low, weight p w_k).
    """

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.R = float(spec["R"])
        self.G = float(spec["G"])
        self.n_risky = int(spec["n_risky"])
        self.logit_cap = float(spec["logit_cap"])
        self.unemp_prob = float(spec["unemp_prob"])
        self.theta_low = float(spec["theta_low"])

    def _core(self, obs, actions):
        obs, actions = self._prep(obs, actions)
        n = self.n_risky

        m = self.cash_on_hand(obs)
        c = th.clamp(self.share(actions) * m, min=EPS_CONSUMPTION)
        rs = th.clamp(m - c, min=0.0)

        # torch copy of PortfolioChoice.alr_safe_pivot_to_weights
        x = th.clamp(self.logit_cap * actions[:, 1:1 + n], -self.logit_cap, self.logit_cap)
        mx = th.clamp(x.max(dim=-1, keepdim=True).values, min=0.0)
        e = th.exp(x - mx)
        pivot = th.exp(-mx)
        denom = pivot + e.sum(dim=-1, keepdim=True)
        w_safe = pivot / denom  # (B, 1)
        w_risky = e / denom  # (B, N)

        z = obs[:, 1:1 + n]  # (B, N)
        feats = th.cat([th.log1p(rs), w_risky, z], dim=1)
        return {"m": m, "c": c, "rs": rs, "w_safe": w_safe, "w_risky": w_risky, "z": z, "feats": feats}

    def next_obs(self, obs, actions):
        core = self._core(obs, actions)
        rs, w_safe, w_risky, z = core["rs"], core["w_safe"], core["w_risky"], core["z"]

        eps = self.tensor("eps_nodes", rs)  # (K0, N)
        eta = self.tensor("eta_nodes", rs)  # (K0, N)
        psi = self.tensor("psi_nodes", rs)  # (K0,)
        theta_emp = self.tensor("theta_nodes", rs)  # (K0,)
        w_nodes = self.tensor("weights", rs)  # (K0,)

        rho_z = self.tensor("rho_z", rs)
        sigma_z = self.tensor("sigma_z", rs)
        kappa_R = self.tensor("kappa_R", rs)
        xi_R = self.tensor("xi_R", rs)
        sigma_R = self.tensor("sigma_R", rs)

        z_b = z[:, None, :]  # (B, 1, N)
        z_next = rho_z * z_b + sigma_z * eps[None, :, :]  # (B, K0, N)
        R_risky = th.exp(kappa_R + xi_R * z_b + sigma_R * eta[None, :, :])  # (B, K0, N)

        Rp = w_safe * self.R + (w_risky[:, None, :] * R_risky).sum(dim=-1)  # (B, K0)
        base = Rp / (self.G * psi[None, :]) * rs  # (B, K0)

        m_emp = base + theta_emp[None, :]
        m_unemp = base + self.theta_low

        obs_emp = th.cat([th.log(m_emp / self.m_ref)[..., None], z_next], dim=-1)
        obs_unemp = th.cat([th.log(m_unemp / self.m_ref)[..., None], z_next], dim=-1)

        next_obs = th.cat([obs_emp, obs_unemp], dim=1)  # (B, 2 K0, 1 + N)
        weights = th.cat([(1.0 - self.unemp_prob) * w_nodes, self.unemp_prob * w_nodes])
        return next_obs, weights


# =============================================================================
# Factory
# =============================================================================
FAMILIES = {
    "growth": GrowthPD,
    "bufferstock": BufferStockPD,
    "portfolio": PortfolioPD,
}


def make_economics(spec: dict) -> PDEconomics:
    if spec is None:
        raise ValueError("A post-decision spec is required (EconomicModel.post_decision_spec).")
    family = str(spec["family"])
    try:
        cls = FAMILIES[family]
    except KeyError as exc:
        raise ValueError(f"Unknown post-decision family {family!r}; known: {sorted(FAMILIES)}") from exc
    return cls(spec)
