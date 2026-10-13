# -----------------------------------------------------------------------------
# Post-decision critic and quadrature target shared by DDPG, TD3 and SAC.
#
#     Q(s, a) = u(c(s, a)) + W_i(post-decision features)
#
# with the exact utility u and one continuation network W_i per critic.
# The target integrates over the model's quadrature nodes:
#
#     target(s, a) = u(c) + gamma * sum_k w_k [ min_i Q_target,i(s'_k, a'_k) - ent_coef * log pi(a'_k | s'_k) ]
#
# so the stored reward, next observation and done flag of a transition are not
# used (infinite horizon; TimeLimit truncations are not terminations).
#
# All economics come from the EconomicModel itself (PostDecisionAdapter is a
# thin adapter): utility, action map and post-decision features run through the
# model's array-operation methods on torch tensors (differentiable in the
# action); the next states at the quadrature nodes are computed in numpy with
# the model's own `transition()` (no autograd needed there).
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import create_mlp

from ecomodrl.envs.envs import EPS
from ecomodrl.models import ModelSpec, model_from_spec

_MAX_EXPANDED_TARGET_ROWS = 2**16


def quadrature_batch_size(n_nodes: int) -> int:
    """Maximum source observations per bounded quadrature target pass."""
    if n_nodes < 1:
        raise ValueError("n_nodes must be at least one.")
    return max(1, _MAX_EXPANDED_TARGET_ROWS // n_nodes)


class PostDecisionAdapter:
    """Adapter between the torch critics and the economic model.

    Plain object (not an nn.Module): the policies assign it before
    `nn.Module.__init__` runs, and it must not appear in state dicts.
    """

    def __init__(self, model, reward_floor: float):
        self.model = model
        self.reward_floor = float(reward_floor)
        self.pd_dim = int(model.post_decision_dim)
        self.n_nodes = int(model.n_quadrature_nodes)
        self._weights_cache: dict[str, th.Tensor] = {}

    def weights(self, weights, device) -> th.Tensor:
        """Quadrature weights as a float32 tensor on `device`.

        Fixed (K,) weights are cached per device; state-dependent (B, K) weights
        (Markov exogenous states) are converted on every call.
        """
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 2:
            return th.as_tensor(w, device=device)
        key = str(device)
        t = self._weights_cache.get(key)
        if t is None:
            t = th.as_tensor(w, device=device)
            self._weights_cache[key] = t
        return t

    def controls(self, obs: th.Tensor, actions: th.Tensor) -> tuple[dict, th.Tensor]:
        """Controls dict (float64 tensors) and post-decision features (float32, (B, pd_dim)).

        Differentiable in `actions`. Mirrors `BaseEconEnv.step`: the model's
        action map, then the consumption floor EPS.
        """
        state = self.model.state_from_obs(obs.to(th.float64))
        controls = self.model.controls_from_action(actions.to(th.float64), state=state)
        controls["consumption"] = th.clamp(controls["consumption"], min=EPS)
        feats = self.model.post_decision_features(state, controls)
        return controls, feats.to(th.float32)

    def utility(self, controls: dict) -> th.Tensor:
        """Period reward floored at -reward_floor (mirrors TransformReward in setup_envs), float32 (B, 1)."""
        u = th.clamp(self.model.utility_from_controls(controls), min=-self.reward_floor)
        return u.to(th.float32).reshape(-1, 1)

    @th.no_grad()
    def next_obs(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Next observations at every quadrature node, shape (B, K, obs_dim), and
        weights, shape (K,) or, for Markov exogenous states, (B, K).

        Computed in numpy with the model's `transition()` and `obs_from_state`,
        exactly like the environment step.
        """
        o = obs.detach().cpu().numpy().astype(np.float64)
        a = actions.detach().cpu().numpy().astype(np.float64)

        state = self.model.state_from_obs(o)
        controls = self.model.controls_from_action(a, state=state)
        controls["consumption"] = np.maximum(controls["consumption"], EPS)

        next_state, weights = self.model.quadrature_next_states(state, controls)
        next_obs = self.model.obs_from_state(next_state)

        return th.as_tensor(next_obs, device=obs.device), self.weights(weights, obs.device)


def make_post_decision_adapter(
    model_spec: ModelSpec,
    reward_floor: float,
) -> PostDecisionAdapter:
    """Rebuild the economic model and wrap its post-decision operations."""
    return PostDecisionAdapter(model_from_spec(model_spec), reward_floor)


class PostDecisionCritic(BaseModel):
    """Drop-in replacement for SB3's ContinuousCritic with the same call interface."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn=nn.ReLU,
        normalize_images=True,
        n_critics=2,
        share_features_extractor=True,
        econ: PostDecisionAdapter | None = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if econ is None:
            raise ValueError("PostDecisionCritic needs a PostDecisionAdapter.")
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.econ = econ

        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            w_net = nn.Sequential(*create_mlp(self.econ.pd_dim, 1, net_arch, activation_fn))
            self.add_module(f"qf{idx}", w_net)
            self.q_networks.append(w_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        controls, feats = self.econ.controls(obs, actions)
        u = self.econ.utility(controls)
        return tuple(u + w_net(feats) for w_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        controls, feats = self.econ.controls(obs, actions)
        return self.econ.utility(controls) + self.q_networks[0](feats)


def quadrature_target(
    econ: PostDecisionAdapter,
    critic_target,
    obs: th.Tensor,
    actions: th.Tensor,
    gamma: float,
    next_action_fn,
    ent_coef=None,
) -> th.Tensor:
    """u(c) + gamma * E_k[ min_i Q_target,i(s'_k, a'_k) - ent_coef * log pi(a'_k | s'_k) ].

    `next_action_fn(next_obs) -> (actions, log_prob or None)` supplies the target
    policy: the smoothed target actor for TD3/DDPG, a sampled action with its
    log-probability for SAC. Call under `torch.no_grad()`.
    """
    controls, _ = econ.controls(obs, actions)
    u = econ.utility(controls)

    rows_per_chunk = quadrature_batch_size(econ.n_nodes)
    expected_values = []
    for start in range(0, obs.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, obs.shape[0])
        next_obs, weights = econ.next_obs(obs[start:stop], actions[start:stop])
        B, K, D = next_obs.shape
        flat = next_obs.reshape(B * K, D)

        a_next, logp = next_action_fn(flat)
        q_next = th.cat(critic_target(flat, a_next), dim=1)
        q_next, _ = th.min(q_next, dim=1, keepdim=True)
        if ent_coef is not None and logp is not None:
            q_next = q_next - ent_coef * logp.reshape(-1, 1)

        q_next = q_next.reshape(B, K)
        w = weights if weights.dim() == 2 else weights[None, :]
        expected_values.append((q_next * w).sum(dim=1, keepdim=True))

    ev = th.cat(expected_values, dim=0)
    return u + gamma * ev
