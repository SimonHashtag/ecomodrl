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
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch as th
from torch import nn

from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.torch_layers import create_mlp

from ecomodrl.algorithms.post_decision.economics import PDEconomics, make_economics


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
        pd_spec=None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.pd_spec = dict(pd_spec) if pd_spec is not None else None
        self.econ: PDEconomics = make_economics(pd_spec)

        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            w_net = nn.Sequential(*create_mlp(self.econ.pd_dim, 1, net_arch, activation_fn))
            self.add_module(f"qf{idx}", w_net)
            self.q_networks.append(w_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        c, feats = self.econ.controls(obs, actions)
        u = self.econ.utility(c)
        return tuple(u + w_net(feats) for w_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        c, feats = self.econ.controls(obs, actions)
        return self.econ.utility(c) + self.q_networks[0](feats)


def quadrature_target(
    econ: PDEconomics,
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
    c, _ = econ.controls(obs, actions)
    u = econ.utility(c)

    next_obs, weights = econ.next_obs(obs, actions)  # (B, K, D), (K,)
    B, K, D = next_obs.shape
    flat = next_obs.reshape(B * K, D)

    a_next, logp = next_action_fn(flat)
    q_next = th.cat(critic_target(flat, a_next), dim=1)
    q_next, _ = th.min(q_next, dim=1, keepdim=True)
    if ent_coef is not None and logp is not None:
        q_next = q_next - ent_coef * logp.reshape(-1, 1)

    q_next = q_next.reshape(B, K)
    ev = (q_next * weights[None, :]).sum(dim=1, keepdim=True)
    return u + gamma * ev
