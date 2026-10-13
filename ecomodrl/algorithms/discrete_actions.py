# -----------------------------------------------------------------------------
# Discrete choices inside the continuous action Box.
#
# A model declares `discrete_action_bins = {coordinate: n}`: that coordinate of
# the action in [-1, 1] is split into n equal-width ordinal bins and the model
# executes bin k (see e.g. DurableGoods.controls_from_action). The environment,
# the analysis and SB3's Box-only DDPG/TD3/SAC therefore stay unchanged.
#
# This module adds the two places where the algorithms need to know about bins:
#
#   * PPO: HybridActorCriticPolicy with a native Categorical head per binned
#     coordinate (times a diagonal Gaussian for the continuous coordinates).
#     A sampled category is written into the Box as its bin centre.
#   * Post-decision DDPG/TD3/SAC: the critic Q = u(controls) + W(features) has a
#     zero gradient with respect to a binned coordinate, so their actor loss uses
#     the expectation of Q over the bins, sum_k P_k Q(s, bin centre k), where
#     P_k are the bin probabilities of a Gaussian around the actor's output
#     (`gaussian_bin_probs`, `expected_bin_q`).
# -----------------------------------------------------------------------------

from __future__ import annotations

from typing import Mapping

import torch as th
from torch import nn
from torch.distributions import Categorical, Normal

from stable_baselines3.common.distributions import DiagGaussianDistribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim


# =============================================================================
# Bin helpers
# =============================================================================

def normalize_bins(bins: Mapping | None) -> dict[int, int]:
    """{coordinate: n_bins} with plain-int keys and values, sorted by coordinate."""
    return {int(k): int(v) for k, v in sorted((bins or {}).items(), key=lambda kv: int(kv[0]))}


def single_discrete_coordinate(bins: Mapping) -> tuple[int, int]:
    """(coordinate, n_bins) of the only binned coordinate."""
    bins = normalize_bins(bins)
    if len(bins) != 1:
        raise NotImplementedError("Exactly one binned action coordinate is supported.")
    return next(iter(bins.items()))


def bin_centres(n: int, like: th.Tensor) -> th.Tensor:
    """Centres -1 + (2k + 1) / n of the n equal-width bins on [-1, 1]."""
    k = th.arange(n, dtype=like.dtype, device=like.device)
    return -1.0 + (2.0 * k + 1.0) / n


def bin_index(a: th.Tensor, n: int) -> th.Tensor:
    """Bin k of a in [-1, 1] (the same rule, in float64, the model uses to execute the action)."""
    return th.clamp(th.floor((a.to(th.float64) + 1.0) / 2.0 * n), 0, n - 1).long()


def gaussian_bin_probs(mu: th.Tensor, sigma: th.Tensor, n: int, *, squashed: bool) -> th.Tensor:
    """Probabilities (B, n) of the n bins under N(mu, sigma^2), outer bins open-ended.

    squashed=True: the Gaussian lives before a tanh squashing (SAC), so the bin
    edges are mapped through atanh. Only the n - 1 interior edges enter the
    normal CDF; the open outer edges are padded with 0 and 1, which keeps the
    gradient with respect to sigma finite.
    """
    edges = -1.0 + 2.0 * th.arange(1, n, dtype=mu.dtype, device=mu.device) / n
    if squashed:
        edges = th.atanh(edges)
    cdf = th.special.ndtr((edges[None, :] - mu[:, None]) / sigma[:, None])
    cdf = th.cat([th.zeros_like(cdf[:, :1]), cdf, th.ones_like(cdf[:, :1])], dim=1)
    return cdf[:, 1:] - cdf[:, :-1]


def expected_bin_q(q_fn, obs: th.Tensor, actions: th.Tensor, dim: int, n: int, probs: th.Tensor) -> th.Tensor:
    """sum_k probs[:, k] * q_fn(obs, actions with coordinate `dim` set to bin centre k), shape (B, 1).

    The gradient flows to the other action coordinates through q_fn and to the
    binned coordinate through `probs`.
    """
    B, A = actions.shape
    mask = th.zeros(A, dtype=th.bool, device=actions.device)
    mask[dim] = True
    centres = bin_centres(n, actions)
    a_k = th.where(mask, centres[None, :, None], actions[:, None, :])  # (B, n, A)
    o_k = obs[:, None, :].expand(B, n, obs.shape[-1])
    q = q_fn(o_k.reshape(B * n, -1), a_k.reshape(B * n, A)).reshape(B, n)
    return (probs * q).sum(dim=1, keepdim=True)


# =============================================================================
# PPO: Categorical x Gaussian action head
# =============================================================================

class HybridBinGaussianDistribution(DiagGaussianDistribution):
    """Categorical over the bins of each binned coordinate x diagonal Gaussian
    over the remaining coordinates.

    Subclasses DiagGaussianDistribution so that SB3's ActorCriticPolicy builds
    and calls it like the Gaussian it replaces; the action net outputs the
    concatenated logits followed by the Gaussian means.
    """

    def __init__(self, action_dim: int, discrete_action_bins: Mapping):
        super().__init__(action_dim)
        self.bins = normalize_bins(discrete_action_bins)
        self.discrete_dims = list(self.bins)
        self.continuous_dims = [i for i in range(action_dim) if i not in self.bins]
        self.n_logits = sum(self.bins.values())
        self.categoricals: list[Categorical] = []

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple[nn.Module, nn.Parameter]:
        action_net = nn.Linear(latent_dim, self.n_logits + len(self.continuous_dims))
        log_std = nn.Parameter(th.ones(len(self.continuous_dims)) * log_std_init, requires_grad=True)
        return action_net, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "HybridBinGaussianDistribution":
        self.categoricals = []
        start = 0
        for dim in self.discrete_dims:
            n = self.bins[dim]
            self.categoricals.append(Categorical(logits=mean_actions[:, start:start + n]))
            start += n
        mean_cont = mean_actions[:, start:]
        self.distribution = Normal(mean_cont, th.ones_like(mean_cont) * log_std.exp())
        return self

    def _assemble(self, categories: list[th.Tensor], continuous: th.Tensor) -> th.Tensor:
        """Action Box vector: bin centres at the binned coordinates, Gaussian values elsewhere."""
        columns: list[th.Tensor | None] = [None] * self.action_dim
        for dim, k in zip(self.discrete_dims, categories):
            columns[dim] = bin_centres(self.bins[dim], continuous)[k]
        for j, dim in enumerate(self.continuous_dims):
            columns[dim] = continuous[:, j]
        return th.stack(columns, dim=-1)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        log_prob = sum_independent_dims(self.distribution.log_prob(actions[:, self.continuous_dims]))
        for dim, cat in zip(self.discrete_dims, self.categoricals):
            log_prob = log_prob + cat.log_prob(bin_index(actions[:, dim], self.bins[dim]))
        return log_prob

    def entropy(self) -> th.Tensor:
        entropy = sum_independent_dims(self.distribution.entropy())
        for cat in self.categoricals:
            entropy = entropy + cat.entropy()
        return entropy

    def sample(self) -> th.Tensor:
        return self._assemble([cat.sample() for cat in self.categoricals], self.distribution.rsample())

    def mode(self) -> th.Tensor:
        return self._assemble([cat.probs.argmax(dim=-1) for cat in self.categoricals], self.distribution.mean)


class HybridActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy whose binned action coordinates get a Categorical head."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, discrete_action_bins=None, **kwargs):
        if not discrete_action_bins:
            raise ValueError(f"{type(self).__name__} needs discrete_action_bins.")
        self.discrete_action_bins = normalize_bins(discrete_action_bins)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule) -> None:
        if self.use_sde:
            raise ValueError("HybridActorCriticPolicy does not support gSDE (use_sde=True).")
        self.action_dist = HybridBinGaussianDistribution(get_action_dim(self.action_space), self.discrete_action_bins)
        super()._build(lr_schedule)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["discrete_action_bins"] = self.discrete_action_bins
        return data
