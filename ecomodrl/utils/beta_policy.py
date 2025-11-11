from __future__ import annotations

from typing import Optional, Tuple, Dict, Type, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta as TorchBeta
from torch.special import digamma

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy, ContinuousCritic

# ============================================================
#                 Factorized Beta on [0,1]^A
# ============================================================
class BetaDistribution(Distribution):
    """
    Factorized Beta distribution over actions in Box([0,1]^A).
    Compatible with SB3 v2.x Distribution API.
    """

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.alpha: Optional[torch.Tensor] = None  # (B, A)
        self.beta: Optional[torch.Tensor] = None   # (B, A)

    # actor head that outputs raw params (alpha_raw, beta_raw)
    def proba_distribution_net(self, latent_dim: int):
        # (module, log_std_param_or_None) -> None for Beta
        return nn.Linear(latent_dim, 2 * self.action_dim), None

    def proba_distribution(self, action_params: torch.Tensor, **kwargs) -> "BetaDistribution":
        alpha_raw, beta_raw = torch.chunk(action_params, 2, dim=-1)
        eps = 1e-6
        self.alpha = F.softplus(alpha_raw) + eps
        self.beta = F.softplus(beta_raw) + eps
        return self

    def sample(self) -> torch.Tensor:
        return TorchBeta(self.alpha, self.beta).rsample()  # reparameterized

    def mode(self) -> torch.Tensor:
        a, b = self.alpha, self.beta
        mode = (a - 1.0) / torch.clamp(a + b - 2.0, min=1e-6)
        use_mean = (a <= 1.0) | (b <= 1.0)
        mean = a / (a + b)
        return torch.where(use_mean, mean, mode)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        a = torch.clamp(actions, eps, 1 - eps)
        logp = (
            (self.alpha - 1.0) * torch.log(a)
            + (self.beta - 1.0) * torch.log1p(-a)
            - torch.lgamma(self.alpha) - torch.lgamma(self.beta)
            + torch.lgamma(self.alpha + self.beta)
        )
        return logp.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        a, b = self.alpha, self.beta
        lnB = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        H = lnB - (a - 1.0) * digamma(a) - (b - 1.0) * digamma(b) + (a + b - 2.0) * digamma(a + b)
        return H.sum(dim=-1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.mode() if deterministic else self.sample()

    def actions_from_params(
        self, action_params: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.proba_distribution(action_params)
        actions = self.get_actions(deterministic=deterministic)
        return actions, self.log_prob(actions)

    def log_prob_from_params(self, action_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # SB3 v2.x: return (actions, log_prob)
        self.proba_distribution(action_params)
        actions = self.sample()
        return actions, self.log_prob(actions)


# ============================================================
#            PPO / A2C: Beta Actor-Critic policy
# ============================================================
class BetaActorCriticPolicy(ActorCriticPolicy):
    """
    Use with: PPO(policy=BetaActorCriticPolicy, ...)
    Expects env.action_space = Box(0.0, 1.0, (A,), dtype=np.float32)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,          # kept for API compat; ignored
        log_std_init: float = 0.0,      # unused for Beta
        full_std: bool = True,          # unused
        use_expln: bool = False,        # unused
        squash_output: bool = False,    # unused (already [0,1])
        features_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None,
        features_extractor_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Union[float, int]]] = None,
    ):
        if features_extractor_class is None:
            features_extractor_class = FlattenExtractor
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=False,  # Beta doesn't use gSDE
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=False,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.action_dist = BetaDistribution(self.action_space.shape[0])

        # Actor head: latent_pi -> 2*A Beta params
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net, _ = self.action_dist.proba_distribution_net(latent_dim_pi)

        if self.ortho_init:
            for m in [self.action_net, self.value_net]:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01 if m is self.action_net else 1.0)
                    nn.init.constant_(m.bias, 0.0)

        # ensure optimizer includes new action head
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[arg-type]

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> BetaDistribution:
        action_params = self.action_net(latent_pi)  # (B, 2*A)
        return self.action_dist.proba_distribution(action_params)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.get_distribution(observation)
        return dist.get_actions(deterministic=deterministic)


# ============================================================
#                     SAC: Beta policy
# ============================================================
class _BetaActor(nn.Module):
    """
    SAC-style actor that outputs a Beta distribution on [0,1]^A.
    Interface matches what SAC expects: forward(obs, deterministic=False) -> (actions, log_prob).
    """

    def __init__(
        self,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module],
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.action_dim = action_dim

        policy_net = create_mlp(features_dim, -1, net_arch, activation_fn)  # -1 = last hidden size
        last_hidden = net_arch[-1] if len(net_arch) > 0 else features_dim
        self.mlp = nn.Sequential(*policy_net) if len(policy_net) > 0 else nn.Identity()

        # Head that produces 2*A params
        self.action_net = nn.Linear(last_hidden if len(policy_net) > 0 else features_dim, 2 * action_dim)
        self.dist = BetaDistribution(action_dim)

        for m in list(self.mlp.modules()) + [self.action_net]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

        # Optional: bias towards moderate alphas/betas (mean 0.5, finite var)
        # with softplus, bias of ~1.0 gives alpha,beta≈softplus(1)≈1.31 (tame)
        nn.init.constant_(self.action_net.bias, 1.0)

    def get_dist(self, obs: torch.Tensor) -> BetaDistribution:
        features = self.features_extractor(obs)
        latent = self.mlp(features)
        params = self.action_net(latent)
        return self.dist.proba_distribution(params)

    @torch.no_grad()
    def _mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.get_dist(obs)
        return dist.mode()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns actions in [0,1]^A (assumes env bounds are [0,1]) and the corresponding log-prob.
        """
        dist = self.get_dist(obs)
        if deterministic:
            actions = dist.mode()
            log_prob = dist.log_prob(actions)
        else:
            actions, log_prob = dist.log_prob_from_params(self.action_net(self.mlp(self.features_extractor(obs))))
            # ^ faster path would reuse params; correctness first.

        return actions, log_prob


class BetaSACPolicy(SACPolicy):
    """
    Drop-in for SAC(policy=BetaSACPolicy, ...), using a Beta actor head.
    Assumes Box action_space in [0,1]^A.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,               # actor hidden sizes
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,                              # ignored
        features_extractor_class: Optional[Type[BaseFeaturesExtractor]] = None,
        features_extractor_kwargs: Optional[Dict[str, Union[int, float, str]]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Union[float, int]]] = None,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=False,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        # --- features extractor ---
        self._build_mlp_extractor()  # sets self.features_extractor and features_dim

        action_dim = self.action_space.shape[0]
        # Actor (Beta)
        self.actor = _BetaActor(
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            action_dim=action_dim,
            net_arch=self.net_arch or [256, 256],
            activation_fn=self.activation_fn,
        )
        # Critic(s) unchanged
        self.critic = ContinuousCritic(
            self.features_extractor, self.features_dim, action_dim,
            net_arch=self.net_arch or [256, 256], activation_fn=self.activation_fn
        )
        self.critic_target = ContinuousCritic(
            self.features_extractor, self.features_dim, action_dim,
            net_arch=self.net_arch or [256, 256], activation_fn=self.activation_fn
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        actor_lr = lr_schedule(1)
        critic_lr = lr_schedule(1)
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=actor_lr, **self.optimizer_kwargs)  # type: ignore[arg-type]
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=critic_lr, **self.optimizer_kwargs)  # type: ignore[arg-type]

    # Predict path used by SAC (env.step)
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        actions, _ = self.actor(observation, deterministic=deterministic)
        return actions
