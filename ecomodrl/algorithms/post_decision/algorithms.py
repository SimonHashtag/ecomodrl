# -----------------------------------------------------------------------------
# Post-decision ("structure-exploiting") versions of the four SB3 algorithms.
#
# PDDDPG, PDTD3 : SB3 TD3 training step with the critic target replaced by the
#                 quadrature target (twin W networks for TD3, one for DDPG).
# PDSAC         : SB3 SAC training step with the target replaced (entropy term
#                 evaluated per node).
# PDPPO         : value head V(s) = u(c(s, mu(s))) + W(post-decision features)
#                 with the policy mean detached; before each update the rollout
#                 buffer's returns and advantages are replaced by the quadrature
#                 target V_tgt(s_t) = u(c_t) + gamma * sum_k w_k V(s'_k) and the
#                 one-step post-decision advantage A_t = V_tgt(s_t) - V(s_t).
#                 GAE / gae_lambda are not used; clipping and entropy unchanged.
#
# All classes take `pd_spec` (EconomicModel.post_decision_spec) in policy_kwargs.
# Observation normalization (VecNormalize) is not supported: the critics invert
# the observation transform obs = log(m / m_ref).
# -----------------------------------------------------------------------------

from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy

from ecomodrl.algorithms.post_decision.critic import PostDecisionCritic, quadrature_target
from ecomodrl.algorithms.post_decision.economics import make_economics


# =============================================================================
# Policies with the post-decision critic (TD3 / DDPG / SAC)
# =============================================================================
class PDTD3Policy(TD3Policy):
    def __init__(self, *args, pd_spec=None, **kwargs):
        self.pd_spec = pd_spec
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PostDecisionCritic(**critic_kwargs, pd_spec=self.pd_spec).to(self.device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["pd_spec"] = self.pd_spec
        return data


class PDSACPolicy(SACPolicy):
    def __init__(self, *args, pd_spec=None, **kwargs):
        self.pd_spec = pd_spec
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PostDecisionCritic(**critic_kwargs, pd_spec=self.pd_spec).to(self.device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["pd_spec"] = self.pd_spec
        return data


# =============================================================================
# TD3 / DDPG: copy of SB3 TD3.train with the target block replaced
# =============================================================================
def _pd_td3_train(self, gradient_steps: int, batch_size: int = 100) -> None:
    self.policy.set_training_mode(True)
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    econ = self.critic.econ
    actor_losses, critic_losses = [], []

    def next_action_fn(next_obs):
        a = self.actor_target(next_obs)
        noise = th.randn_like(a) * self.target_policy_noise
        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
        return (a + noise).clamp(-1.0, 1.0), None

    for _ in range(gradient_steps):
        self._n_updates += 1
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

        with th.no_grad():
            target_q_values = quadrature_target(
                econ, self.critic_target,
                replay_data.observations, replay_data.actions,
                self.gamma, next_action_fn,
            )

        current_q_values = self.critic(replay_data.observations, replay_data.actions)
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        critic_losses.append(critic_loss.item())

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        if self._n_updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    if len(actor_losses) > 0:
        self.logger.record("train/actor_loss", np.mean(actor_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))


class PDTD3(TD3):
    """TD3 with post-decision twin critics and quadrature targets."""

    policy_aliases = {"MlpPolicy": PDTD3Policy}
    train = _pd_td3_train


class PDDDPG(DDPG):
    """DDPG with a post-decision critic and quadrature target."""

    policy_aliases = {"MlpPolicy": PDTD3Policy}
    train = _pd_td3_train


# =============================================================================
# SAC: copy of SB3 SAC.train with the target block replaced
# =============================================================================
class PDSAC(SAC):
    """SAC with post-decision twin critics and quadrature targets."""

    policy_aliases = {"MlpPolicy": PDSACPolicy}

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        econ = self.critic.econ
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        def next_action_fn(next_obs):
            a, logp = self.actor.action_log_prob(next_obs)
            return a, logp.reshape(-1, 1)

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                target_q_values = quadrature_target(
                    econ, self.critic_target,
                    replay_data.observations, replay_data.actions,
                    self.gamma, next_action_fn, ent_coef=ent_coef,
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


# =============================================================================
# PPO: post-decision value head + quadrature returns / advantages
# =============================================================================
def _split_net_arch(net_arch):
    """(actor arch, W arch) from an SB3 net_arch spec (None, list, or dict)."""
    if net_arch is None:
        pi_arch, w_arch = [64, 64], [64, 64]
    elif isinstance(net_arch, dict):
        pi_arch = list(net_arch.get("pi", []))
        w_arch = list(net_arch.get("vf", []))
    else:
        pi_arch = list(net_arch)
        w_arch = list(net_arch)
    if not w_arch:
        w_arch = list(pi_arch)
    return pi_arch, w_arch


class PDPPOPolicy(ActorCriticPolicy):
    """ActorCriticPolicy whose value is V(s) = u(c(s, mu(s))) + W(post-decision features).

    The policy mean mu(s) is detached, so the value loss trains only W. The
    network arch given as `net_arch` is interpreted as {"pi": actor, "vf": W};
    the parent's value branch is built empty and left unused.
    """

    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, pd_spec=None, **kwargs):
        if pd_spec is None:
            raise ValueError("PDPPOPolicy needs pd_spec (EconomicModel.post_decision_spec).")
        self.pd_spec = dict(pd_spec)
        self.econ = make_economics(pd_spec)
        self._pd_net_arch = net_arch
        pi_arch, self.w_arch = _split_net_arch(net_arch)
        super().__init__(
            observation_space, action_space, lr_schedule,
            net_arch={"pi": pi_arch, "vf": []},
            **kwargs,
        )

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.w_net = nn.Sequential(*create_mlp(self.econ.pd_dim, 1, self.w_arch, self.activation_fn))
        if self.ortho_init:
            self.w_net.apply(partial(self.init_weights, gain=np.sqrt(2)))
            last_linear = [m for m in self.w_net if isinstance(m, nn.Linear)][-1]
            self.init_weights(last_linear, gain=1)
        # Re-create the optimizer so it covers w_net as well.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(net_arch=self._pd_net_arch, pd_spec=self.pd_spec)
        return data

    # --- value head -------------------------------------------------------------
    def _pd_values(self, obs: th.Tensor, latent_pi: th.Tensor) -> th.Tensor:
        mean_actions = self.action_net(latent_pi).detach()
        c, feats = self.econ.controls(obs, mean_actions)
        return self.econ.utility(c) + self.w_net(feats)

    def _latent_pi(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features[0])
        return latent_pi

    # --- overrides ----------------------------------------------------------------
    def forward(self, obs: th.Tensor, deterministic: bool = False):
        latent_pi = self._latent_pi(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._pd_values(obs, latent_pi)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        latent_pi = self._latent_pi(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._pd_values(obs, latent_pi)
        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._pd_values(obs, latent_pi)


class PDPPO(PPO):
    """PPO with the post-decision value head; returns and advantages from quadrature.

    Before each update, with the current network on the whole rollout buffer:
    V_tgt(s_t) = u(c_t) + gamma * sum_k w_k V(s'_k(s_t, a_t)), A_t = V_tgt(s_t) - V(s_t).
    The GAE values written by collect_rollouts are overwritten; rewards are unused.
    """

    policy_aliases = {"MlpPolicy": PDPPOPolicy}

    def train(self) -> None:
        buf = self.rollout_buffer
        econ = self.policy.econ

        with th.no_grad():
            obs = th.as_tensor(buf.observations, device=self.device)
            act = th.as_tensor(buf.actions, device=self.device)
            n_steps, n_envs = obs.shape[0], obs.shape[1]
            obs_f = obs.reshape(n_steps * n_envs, -1)
            act_f = act.reshape(n_steps * n_envs, -1)

            c, _ = econ.controls(obs_f, act_f)
            u = econ.utility(c)

            nxt, weights = econ.next_obs(obs_f, act_f)
            B, K, D = nxt.shape
            v_next = self.policy.predict_values(nxt.reshape(B * K, D)).reshape(B, K)
            ev = (v_next * weights[None, :]).sum(dim=1, keepdim=True)

            v_tgt = u + self.gamma * ev
            v_now = self.policy.predict_values(obs_f)

        buf.returns = v_tgt.reshape(n_steps, n_envs).cpu().numpy().astype(np.float32)
        buf.values = v_now.reshape(n_steps, n_envs).cpu().numpy().astype(np.float32)
        buf.advantages = (buf.returns - buf.values).astype(np.float32)

        super().train()
