# -----------------------------------------------------------------------------
# Post-decision versions of the four SB3 algorithms.
#
# PDDDPG, PDTD3 : SB3 TD3 training step with the critic target replaced by the
#                 quadrature target (twin W networks for TD3, one for DDPG).
# PDSAC         : SB3 SAC training step with the target replaced (entropy term
#                 evaluated per node).
# PDPPO         : standard SB3 PPO with the ordinary pre-decision value net V(s).
#                 The one-step bootstrap V(s_{t+1}) in the GAE TD error is
#                 replaced by its quadrature expectation over the model's shock
#                 nodes at the visited post-decision state,
#                     y_t     = sum_k w_k V(s'_k(s_t, a_t)),
#                     delta_t = r_t + gamma * y_t - V(s_t),
#                     A_t     = sum_l (gamma lambda)^l delta_{t+l},
#                 computed once at rollout end with the detached value net.
#                 SB3's time-limit reward bootstrap is disabled (every step
#                 already bootstraps through y_t); clipping, entropy and the
#                 value regression on A_t + V(s_t) are unchanged.
#
# All classes store a model_spec plus reward_floor in policy_kwargs,
# so checkpoints reconstruct their economic model without an environment.
# Observation normalization (VecNormalize with norm_obs=True) is not supported:
# the critics and the quadrature transition invert the observation transform.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor, polyak_update
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.td3.policies import TD3Policy

from ecomodrl.algorithms.discrete_actions import (
    HybridActorCriticPolicy,
    expected_bin_q,
    gaussian_bin_probs,
    single_discrete_coordinate,
)
from ecomodrl.algorithms.post_decision.critic import (
    PostDecisionCritic,
    make_post_decision_adapter,
    quadrature_batch_size,
    quadrature_target,
)
from ecomodrl.models import ModelSpec


# =============================================================================
# Shared policy construction and checkpoint contract
# =============================================================================
class EconomicModelPolicyMixin:
    """Give an SB3 policy its reconstructed post-decision economic model."""

    def __init__(
        self,
        *args,
        model_spec: ModelSpec | None = None,
        reward_floor: float | None = None,
        **kwargs,
    ):
        if model_spec is None or reward_floor is None:
            raise ValueError(
                f"{type(self).__name__} needs model_spec and reward_floor."
            )
        self.model_spec = model_spec
        self.reward_floor = float(reward_floor)
        self.econ = make_post_decision_adapter(
            self.model_spec,
            self.reward_floor,
        )
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["model_spec"] = self.model_spec
        data["reward_floor"] = self.reward_floor
        return data


class PostDecisionCriticMixin(EconomicModelPolicyMixin):
    """Shared post-decision critic factory for TD3, DDPG and SAC."""
    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PostDecisionCritic(**critic_kwargs, econ=self.econ).to(self.device)


class PDTD3Policy(PostDecisionCriticMixin, TD3Policy):
    pass


class PDSACPolicy(PostDecisionCriticMixin, SACPolicy):
    pass


# =============================================================================
# TD3 / DDPG: copy of SB3 TD3.train with the target block replaced
# =============================================================================
def _pd_td3_train(self, gradient_steps: int, batch_size: int = 100) -> None:
    self.policy.set_training_mode(True)
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    econ = self.critic.econ
    bins = econ.model.discrete_action_bins
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
            if bins:
                # Binned (discrete) coordinate: Q has a zero gradient there, so ascend
                # the expectation of Q over the bins under a Gaussian of half a bin
                # width around the actor's output (see algorithms/discrete_actions.py).
                obs = replay_data.observations
                actions_pi = self.actor(obs)
                dim, n = single_discrete_coordinate(bins)
                probs = gaussian_bin_probs(
                    actions_pi[:, dim], th.full_like(actions_pi[:, dim], 1.0 / n), n, squashed=False
                )
                actor_loss = -expected_bin_q(self.critic.q1_forward, obs, actions_pi, dim, n, probs).mean()
            else:
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
        bins = econ.model.discrete_action_bins
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
            # Pre-tanh Gaussian of this sample (quadrature_target below overwrites it)
            pi_gauss = self.actor.action_dist.distribution if bins else None

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

            if bins:
                # Binned (discrete) coordinate: replace Q by its expectation over the
                # bins under the policy's own pre-tanh Gaussian for that coordinate.
                dim, n = single_discrete_coordinate(bins)
                probs = gaussian_bin_probs(pi_gauss.mean[:, dim], pi_gauss.stddev[:, dim], n, squashed=True)
                min_qf_pi = expected_bin_q(
                    lambda o, a: th.min(th.cat(self.critic(o, a), dim=1), dim=1, keepdim=True)[0],
                    replay_data.observations, actions_pi, dim, n, probs,
                )
            else:
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
# PPO: standard actor-critic policy + quadrature bootstrap in the GAE TD error
# =============================================================================
class PDPPOPolicy(EconomicModelPolicyMixin, ActorCriticPolicy):
    """SB3's ActorCriticPolicy (ordinary pre-decision value net V(s)) that carries
    a PostDecisionAdapter. It provides the quadrature transition over the
    model's shock nodes that PDPPO uses for the bootstrap.
    """


class PDHybridPPOPolicy(EconomicModelPolicyMixin, HybridActorCriticPolicy):
    """PDPPOPolicy with a Categorical head for binned (discrete) action coordinates."""


class PDPPO(PPO):
    """PPO whose GAE TD error bootstraps through the quadrature expectation of V.

    For every stored transition (s_t, a_t) the post-decision state is rebuilt
    from the executed (clipped) action and the next observations at the model's
    shock nodes are evaluated with the detached value net:

        y_t     = sum_k w_k V(s'_k(s_t, a_t))
        delta_t = r_t + gamma * y_t * (1 - terminated_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - episode_start_{t+1}) * A_{t+1}

    with r_t the rollout-buffer reward, and returns A_t + V(s_t) as in SB3.
    `collect_rollouts` is SB3's with the time-limit reward bootstrap removed
    (delta_t already bootstraps at every step, so truncations only cut the GAE
    recursion) and true terminations recorded. The B*K next states are
    evaluated in bounded chunks.
    """

    policy_aliases = {"MlpPolicy": PDPPOPolicy}

    # --- rollout: copy of OnPolicyAlgorithm.collect_rollouts (SB3 2.6 / 2.7) --------
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        terminated = np.zeros((n_rollout_steps, env.num_envs), dtype=np.float32)
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # No time-limit reward bootstrap here (SB3 adds gamma * V(terminal_obs)
            # to the reward): the quadrature bootstrap covers every step, so a
            # truncation only cuts the GAE recursion. Record true terminations.
            for idx, done in enumerate(dones):
                if done and not infos[idx].get("TimeLimit.truncated", False):
                    terminated[n_steps - 1, idx] = 1.0

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        # Quadrature bootstrap + GAE replace compute_returns_and_advantage; the
        # value of the final observation is not needed.
        self._pd_returns_and_advantages(rollout_buffer, terminated)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    # --- quadrature bootstrap -----------------------------------------------------
    @th.no_grad()
    def _pd_bootstrap(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """y = sum_k w_k V(s'_k) for flat (B, obs_dim) observations and (B, action_dim)
        buffer actions (unclipped; the model clips them like the env does), evaluated
        in bounded chunks of next states."""
        econ = self.policy.econ
        obs_t = th.as_tensor(obs, device=self.device)
        act_t = th.as_tensor(actions, device=self.device)
        n = obs_t.shape[0]

        rows = quadrature_batch_size(econ.n_nodes)

        out = th.empty(n, device=self.device)
        for start in range(0, n, rows):
            nxt, weights = econ.next_obs(obs_t[start:start + rows], act_t[start:start + rows])  # (b, K, D), (K,) or (b, K)
            b, K, D = nxt.shape
            v_next = self.policy.predict_values(nxt.reshape(b * K, D)).reshape(b, K)
            w = weights if weights.dim() == 2 else weights[None, :]
            out[start:start + b] = (v_next * w).sum(dim=1)
        return out.cpu().numpy().astype(np.float32)

    def _pd_returns_and_advantages(self, buf: RolloutBuffer, terminated: np.ndarray) -> None:
        """Write GAE advantages and returns into `buf` using the quadrature bootstrap."""
        n_steps, n_envs = buf.rewards.shape
        obs = buf.observations.reshape(n_steps * n_envs, -1)
        act = buf.actions.reshape(n_steps * n_envs, -1)
        y = self._pd_bootstrap(obs, act).reshape(n_steps, n_envs)

        deltas = buf.rewards + self.gamma * y * (1.0 - terminated) - buf.values

        last_gae = np.zeros(n_envs, dtype=np.float32)
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 0.0  # the last row already bootstraps through y
            else:
                next_non_terminal = 1.0 - buf.episode_starts[step + 1]
            last_gae = deltas[step] + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            buf.advantages[step] = last_gae
        buf.returns = buf.advantages + buf.values
