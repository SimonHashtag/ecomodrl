# -----------------------------------------------------------------------------
# Common policy interface for EGM and RL.
#
# A Policy exposes two views of its decision rule:
#   - policy.share(state)        -> consumption share s in (0, 1)
#   - policy.consumption(state)  -> consumption c = s * m
#   - policy(state)              -> alias for policy.consumption(state)
#
# `state` is a flat numpy array with last axis equal to model.STATE_DIM:
#   - 1-D models (CK, BS) : state shape (..., 1) with [m]
#   - 2-D models (BM)     : state shape (..., 2) with [m, z]
# Both batched and single-state inputs are supported.
#
# Both producers natively generate a share: EGM stores c on m_grid (s = c/m),
# and RL applies the linear share map (envs.share_from_action) to the
# action head. Exposing `.share` directly avoids the multiply-then-divide
# round-trip when consumers (plotting, diagnostics) only need the share.
# -----------------------------------------------------------------------------

import numpy as np

from envs import obs_from_state, share_from_action


class EGMPolicy:
    """Wraps a solved EGM model so it satisfies the common policy interface."""

    def __init__(self, model):
        self.model = model
        self.has_exo = model.HAS_EXO

    def consumption(self, state):
        state = np.asarray(state)
        if self.has_exo:
            return self.model.consumption(state[..., 0], state[..., 1])
        return self.model.consumption(state[..., 0])

    def share(self, state):
        state = np.asarray(state)
        if self.has_exo:
            return self.model.share(state[..., 0], state[..., 1])
        return self.model.share(state[..., 0])

    def __call__(self, state):
        return self.consumption(state)


class RLPolicy:
    """Wraps a trained SB3 model as a policy.

    The env applies a linear action -> share map (no env-side squash);
    we replicate the same map here so the policy returns the same share /
    consumption as EGMPolicy.

    `vecnormalize` is kept in the signature for backward compatibility with
    callers that still pass one in, but is unused: the env-side observation
    transform (envs.obs_from_state) is fixed and deterministic, so no
    running normalization stats are needed.
    """

    def __init__(self, sb3_model, vecnormalize, has_exo, model=None):
        self.sb3_model = sb3_model
        self.vecnormalize = vecnormalize  # kept for backward compat; unused
        self.has_exo = has_exo
        # `model` is the solved economic model; needed by obs_from_state to
        # look up m_ref.  Falls back to the env stored on the SB3 model.
        if model is None:
            inner = sb3_model.get_env()
            try:
                inner_env = inner.envs[0]  # DummyVecEnv
                while hasattr(inner_env, "env"):
                    inner_env = inner_env.env
                model = inner_env.model
            except Exception:
                model = None
        self.model = model

    def share(self, state):
        state = np.asarray(state, dtype=np.float32)
        single = (state.ndim == 1)
        if single:
            state = state[None, :]

        flat_shape = state.shape[:-1]
        state_flat = state.reshape(-1, state.shape[-1])

        # Mirror the env-side observation transform.
        obs_flat = obs_from_state(state_flat, self.model)

        action, _ = self.sb3_model.predict(obs_flat, deterministic=True)
        action = np.asarray(action).reshape(flat_shape + (-1,))
        s = share_from_action(action[..., 0])

        if single:
            return float(s.ravel()[0])
        return s

    def consumption(self, state):
        state_arr = np.asarray(state, dtype=np.float32)
        s = self.share(state_arr)
        m = state_arr[..., 0]
        c = s * m
        if state_arr.ndim == 1:
            return float(c)
        return c

    def __call__(self, state):
        return self.consumption(state)