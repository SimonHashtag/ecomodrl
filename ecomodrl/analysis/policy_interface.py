# -----------------------------------------------------------------------------
# Common policy interface for EGM and RL.
#
# A Policy exposes:
#   - policy.share(state)        -> consumption share s
#   - policy.consumption(state)  -> consumption c
#   - policy(state)              -> alias for policy.consumption(state)
#
# State convention:
#   state = (m, x1, ..., xd)
#
# Both single-state inputs with shape (state_dim,) and batched inputs with shape
# (..., state_dim) are supported.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from ecomodrl.envs.envs import obs_from_state

def is_vecnormalize(obj) -> bool:
    """Return True if `obj` looks like an SB3 VecNormalize wrapper."""
    return (
        obj is not None
        and hasattr(obj, "normalize_obs")
        and hasattr(obj, "obs_rms")
    )


def get_attached_vecnormalize(sb3_model):
    """Return the top-level VecNormalize attached to an SB3 model, if present.

    In this project, build_envs() returns VecNormalize as the top-level vectorized
    environment, so no recursive wrapper search is needed.
    """
    try:
        env = sb3_model.get_env()
    except Exception:
        return None

    return env if is_vecnormalize(env) else None


def as_state_batch(state):
    """Convert a state array to batched form.

    Returns
    -------
    state_batch: Array with shape (N, state_dim) or more generally (..., state_dim).
    single: True if the input was a single state with shape (state_dim,).
    """
    state_arr = np.asarray(state, dtype=np.float64)
    single = state_arr.ndim == 1

    if single:
        state_arr = state_arr[None, :]

    return state_arr, single


class EGMPolicy:
    """Wrap a solved EGM model in the common policy interface."""

    def __init__(self, model):
        self.model = model

    def consumption(self, state):
        return self.model.consumption(state)

    def share(self, state):
        state = np.asarray(state, dtype=np.float64)
        c = self.consumption(state)
        m = state[..., 0]
        return c / m

    def __call__(self, state):
        return self.consumption(state)


class RLPolicy:
    """Wrap a trained SB3 model in the common policy interface.

    Parameters
    ----------
    sb3_model: Trained Stable-Baselines3 model.

    model: Economic model instance.

    vecnormalize:
        VecNormalize object containing observation-normalization statistics.
        If None, this class tries to use `sb3_model.get_env()` if it is a
        top-level VecNormalize.
    """

    def __init__(
        self,
        sb3_model,
        model,
        vecnormalize=None,
    ):
        self.sb3_model = sb3_model
        self.model = model

        if vecnormalize is None:
            vecnormalize = get_attached_vecnormalize(sb3_model)

        self.vecnormalize = vecnormalize

    def share(self, state):
        controls, single = self._controls(state)

        if "share" not in controls:
            raise KeyError(
                "model.controls_from_action(...) must return a 'share' entry "
                "for RLPolicy.share()."
            )

        share = np.asarray(controls["share"], dtype=np.float64)

        if single:
            return float(share.reshape(-1)[0])

        return share

    def consumption(self, state):
        controls, single = self._controls(state)

        if "consumption" in controls:
            c = np.asarray(controls["consumption"], dtype=np.float64)
        else:
            if "share" not in controls:
                raise KeyError(
                    "model.controls_from_action(...) must return either "
                    "'consumption' or 'share'."
                )

            state_arr, _ = as_state_batch(state)
            c = np.asarray(controls["share"], dtype=np.float64) * state_arr[..., 0]

        if single:
            return float(c.reshape(-1)[0])

        return c

    def __call__(self, state):
        return self.consumption(state)

    def _controls(self, state):
        """Predict SB3 action and map it to economic controls."""
        state_arr, single = as_state_batch(state)
        actions = self._predict_actions(state_arr)

        controls = self.model.controls_from_action(
            actions,
            state=state_arr,
        )

        return controls, single

    def _predict_actions(self, state_arr):
        """Predict actions from raw economic states."""
        flat_shape = state_arr.shape[:-1]
        state_flat = state_arr.reshape(-1, state_arr.shape[-1])

        obs = obs_from_state(state_flat, self.model)
        obs = self._normalize_obs(obs)

        actions, _ = self.sb3_model.predict(obs, deterministic=True)
        actions = np.asarray(actions, dtype=np.float64)

        return actions.reshape(flat_shape + (self.model.action_dim,))

    def _normalize_obs(self, obs):
        """Apply VecNormalize observation statistics if available."""
        obs = np.asarray(obs, dtype=np.float64)

        if self.vecnormalize is None:
            return obs

        return np.asarray(
            self.vecnormalize.normalize_obs(obs.copy()),
            dtype=np.float64,
        )