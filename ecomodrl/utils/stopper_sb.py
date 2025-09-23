from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
import numpy as np
from typing import Optional

def unwrap_vecnormalize(env):
    """Return the inner VecNormalize wrapper if present, else None."""
    e = env
    while isinstance(e, VecEnvWrapper):
        if isinstance(e, VecNormalize):
            return e
        e = e.venv
    return None

def sync_vecnormalize(dst: VecNormalize, src: VecNormalize):
    """keep eval stats in lockstep with training stats"""
    dst.obs_rms = src.obs_rms
    dst.ret_rms = src.ret_rms

def evaluate_policy_discounted(model, eval_env, beta, n_episodes=1, deterministic=True):
    """fixed start + deterministic => 1 episode is fine."""
    eval_env.training = False
    avg_return = 0
    for _ in range(n_episodes):
        obs = eval_env.reset()
        dones = np.zeros(eval_env.num_envs, dtype=bool)#np.array([False])
        G, power = 0.0, 1.0
        while not dones.all():
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, done_step, _ = eval_env.step(action)
            G += float(rewards.mean()) * power
            power *= beta
            dones |= done_step
        avg_return += G/n_episodes
    return avg_return

class EarlyStoppingEvalCallback(BaseCallback):
    """
    Syncs VecNormalize, runs discounted evaluation, saves best, and early-stops,
    logs metrics, and reloads the best model at the end—matching the manual loop.
    """
    def __init__(
        self,
        eval_env,
        beta: float,
        eval_freq: int,
        patience: int,
        min_delta: float,
        n_eval_episodes: int = 1,
        deterministic_eval: bool = True,
        best_model_path: str = "best_model.zip",
        vecnorm_path: Optional[str] = "vecnormalize.pkl",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.beta = beta
        self.eval_freq = int(eval_freq)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic_eval = bool(deterministic_eval)
        self.best_model_path = best_model_path
        self.vecnorm_path = vecnorm_path

        self.best_score = -np.inf
        self.patience_counter = 0
        self._last_eval_step = 0
        self.return_history = []
    

    def _on_training_start(self) -> None:
        self._last_eval_step = 0
        if self.verbose:
            print("────────────────────────────────────────────")
            print("     PPO TRAINING LOOP WITH EARLY STOPPING")
            print("────────────────────────────────────────────", flush=True)
        
    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps

        # 1) sync VecNormalize stats if available
        train_norm = unwrap_vecnormalize(self.model.get_env())
        eval_norm  = unwrap_vecnormalize(self.eval_env)
        if train_norm and eval_norm:
            sync_vecnormalize(eval_norm, train_norm)
            eval_norm.training = False
            eval_norm.norm_reward = False
        else:
            # if not using VecNormalize, make sure eval env is in eval mode
            if hasattr(self.eval_env, "training"):
                self.eval_env.training = False

        # 2) evaluate
        avg_return = evaluate_policy_discounted(
            self.model, self.eval_env, beta=self.beta,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic_eval
        )
        self.return_history.append((self.num_timesteps, float(avg_return)))
        
        if self.verbose:
            print(
                f"[{self.num_timesteps:7d}] avg true return: {avg_return: .6f} | "
                f"best: {self.best_score: .6f} | "
                f"PATIENCE: {self.patience_counter}/{self.patience}",
                flush=True,
            )

        # 3) early stopping logic + saving
        if avg_return > self.best_score + self.min_delta:
            self.best_score = float(avg_return)
            self.patience_counter = 0
            self.model.save(self.best_model_path)
            # save vecnorm if available
            if self.vecnorm_path and train_norm is not None and hasattr(train_norm, "save"):
                train_norm.save(self.vecnorm_path)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                if self.verbose:
                    print("Early stopping.")
                return False  # signal SB3 to stop training
            
        # 4) log results
        self.logger.record("eval/avg_discounted_return", float(avg_return))
        self.logger.record("eval/best_discounted_return", float(self.best_score))
        self.logger.dump(self.num_timesteps)
        return True
    
    def _on_training_end(self) -> None:
        # Reload the best policy so the model after learn() matches your loop
        try:
            if self.best_model_path:
                if self.verbose:
                    print(f"Loading best model from {self.best_model_path}", flush=True)
                self.model.set_parameters(self.best_model_path)
        except Exception as e:
            if self.verbose:
                print(f"Warning: could not reload best model: {e}", flush=True)
