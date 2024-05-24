import numpy as np
from collections import defaultdict

class Stopper():
    """Early Stopping Mechanism
    
    Stop trials according to following criteria:
        1. if maximum numbers of episodes is reached
        or
        2. if both early stopping criteria (ESC) are met for [early_stopping] number of periods:
            - each episode has more than [min_episodes_length] periods
            - the average reward per episode does not change by more
              than [max deviation]% during the [early_stopping] periods

    Args:
        max_iter (int): Maximum number of iterations before stopping a trial.
        min_iter (int): Minimum number of iterations before stopping a trial.
        early_stopping (int): Number of episodes both early stopping criteria must be fulfilled.
        min_episode_length (int): 1st ESC: Minimum no. of periods per episode
        max_deviation (float): 2nd ESC: Maximum percentage points of deviation during the last [early_stopping] periods
    """
    
    def __init__(
        self,
        max_iter: int = 500,
        min_iter: int = 100,
        early_stopping: int = 15,
        min_episode_length: int = 475,
        max_deviation: float = 0.05,
    ):
        self._iter = 0
        self._max_iter = max_iter
        self._min_iter = min_iter
        self._condition_iter = 0
        self._early_stopping = early_stopping
        self._min_episode_length = min_episode_length
        self._reward_benchmark = np.inf
        self._max_deviation = max_deviation
    
    def __call__(self, episode_results: defaultdict):
        self._iter += 1
        convergence_criterion = (
            abs((np.mean(episode_results["episode_returns"]) - self._reward_benchmark) / self._reward_benchmark)
            <= self._max_deviation
        )
        min_iter_crit = self._iter > self._min_iter - self._early_stopping
        min_eps_len_crit = np.mean(episode_results["episode_lens"]) >= self._min_episode_length
        if convergence_criterion and min_iter_crit and min_eps_len_crit:
            self._condition_iter += 1
        else:
            self._condition_iter = 0
            self._reward_benchmark = np.mean(episode_results["episode_returns"])
        return (self._condition_iter >= self._early_stopping) or (self._iter >= self._max_iter)