import numpy as np
from ecomodrl.envs.casskoopmans import CassKoopmansEnv

def test_casskoopmans():
    capital_0 = 10.0
    env = CassKoopmansEnv(max_episode_steps=10, RESETSTATE=capital_0, finite=True)
    obs, _ = env.reset()
    assert obs[0] == capital_0, "Reset function returns wrong observation."
    
    for _ in range(0,10):
        _, _, terminated, _, _ = env.step([0.001])
    assert terminated, "Step function does not terminate"
    
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step([1.0])
    assert truncated, "Step function does not truncate"
    assert not terminated, "Step function terminates incorrectly"
    assert type(obs[0])==np.float32, f"Incorrect type of observation: {type(obs[0])}"
    assert type(reward)==np.float32, f"Incorrect type of reward: {type(reward)}"
    
test_casskoopmans()
