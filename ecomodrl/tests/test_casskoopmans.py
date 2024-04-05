import numpy as np
from ecomodrl.envs.casskoopmans import CassKoopmansEnv

import subprocess


def test_ppo():
    subprocess.run(
        "python ecomodrl/algorithms/casskoopmans_ppo.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )
    
def test_ddpg():
    subprocess.run(
        "python ecomodrl/algorithms/casskoopmans_ddpg.py --total-timesteps 256 --learning-starts 128",
        shell=True,
        check=True,
    )
    
def test_td3():
    subprocess.run(
        "python ecomodrl/algorithms/casskoopmans_td3.py --total-timesteps 256 --learning-starts 128",
        shell=True,
        check=True,
    )
    
def test_sac():
    subprocess.run(
        "python ecomodrl/algorithms/casskoopmans_sac.py --total-timesteps 256 --learning-starts 128",
        shell=True,
        check=True,
    )

def test_casskoopmans():
    capital_0 = 10.0
    env = CassKoopmansEnv(max_episode_steps=10, RESETSTATE=capital_0, finite=True)
    obs, _ = env.reset()
    assert obs[0] == capital_0, "Reset function returns wrong observation."
    
    for _ in range(0,10):
        _, _, terminated, _, _ = env.step([0.001])
    assert terminated, "Step function does not terminate"
    
    obs, _ = env.reset()
    obs, reward, _, _, _ = env.step([0.5])
    assert type(obs[0])==np.float32, f"Incorrect type of observation: {type(obs[0])}"
    assert type(reward)==np.float32, f"Incorrect type of reward: {type(reward)}"
    
test_casskoopmans()
test_ppo()
test_ddpg()
test_td3()
test_sac()
