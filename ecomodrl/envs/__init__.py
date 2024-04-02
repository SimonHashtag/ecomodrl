from gymnasium.envs.registration import register

from ecomodrl.envs.casskoopmans import CassKoopmansEnv

register(
    id="CassKoopmans-v0",
    entry_point="ecomodrl.envs:CassKoopmansEnv",
)