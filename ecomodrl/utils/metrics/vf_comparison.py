import torch
import tyro
import gymnasium as gym
import numpy as np

from ecomodrl.algorithms.casskoopmans_td3 import make_env
from ecomodrl.algorithms.casskoopmans_td3 import Actor
from ecomodrl.algorithms.casskoopmans_td3 import QNetwork
from ecomodrl.algorithms.casskoopmans_td3 import Args
from ecomodrl.utils.hyperparam_import import import_tuned_hp
from ecomodrl.algorithms.casskoopmans_benchmark import vfi, problem

def load_model(run_name, device, args):
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.max_episode_steps, 1)])
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    checkpoints = torch.load(model_path)
    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    models = [actor, qf1, qf2]
    for model, checkpoint in zip(models, checkpoints):
        model.load_state_dict(checkpoint)
    return actor, qf1, qf2 
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args = import_tuned_hp(args, args.exp_name)
    run_name = "CassKoopmans-v0__casskoopmans_td3__15__1721743725"
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # vfi solution
    p = problem()
    grid, V_vfi, actions_vfi = vfi(p, conv=0.01, tolerance=0.01, verbose=False)
    
    actor, qf1, qf2 = load_model(run_name, device, args)
    with torch.no_grad():
        eval_space = torch.transpose(torch.Tensor([grid]), 0, 1).to(device)
        actions = actor(eval_space)
        values = qf1(eval_space, actions)
    actions_RL = actions.cpu().numpy()
    V_RL = values.cpu().numpy()
    print(f"Mean squared difference (VF): {np.mean((V_RL-V_vfi)**2)}")
    print(f"Mean absolute difference (VF): {np.mean(np.abs(V_RL-V_vfi))}")
    print(f"Mean squared difference (Policy): {np.mean((actions_RL-actions_vfi)**2)}")
    print(f"Mean absolute difference (Policy): {np.mean(np.abs(actions_RL-actions_vfi))}")
    
    print("Finished")