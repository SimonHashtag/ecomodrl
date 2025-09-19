import torch
import tyro
import gymnasium as gym
import numpy as np
from scipy.interpolate import interp1d

from ecomodrl.experiments.casskoopmans_td3 import make_env
from ecomodrl.experiments.casskoopmans_td3 import Actor as td3_actor
from ecomodrl.experiments.casskoopmans_td3 import QNetwork as td3_critic
from ecomodrl.experiments.casskoopmans_td3 import Args
from ecomodrl.utils.hyperparam_import import import_tuned_hp
from ecomodrl.benchmarks.casskoopmans_benchmark import vfi, problem

def load_model(envs, run_name, device, args):
    """Load the actor and critic(s) from the checkpoint"""
    exp, algo = args.exp_name.split("_")
    model_path = f"runs/{run_name}/{args.exp_name}.ecomodrl_model"
    checkpoints = torch.load(model_path)
    if exp == "casskoopmans":
        if algo == "td3":
            actor = td3_actor(envs).to(device)
            qf1 = td3_critic(envs).to(device)
            qf2 = td3_critic(envs).to(device)
            models = [actor, qf1, qf2]
            for model, checkpoint in zip(models, checkpoints):
                model.load_state_dict(checkpoint)
            return actor, qf1
        else:
            NotImplementedError
    else:
        NotImplementedError

def comparison_on_grid(grid, actor, q_fun, V_vfi, actions_vfi, device):
    """Calculate MSE and MAE between value functions and policies determined by 
    Value Function Iteration (VFI) and RL on grid points used for VFI"""
    with torch.no_grad():
        eval_space = torch.transpose(torch.Tensor([grid]), 0, 1).to(device)
        actions = actor(eval_space)
        values = q_fun(eval_space, actions)
    actions_RL = actions.cpu().numpy()
    V_RL = values.cpu().numpy()
    print(f"Mean squared difference (VF): {np.mean((V_RL-V_vfi)**2)}")
    print(f"Mean absolute difference (VF): {np.mean(np.abs(V_RL-V_vfi))}")
    print(f"Mean squared difference (Policy): {np.mean((actions_RL-actions_vfi)**2)}")
    print(f"Mean absolute difference (Policy): {np.mean(np.abs(actions_RL-actions_vfi))}")
    
def comparison_on_trajectories(envs, actor, q_fun, eval_steps, grid, V_vfi, actions_vfi, device):
    """Calculate MSE and MAE between value functions and policies determined by 
    Value Function Iteration (VFI) and RL on observations sampled by RL actor"""
    V = interp1d(grid, V_vfi, fill_value="extrapolate")
    policy_vfi = interp1d(grid, actions_vfi, fill_value="extrapolate")
    obs_list = []
    obs, _ = envs.reset(seed=1)
    for _ in range(1, eval_steps + 1):
        obs_list.append(obs[0][0])
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
        next_obs, _, _, _, _ = envs.step(actions.cpu().numpy())
        obs = next_obs
    comparison_on_grid(obs_list, actor, q_fun, V(obs_list), policy_vfi(obs_list), device)

def main(run_name):
    args = tyro.cli(Args)
    args = import_tuned_hp(args, args.exp_name)
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.max_episode_steps, 1)])
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # vfi solution
    exp = args.exp_name.split("_")[0]
    if exp=="casskoopmans":
        p = problem()
        grid, V_vfi, actions_vfi = vfi(p, conv=0.01, tolerance=0.01, verbose=False)
    else:
        NotImplementedError
    
    # RL solution
    actor, qf = load_model(envs, run_name, device, args)
    
    # VF and policy comparisons
    print("Comparison on vfi grid: \n")
    comparison_on_grid(grid, actor, qf, V_vfi, actions_vfi, device)
    
    print("Comparison on trajectories sampled by RL agent: \n")
    comparison_on_trajectories(envs, actor, qf, 10000, grid, V_vfi, actions_vfi, device)

if __name__ == "__main__":
    run_name = "CassKoopmans-v0__casskoopmans_td3__15__1721743725"
    main(run_name)