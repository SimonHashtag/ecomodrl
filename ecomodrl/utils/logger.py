import numpy as np
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class BaseLogger(SummaryWriter):
    def __init__(
        self,
        log_dir):
        super().__init__(log_dir=log_dir)
        self.eps_buffer = defaultdict(list)
        self.is_eval(False)
    
    def log_env_stats(
        self, 
        global_step, 
        infos,
        ):
        current_walltime = time.time()
        if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        self.add_scalar(f"{self.eps_label}/episodic_return", info["episode"]["r"], global_step, current_walltime)
                        self.add_scalar(f"{self.eps_label}/episodic_length", info["episode"]["l"], global_step, current_walltime)
                        self.add_scalar(f"{self.eps_label}/seed", info["seed"], global_step, current_walltime)
                        if self.eps_label == "eval/eps_stats":
                            self.eps_buffer["episode_returns"].append(info["episode"]["r"])
                            self.eps_buffer["episode_lens"].append(info["episode"]["l"])   
                        for k, v in info.items():
                            if k == "episode" or k.startswith("_"):
                                continue
                            self.add_scalar(f"{self.period_label}/{k}", v, global_step, current_walltime)
        else:
            for k, v in infos.items():
                if k == "episode" or k.startswith("_"):
                    continue
                self.add_scalar(f"{self.period_label}/{k}", v.mean(), global_step, current_walltime)

    def empty_eps_buffer(self):
        self.eps_buffer.clear()
        
    def is_eval(self, eval=True):
        self.eps_label = "eval/eps_stats" if eval else "eps_stats"
        self.period_label = "eval/period_stats" if eval else "period_stats"
        


class PPOLogger(BaseLogger):
    def __init__(
        self,
        log_dir):
        super().__init__(log_dir=log_dir)
    
    def log_training_stats(
        self,
        global_step,
        v_loss,
        pg_loss,
        entropy_loss,
        old_approx_kl,
        approx_kl,
        clipfracs,
        explained_var,
        start_time,
        lr,
        ):
        current_walltime = time.time()
        self.add_scalar("performance/value_loss", v_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/policy_loss", pg_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/entropy", entropy_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/old_approx_kl", old_approx_kl.item(), global_step, current_walltime)
        self.add_scalar("performance/approx_kl", approx_kl.item(), global_step, current_walltime)
        self.add_scalar("performance/clipfrac", np.mean(clipfracs), global_step, current_walltime)
        self.add_scalar("performance/explained_variance", explained_var, global_step, current_walltime)
        print("SPS:", int(global_step / (current_walltime - start_time)))
        self.add_scalar("performance/SPS", int(global_step / (current_walltime - start_time)), global_step, current_walltime)
        self.add_scalar("performance/learning_rate", lr, global_step, current_walltime)


class DDPGLogger(BaseLogger):
    def __init__(
        self,
        log_dir):
        super().__init__(log_dir=log_dir)
    
    def log_training_stats(
        self,
        global_step,
        qf1_a_values,
        qf1_loss,
        actor_loss,
        start_time,
        walltime=None,
        ):
        current_walltime = time.time() if walltime is None else walltime
        self.add_scalar("performance/qf1_values", qf1_a_values.mean().item(), global_step, current_walltime)
        self.add_scalar("performance/qf1_loss", qf1_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/actor_loss", actor_loss.item(), global_step, current_walltime)
        print("SPS:", int(global_step / (current_walltime - start_time)))
        self.add_scalar("performance/SPS", int(global_step / (current_walltime - start_time)), global_step, current_walltime)


class TD3Logger(DDPGLogger):
    def __init__(
        self,
        log_dir):
        super().__init__(log_dir=log_dir)
    
    def log_training_stats(
        self,
        global_step,
        qf1_a_values,
        qf2_a_values,
        qf1_loss,
        qf2_loss,
        qf_loss,
        actor_loss,
        start_time,
        ):
        current_walltime = time.time()
        super().log_training_stats(
            global_step,
            qf1_a_values,
            qf1_loss,
            actor_loss,
            start_time,
            current_walltime,)
        self.add_scalar("performance/qf2_values", qf2_a_values.mean().item(), global_step, current_walltime)
        self.add_scalar("performance/qf2_loss", qf2_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/qf_loss", qf_loss.item() / 2.0, global_step, current_walltime)


class SACLogger(BaseLogger):
    def __init__(
        self,
        log_dir):
        super().__init__(log_dir=log_dir)
    
    def log_training_stats(
        self,
        global_step,
        qf1_a_values,
        qf2_a_values,
        qf1_loss,
        qf2_loss,
        qf_loss,
        actor_loss,
        alpha,
        autotune,
        alpha_loss,
        start_time,
        ):
        current_walltime = time.time()
        self.add_scalar("performance/qf1_values", qf1_a_values.mean().item(), global_step, current_walltime)
        self.add_scalar("performance/qf2_values", qf2_a_values.mean().item(), global_step, current_walltime)
        self.add_scalar("performance/qf1_loss", qf1_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/qf2_loss", qf2_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/qf_loss", qf_loss.item() / 2.0, global_step, current_walltime)
        self.add_scalar("performance/actor_loss", actor_loss.item(), global_step, current_walltime)
        self.add_scalar("performance/alpha", alpha, global_step, current_walltime)
        if autotune:
            self.add_scalar("performance/alpha_loss", alpha_loss.item(), global_step, current_walltime)
        print("SPS:", int(global_step / (current_walltime - start_time)))
        self.add_scalar("performance/SPS", int(global_step / (current_walltime - start_time)), global_step, current_walltime)