import numpy as np
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

class PPOLogger(SummaryWriter):
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
        if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        self.add_scalar(f"{self.eps_label}/episodic_return", info["episode"]["r"], global_step)
                        self.add_scalar(f"{self.eps_label}/episodic_length", info["episode"]["l"], global_step)
                        self.eps_buffer["episode_returns"].append(info["episode"]["r"])
                        self.eps_buffer["episode_lens"].append(info["episode"]["l"])   
                        for k, v in info.items():
                            if k == "episode" or k.startswith("_"):
                                continue
                            self.add_scalar(f"{self.period_label}/{k}", v, global_step)   
        else:
            for k, v in infos.items():
                if k == "episode" or k.startswith("_"):
                    continue
                self.add_scalar(f"{self.period_label}/{k}", v.mean(), global_step)
    
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
        self.add_scalar("performance/value_loss", v_loss.item(), global_step)
        self.add_scalar("performance/policy_loss", pg_loss.item(), global_step)
        self.add_scalar("performance/entropy", entropy_loss.item(), global_step)
        self.add_scalar("performance/old_approx_kl", old_approx_kl.item(), global_step)
        self.add_scalar("performance/approx_kl", approx_kl.item(), global_step)
        self.add_scalar("performance/clipfrac", np.mean(clipfracs), global_step)
        self.add_scalar("performance/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        self.add_scalar("performance/SPS", int(global_step / (time.time() - start_time)), global_step)
        self.add_scalar("performance/learning_rate", lr, global_step)
    
    def empty_eps_buffer(self):
        self.eps_buffer.clear()
        
    def is_eval(self, eval=True):
        self.eps_label = "eval/eps_stats" if eval else "eps_stats"
        self.period_label = "eval/period_stats" if eval else "period_stats"