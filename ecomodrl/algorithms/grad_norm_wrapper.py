from torch.nn.utils import clip_grad_norm_

def add_grad_norm_rescaling(
    model, 
    max_grad_norm: float,
    clip_actor: bool = True,
    clip_critic: bool = True):
    """
    Safely adds gradient rescaling to SB3 off-policy algorithms by wrapping 
    the optimizer's .step() method.
    Args:
        model: Stable Baselines 3 RL model (TD3, DDPG, SAC).
        max_grad_norm: Maximum norm for gradient rescaling.
        clip_actor: Whether to rescale actor network gradients.
        clip_critic: Whether to rescale critic network gradients.
        
    Note: This function modifies the model in place.
    """
    
    def _wrap_step(optimizer, params_iterable):
        params_list = list(params_iterable) 
        orig_step = optimizer.step

        def clipped_step(*args, **kwargs):
            clip_grad_norm_(params_list, max_grad_norm)
            return orig_step(*args, **kwargs)
        
        optimizer.step = clipped_step
        optimizer._has_grad_clip_wrapper = True

    if clip_actor and hasattr(getattr(model, "policy", None), "actor") and hasattr(getattr(model.policy, "actor", None), "optimizer"):
        _wrap_step(model.policy.actor.optimizer, model.policy.actor.parameters())

    if clip_critic and hasattr(getattr(model, "policy", None), "critic") and hasattr(getattr(model.policy, "critic", None), "optimizer"):
        _wrap_step(model.policy.critic.optimizer, model.policy.critic.parameters())