from typing import Dict
import torch
from torch._tensor import Tensor
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import dill
import hydra

class DiffusionVisualizeBufferPolicy(TEDiUnetLowdimPolicy):
    def __init__(self, checkpoint_path: str):
        checkpoint_policy,cfg = self.load_policy(checkpoint_path)
        
        # Initialize parent class
        super().__init__(
            model = cfg.policy.model, 
            noise_scheduler = cfg.policy.noise_scheduler,
            horizon = cfg.policy.horizon, 
            obs_dim = cfg.policy.obs_dim,
            action_dim=cfg.policy.action_dim,
            n_action_steps=cfg.policy.n_action_steps,
            n_obs_steps=cfg.policy.n_obs_steps,
            num_inference_steps=cfg.policy.num_inference_steps,
            obs_as_local_cond=cfg.policy.obs_as_local_cond,
            obs_as_global_cond=cfg.policy.obs_as_global_cond,
            pred_action_steps_only=cfg.policy.pred_action_steps_only,
            oa_step_convention=cfg.policy.oa_step_convention,
        )
        self.normalizer = checkpoint_policy.normalizer
        self.noise_scheduler = checkpoint_policy.noise_scheduler
        self.model = checkpoint_policy.model
    
    def load_policy(self, checkpoint_path: str):
        # Load workspace from checkpoint
        with open(checkpoint_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill)
        
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        return policy, cfg
    
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            env=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        img_frames = []

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            
            # Get the unnormalized actions in the buffer
            
            naction_pred = trajectory[..., :self.action_dim]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)
            env.set_buffer(action_pred.detach().to('cpu').numpy()[0])
            # Set buffer as t repeated T_p times
            env.set_buffer_diff_steps(t.repeat(self.horizon))
            img_frames.append(env.render(mode='rgb_array'))
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory, img_frames


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], env=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs']) #(B, To, Do)
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1) # (B, To*Do)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)    # (B, Ta, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, img_frames = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            env=env,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result, img_frames