from typing import Dict
import torch
from torch._tensor import Tensor
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import dill
import hydra

class TEDiVisualizeBufferPolicy(TEDiUnetLowdimPolicy):
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
            temporally_constant_weight=cfg.policy.temporally_constant_weight,
            temporally_increasing_weight=cfg.policy.temporally_increasing_weight,
            temporally_random_weights=cfg.policy.temporally_random_weights,
            chunk_wise_weight=cfg.policy.chunk_wise_weight,
            buffer_init=cfg.policy.buffer_init 
        )

        self.model = checkpoint_policy.model
        self.normalizer = checkpoint_policy.normalizer
        self.noise_scheduler = checkpoint_policy.noise_scheduler
        
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

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        env=None,
        **kwargs,
    ):
        Tp = condition_data.shape[1]
        Ta = self.n_action_steps
        To = self.n_obs_steps
        N = self.num_inference_steps

        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(N)

        img_frames = []

        while torch.max(self.buffer_diff_steps[:, (To - 1) + Ta - 1]) != -1:
            self.action_buffer[condition_mask] = condition_data[condition_mask]
            model_output = model(
                self.action_buffer,
                self.buffer_diff_steps,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            self.action_buffer = scheduler.step(
                model_output,
                self.buffer_diff_steps,
                self.action_buffer,
                generator=generator,
                **kwargs
            ).prev_sample
            self.buffer_diff_steps = torch.clamp(self.buffer_diff_steps - 1, min=-1)

            # Get the unnormalized actions in the buffer
            
            naction_pred = self.action_buffer[..., :self.action_dim]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)
            env.set_buffer(action_pred.detach().to('cpu').numpy()[0])

            env.set_buffer_diff_steps(self.buffer_diff_steps[0])
            #img_frames.append(env.render(mode='rgb_array'))

        self.action_buffer[condition_mask] = condition_data[condition_mask]
        action_pred = self.action_buffer

        print(f"Diffusion steps after sampling {self.buffer_diff_steps[0, :]}")

        self.action_buffer = self.action_buffer[:, Ta:]
        self.buffer_diff_steps = self.buffer_diff_steps[:, Ta:]

        if (Tp - (To - 1)) % Ta != 0:
            self.action_buffer = self.action_buffer[:, :-((Tp - (To - 1)) % Ta)]
            self.buffer_diff_steps = self.buffer_diff_steps[:, :-((Tp - (To - 1)) % Ta)]

        B = condition_data.shape[0]
        num_new_actions = Ta + ((Tp - (To - 1)) % Ta)
        new_noise = torch.randn(
            size=(B, num_new_actions, self.action_buffer.shape[-1]),
            dtype=self.dtype,
            device=self.device,
        )
        new_sigma_indices = torch.ones(
            size=(B, num_new_actions), dtype=torch.long, device=self.device
        ) * (N - 1)
        self.push_buffer(new_noise, new_sigma_indices)

        next_chunk_diff_step = self.buffer_diff_steps[:, To - 1]
        next_chunk_diff_step = next_chunk_diff_step.unsqueeze(1).repeat(1, To - 1)
        self.buffer_diff_steps[:, :To - 1] = next_chunk_diff_step

        noise = torch.randn(
            size=self.action_buffer[:, :To - 1].shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.noise_scheduler.add_noise(
            self.action_buffer[:, :To - 1], noise, next_chunk_diff_step
        )

        return action_pred, img_frames

    @torch.no_grad()
    def predict_action(
        self, obs_dict: Dict[str, Tensor], env
    ) -> Dict[str, Tensor]:
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        Ta = self.n_action_steps
        N = self.num_inference_steps
        scheduler = self.noise_scheduler

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        if self.action_buffer is None:
            if self.buffer_init == "zero":
                self.initialize_buffer_as_zero(shape, generator=None)
            elif self.buffer_init == "constant":
                self.initialize_buffer_as_constant(shape, cond_data, generator=None)
            elif self.buffer_init == "denoise":
                self.initialize_buffer(
                    shape,
                    cond_data,
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    generator=None,
                    env=env,
                    **self.kwargs,
                )
            else:
                raise ValueError(f"Unsupported buffer initialization {self.buffer_init}")

        nsample, img_frames = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            env=env,
            **self.kwargs,
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            "action": action,
            "action_pred": action_pred,
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result["action_obs_pred"] = action_obs_pred
            result["obs_pred"] = obs_pred

        return result, img_frames
