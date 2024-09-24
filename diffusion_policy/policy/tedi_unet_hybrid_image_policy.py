if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from typing import Dict
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusion_policy.policy.schedulers import DDPMTEDiScheduler
from diffusion_policy.policy.schedulers import DDIMTEDiScheduler
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d_tedi import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class TEDIUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMTEDiScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=False,
        eval_fixed_crop=False,
        temporally_constant_weight=0.0,
        temporally_increasing_weight=0.0,
        temporally_random_weights=0.0,
        chunk_wise_weight=1.0,
        buffer_init="zero",
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            type = attr.get("type", "low_dim")
            if type == "rgb":
                obs_config["rgb"].append(key)
            elif type == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name="bc_rnn", hdf5_type="image", task_name="square", dataset_type="ph"
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        obs_encoder = policy.nets["policy"].nets["encoder"].nets["obs"]

        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            local_cond_dim=None,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            horizon=horizon,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        if (horizon < n_obs_steps - 1 + n_action_steps) or (horizon % 4 != 0):
            raise ValueError(
                "Horizon must be longer than (To-1) + Ta \n Also, the horizon must be divisible by 4 for the UNet to accept it."
                % (horizon - n_obs_steps, n_action_steps)
            )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # TEDi action buffer
        self.action_buffer = None
        self.buffer_diff_steps = None

        self.temporally_constant_weight = temporally_constant_weight
        self.temporally_increasing_weight = temporally_increasing_weight
        self.temporally_random_weights = temporally_random_weights
        self.chunk_wise_weight = chunk_wise_weight

        # Check that the weights sum to 1
        assert math.isclose(
            self.temporally_constant_weight
            + self.temporally_increasing_weight
            + self.temporally_random_weights
            + self.chunk_wise_weight,
            1,
            rel_tol=1e-2,
        )

        self.buffer_init = buffer_init
        assert self.buffer_init in ["zero", "constant", "denoise"]

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print(
            "Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters())
        )

    # ========= common  ============
    def reset_buffer(self):
        self.action_buffer = None
        self.buffer_diff_steps = None

    def push_buffer(self, new_value, new_sigma_indices):
        """
        Adds the new value to the end of the action buffer like a FIFO queue.
        Also add the corresponding values to buffer_sigma_indices.
        Args:
            new_value: (B, N, Da)
        """
        self.action_buffer = torch.cat([self.action_buffer, new_value], dim=1)
        self.buffer_diff_steps = torch.cat(
            [self.buffer_diff_steps, new_sigma_indices], dim=1
        )

    # ========= sampling ============
    @torch.no_grad()
    def initialize_buffer(
        self,
        shape,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        """
        Args:
            shape: The shape of the action buffer, for obs_as_global_cond, it is (B, T, Da), for inpainting (B, T, Da+Do)
            condition_data: The conditioning data, shape corresponds to shape
            condition_mask: The mask for the conditioning data, shape corresponds to shape
        """
        # if first step, initialize action buffer as pure noise with std. dev.
        # sigma_max, and do T_a number of steps

        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps
        N = self.num_inference_steps
        scheduler = self.noise_scheduler
        if self.action_buffer is None:
            trajectory = torch.randn(
                size=shape, dtype=self.dtype, device=self.device, generator=generator
            )

            # set step values
            scheduler.set_timesteps(self.num_inference_steps)

            # 1. Denoise buffer
            for t in scheduler.timesteps.to(device=self.device, dtype=torch.long):
                # 1. apply conditioning
                trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = self.model(
                    trajectory,
                    t.repeat(B, Tp),
                    local_cond=local_cond,
                    global_cond=global_cond,
                )

                # 3. compute previous image: x_t -> x_t-1
                trajectory = scheduler.step(
                    model_output,
                    t.repeat(B, Tp),
                    trajectory,
                    generator=generator,
                    **kwargs,
                ).prev_sample

            self.action_buffer = trajectory

            # 2. Find the diffusion steps (B, T) for each element in the buffer

            # Calculate chunk indices i
            num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
            incomplete_index = Ta * num_complete_chunks + (To - 1)
            indices = torch.arange(0, Tp, device=self.device)
            chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
            chunk_indices = torch.where(indices < To, 0, chunk_indices)
            chunk_indices = torch.where(
                indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
            )
            chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

            # Calculate corresponding diffusion levels
            diff_steps = (
                torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
            ).long()
            self.buffer_diff_steps = diff_steps

            # 3. Noise action buffer corresponding to the sigma_indices
            # Note: we don't noise the observation steps + 1st chunk
            buffer_to_be_noised = self.action_buffer[:, Ta + (To - 1) :]
            diff_steps = self.buffer_diff_steps[:, Ta + (To - 1) :]
            noise = torch.randn(
                size=buffer_to_be_noised.shape, dtype=self.dtype, device=self.device
            )
            self.action_buffer[:, Ta + (To - 1) :] = scheduler.add_noise(
                buffer_to_be_noised, noise, diff_steps
            )

            # Apply conditioning
            self.action_buffer[condition_mask] = condition_data[condition_mask]

    def initialize_buffer_as_zero(self, shape, generator=None):
        N = self.num_inference_steps
        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps

        action_buffer = torch.zeros(shape, dtype=self.dtype, device=self.device)
        scheduler = self.noise_scheduler

        # Find diffusion steps
        # Calculate chunk indices i
        num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
        incomplete_index = Ta * num_complete_chunks + (To - 1)
        indices = torch.arange(0, Tp, device=self.device)
        chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
        chunk_indices = torch.where(indices < To, 0, chunk_indices)
        chunk_indices = torch.where(
            indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
        )
        chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

        diff_steps = (
            torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
        ).long()
        self.buffer_diff_steps = diff_steps

        # Add noise
        noise = torch.randn(
            size=action_buffer.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        self.action_buffer = scheduler.add_noise(action_buffer, noise, diff_steps)

    def initialize_buffer_as_constant(self, shape, cond_data, generator=None):
        N = self.num_inference_steps
        B, Tp, _ = shape
        Ta = self.n_action_steps
        To = self.n_obs_steps
        Da = self.action_dim

        first_agent_pos = cond_data[:, 0, :Da]

        constant_action = first_agent_pos.unsqueeze(1).repeat(1, Tp, 1)
        action_buffer = torch.zeros(shape, dtype=self.dtype, device=self.device)
        action_buffer[:, :, :Da] = constant_action
        scheduler = self.noise_scheduler

        # Find diffusion steps
        # Calculate chunk indices i
        num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
        incomplete_index = Ta * num_complete_chunks + (To - 1)
        indices = torch.arange(0, Tp, device=self.device)
        chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
        chunk_indices = torch.where(indices < To, 0, chunk_indices)
        chunk_indices = torch.where(
            indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
        )
        chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

        diff_steps = (
            torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1
        ).long()
        self.buffer_diff_steps = diff_steps

        # Add noise
        noise = torch.randn(
            size=action_buffer.shape,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        self.action_buffer = scheduler.add_noise(action_buffer, noise, diff_steps)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        Sample from the diffusion model conditioned on condition_data. Unlike EDM, we just do one
        condition_data: (B, T, C) The conditioning data
        Returns:
            action_pred: (B, T, Da) The predicted action including the observation steps
        """
        Tp = condition_data.shape[1]
        Ta = self.n_action_steps
        To = self.n_obs_steps
        N = self.num_inference_steps

        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(N)

        # We need to denoise the first T_o+T_a steps, i.e. push their sigma to 0
        while torch.max(self.buffer_diff_steps[:, (To - 1) + Ta - 1]) != -1:
            # 1. apply conditioning
            self.action_buffer[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                self.action_buffer,
                self.buffer_diff_steps,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            # 3. compute previous image: x_t -> x_t-1
            self.action_buffer = scheduler.step(
                model_output,
                self.buffer_diff_steps,
                self.action_buffer,
                generator=generator,
                **kwargs,
            ).prev_sample

            # 4. update the diffusion step for the buffer
            self.buffer_diff_steps = torch.clamp(self.buffer_diff_steps - 1, min=-1)

        # Finally, make sure conditioning is enforced
        self.action_buffer[condition_mask] = condition_data[condition_mask]

        # Return whole buffer as prediction, we slice later
        action_pred = self.action_buffer  # (B, T, Da) or (B, T, Da+Do)

        # Remove the first T_a steps from the buffer,
        # as they are not needed for the next iteration
        self.action_buffer = self.action_buffer[:, Ta:]
        self.buffer_diff_steps = self.buffer_diff_steps[:, Ta:]

        # Remove excess steps (not part of complete chunks) that are partly denoised
        if (Tp - (To - 1)) % Ta != 0:
            self.action_buffer = self.action_buffer[:, : -((Tp - (To - 1)) % Ta)]
            self.buffer_diff_steps = self.buffer_diff_steps[
                :, : -((Tp - (To - 1)) % Ta)
            ]

        # Add the actions with sigma_max to noise to the end
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

        # Noise the first T_o-1 steps with same noise as for index T_o-1
        next_chunk_diff_step = self.buffer_diff_steps[:, To - 1]  # (B, 1)
        next_chunk_diff_step = next_chunk_diff_step.unsqueeze(1).repeat(1, To - 1)
        self.buffer_diff_steps[:, : To - 1] = next_chunk_diff_step

        noise = torch.randn(
            size=self.action_buffer[:, : To - 1].shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.noise_scheduler.add_noise(
            self.action_buffer[:, : To - 1], noise, next_chunk_diff_step
        )

        return action_pred

    @torch.no_grad()
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
        result:
            "action": (B, num_action_steps, action_dim) The action applied to the MultiStepWrapper
            "action_pred": (B, horizon, action_dim) The action predicted by the model, including the steps with the observations. horizon = num_action_steps + n_obs_steps
        """
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            shape = (B, T, Da)
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        ## SAMPLING
        # If the buffer is reset, initialize it
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
                    **self.kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported buffer initialization {self.buffer_init}"
                )

        # Produce clean action chunk
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            "action": action,  # (B, Ta, Da)
            "action_pred": action_pred,  # (B, T, Da)
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_diff_steps_training(self, B, Tp, To, Ta):
        """
        Args:
            B: batch size
            T: number of timesteps
            To: number of observation steps
            Ta: number of action steps
        Returns:
            diff_steps: (T,) or (B, T) The indices of the noise injection regime for each timestep
        """
        N = self.noise_scheduler.config.num_train_timesteps

        # Sample the noise injection regime for this batch from a categoriecal distribution
        # over the different noise injection regimes.
        probabilities = torch.tensor(
            [
                self.temporally_constant_weight,
                self.temporally_increasing_weight,
                self.temporally_random_weights,
                self.chunk_wise_weight,
            ]
        )

        # Sample from the categorical distribution based on the defined probabilities
        noise_regime = torch.multinomial(
            probabilities, num_samples=1, replacement=True
        ).item()

        # Select the noise injection regime based on the sampled value
        if noise_regime == 0:  # Constant scheme
            diff_steps = (
                torch.randint(0, N, (B, 1), device=self.device).long().repeat(1, Tp)
            )
        elif noise_regime == 1:  # Linearly increasing scheme
            diff_steps = (
                torch.linspace(0, N - 1, Tp, device=self.device, dtype=self.dtype)
                .long()
                .repeat(B, 1)
            )
        elif noise_regime == 2:  # Random scheme
            diff_steps = timesteps = torch.randint(
                0, N, (B, Tp), device=self.device
            ).long()
        elif noise_regime == 3:  # Chunk-wise scheme

            num_complete_chunks = math.floor((Tp - (To - 1)) / Ta)
            incomplete_index = Ta * num_complete_chunks + (To - 1)

            # Calculate chunk indices i
            indices = torch.arange(0, Tp, device=self.device)
            chunk_indices = torch.div((indices - (To - 1)), Ta, rounding_mode="floor")
            chunk_indices = torch.where(indices < To, 0, chunk_indices)
            chunk_indices = torch.where(
                indices >= incomplete_index, num_complete_chunks - 1, chunk_indices
            )
            chunk_indices = chunk_indices.repeat(B, 1)  # (B,T)

            # 1. Sample j from [0, upper_limit]
            upper_limit = math.floor(N / num_complete_chunks)
            j = torch.randint(
                0, upper_limit, (B, 1), device=self.device
            )  # Don't include N/h, as that corresponds to a clean first chunk.
            j = j.repeat(1, Tp)  # (B,T)

            # 2. Evaluate k(i,j)=round((N-1) * ((i+1)/h)) - j
            diff_steps = (
                torch.floor((N * (chunk_indices + 1)) / num_complete_chunks) - 1 - j
            ).long()

        return diff_steps

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(
                nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
            )
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        B = trajectory.shape[0]
        Tp = trajectory.shape[1]
        To = self.n_obs_steps
        Ta = self.n_action_steps

        # Get noise on shape (B, T, D)
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        diff_steps = self.get_diff_steps_training(B, Tp, To, Ta)

        # Add noise using noise scheduler
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, diff_steps
        )  # (T, B, Da)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the denoised image from t_1
        pred = self.model(
            noisy_trajectory, diff_steps, local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss


if __name__ == "__main__":
    # Test the TEDIUnetHybridImagePolicy class
    # Config taken from "train_diffusion_unet_hybrid_workspace.yaml"
    Tp = 16
    To = 2
    Ta = 1
    Da = 2
    obs_as_global_cond = True

    # Create dataset and normalizer
    dataset = PushTImageDataset(
        zarr_path="data/pusht/pusht_cchi_v7_replay.zarr", horizon=Tp
    )
    shape_meta = {
        "obs": {
            "image": {
                "shape": (3, 96, 96),
                "type": "rgb",
            },  # Assuming *image_shape resolves to (3, 64, 64)
            "agent_pos": {"shape": (2,), "type": "low_dim"},
        },
        "action": {"shape": (2,)},
    }

    # Create a dummy model, noise scheduler and policy
    ddpm_scheduler = DDPMTEDiScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )

    ddim_scheduler = DDIMTEDiScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )

    policy = TEDIUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=ddim_scheduler,
        horizon=Tp,
        n_action_steps=Ta,
        n_obs_steps=To,
        num_inference_steps=16,
        obs_as_global_cond=obs_as_global_cond,
        crop_shape=(76, 76),
        diffusion_step_embed_dim=128,
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
        temporally_constant_weight=0.0,
        temporally_increasing_weight=0.0,
        temporally_random_weights=0.0,
        chunk_wise_weight=1.0,
    )
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to("cuda")

    # Create a dummy batch
    B = 1
    batch = {
        "obs": {
            "image": torch.randn(B, Tp, 3, 96, 96),
            "agent_pos": torch.randn(B, Tp, 2),
        },
        "action": torch.randn(B, Tp, 2),
    }
    # Test compute_loss
    loss = policy.compute_loss(batch)
    print(loss)

    # Test predict_action
    obs_dict = {
        "image": torch.randn(B, Tp, 3, 96, 96),
        "agent_pos": torch.randn(B, Tp, 2),
    }
    result = policy.predict_action(obs_dict)
    print(result["action"].shape)
    print(result["action_pred"].shape)
    # Test predict_action
    result = policy.predict_action(obs_dict)
    print(result["action"].shape)
    print(result["action_pred"].shape)
