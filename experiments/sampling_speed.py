if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


import torch
import timeit
import matplotlib.pyplot as plt

from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.model.diffusion.conditional_unet1d_tedi import ConditionalUnet1D as ConditionalUnet1DTEDi
from diffusion_policy.policy.schedulers import DDPMTEDiScheduler
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

# Common parameters
T, To, Ta, Do, Da = 16, 2, 5, 20, 2
B = 64
obs_as_local_cond = False
obs_as_global_cond = False
pred_action_steps_only = False
num_predicitons = 10
num_inference_steps = 100


# Setup TEDi Policy

# Create dataset and normalizer
dataset = PushTLowdimDataset(
    zarr_path="data/pusht/pusht_cchi_v7_replay.zarr", horizon=T
)

# Create a dummy model, noise scheduler and policy
model = ConditionalUnet1DTEDi(
    input_dim=Da if (obs_as_local_cond or obs_as_global_cond) else Do + Da,
    local_cond_dim=Do if obs_as_local_cond else None,
    global_cond_dim=Do*To if obs_as_global_cond else None,
    diffusion_step_embed_dim=256,
    down_dims=(256, 512, 1024),
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True,
    horizon=T,
)
tedi_noise_scheduler = DDPMTEDiScheduler(
    num_train_timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2",
    variance_type="fixed_small",
    clip_sample=True,
    prediction_type="epsilon",
)
tedi_policy = TEDiUnetLowdimPolicy(
    model=model,
    noise_scheduler=tedi_noise_scheduler,
    horizon=T,
    obs_dim=Do,
    action_dim=Da,
    n_action_steps=Ta,
    n_obs_steps=To,
    num_inference_steps=num_inference_steps,
    obs_as_local_cond=obs_as_local_cond,
    obs_as_global_cond=obs_as_global_cond,
    pred_action_steps_only=pred_action_steps_only,
    oa_step_convention=True,
    temporally_constant_weight=0.0,
    temporally_increasing_weight=0.0,
    temporally_random_weights=0.0,
    chunk_wise_weight=1.0,
    buffer_init="denoise"
)
normalizer = dataset.get_normalizer()
tedi_policy.set_normalizer(normalizer)


# Setup Diffusion Policy
dp_model = ConditionalUnet1D(
    input_dim=Da if (obs_as_local_cond or obs_as_global_cond) else Do + Da,
    local_cond_dim=Do if obs_as_local_cond else None,
    global_cond_dim=Do*To if obs_as_global_cond else None,
    diffusion_step_embed_dim=256,
    down_dims=(256, 512, 1024),
    kernel_size=5,
    n_groups=8,
    cond_predict_scale=True
)

dp_noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2",
    variance_type="fixed_small",
    clip_sample=True,
    prediction_type="epsilon",
)

diffusion_policy = DiffusionUnetLowdimPolicy(
    model=dp_model,
    noise_scheduler=dp_noise_scheduler,
    horizon=T,
    obs_dim=Do,
    action_dim=Da,
    n_action_steps=Ta,
    n_obs_steps=To,
    num_inference_steps=num_inference_steps,
    obs_as_local_cond=obs_as_local_cond,
    obs_as_global_cond=obs_as_global_cond,
    pred_action_steps_only=pred_action_steps_only,
    oa_step_convention=True,
)
diffusion_policy.set_normalizer(normalizer)

#buffer_lengths = [i for i in range(6, 49) if (i%4 == 0)]
buffer_lengths = [20]

dp_times = {}
tedi_times = {}

for buffer_length in buffer_lengths:
    # Re-initialize the tedi-network with the new buffer lenght
    tedi_model = ConditionalUnet1DTEDi(
        input_dim=Da if (obs_as_local_cond or obs_as_global_cond) else Do + Da,
        local_cond_dim=Do if obs_as_local_cond else None,
        global_cond_dim=Do*To if obs_as_global_cond else None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        horizon=buffer_length,
    )
    # Add this network to the policy and update the horizon
    tedi_policy.model = tedi_model

    # Update the horizon for both policies
    tedi_policy.horizon = buffer_length
    diffusion_policy.horizon = buffer_length

    # Move the policies to the GPU
    tedi_policy.to("cuda")
    diffusion_policy.to("cuda")
    tedi_policy.eval()
    diffusion_policy.eval()

    # Make dummy input
    obs_dict = {
        "obs": torch.randn((B, To, Do)),
        "obs_mask": torch.randn((B, To, Do)),
    }

    # Time TEDi Policy prediction
    lambda_forward = lambda: tedi_policy.predict_action(obs_dict)
    # Initialize the buffer
    #tedi_policy.predict_action(obs_dict)
    duration_tedi = timeit.timeit(lambda_forward, number=num_predicitons)/num_predicitons
    print(f"Time taken for TEDi Policy with buffer length={buffer_length}: {duration_tedi}")
    tedi_policy.reset_buffer()
    tedi_times[buffer_length] = duration_tedi

    # Time Diffusion Policy prediction
    print(f"GBs allocated before Diffusion Policy: {torch.cuda.memory_allocated()/1e9}")
    diffusion_policy.predict_action(obs_dict)
    lambda_forward = lambda: diffusion_policy.predict_action(obs_dict)
    duration_dp = timeit.timeit(lambda_forward, number=num_predicitons)/num_predicitons
    print(f"Time taken for Diffusion Policy with buffer length={buffer_length}: {duration_dp}")
    dp_times[buffer_length] = duration_dp

# Plot the results

# Sorting the dictionaries by key to ensure the plot is ordered
keys = sorted(dp_times)
dp_values = [dp_times[key] for key in keys]
tedi_values = [tedi_times[key] for key in keys]

plt.figure(figsize=(10, 6))
plt.plot(keys, dp_values, marker='o', linestyle='-', color='blue', label='Diffusion Policy', linewidth=2, markersize=8)
plt.plot(keys, tedi_values, marker='s', linestyle='--', color='green', label='TEDi Policy', linewidth=2, markersize=8)

plt.title('Comparison of Prediction Time', fontsize=16)
plt.xlabel('Buffer length $T_p$', fontsize=14)
plt.ylabel('Prediction time [s]', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(keys, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

# Save the plot
plt.savefig('experiments/sampling_speed_new.png')

# Save the results to file
with open('experiments/sampling_speed_new.txt', 'w') as f:
    f.write("Buffer Length\tDiffusion Policy Time\tTEDi Policy Time\n")
    for key in keys:
        f.write(f"{key}\t{dp_times[key]}\t{tedi_times[key]}\n")

# Check model size
print("Number of parameter of TEDi model: ", sum(p.numel() for p in tedi_policy.model.parameters()))
print("Number of parameter of Diffusion model: ", sum(p.numel() for p in diffusion_policy.model.parameters()))


# Test time for TEDi NFE
x = torch.randn(B, 8, Da + Do).to("cuda")
t = torch.randint(0, 8, (B,8)).to("cuda")
model_lambda = lambda: tedi_policy.model(x, t)
num = 100
duration_nfe = timeit.timeit(model_lambda, number=num)
print(f"Duration per NFE for TEDi: {duration_nfe}")

# Time for DP NFE
x = torch.randn(B, 8, Da + Do).to("cuda")
t = torch.randint(0, 8, (B,)).to("cuda")

model_lambda = lambda: diffusion_policy.model(x, t)
num = 100
duration_nfe = timeit.timeit(model_lambda, number=num)
print(f"Duration per NFE for Diffusion Policy: {duration_nfe}")




