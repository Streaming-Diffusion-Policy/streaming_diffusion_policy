"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import matplotlib.pyplot as plt
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, output_dir, device):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    num_sampling_iterations_list = [1, 10, 100, 1000]
    mean_score = []
    for num_sampling_iterations in num_sampling_iterations_list:

        # set num_sampling_iterations
        policy.num_inference_steps = num_sampling_iterations
    
        # run eval
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)
        runner_log = env_runner.run(policy)
        
        # dump log to json
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value

        mean_score.append((num_sampling_iterations, json_log['test/mean_score']))
        #out_path = os.path.join(output_dir, 'eval_log.json')
        #json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    # Plotting
    plot_and_save(mean_score, output_dir)

def plot_and_save(mean_score_data, output_dir):
    sampling_steps, scores = zip(*mean_score_data)
    plt.figure()
    plt.plot(sampling_steps, scores, marker='o')
    plt.xlabel('Number of Sampling Iterations')
    plt.ylabel('Mean Score')
    plt.title('Sampling Steps vs Mean Score')
    plt.xscale('log')  # Use logarithmic scale for x-axis
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'sampling_steps_vs_mean_score.svg'), format='svg')  # Save in SVG format


if __name__ == '__main__':
    main()
