"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

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


def plot_results(json_dict, output_dir):
    plt.figure()
    markers = [
        "o",
        "^",
        "s",
        "p",
        "*",
        "+",
        "x",
    ]  # Different markers for different checkpoints
    for idx, (name, checkpoints) in enumerate(json_dict.items()):
        color = plt.cm.tab10(idx)  # Different colors for different names
        for i, (checkpoint, results) in enumerate(checkpoints.items()):
            sampling_steps = list(results.keys())
            scores = list(results.values())
            plt.plot(
                sampling_steps,
                scores,
                marker=markers[i % len(markers)],
                color=color,
                label=f"{name}_{i+1}",
            )

    plt.xlabel("Number of Sampling Iterations")
    plt.ylabel("Mean Score")
    plt.title("Evaluation Results")
    #plt.xscale("log")  # Use logarithmic scale for x-axis
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "eval_results.pdf"), format="pdf")


@click.command()
@click.option("-o", "--output_dir", default="data/pusht_eval_vs_steps")
@click.option("-d", "--device", default="cuda:0")
def main(output_dir, device):
    # if os.path.exists(output_dir):
    #    click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    name_checkpoint_dict = {
        "diffusion_policy": "data/outputs/2024.01.21_train_diffusion_unet_hybrid_pusht_image/train_2/checkpoints/epoch=0300-test_mean_score=0.903.ckpt",
        "edm": "data/outputs/2024.01.21_train_edm_test_score_unet_hybrid_pusht_image_test_score/train_2/checkpoints/epoch=1700-test_mean_score=0.925.ckpt",
        # "cm": "",
        "ctm": "data/outputs/2024.01.29/17.52.11_train_ctm_unet_hybrid_euler_pusht_image/train_1/checkpoints/epoch=0200-test_mean_score=0.828.ckpt",
        "ddim": "data/outputs/2024.02.01_train_diffusion_unet_hybrid_pusht_image/epoch=0650-test_mean_score=0.911.ckpt"
    }
    num_sampling_iterations_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100]

    # load json with results if it exists
    json_path = os.path.join(output_dir, "eval_vs_steps.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            json_dict = json.load(f)
    else:
        json_dict = dict()

    for name, checkpoint in name_checkpoint_dict.items():
        # json format: {name: {checkpoint: {num_sampling_iterations: mean_score}}}
        print("Running eval for " + name + " with checkpoint " + checkpoint)

        # If name not present in json_dict, add it
        if name not in json_dict:
            json_dict[name] = dict()
        # If checkpoint not present in json_dict[name], add it
        if checkpoint not in json_dict[name]:
            json_dict[name][checkpoint] = dict()

        # Get the list of num_sampling_iterations for which results are not present in json_dict[name][checkpoint]
        existing_iterations = {int(k) for k in json_dict[name][checkpoint].keys()}
        remaining_n = [
            n for n in num_sampling_iterations_list if n not in existing_iterations
        ]
        print(
            "Remaining num_sampling_iterations: "
            + str(remaining_n)
            + " for "
            + name
            + " with checkpoint "
            + checkpoint
        )
        if len(remaining_n) == 0:
            print(
                "All results present in json_dict. Skipping eval for this checkpoint."
            )
            continue

        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
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

        # run eval for the remaining num_sampling_iterations
        for num_sampling_iterations in remaining_n:
            # set num_sampling_iterations
            policy.num_inference_steps = num_sampling_iterations

            # run eval
            video_output_dir = os.path.join(
                output_dir, name, str(num_sampling_iterations)
            )
            pathlib.Path(video_output_dir).mkdir(parents=True, exist_ok=True)
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=video_output_dir,
                n_test_vis=1,
                n_train_vis=0,
            )

            runner_log = env_runner.run(policy)

            # log output dir
            print("Video output dir: " + env_runner.output_dir)

            # dump log to json
            # json_log = dict()
            # for key, value in runner_log.items():
            #    if isinstance(value, wandb.sdk.data_types.video.Video):
            #        json_log[key] = value._path
            #    else:
            #        json_log[key] = value
            print(
                "Mean score for "
                + name
                + " with checkpoint "
                + checkpoint
                + " and num_sampling_iterations "
                + str(num_sampling_iterations)
                + " is "
                + str(runner_log["test/mean_score"])
            )
            json_dict[name][checkpoint][num_sampling_iterations] = runner_log[
                "test/mean_score"
            ]

            # save json
            with open(json_path, "w") as f:
                json.dump(json_dict, f, indent=4)

    # Plotting results for all checkpoints
    plot_results(json_dict, output_dir)


if __name__ == "__main__":
    main()
