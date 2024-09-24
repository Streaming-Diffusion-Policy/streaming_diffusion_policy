if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import torch
import hydra
import dill
import time
from tqdm import tqdm
from gym import spaces
import collections
import numpy as np
import pymunk.pygame_util
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from typing import Dict, Sequence, Union, Optional
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import cv2
from skvideo.io import vwrite

from diffusion_policy.policy.tedi_visualize_buffer import TEDiVisualizeBufferPolicy
from diffusion_policy.policy.diffusion_visualize_buffer import DiffusionVisualizeBufferPolicy
from diffusion_policy.policy.tedi_unet_lowdim_policy import TEDiUnetLowdimPolicy


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl


# Function to add the legend to each frame
def add_legend_to_frame(frame, legend, position=(12, int(768/2))):
    # Resize the legend to fit the frame without distorting aspect ratio
    h, w = frame.shape[:2]
    legend_aspect_ratio = legend.shape[1] / legend.shape[0]
    legend_width = int(w * 0.3)  # e.g., 10% of frame width
    legend_height = int(legend_width / legend_aspect_ratio)

    legend_scaled = cv2.resize(legend, (legend_width, legend_height))

    # Position the legend on the frame
    lx, ly = (position[0], int(h / 2 - legend_height / 2))
    sx, sy = lx + legend_scaled.shape[1], ly + legend_scaled.shape[0]

    if ly + legend_height > h or lx + legend_width > w:
        print("Legend dimensions exceed frame dimensions. Check the scaling and position.")
        return frame

    frame[ly:sy, lx:sx] = legend_scaled
    return frame


class PushTKeypointsEnvVisualizeBuffer(PushTEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None
        self.buffer = None
        self.color_options = {
            "carrot_orange": (55, 152, 240),  # BGR
            "robin_egg_blue": (205, 197, 13),  # BGR
            "sgbus_green": (7, 233, 0),  # BGR
            "slate_blue": (222, 83, 125),  # BGR
            "penn_blue": (69, 16, 10)  # BGR
        }
        self.current_color = "carrot_orange"  # Default color

    def set_path_color(self, color_name):
        if color_name in self.color_options:
            self.path_color = self.color_options[color_name]
        else:
            print(f"Color {color_name} not found. Using default color.")
            self.path_color = self.color_options["slate_blue"]

    def plot_path(self, img, path):
        if not path:
            return img

        path = np.array(path)
        path = (path / 512 * self.render_size).astype(np.int32)

        for i in range(1, len(path)):
            coord_start = tuple(path[i-1])
            coord_end = tuple(path[i])
            marker_size = int(2/96*self.render_size)
            
            overlay = img.copy()
            cv2.line(overlay, coord_start, coord_end, self.path_color, marker_size)
            img = cv2.addWeighted(overlay, 1, img, 0, 0)

        return img

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs
    
    # Function to create the legend with gradient and min/max annotations
    def create_legend(self, diff_steps_min, diff_steps_max):
        # Enable LaTeX text rendering
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'  # Can specify other fonts here
        plt.rcParams['font.serif'] = 'Computer Modern'
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
        
        # Create a color map from green to red
        self.cmap = plt.get_cmap("rainbow")
        #self.cmap = LinearSegmentedColormap.from_list("custom_cmap", ["lime", "yellow", "red"], N=256)
        
        # Create a gradient bar for the legend
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack(np.repeat(gradient, 256))

        # Setting the DPI higher to improve resolution
        dpi = 300
        fig_width, fig_height = 4, 7  # dimensions in inches for better control
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax.imshow(gradient, aspect='auto', cmap=self.cmap, origin='lower')

        # Set title and labels
        #ax.set_title('Diffusion Step', fontsize=52, pad=20)
        ax.set_ylabel('Diffusion Step $k$', fontsize=52)
        ax.set_yticks([0, gradient.shape[0] - 1])
        ax.set_yticklabels([int(diff_steps_min), int(diff_steps_max)], fontsize=32)
        ax.set_xticks([])

        # Remove extra margins
        plt.subplots_adjust(left=0.3, right=0.7, top=0.85, bottom=0.15)

        # Convert Matplotlib figure to an image in memory (without saving to disk)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Save the figure to disk
        fig.savefig('legend.pdf', dpi=dpi)

        plt.close(fig)

        return image_from_plot


    def _get_obs(self):
        # get keypoints
        obj_map = {
            'block': self.block
        }
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map
        
        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([
                obs, agent_pos
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool)
            ])

        # obs, obs_mask
        #obs = np.concatenate([
        #    obs, obs_mask.astype(obs.dtype)
        #], axis=0)
        return obs
    
    def set_buffer(self, buffer):
        self.buffer = buffer
    
    def set_buffer_diff_steps(self, buffer_diff_steps, diff_steps_max=99, diff_steps_min=-1):
        self.buffer_diff_steps = buffer_diff_steps
        self.diff_steps_max = diff_steps_max
        self.diff_steps_min = diff_steps_min

    def set_buffer_color(self, color_name):
        if color_name in self.color_options:
            self.current_color = color_name
        else:
            print(f"Color {color_name} not found. Using default color.")

    def draw_buffer(self, img, buffer):
        if buffer is None:
            return img
        
        n = buffer.shape[0]
        color = self.color_options[self.current_color]
        
        for i in range(n):
            coord = buffer[i]
            coord = (coord / 512 * self.render_size).astype(np.int32)
            marker_size = int(2/96*self.render_size)
            
            if img.shape[2] < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            overlay = img.copy()
            cv2.circle(overlay, coord, marker_size, color, -1)
            img = cv2.addWeighted(overlay, 1, img, 0, 0)
            if img is None: 
                print("img is None after cv2")

        return img
    
    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        img = self.draw_buffer(img, self.buffer)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img


if __name__ == "__main__":
    
    # 1. Load policy
    #checkpoint = "/home/sigmund/Code/Spring24/diffusion_policy/data/outputs/2024.03.23/16.46.34_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=0500-test_mean_score=0.925.ckpt"
    checkpoint = "/home/sigmund/Code/Spring24/diffusion_policy/data/outputs/2024.09.11/10.18.24_train_tedi_ddim_unet_lowdim_pusht_lowdim/checkpoints/epoch=0200-test_mean_score=0.809.ckpt"
    
    vis_policy = TEDiVisualizeBufferPolicy(checkpoint)
    #vis_policy = DiffusionVisualizeBufferPolicy(checkpoint)
    device = torch.device("cuda:0")
    vis_policy.to(device)
    vis_policy.eval()
    obs_horizon = vis_policy.n_obs_steps

    # limit enviornment interaction to 200 steps before termination
    max_steps = 50
    env = PushTKeypointsEnvVisualizeBuffer(render_size=768)
    env.set_buffer_color("robin_egg_blue")
    env.set_path_color("slate_blue")  # Default color for the path
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(100000)

    # get first observation
    obs = env.reset()

    # Create the legend (you might need to adjust these based on the expected range)
    #legend_image = env.create_legend(0, 99)  # Update min and max values based on your application

    #legend_image = cv2.imread('legend.png', cv2.IMREAD_COLOR)

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = []
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            marker_path = [env.agent.position]
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_dict = {
                "obs": torch.from_numpy(np.stack(obs_deque, axis=0)).to(device, dtype=torch.float32).unsqueeze(0),
            }
            action, img_frames = vis_policy.predict_action(obs_dict, env)
            imgs.extend(img_frames)
            buffer = action['action_pred'].detach().to('cpu').numpy()[0]
            env.set_buffer(buffer)
            if type(vis_policy) == TEDiUnetLowdimPolicy:
                modified_diff_steps = vis_policy.buffer_diff_steps[0] -  vis_policy.buffer_diff_steps[0, 0]
                env.set_buffer_diff_steps(modified_diff_steps, diff_steps_max=vis_policy.num_inference_steps-1)
            else:
                env.set_buffer_diff_steps(torch.ones(buffer.shape[0])*(-1)) # (Ta,)

            action = action['action'].detach().to('cpu').numpy()[0]
            #Sleep a tiny bit so that we can see the prediciton
            #time.sleep(0.1)

            # Before moving, plot the current plan
            # Remove first (obs) action from buffer
            env.buffer = env.buffer[1:]
            imgs.append(env.render(mode='rgb_array'))

            

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                
                ## Render
                # Remove the leftmost action from the env buffer
                marker_path.append(env.agent.position)
                env.buffer = env.buffer[1:]
                frame = env.render(mode='rgb_array')
                frame = env.plot_path(frame, marker_path)
                imgs.append(frame)

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break
            
            # Plot the path
            img = env.render(mode='rgb_array')
            img = env.plot_path(img, marker_path)
            imgs.append(img)
            print(f"Len of marker path: {len(marker_path)}")

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    from IPython.display import Video
    video_path = 'vis_buffer_tedi.mp4'
    vwrite(video_path, imgs)
    print('Done saving to ', video_path)

    # Save the 2nd frame as an image
    img_path = 'vis_buffer_tedi.png'
    cv2.imwrite(img_path, imgs[7])
    print('Done saving to ', img_path)