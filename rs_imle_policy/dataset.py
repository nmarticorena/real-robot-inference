import os
import pathlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
import pandas as pd
import roboticstoolbox as rtb
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import h5py
import pickle as pkl

import rs_imle_policy.utilities as utils
import spatialmath as sm
from rs_imle_policy.configs.train_config import VisionConfig
from collections import defaultdict


# Helper function to get normalization stats
def get_data_stats(data):
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    return stats


# Normalize data to [-1, 1]
def normalize_data(data, stats):
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])  # Normalize to [0,1]
    ndata = ndata * 2 - 1  # Normalize to [-1,1]
    return ndata


# Unnormalize data
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


class PolicyDataset(Dataset):
    def __init__(
        self,
        dataset_path: pathlib.Path,
        pred_horizon: int = 1,
        obs_horizon: int = 1,
        action_horizon: int = 1,
        transform: Optional[Callable] = None,
        low_dim_obs_keys: tuple[str, ...] = (),
        action_keys: tuple[str, ...] = (),
        vision_config: VisionConfig = VisionConfig(),
        visualize: bool = False,
    ):
        self.dataset_path = dataset_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        self.low_dim_obs_keys = low_dim_obs_keys
        self.action_keys = action_keys

        self.vision_config = vision_config

        self.transform = transform
        self.robot = rtb.models.Panda()

        # tranform to finray tcp
        X_FE = np.array(
            [
                [0.70710678, 0.70710678, 0.0, 0.0],
                [-0.70710678, 0.70710678, 0, 0],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.X_FE = sm.SE3(X_FE, check=False).norm()

        # Load all episodes and create sample indices
        self.rlds = self.create_rlds_dataset()
        # Store video paths
        # self.video_paths = self.get_video_paths()
        # Compute statistics for normalization
        self.stats = defaultdict(lambda: dict)
        self.compute_normalization_stats()

        if not visualize:
            # save stats
            with open(os.path.join(self.dataset_path, "stats.pkl"), "wb") as f:
                pkl.dump(dict(self.stats), f)

        # Create sample indices
        self.indices = self.create_sample_indices(
            self.rlds, sequence_length=pred_horizon
        )
        # Normalize the data
        self.normalize_rlds()

        # if mode == 'train':
        self.cached_dataset = h5py.File(f"{self.dataset_path}/images.h5", "r")

        # if mode == 'test':
        # self.cached_dataset = h5py.File('../../data/t_block_1/t_block_1.h5', 'r')

        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomCrop((216, 288)),
                    transforms.ToTensor(),
                ]
            )

        self.low_dim_obs_shape = self.rlds[0]["state"].shape[1]
        self.img_shape = vision_config.vision_features_dim * len(vision_config.cameras)
        self.obs_shape = self.low_dim_obs_shape + self.img_shape

        self.action_shape = self.rlds[0]["action"].shape[1]

    def create_rlds_dataset(self):
        rlds = {}
        episodes = sorted(
            os.listdir(os.path.join(self.dataset_path, "episodes")),
            key=lambda x: int(x),
        )

        for episode_index, episode in enumerate(episodes):
            episode_path = os.path.join(
                self.dataset_path, "episodes", episode, "state.json"
            )

            with open(episode_path, "r") as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            df["idx"] = range(len(df))

            X_BE_follower = df["X_BE"].tolist()
            X_BE_leader = [(self.robot.fkine(np.array(q))).A for q in df["gello_q"]]
            gripper_width = df["gripper_state"].tolist()
            gripper_width = np.array(gripper_width).reshape(-1, 1)
            gripper_action = df["gripper_action"].tolist()
            gripper_action = np.array(gripper_action).reshape(-1, 1)

            X_BE_follower_pos, X_BE_follower_orien = utils.extract_robot_pos_orien(
                X_BE_follower
            )
            X_BE_leader_pos, X_BE_leader_orien = utils.extract_robot_pos_orien(
                X_BE_leader
            )

            rlds[episode_index] = {
                "robot_pos": X_BE_follower_pos,
                "robot_orien": X_BE_follower_orien,
                "gripper_state": gripper_width,
                "action_orien": X_BE_leader_orien,
                "action_pos": X_BE_leader_pos,
                "action_gripper": gripper_action,
                "X_BE": X_BE_follower,
                "gello_q": df["gello_q"].tolist(),
                "robot_q": df["robot_q"].tolist(),
            }

            if len(self.low_dim_obs_keys) != 0:
                state = np.concatenate(
                    [rlds[episode_index][key] for key in self.low_dim_obs_keys], axis=-1
                )
                action = np.concatenate(
                    [rlds[episode_index][key] for key in self.action_keys], axis=-1
                )

                rlds[episode_index]["state"] = state
                rlds[episode_index]["action"] = action
        return rlds

    def normalize_rlds(self):
        for episode in self.rlds.keys():
            for key in self.rlds[episode].keys():
                self.rlds[episode][key] = normalize_data(
                    np.array(self.rlds[episode][key]), self.stats[key]
                )
            # self.rlds[episode]["robot_pos"] = normalize_data(
            #     np.array(self.rlds[episode]["robot_pos"]), self.stats["robot_pos"]
            # )
            # self.rlds[episode]["robot_orien"] = normalize_data(
            #     np.array(self.rlds[episode]["robot_orien"]), self.stats["robot_orien"]
            # )
            # self.rlds[episode]["gripper_state"] = normalize_data(
            #     np.array(self.rlds[episode]["gripper_state"]),
            #     self.stats["gripper_state"],
            # )
            #
            # self.rlds[episode]["action_pos"] = normalize_data(
            #     np.array(self.rlds[episode]["action_pos"]), self.stats["action_pos"]
            # )
            # self.rlds[episode]["action_orien"] = normalize_data(
            #     np.array(self.rlds[episode]["action_orien"]), self.stats["action_orien"]
            # )
            # self.rlds[episode]["action_gripper"] = normalize_data(
            #     np.array(self.rlds[episode]["action_gripper"]),
            #     self.stats["action_gripper"],
            # )
        return

    def create_sample_indices(self, rlds_dataset, sequence_length=16):
        indices = []
        for episode in rlds_dataset.keys():
            if int(episode) > 34:
                break
            episode_length = len(rlds_dataset[episode]["robot_pos"])
            range_idx = episode_length - (sequence_length + 2)
            for idx in range(range_idx):
                buffer_start_idx = idx
                buffer_end_idx = idx + sequence_length
                indices.append([episode, buffer_start_idx, buffer_end_idx])
        indices = np.array(indices)

        return indices

    def compute_normalization_stats(self):
        def get_data(key: str):
            data = np.concatenate(
                [np.array(self.rlds[episode][key]) for episode in self.rlds.keys()],
                axis=0,
            )
            self.stats[key] = get_data_stats(data)

        for keys in self.rlds[0].keys():
            get_data(keys)

    def sample_sequence(self, episode, buffer_start_idx, buffer_end_idx):
        agent_pos = self.rlds[episode]["robot_pos"][buffer_start_idx:buffer_end_idx]
        agent_orien = self.rlds[episode]["robot_orien"][buffer_start_idx:buffer_end_idx]
        agent_gripper = self.rlds[episode]["gripper_state"][
            buffer_start_idx:buffer_end_idx
        ]

        action_pos = self.rlds[episode]["robot_pos"][
            buffer_start_idx + 1 : buffer_end_idx + 1
        ]
        action_orien = self.rlds[episode]["robot_orien"][
            buffer_start_idx + 1 : buffer_end_idx + 1
        ]
        action_gripper = self.rlds[episode]["action_gripper"][
            buffer_start_idx + 1 : buffer_end_idx + 1
        ]
        frames = self.read_video_frames(episode, buffer_start_idx, buffer_end_idx)

        robot_state = np.concatenate([agent_pos, agent_orien, agent_gripper], axis=-1)
        robot_action = np.concatenate(
            [action_pos, action_orien, action_gripper], axis=-1
        )

        seq = {"state": robot_state, "action": robot_action, "frames": frames}
        return seq

    def read_video_frames(self, episode, start_frame, end_frame):
        frames = {}
        video = self.cached_dataset[str(episode)]
        # only need 2 frames
        for key in video.keys():
            frame = video[key][start_frame : start_frame + self.obs_horizon]
            frames[key] = np.array([self.transform(f) for f in frame])

        return frames

    def __len__(self):
        return len(self.indices)

    def visualize_images_in_row(self, tensor):
        # Ensure the input tensor is in the right shape
        assert tensor.shape == (16, 3, 216, 288), (
            "Tensor should have shape (16, 3, 216, 288)"
        )

        # Create a grid of images in a single row
        grid_img = torchvision.utils.make_grid(
            tensor, nrow=16
        )  # Arrange 16 images in a single row
        # Convert the tensor to a numpy array for displaying
        plt.figure(figsize=(20, 5))  # Adjust figure size if necessary
        plt.imshow(
            grid_img.permute(1, 2, 0).cpu().numpy()
        )  # Permute to get (H, W, C) for display
        plt.axis("off")  # Hide axis
        plt.show()

    def __getitem__(self, idx):
        episode, buffer_start_idx, buffer_end_idx = self.indices[idx]
        seq = self.sample_sequence(episode, buffer_start_idx, buffer_end_idx)

        frames = seq["frames"]

        # Convert to tensors
        state = torch.tensor(seq["state"], dtype=torch.float32)
        action = torch.tensor(seq["action"], dtype=torch.float32)
        frames_wrist = frames["wrist"]
        frames_side = frames["side"]

        # self.visualize_images_in_row(frames_wrist)

        # discard unused observations
        state = state[: self.obs_horizon]

        return {
            "state": state,
            "action": action,
            "frames_wrist": frames_wrist,
            "frames_side": frames_side,
        }


if __name__ == "__main__":
    dataset = PolicyDataset(
        "data/t_block_1", pred_horizon=16, obs_horizon=2, action_horizon=8
    )

    idx = 0
    while True:
        start_time = time.time()
        dataset.__getitem__(idx)
        print(f"Time taken: {time.time() - start_time}")
        idx += 1
