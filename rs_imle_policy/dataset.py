import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
import pdb
import pandas as pd
import roboticstoolbox as rtb
import time
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import h5py
import pickle as pkl

# Helper function to get normalization stats
def get_data_stats(data):
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

# Normalize data to [-1, 1]
def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])  # Normalize to [0,1]
    ndata = ndata * 2 - 1  # Normalize to [-1,1]
    return ndata

# Unnormalize data
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class PolicyDataset(Dataset):
    def __init__(self, dataset_path: str, pred_horizon: int, obs_horizon: int,
                 action_horizon: int, transform: Optional[Callable] = None, mode='train'):
        self.dataset_path = dataset_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.transform = transform
        self.robot = rtb.models.Panda()

        # Load all episodes and create sample indices
        self.rlds = self.create_rlds_dataset()
        # Store video paths
        self.video_paths = self.get_video_paths()
        # Compute statistics for normalization
        self.stats = {'agent_pos': None, 'action': None}
        self.compute_normalization_stats()

        # save stats
        with open(os.path.join(self.dataset_path, "stats.pkl"), 'wb') as f:
            pkl.dump(self.stats, f)

        # Create sample indices
        self.indices = self.create_sample_indices(self.rlds, sequence_length=pred_horizon)


        # Normalize the data
        self.normalize_rlds()

        if mode == 'train':
            self.cached_dataset = h5py.File('data/t_block_1/t_block_1.h5', 'r')

        if mode == 'test':
            self.cached_dataset = h5py.File('../../data/t_block_1/t_block_1.h5', 'r')


        if self.transform is None:
            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop((216, 288)),
                            transforms.ToTensor()])
            

    def get_video_paths(self):
        video_paths = {}
        episodes = sorted(os.listdir(os.path.join(self.dataset_path, "episodes")), key=lambda x: int(x))
        for episode in episodes:
            video_file_wrist = os.path.join(self.dataset_path, "episodes", episode, "video", "0.mp4")
            video_file_top = os.path.join(self.dataset_path, "episodes", episode, "video", "1.mp4")
            video_file_side = os.path.join(self.dataset_path, "episodes", episode, "video", "2.mp4")

            all_videos = {
                'wrist': video_file_wrist,
                'top': video_file_top,
                'side': video_file_side
            }
            video_paths[int(episode)] = all_videos
        return video_paths
    
    
    def create_rlds_dataset(self):
        rlds = {}
        episodes = sorted(os.listdir(os.path.join(self.dataset_path, "episodes")), key=lambda x: int(x))

        for episode_index, episode in enumerate(episodes):
            episode_path = os.path.join(self.dataset_path, "episodes", episode, "state.json")

            with open(episode_path, "r") as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            df['idx'] = range(len(df))

            X_BE_follower = df['X_BE'].tolist()
            X_BE_leader = [(self.robot.fkine(np.array(q), "panda_link8")).A for q in df['gello_q']]

            # Extract only xy coordinates from the 4x4 matrix
            X_BE_follower_xy = [np.array(x)[:2, 3] for x in X_BE_follower]
            X_BE_leader_xy = [np.array(x)[:2, 3] for x in X_BE_leader]

            rlds[episode_index] = {
                'robot_pos': X_BE_follower_xy,
                'action': X_BE_leader_xy,
                'gello_q': df['gello_q'].tolist(),
                'robot_q': df['robot_q'].tolist()}

        return rlds
    
    def normalize_rlds(self):
        for episode in self.rlds.keys():
            self.rlds[episode]['robot_pos'] = normalize_data(np.array(self.rlds[episode]['robot_pos']), self.stats['agent_pos'])
            self.rlds[episode]['action'] = normalize_data(np.array(self.rlds[episode]['action']), self.stats['action'])
        return

    def create_sample_indices(self, rlds_dataset, sequence_length=16):
        indices = []
        for episode in rlds_dataset.keys():
            if int(episode) > 68:
                break
            episode_length = len(rlds_dataset[episode]['robot_pos'])
            range_idx = episode_length - (sequence_length + 2)
            for idx in range(range_idx):
                buffer_start_idx = idx
                buffer_end_idx = idx + sequence_length
                indices.append([episode, buffer_start_idx, buffer_end_idx])
        indices = np.array(indices)

        return indices

    def compute_normalization_stats(self):
        agent_pos_data = np.concatenate([np.array(self.rlds[episode]['robot_pos']) for episode in self.rlds.keys()], axis=0)
        action_data = np.concatenate([np.array(self.rlds[episode]['action']) for episode in self.rlds.keys()], axis=0)

        self.stats['agent_pos'] = get_data_stats(agent_pos_data)
        self.stats['action'] = get_data_stats(action_data)

    def sample_sequence(self, episode, buffer_start_idx, buffer_end_idx):
        agent_pos = self.rlds[episode]['robot_pos'][buffer_start_idx:buffer_end_idx]
        # action = self.rlds[episode]['action'][buffer_start_idx:buffer_end_idx]
        action = self.rlds[episode]['robot_pos'][buffer_start_idx+1:buffer_end_idx+1]
        frames = self.read_video_frames(episode, buffer_start_idx, buffer_end_idx)

        seq = {
            'agent_pos': np.array(agent_pos),
            'action': np.array(action),
            'frames': frames
        }
        return seq

    
    def read_video_frames(self, episode, start_frame, end_frame):
        frames = {}
        video = self.cached_dataset[str(episode)]
        # only need 2 frames
        wrist_frames = video['wrist'][start_frame:start_frame+self.obs_horizon]
        side_frames = video['side'][start_frame:start_frame+self.obs_horizon]
        # apply transform
        frames['wrist']  = np.array([self.transform(frame) for frame in wrist_frames])
        frames['side']  = np.array([self.transform(frame) for frame in side_frames])

        return frames
    
    def __len__(self):
        return len(self.indices)
    
    def visualize_images_in_row(self, tensor):
        # Ensure the input tensor is in the right shape
        assert tensor.shape == (16, 3, 216, 288), "Tensor should have shape (16, 3, 216, 288)"
        
        # Create a grid of images in a single row
        grid_img = torchvision.utils.make_grid(tensor, nrow=16)  # Arrange 16 images in a single row      
        # Convert the tensor to a numpy array for displaying
        plt.figure(figsize=(20, 5))  # Adjust figure size if necessary
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())  # Permute to get (H, W, C) for display
        plt.axis('off')  # Hide axis
        plt.show()

    def __getitem__(self, idx):
        episode, buffer_start_idx, buffer_end_idx = self.indices[idx]
        seq = self.sample_sequence(episode, buffer_start_idx, buffer_end_idx)

        agent_pos = seq['agent_pos']
        action = seq['action']
        frames = seq['frames']

        # Convert to tensors
        agent_pos = torch.tensor(agent_pos, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        frames_wrist = frames['wrist']
        frames_side = frames['side']

        # self.visualize_images_in_row(frames_wrist)
        
        # discard unused observations
        agent_pos = agent_pos[:self.obs_horizon]

        return {
            'agent_pos': agent_pos,
            'action': action,
            'frames_wrist': frames_wrist,
            'frames_side': frames_side
        }
    


if __name__ == '__main__':
    dataset = PolicyDataset('data/t_block_1', pred_horizon=16, obs_horizon=2, action_horizon=8)
    
    idx=0
    while True:
        start_time = time.time()
        dataset.__getitem__(idx)
        print(f"Time taken: {time.time() - start_time}")
        idx += 1
