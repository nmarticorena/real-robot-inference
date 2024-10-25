import os
import tqdm
import pims
import numpy as np
import h5py
import cv2
import pdb
import matplotlib.pyplot as plt
import importlib.util
import os
import sys


def get_config(config_file):
    config_file += '.py'
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one directory up
    config_file = os.path.join(script_dir, config_file)
    module_name = os.path.basename(config_file).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def cache_pims_video(dataset_path):
    # Open an HDF5 file in write mode
    with h5py.File('t_block_1.h5', 'w') as h5f:
        episodes = sorted(os.listdir(os.path.join(dataset_path, "episodes")), key=lambda x: int(x))
        for episode in tqdm.tqdm(episodes):
            video_wrist = os.path.join(dataset_path, "episodes", episode, "video", "0.mp4")
            video_side = os.path.join(dataset_path, "episodes", episode, "video", "2.mp4")

            # Use PIMS to read the video files
            pims_video_wrist = pims.PyAVReaderIndexed(video_wrist)
            pims_video_side = pims.PyAVReaderIndexed(video_side)

            # Convert frames to NumPy arrays and resize
            wrist_frames = np.array([cv2.resize(np.array(frame), (320, 240)) for frame in pims_video_wrist])
            side_frames = np.array([cv2.resize(np.array(frame), (320, 240)) for frame in pims_video_side])

            # Create a group for each episode in HDF5
            grp = h5f.create_group(f"{episode}")

            # Chunking size is critical: Use (chunk_frames, height, width, channels)
            # Assuming typical access of 16 frames at a time.
            chunk_size = (16, 240, 320, 3)

            # Store wrist and side video frames in separate datasets with chunking
            grp.create_dataset('wrist', data=wrist_frames, compression="gzip", chunks=chunk_size)
            grp.create_dataset('side', data=side_frames, compression="gzip", chunks=chunk_size)

    print("Video data saved to video_data.h5")


if __name__ == "__main__":
    cache_pims_video('data/t_block_1')
