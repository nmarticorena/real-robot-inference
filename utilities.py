import os
import tqdm
import pims
import numpy as np
import h5py
import cv2
import pdb
import matplotlib.pyplot as plt

class VideoCaching:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def cache_pims_video(self, video_data=None):
        if video_data is not None:
            return video_data

        # Open an HDF5 file in write mode
        with h5py.File('t_block_1.h5', 'w') as h5f:
            episodes = sorted(os.listdir(os.path.join(self.dataset_path, "episodes")), key=lambda x: int(x))
            for episode in tqdm.tqdm(episodes):
                video_wrist = os.path.join(self.dataset_path, "episodes", episode, "video", "0.mp4")
                video_side = os.path.join(self.dataset_path, "episodes", episode, "video", "2.mp4")

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

# Usage
video_caching = VideoCaching(dataset_path='data/t_block_1')
video_caching.cache_pims_video()