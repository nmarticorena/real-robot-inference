from rs_imle_policy.visualizer.rtb import RobotViz
import spatialmath as sm
import numpy as np
from rs_imle_policy.dataset import PolicyDataset, unnormalize_data
from rs_imle_policy.configs.train_config import LoaderConfig, VisionConfig
import rs_imle_policy.utilities as utils
import cv2
import torch
import spatialgeometry as sg

import tyro

args = tyro.cli(LoaderConfig)

vision_config = tyro.extras.from_yaml(
    VisionConfig, open(args.dataset_path / "vision_config.yml")
)

dataset = PolicyDataset(
    args.dataset_path,
    vision_config=vision_config,
    visualize=True,
)
rlds = dataset.rlds
env = RobotViz()

robot_pos = sg.Axes(0.05)
env.env.add(robot_pos)

action_pos = sg.Axes(0.05)
env.env.add(action_pos)


for episode in rlds:
    ep_data = rlds[episode]
    ep_data_video = dataset.cached_dataset[str(episode)]

    for idx in range(len(ep_data["robot_pos"])):
        if idx % 2 != 0:
            continue

        gello_q = unnormalize_data(ep_data["gello_q"][idx], dataset.stats["gello_q"])
        robot_q = unnormalize_data(ep_data["robot_q"][idx], dataset.stats["robot_q"])

        X_BE_gello = env.robot.fkine(np.array(gello_q), "panda_link8")

        # create a pose at 0,0,0
        pose = sm.SE3(0, 0, 0)
        orien = unnormalize_data(
            ep_data["action_orien"][idx], dataset.stats["action_orien"]
        )
        R = utils.rotation_6d_to_matrix(torch.from_numpy(orien))
        t = unnormalize_data(ep_data["action_pos"][idx], dataset.stats["action_pos"])
        #
        pose.R = sm.SO3(R.numpy(), check=False).norm()
        pose.t = t
        action_pos.T = pose

        orien = unnormalize_data(
            ep_data["robot_orien"][idx], dataset.stats["robot_orien"]
        )
        R = utils.rotation_6d_to_matrix(torch.from_numpy(orien))
        t = unnormalize_data(ep_data["robot_pos"][idx], dataset.stats["robot_pos"])
        r_pos = sm.SE3()

        r_pos.R = sm.SO3(R.numpy(), check=False).norm()
        r_pos.t = t

        robot_pos.T = r_pos

        env.step(robot_q, gello_q)

        # show image
        frames = []
        for camera in vision_config.cameras:
            frame = ep_data_video[camera.name][idx]
            frames.append(frame)
        merged = np.hstack(frames)

        # show one combined window
        cv2.imshow("viewer", merged)
        cv2.waitKey(1)

        # time.sleep(0.001)
