from diffrobot.robot.visualizer import RobotViz
import spatialmath as sm
import numpy as np
from rs_imle_policy.dataset import PolicyDataset, unnormalize_data
import rs_imle_policy.utilities as utils
import cv2
import torch
import spatialgeometry as sg

dataset = PolicyDataset(
    "/media/nmarticorena/DATA/imitation_learning/pick_and_place_ball_new_camera/",
    pred_horizon=16,
    obs_horizon=2,
    action_horizon=8,
    mode="test",
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

        gello_q = ep_data["gello_q"][idx]
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

        print("norm: ", pose.t)
        # print('actual: ', ep_data['action'][idx])

        # action = ep_data['action'][idx]
        # robot_pos = ep_data['robot_pos'][idx]

        env.step(ep_data["robot_q"][idx], ep_data["gello_q"][idx])

        # show image
        wrist_frames = ep_data_video["wrist"][idx : idx + 2]
        side_frames = ep_data_video["side"][idx : idx + 2]
        merged = np.hstack([wrist_frames[0], side_frames[0]])

        # show one combined window
        cv2.imshow("viewer", merged)
        cv2.waitKey(1)

        # time.sleep(0.001)
