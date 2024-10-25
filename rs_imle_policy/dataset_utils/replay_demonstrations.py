from diffrobot.robot.visualizer import RobotViz
import pdb
import time
import spatialmath as sm
import numpy as np
from scipy.spatial.transform import Rotation as R
from rs_imle_policy.dataset import PolicyDataset


rlds = PolicyDataset('../../data/t_block_1', pred_horizon=16, obs_horizon=2, action_horizon=8, mode='test').rlds
env = RobotViz()

for episode in rlds:
    ep_data = rlds[episode]


    for idx in range(len(ep_data['robot_pos'])):

        if idx%2!= 0:
            continue

        gello_q = ep_data['gello_q'][idx]
        X_BE_gello = env.robot.fkine(np.array(gello_q), "panda_link8") 

        # create a pose at 0,0,0
        pose = sm.SE3(0,0,0)
        # set x and y to gello position
        pose.t[0] = X_BE_gello.t[0]
        pose.t[1] = X_BE_gello.t[1]
        # set z to 0.3
        pose.t[2] = 0.0


        print('norm: ', pose.t)

        print(ep_data['action'][idx])

        action = ep_data['action'][idx]
        robot_pos = ep_data['robot_pos'][idx]

        # print(f'Action: {action}')
        # print(f'Robot pos: {robot_pos}')
# 
        env.policy_pose.T = sm.SE3(pose, check=False).norm()
        env.step(ep_data['robot_q'][idx])

        time.sleep(0.1)

  

    
    
    
    

   