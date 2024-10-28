from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import time

from rs_imle_policy.policy import Policy
from rs_imle_policy.dataset import normalize_data, unnormalize_data

import numpy as np
import time
import collections
import reactivex as rx
from reactivex import operators as ops

# from diffrobot.robot.robot import Robot, to_affine, matrix_to_pos_orn
from diffrobot.robot.robot import to_affine, matrix_to_pos_orn
from frankx import Robot, Waypoint, WaypointMotion, JointMotion, Affine, LinearMotion, Kinematics

from rs_imle_policy.realsense.multi_realsense import MultiRealsense

import matplotlib.pyplot as plt

import pdb


class PerceptionSystem:
    def __init__(self):
        self.cams = MultiRealsense(
            serial_numbers=['123622270136', # wrist
                            '036422070913'], # side
            resolution=(640,480),
            enable_depth=False, 
        )
    def start(self):
        self.cams.start()
        self.cams.cameras['123622270136'].set_exposure(exposure=10000.0, gain=16) # d405
        self.cams.cameras['036422070913'].set_exposure(exposure=130.00, gain=13) #d435
   
    def stop(self):
        self.cams.stop()


class RobotInferenceController:
    def __init__(self):
        
        self.robot = self.create_robot()
        self.perception_system = PerceptionSystem()
        self.perception_system.start()
        self.setup_diffusion_policy()
        self.move_to_start()

    def move_to_start(self):
        self.robot.move(JointMotion([-1.9953644495495173, -0.07019201069593659, 0.051291523464672376, -2.4943418327817803, -0.042134962130810624, 2.385776886145273, 0.35092161391247345]))


    def create_robot(self, ip:str = "172.16.0.2", dynamic_rel: float=0.4): #0.4
        # panda = Robot(ip)
        panda = Robot(ip, repeat_on_error=True, dynamic_rel=dynamic_rel)
        panda.recover_from_errors()
        panda.accel_rel = 0.1
        panda.jerk_rel = 0.01
        return panda

    def setup_diffusion_policy(self):
        torch.cuda.empty_cache()
        self.policy = Policy(config_file='vision_config',
                        saved_run_name=None, 
                        mode='infer')

        self.obs_horizon = self.policy.params.obs_horizon
        self.obs_deque = collections.deque(maxlen=self.policy.params.obs_horizon)


    def process_inference_vision(self, obs_deque):
        image_side = np.stack([x['image_side'] for x in obs_deque])
        image_wrist = np.stack([x['image_wrist'] for x in obs_deque])
        agent_pos = np.stack([x['agent_pos'] for x in obs_deque])

        nagent_pos = normalize_data(agent_pos, stats=self.policy.stats['agent_pos'])

        nimage_side = torch.stack([self.policy.transform(img) for img in image_side])
        nimage_wrist = torch.stack([self.policy.transform(img) for img in image_wrist])

        nagent_pos = torch.from_numpy(nagent_pos).to(self.policy.device, dtype=self.policy.precision)
        nimage_side = nimage_side.to(self.policy.device, dtype=self.policy.precision)
        nimage_wrist = nimage_wrist.to(self.policy.device, dtype=self.policy.precision)

        image_features_side = self.policy.ema_nets['vision_encoder_side'](nimage_side)
        image_features_wrist = self.policy.ema_nets['vision_encoder_wrist'](nimage_wrist)

        obs_features = torch.cat([image_features_side, image_features_wrist, nagent_pos], dim=-1)                
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
        
        return obs_cond
    
    
    def get_observation(self):
        # s = self.robot.get_state()
        s = self.motion.get_robot_state()
        # s = self.robot.read_once()
        X_BE = np.array(s.O_T_EE).reshape(4,4).T

        images = self.perception_system.cams.get()
        images_wrist = images[0]['color']
        images_side = images[1]['color']

        return {"agent_pos": X_BE[:2, 3], 
                "image_wrist": images_wrist,
                "image_side": images_side}
    
    
    def infer_action(self, obs_deque):

        with torch.no_grad():

            obs_cond = self.process_inference_vision(obs_deque)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, self.policy.params.pred_horizon, self.policy.params.action_dim), device=self.policy.device, dtype=self.policy.precision)
            naction = noisy_action

            # init scheduler
            self.policy.noise_scheduler.set_timesteps(self.policy.params.num_diffusion_iters)
            # self.noise_scheduler.set_timesteps(20)

            for k in self.policy.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.policy.ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.policy.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
        
        naction = naction.detach().to('cpu').numpy()[0]

        # unnormalize action
        action_pos = unnormalize_data(naction, stats=self.policy.stats['agent_pos'])

        # only take action_horizon number of actions
        start = self.policy.params.obs_horizon - 1
        end = start + self.policy.params.action_horizon
        action = action_pos[start:end]

        return {"action": action}


    def start_inference(self):
        obs_stream = rx.interval(0.1, scheduler=rx.scheduler.NewThreadScheduler()) \
                    .pipe(ops.map(lambda _: self.get_observation())) \
                    .subscribe(lambda x: self.obs_deque.append(x))  
        
        current_pose = self.robot.current_pose()
        height = current_pose.translation()[2]
        q = current_pose.quaternion()
        
        # motion = robot.start_impedance_controller(1000, 40, 1)
        self.motion = WaypointMotion([Waypoint(current_pose)], return_when_finished=False)
        thread = self.robot.move_async(self.motion)
        done = False

        time.sleep(2)

        while not done:
            # wait for obs_deque to have len 2
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(0.1)
                # print("Waiting for observation")

            out = self.infer_action(self.obs_deque.copy())
            action = out['action']

            
            for i in range(4,len(action)):

                print(action[i])

                # keep everyting the same except xy which is the action
                trans = np.array([action[i][0], action[i][1], height])

                self.motion.set_next_waypoint(Waypoint(Affine(trans[0], trans[1], height, q[0], q[1], q[2], q[3])))

                # robot_state = motion.get_robot_state()
                # self.robot_visualiser.policy_pose.T = action[i] 
                # self.robot_visualiser.step(robot_state.q)

                time.sleep(0.1)





if __name__ == '__main__':
    


    controller = RobotInferenceController()
    controller.start_inference()
    controller.perception_system.stop()
