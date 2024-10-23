from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os
import time

from policy import Policy
from dataset import normalize_data, unnormalize_data

import numpy as np
import time
import json
import collections
import reactivex as rx
from reactivex import operators as ops

from diffrobot.robot.robot import Robot, to_affine, matrix_to_pos_orn





class PerceptionSystem:
    def __init__(self):
        self.cams = MultiRealsense(
            serial_numbers=['128422271784', '123622270136'],
            resolution=(640,480),
        )
    def start(self):
        self.cams.start()
        self.cams.set_exposure(exposure=5000, gain=60)
   
    def stop(self):
        self.cams.stop()


class RobotInferenceController:
    def __init__(self):
        
        self.robot = self.create_robot()
        self.perception_system = PerceptionSystem()
        self.perception_system.start()
        self.setup_diffusion_policy()

    def create_robot(self, ip:str = "172.16.0.2", dynamic_rel: float=0.4):
        panda = Robot(ip)
        panda.gripper.open()
        panda.set_dynamic_rel(dynamic_rel, accel_rel=0.2, jerk_rel=0.05)
        panda.frankx.set_collision_behavior(
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            [30.0, 30.0, 30.0, 30.0, 30.0, 30.0])
        return panda

    def setup_diffusion_policy(self):
        torch.cuda.empty_cache()
        self.policy = Policy(config_file='vision_config',
                        saved_run_name='data/t_block_1', 
                        mode='infer')

        self.obs_horizon = self.policy.params.obs_horizon
        self.obs_deque = collections.deque(maxlen=self.policy.params.obs_horizon)


    def process_inference_vision(self, obs_deque):
        image_side = np.stack([x['image_side'] for x in obs_deque])
        image_wrist = np.stack([x['image_wrist'] for x in obs_deque])
        agent_pos = np.stack([x['agent_pos'] for x in obs_deque])

        nagent_pos = self.dutils.normalize_data(agent_pos, stats=self.stats['states'])
        nimage_side = self.policy.transform(image_side)
        nimage_wrist = self.policy.transform(image_wrist)

        nagent_pos = torch.from_numpy(nagent_pos).to(self.device, dtype=self.precision)
        nimage_side = torch.from_numpy(nimage_side).to(self.device, dtype=self.precision)
        nimage_wrist = torch.from_numpy(nimage_wrist).to(self.device, dtype=self.precision)

        nimage_side = torch.stack([self.transform(img) for img in nimage_side])
        nimage_wrist = torch.stack([self.transform(img) for img in nimage_wrist])

        image_features_side = self.policy.ema_nets['vision_encoder_side'](nimage_side)
        image_features_wrist = self.policy.ema_nets['vision_encoder_wrist'](nimage_wrist)

        obs_features = torch.cat([image_features_side, image_features_wrist, nagent_pos], dim=-1)                
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
        
        return obs_cond
    
    def get_observation(self):
        s = self.robot.get_state()
        X_BE = np.array(s.O_T_EE).reshape(4,4).T
        self.detect_objects()

        # get frames
        image_side = self.perception_system.cams.get_frame(0)
        image_wrist = self.perception_system.cams.get_frame(1)

        X_OO_O = np.dot(np.linalg.inv(X_B_OO), X_BO) 
        # self.robot_visualiser.object_pose.T = self.X_BO
        return {"X_BE": X_BE, 
                "phase": task.phase}
    
    
    def infer_action(self, obs_deque):

        with torch.no_grad():

            obs_cond = self.process_inference_vision(obs_deque)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, self.params.pred_horizon, self.params.action_dim), device=self.device, dtype=self.precision)
            naction = noisy_action

            # init scheduler
            self.policy.noise_scheduler.set_timesteps(self.params.num_diffusion_iters)
            # self.noise_scheduler.set_timesteps(20)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.policy.ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
        
        naction = naction.detach().to('cpu').numpy()[0]

        # unnormalize action
        action_pos = self.dutils.unnormalize_data(naction, stats=self.stats['action'])

        # only take action_horizon number of actions
        start = self.params.obs_horizon - 1
        end = start + self.params.action_horizon
        action = action_pos[start:end]


    def start_inference(self):
        robot = self.robot
        obs_stream = rx.interval(0.1, scheduler=rx.scheduler.NewThreadScheduler()) \
                    .pipe(ops.map(lambda _: self.task.get_observation())) \
                    .pipe(ops.filter(lambda x: x["X_BO"] is not None)) \
                    .subscribe(lambda x: self.obs_deque.append(x))  
      
        motion = robot.start_impedance_controller(830, 40, 1)
        done = False

        while not done:
            # wait for obs_deque to have len 2
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(0.1)
                # print("Waiting for observation")

            out = self.infer_action(self.obs_deque.copy())
            action = out['action']
            action_gripper = out['action_gripper']
            progress = out['progress']

            X_BE = self.obs_deque[-1]["X_BE"]

            for i in range(len(action)):

                trans, orien = matrix_to_pos_orn(action[i])
                motion.set_target(to_affine(trans, orien))

                robot_state = motion.get_robot_state()

                self.robot_visualiser.object_pose.T = self.obs_deque[-1]["X_BO"]
                self.robot_visualiser.policy_pose.T = action[i] 
                self.robot_visualiser.step(robot_state.q)

                time.sleep(0.2)






if __name__ == '__main__':
    


    controller = RobotInferenceController(
        perception_system=perception_system, 
        )
        
    #     #saved_run_name={'cup_rotate': 'golden-grass-127_state', #'fiery-pond-126_state
    #     #                                                  'place_saucer': 'laced-cosmos-124_state'}, 
    #                                     #   robot_ip='172.16.0.2')
    controller.start_inference()

    perception_system.stop()
