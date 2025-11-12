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
import cv2
import os
# env import
import time

from rs_imle_policy.policy import Policy
from rs_imle_policy.dataset import normalize_data, unnormalize_data
import rs_imle_policy.utilities as utils

import numpy as np
import time
import collections
import reactivex as rx
from reactivex import operators as ops

# from diffrobot.robot.robot import Robot, to_affine, matrix_to_pos_orn
from frankx import Robot, Waypoint, WaypointMotion, JointMotion, Affine, LinearMotion, Kinematics

from rs_imle_policy.realsense.multi_realsense import MultiRealsense

import matplotlib.pyplot as plt

import pdb


class PerceptionSystem:
    def __init__(self):
        self.cams = MultiRealsense(
            serial_numbers=['123622270136', '035122250692'],
            resolution=(640,480),
            enable_depth=False, 
        )
    def start(self):
        self.cams.start()
        self.cams.cameras['123622270136'].set_exposure(exposure=5000, gain=60)
        self.cams.cameras['035122250692'].set_exposure(exposure=100, gain=60)
   
    def stop(self):
        self.cams.stop()


class RobotInferenceController:
    def __init__(self, eval_name, idx):
        
        self.seed(42)
        self.robot = self.create_robot()
        self.perception_system = PerceptionSystem()
        self.perception_system.start()
        self.setup_diffusion_policy()
        self.move_to_start()

        self.eval_name = eval_name
        self.all_frames = []
        self.done = False
        self.idx = str(idx)

        #create folder to save images
        os.makedirs(f"saved_evaluation_media/{self.eval_name}", exist_ok=True)

    
    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def move_to_start(self):
        self.robot.move(JointMotion([-1.829, 0.008, 0.172, -1.594, 0.001, 1.559, 0.718]))
        # self.robot.move(JointMotion([-1.9953644495495173, -0.07019201069593659, 0.051291523464672376, -2.4943418327817803, -0.042134962130810624, 2.385776886145273, 0.35092161391247345]))
        # self.robot.move(JointMotion([-1.4257584505685634, -0.302815655026201, 0.05126683842989545, -2.7415479229025443, 0.011030055865001531, 2.3881221201502796, 0.8777110836404692]))


    def create_robot(self, ip:str = "172.16.0.2", dynamic_rel: float=0.4): #0.4
        # panda = Robot(ip)
        panda = Robot(ip, repeat_on_error=True, dynamic_rel=dynamic_rel)
        panda.recover_from_errors()
        panda.accel_rel = 0.1
        panda.jerk_rel = 0.01
        return panda

    def setup_diffusion_policy(self):
        torch.cuda.empty_cache()
        self.policy = Policy(config_file='pick_place_config',
                        saved_run_name=None, 
                        mode='infer')

        self.obs_horizon = self.policy.params.obs_horizon
        self.obs_deque = collections.deque(maxlen=self.policy.params.obs_horizon)


    def process_inference_vision(self, obs_deque):
        image_side = np.stack([x['image_side'] for x in obs_deque])
        image_wrist = np.stack([x['image_wrist'] for x in obs_deque])
        agent_pos = np.stack([x['state'] for x in obs_deque])
        print(self.policy.stats.keys())


        nagent_pos = normalize_data(agent_pos, stats=self.policy.stats['state'])

        nimage_side = torch.stack([self.policy.transform(img) for img in image_side])
        nimage_wrist = torch.stack([self.policy.transform(img) for img in image_wrist])

        nagent_pos = torch.from_numpy(nagent_pos).to(self.policy.device, dtype=self.policy.precision)
        nimage_side = nimage_side.to(self.policy.device, dtype=self.policy.precision)
        nimage_wrist = nimage_wrist.to(self.policy.device, dtype=self.policy.precision)

        if self.policy.params.method == 'diffusion':
            image_features_side = self.policy.ema_nets['vision_encoder_side'](nimage_side)
            image_features_wrist = self.policy.ema_nets['vision_encoder_wrist'](nimage_wrist)
        elif self.policy.params.method == 'rs_imle':
            image_features_side = self.policy.nets['vision_encoder_side'](nimage_side)
            image_features_wrist = self.policy.nets['vision_encoder_wrist'](nimage_wrist)

        obs_features = torch.cat([image_features_side, image_features_wrist, nagent_pos], dim=-1)                
        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)
        
        return obs_cond
    
    
    def get_observation(self):
        s = self.robot.get_state(read_once = False)
 
        # s = self.robot.read_once()
        X_BE = np.array(s.O_T_EE).reshape(4,4).T
        t = X_BE[:3,3]
        rot = utils.matrix_to_rotation_6d(X_BE[:3,:3])
        state = np.concat([t,rot])

        images = self.perception_system.cams.get()
        images_wrist = images[0]['color']
        images_side = images[1]['color']
        # images_top = images[2]['color']

        # save images_top
        # plt.imsave("images_top.png", images_top)

        self.all_frames.append(images_side)

        cv2.imshow('image', images_side)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # write all frames to video
            save_path = f"saved_evaluation_media/{self.eval_name}/{self.idx}.mp4"
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
            for frame in self.all_frames:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                out.write(rgb_frame)
            out.release()
            cv2.destroyAllWindows()
            self.perception_system.stop()
            self.done = True
            print("Done")


        return {"state": state, 
                "image_wrist": images_wrist,
                "image_side": images_side}
    
    
    def infer_action(self, obs_deque):

        with torch.no_grad():

            obs_cond = self.process_inference_vision(obs_deque)

            if self.policy.params.method == 'diffusion':

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
            
            elif self.policy.params.method == 'rs_imle':
                noise = torch.randn((1, self.policy.params.pred_horizon, self.policy.params.action_dim), device=self.policy.params.device)
                # clip noise
                noise = torch.clamp(noise, -1, 1)
                naction = self.policy.nets['generator'](noise, global_cond=obs_cond)
        
        naction = naction.detach().to('cpu').numpy()[0]

        # unnormalize action
        action_pos = unnormalize_data(naction, stats=self.policy.stats['action'])

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
        
        self.motion = WaypointMotion([Waypoint(current_pose)], return_when_finished=False)
        thread = self.robot.move_async(self.motion)

        start_time = time.time()    

        time.sleep(2)

        while not self.done:
            # wait for obs_deque to have len 2
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(0.1)
                print("Waiting for observation")

            out = self.infer_action(self.obs_deque.copy())
            action = out['action']

            print("elapsed time: ", time.time() - start_time)

            

            
            for i in range(4,len(action)):
                # keep everyting the same except xy which is the action
                trans = np.array([action[i][0], action[i][1], action[i][2]])
                self.motion.set_next_waypoint(Waypoint(Affine(trans[0], trans[1], trans[2], q[0], q[1], q[2], q[3])))
                time.sleep(0.1)





if __name__ == '__main__':
    


    controller = RobotInferenceController(eval_name='rs_imle_policy_10', idx=0)
    controller.start_inference()
    controller.perception_system.stop()
