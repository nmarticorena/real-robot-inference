from collections import defaultdict
import roboticstoolbox as rtb
import rerun as rr

import spatialmath as sm
import spatialmath.base as smb
import numpy as np
import torch
import collections
import cv2
import os

import time
from rs_imle_policy.visualizer.rerun_tools import ReRunRobot

from rs_imle_policy.configs.default_configs import ExperimentConfigChoice  # noqa: F401
from rs_imle_policy.configs.train_config import (
    ExperimentConfig,
    VisionConfig,
    Diffusion,
    RSIMLE,
)
from rs_imle_policy.policy import Policy
from rs_imle_policy.dataset import normalize_data, unnormalize_data
import rs_imle_policy.utilities as utils

import reactivex as rx
from reactivex import operators as ops

from frankx import (
    Robot,
    Waypoint,
    WaypointMotion,
    JointMotion,
    Affine,
    Gripper,
)


from rs_imle_policy.realsense.multi_realsense import MultiRealsense


class PerceptionSystem:
    def __init__(self, vision_config: VisionConfig):
        serial_numbers = [cam.serial_number for cam in vision_config.cameras_params]
        self.cams = MultiRealsense(
            serial_numbers=serial_numbers,
            resolution=vision_config.cameras_params[0].resolution,
            record_fps=vision_config.cameras_params[0].frame_rate,
            depth_resolution=vision_config.cameras_params[0].depth_resolution,
            enable_depth=vision_config.cameras_params[0].depth_enabled,
        )
        self.cams_config = vision_config

    def start(self):
        for cam_params in self.cams_config.cameras_params:
            self.cams.cameras[cam_params.serial_number].set_exposure(
                exposure=cam_params.exposure, gain=cam_params.gain
            )
        self.cams.start()

    def stop(self):
        self.cams.stop()


class RobotInferenceController:
    def __init__(self, config: ExperimentConfig, eval_name: str, idx: int):
        self.seed(42)
        self.config = config
        self.config.training = False
        self.robot = self.create_robot()

        rtb_panda = rtb.models.Panda()
        self.gui = ReRunRobot(rtb_panda)
        self.gripper = Gripper("172.16.0.2", speed=0.1)
        self.perception_system = PerceptionSystem(config.data.vision)
        self.perception_system.start()
        self.gripper_open = False
        self.setup_diffusion_policy()
        self.move_to_start()

        self.eval_name = eval_name
        self.all_frames = defaultdict(list)
        self.done = False
        self.idx = str(idx)

        # create folder to save images
        os.makedirs(f"saved_evaluation_media/{self.eval_name}", exist_ok=True)

    def seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def move_to_start(self):
        self.open_gripper_if_closed()
        self.robot.move(JointMotion(np.deg2rad([-90, 0, 0, -90, 0, 90, 45])))

    def create_robot(self, ip: str = "172.16.0.2", dynamic_rel: float = 0.4):  # 0.4
        panda = Robot(ip, repeat_on_error=True, dynamic_rel=dynamic_rel)
        panda.recover_from_errors()
        impedance = [400.0, 400.0, 400.0, 40.0, 40.0, 40.0]

        # impedance = np.diag(impedance)
        panda.set_cartesian_impedance(impedance)
        panda.accel_rel = 0.1
        panda.jerk_rel = 0.01
        return panda

    def setup_diffusion_policy(self):
        torch.cuda.empty_cache()
        self.policy = Policy(self.config)

        self.obs_horizon = self.config.model.obs_horizon
        self.obs_deque = collections.deque(maxlen=self.config.model.obs_horizon)

    def process_inference_vision(self, obs_deque):
        images = []
        for cam_name in self.config.data.vision.cameras:
            image = np.stack([x[cam_name] for x in obs_deque])
            input_image = torch.stack([self.policy.transform(img) for img in image])
            images.append(
                input_image.to(self.policy.device, dtype=self.policy.precision)
            )

        agent_pos = np.stack([x["state"] for x in obs_deque])
        nagent_pos = normalize_data(agent_pos, stats=self.policy.stats["state"])
        nagent_pos = torch.from_numpy(nagent_pos).to(
            self.policy.device, dtype=self.policy.precision
        )

        image_features = []
        if isinstance(self.config.model, Diffusion):
            for ix, cam_name in enumerate(self.config.data.vision.cameras):
                image_feature = self.policy.ema_nets[f"vision_encoder_{cam_name}"](
                    images[ix]
                )
                image_features.append(image_feature)
        elif isinstance(self.config.model, RSIMLE):
            for ix, cam_name in enumerate(self.config.data.vision.cameras):
                image_feature = self.policy.nets[f"vision_encoder_{cam_name}"](
                    images[ix]
                )
                image_features.append(image_feature)
        obs_features = torch.cat(image_features + [nagent_pos], dim=-1)

        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

        return obs_cond

    def get_observation(self):
        # s = self.robot.get_robo_state(read_once=False)
        s = self.motion.get_robot_state()

        self.gui.step_robot(s.q)
        X_BE = np.array(s.O_T_EE).reshape(4, 4, order="F")

        rr.log("state/O_T_ee", rr.Transform3D(
            translation=X_BE[:3, 3], mat3x3=X_BE[:3, :3],
            axis_length=0.1)
        )

        rot = utils.matrix_to_rotation_6d(X_BE[:3, :3])
        t = X_BE[:3, 3]
        width = self.gripper.width()

        state = np.concat([t, rot, (width,)])

        images = self.perception_system.cams.get()

        frames = {}
        for ix, cam_name in enumerate(self.config.data.vision.cameras):
            frames[f"{cam_name}"] = images[ix]["color"]
            self.all_frames[f"{cam_name}"].append(images[ix]["color"])
            rr.log(f"/{cam_name}", rr.Image(images[ix]["color"]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.record_videos()  # Improve here

        return {"state": state, **frames}

    def record_videos(self):
        for cam_name in self.config.data.vision.cameras:
            save_path = (
                f"saved_evaluation_media/{self.eval_name}/{self.idx}_{cam_name}.mp4"
            )
            out = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480)
            )
            for frame in self.all_frames[cam_name]:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(rgb_frame)
            out.release()
        cv2.destroyAllWindows()
        self.perception_system.stop()
        self.done = True

    def infer_action(self, obs_deque):
        low_dim_state = np.stack([x["state"] for x in obs_deque])
        pos = low_dim_state[:, :3]
        orien_6d = low_dim_state[:, 3:9]

        pose = utils.pos_rot_to_se3(torch.from_numpy(pos), torch.from_numpy(orien_6d))

        with torch.no_grad():
            obs_cond = self.process_inference_vision(obs_deque)

            if isinstance(self.config.model, Diffusion):
                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (1, self.config.model.pred_horizon, self.config.action_shape),
                    device=self.policy.device,
                    dtype=self.policy.precision,
                )
                naction = noisy_action
                # init scheduler
                self.policy.noise_scheduler.set_timesteps(
                    self.config.model.num_diffusion_iters
                )

                actions = []
                for k in self.policy.noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = self.policy.ema_nets["noise_pred_net"](
                        sample=naction, timestep=k, global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.policy.noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=naction
                    ).prev_sample
                    actions.append(naction.cpu().numpy())

            elif isinstance(self.config.model, RSIMLE):
                noise = torch.randn(
                    (1, self.config.model.pred_horizon, self.config.action_shape),
                    device=self.config.model.device,
                )
                # clip noise
                noise = torch.clamp(noise, -1, 1)
                naction = self.policy.nets["generator"](noise, global_cond=obs_cond)
            else:
                raise NotImplementedError("Model not supported for inference.")

        naction = naction.detach().to("cpu").numpy()[0]

        # unnormalize action
        action_pos = unnormalize_data(naction, stats=self.policy.stats["action"])

        # only take action_horizon number of actions
        start = self.config.model.obs_horizon - 1
        end = start + self.config.model.action_horizon
        action = action_pos[start:end]

        return {"action": action}

    def close_gripper_if_open(self) -> bool:
        if self.gripper_open:
            self.gripper.close()
            self.gripper_open = False
            return True
        return False

    def open_gripper_if_closed(self) -> bool:
        if not self.gripper_open:
            self.gripper.open()
            self.gripper_open = True
            return True
        return False

    def log_poses(self, trans, quads):
        n_actions = len(trans)
        for i in range(n_actions):
            rr.log(
                f"action/pose_{i}/transform",
                rr.Transform3D(
                        translation = trans[i],
                        mat3x3 = quads[i],
                        axis_length=0.1
                    )
            )
            

    def start_inference(self):
        obs_stream = (  # noqa: F841
            rx.interval(0.1, scheduler=rx.scheduler.NewThreadScheduler())
            .pipe(ops.map(lambda _: self.get_observation()))
            .subscribe(lambda x: self.obs_deque.append(x))
        )

        current_pose = self.robot.current_pose()

        self.motion = WaypointMotion(
            [Waypoint(current_pose)], return_when_finished=False
        )
        thread = self.robot.move_async(self.motion)  # noqa: F841

        start_time = time.time()

        time.sleep(2)

        all_actions = np.zeros((0, self.config.action_shape))

        while not self.done:
            while len(self.obs_deque) < self.obs_horizon:
                time.sleep(0.001)
                print("Waiting for observation")

            obs = self.obs_deque.copy()
            out = self.infer_action(obs)
            action = out["action"]
            all_actions = np.concatenate([all_actions, action], axis=0)
            # Im not proud of this, but currently we only do
            # either relative or absolute actions, Here we
            # need to add any fix

            print("elapsed time: ", time.time() - start_time)
            low_level_obs = obs[-1]["state"]
            low_level_pos = low_level_obs[:3]
            low_level_orien = low_level_obs[3:9]

            if self.config.data.action_relative:
                current_pose = utils.pos_rot_to_se3(low_level_pos, low_level_orien)
                n_trans, n_quads, poses = self.convert_actions(action)
                r = utils.rotation_6d_to_matrix(action[:, 3:9])
            else:
                n_trans, n_quads, poses = self.convert_actions(action)
                r = utils.rotation_6d_to_matrix(action[:, 3:9])

            self.log_poses(n_trans, r.numpy())
            rr.log(f"action/gripper", rr.Scalars(action[0, -2].tolist()))
            rr.log(f"action/progress", rr.Scalars(action[0, -1].tolist()))

            waypoints = []
            extra_poses = []
            for i in range(0, int(len(action) / 2)):
                trans = n_trans[i]
                q = n_quads[i]
                pose = Affine(trans[0], trans[1], trans[2], q[0], q[1], q[2], q[3])

                extra_poses.append(np.array(pose.array()).reshape((4, 4), order="F"))

                waypoints.append(
                    Waypoint(
                        Affine(trans[0], trans[1], trans[2], q[0], q[1], q[2], q[3])
                    )
                )
                if action[i][-2] > 0.5:
                    print("clossing")
                    self.close_gripper_if_open()
                else:
                    print("openning")
                    self.open_gripper_if_closed()

            # print("progress: ",action[i][-1])
            # if action[0][-1] > 0.9:
                # self.done = True
            self.motion.set_next_waypoints(waypoints)
            time.sleep(0.5 * len(waypoints))

    def relative_to_absolute(
        self, action: np.ndarray, current_pose: sm.SE3
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[sm.SE3]]:
        n = action.shape[0]
        t = action[:, :3]
        rot = action[:, 3:-2]

        rel_poses = utils.pos_rot_to_se3(torch.from_numpy(t), rot)
        trans = []
        quads = []
        poses = [current_pose]
        for i in range(n):
            poses.append(poses[-1] * rel_poses[i])

            trans.append(poses[-1].t)
            quads.append(smb.r2q(poses[-1].A[:3, :3], order="sxyz"))
        return trans, quads, poses

    def convert_actions(
        self, action: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[sm.SE3]]:
        global idx_plot
        t = action[:, :3]
        rot = action[:, 3:-2]

        poses = utils.pos_rot_to_se3(torch.from_numpy(t), rot)
        trans = t.tolist()
        quads = utils.rotation_6d_to_quat(torch.from_numpy(rot)).numpy()

        return trans, quads, poses


if __name__ == "__main__":
    from rs_imle_policy.configs.train_config import LoaderConfig
    import re
    

    import tyro
    from InquirerPy import inquirer

    exp_name = inquirer.text("Enter the experiment name: ").execute()
    exp_name = re.sub(r"\s+", "_", exp_name.strip())

    args = tyro.cli(LoaderConfig)
    config = tyro.extras.from_yaml(ExperimentConfig, open(args.path / "config.yaml"))
    config.epoch = args.epoch
    
    rr.init("Robot Inference ",recording_id = exp_name)
    rr.serve_web()

    controller = RobotInferenceController(config, eval_name=exp_name, idx=0)

    controller.start_inference()
    controller.perception_system.stop()
