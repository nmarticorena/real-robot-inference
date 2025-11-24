from rs_imle_policy.configs.train_config import TrainConfig, RSIMLE, Diffusion
from dataclasses import dataclass, field


@dataclass
class PickPlaceRSMLEConfig(TrainConfig):
    model: RSIMLE | Diffusion = field(
        default_factory=lambda: RSIMLE(
            lowdim_obs_keys=("robot_pos", "robot_orien", "gripper_state"),
            action_keys=("action_pos", "action_orien", "action_gripper"),
        )
    )


@dataclass
class PickPlaceRelativeRSMLEConfig(TrainConfig):
    model: RSIMLE | Diffusion = field(
        default_factory=lambda: RSIMLE(
            lowdim_obs_keys=("robot_pos", "robot_orien", "gripper_state"),
            action_keys=("relative_pos", "relative_orien", "action_gripper"),
            action_relative=True,
        )
    )


@dataclass
class PickPlaceDiffusionConfig(TrainConfig):
    model: RSIMLE | Diffusion = field(
        default_factory=lambda: Diffusion(
            lowdim_obs_keys=("robot_pos", "robot_orien", "gripper_state"),
            action_keys=("action_pos", "action_orien", "action_gripper"),
        )
    )


@dataclass
class PickPlaceDiffusionRelativeConfig(TrainConfig):
    model: RSIMLE | Diffusion = field(
        default_factory=lambda: Diffusion(
            lowdim_obs_keys=("robot_pos", "robot_orien", "gripper_state"),
            action_keys=("relative_pos", "relative_orien", "action_gripper"),
            action_relative=True,
        )
    )
