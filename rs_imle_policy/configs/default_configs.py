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
class PickPlaceDiffusionConfig(TrainConfig):
    model: RSIMLE | Diffusion = field(
        default_factory=lambda: Diffusion(
            lowdim_obs_keys=("robot_pos", "robot_orien", "gripper_state"),
            action_keys=("action_pos", "action_orien", "action_gripper"),
        )
    )
