from dataclasses import dataclass, field
import pathlib
from typing import Literal


@dataclass
class TrainingParams:
    """Learning parameters"""

    lr: float = 1e-4
    "Learning rate"
    weight_decay: float = 1e-6
    "weight_decay"
    num_epochs: int = 1200
    "Total number of epochs"
    batch_size: int = 64
    "Batch Size"
    num_workers: int = 11
    "Num of workers"
    lr_scheduler_profile: str = "cosine"
    "Learning schedule profile"
    num_warmup_steps: int = 500
    eval_interval: int = 10
    "Interval for evaluation during training"
    num_eval_episodes: int = 1
    save_period: int = 10
    "Period (in epochs) to save model checkpoints"


@dataclass
class CameraConfig:
    """Realsense Camera configuration"""

    name: str
    "Name of the camera"
    serial_number: str
    "Serial number of the camera"
    exposure: int
    gain: int
    resolution: tuple[int, int] = (640, 480)
    depth_resolution: tuple[int, int] = (640, 480)
    frame_rate: int = 10
    rgb_enabled: bool = True
    depth_enabled: bool = False


@dataclass
class WristCamera(CameraConfig):
    name: str = "wrist"
    serial_number: str = "123622270136"
    exposure: int = 5000
    gain: int = 60


class TopCamera(CameraConfig):
    name: str = "top"
    serial_number: str = "035122250388"
    exposure: int = 100
    gain: int = 60


@dataclass
class SideCamera(CameraConfig):
    name: str = "side"
    serial_number: str = "035122250692"
    exposure: int = 100
    gain: int = 60


@dataclass
class VisionConfig:
    """Vision feature configuration"""

    vision_features_dim: int = 512
    "Feature dimension of the vision encoder"
    cameras: tuple[str, ...] = ("wrist", "side", "top")
    # Name and order of the cameras used
    img_shape: tuple[int, int] = (240, 320)
    # resolution used as input to the model


@dataclass
class BaseModel:
    """Base model with common attributes between methods"""

    device: Literal["cuda", "cpu"] = "cuda"

    pred_horizon: int = 16
    action_horizon: int = 8
    obs_horizon: int = 2
    action_relative: bool = False

    lowdim_obs_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = ()

    vision_config: VisionConfig = field(default_factory=VisionConfig)


@dataclass
class RSIMLE(BaseModel):
    name: str = "rs_iml"
    "Name used to distinguish between small variations of the method"
    n_samples_per_condition: int = 10
    "Number od sampled per condition for RS-IMLE"
    epsilon: float = 0.1
    "Epsilon for RS-IMLE loss"


@dataclass
class Diffusion(BaseModel):
    name = "diffusion"
    num_diffusion_iters = 100
    "diffusion iterations"
    beta_schedule: str = "squaredcos_cap_v2"
    "beta schedule"
    clip_sample: bool = True
    "enable clip sample"
    prediction_type: str = "epsilon"
    "prediction_type"


@dataclass
class TrainConfig:
    exp_name: str
    "Name for experiment, the method will be appended in the end"
    dataset_path: pathlib.Path
    "Path to dataset"
    training_params: TrainingParams
    "Training parameters"
    model: Diffusion | RSIMLE = field(default_factory=RSIMLE)
    "Model used, either RS_IMLE or Diffusion"
    task_name: str = "default"
    "Name of the task to train on"


@dataclass
class LoaderConfig:
    dataset_path: pathlib.Path
    "Path to dataset"


@dataclass
class InferenceConfig:
    exp_name: str
    "Name for experiment, the method will be appended in the end"
    weights_path: pathlib.Path
    "Path to weights"
    model: Diffusion | RSIMLE = field(default_factory=RSIMLE)
    "Model used, either RS_IMLE or Diffusion"
    task_name: str = "default"
    "Name of the task to train on"


if __name__ == "__main__":
    import tyro

    args = tyro.cli(TrainConfig)
    config = tyro.extras.to_yaml(args)
    with open("example.yml", "w") as outfile:
        outfile.write(config)
