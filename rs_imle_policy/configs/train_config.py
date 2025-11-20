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


@dataclass
class DiffusionConfig:
    """DDPM parameters"""

    num_diffusion_iters = 100
    "diffusion iterations"
    beta_schedule: str = "squaredcos_cap_v2"
    "beta schedule"
    clip_sample: bool = True
    "enable clip sample"
    prediction_type: str = "epsilon"
    "prediction_type"


@dataclass
class VisionConfig:
    """Vision feature configuration"""

    vision_features_dim: int = 512
    "Feature dimension of the vision encoder"
    num_cameras: int = 2
    "Cameras view used"


@dataclass
class BaseModel:
    """Base model with common attributes between methods"""

    device: Literal["cuda", "cpu"] = "cuda"

    pred_horizon: int = 16
    action_horizon: int = 8
    obs_horizon: int = 2

    lowdim_obs_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = ()

    vision_config: VisionConfig = field(default_factory=VisionConfig)


@dataclass
class RSIMLE(BaseModel):
    name: str = "rs_iml"


@dataclass
class Diffusion(BaseModel):
    ddpm: DiffusionConfig = field(default_factory=DiffusionConfig)
    name = "diffusion"


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


if __name__ == "__main__":
    import tyro

    args = tyro.cli(TrainConfig)
    config = tyro.extras.to_yaml(args)
    with open("example.yml", "w") as outfile:
        outfile.write(config)
