from rs_imle_policy.network import (
    get_resnet,
    replace_bn_with_gn,
    DiffusionConditionalUnet1D,
    GeneratorConditionalUnet1D,
)
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from rs_imle_policy.dataset import PolicyDataset
import os
import copy
import numpy as np
import torchvision.transforms as transforms
from rs_imle_policy.configs.train_config import (
    TrainConfig,
    InferenceConfig,
    Diffusion,
    RSIMLE,
)
from typing import Union


class Policy:
    def __init__(self, config: Union[InferenceConfig, TrainConfig]):
        self.config = config
        self.model_config = config.model

        if isinstance(self.model_config, Diffusion):
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_config.num_diffusion_iters,
                beta_schedule=self.model_config.beta_schedule,
                clip_sample=self.model_config.clip_sample,
                prediction_type=self.model_config.prediction_type,
            )  # TODO: Check if I can do *kwargs here
        else:
            self.noise_scheduler = None

        self.precision = torch.float32
        self.device = self.model_config.device
        if isinstance(self.config, TrainConfig):
            self.dataset = PolicyDataset(
                self.config.dataset_path,
                self.model_config.pred_horizon,
                self.model_config.obs_horizon,
                self.model_config.action_horizon,
                low_dim_obs_keys=self.model_config.lowdim_obs_keys,
                action_keys=self.model_config.action_keys,
            )  # TODO: Check why does we initialize this twice
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.config.training_params.batch_size,
                num_workers=self.config.training_params.num_workers,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True,
            )
            self.nets = self.create_networks()
            self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

            self.optimizer = torch.optim.AdamW(
                params=self.nets.parameters(),
                lr=self.config.training_params.lr,
                weight_decay=self.config.training_params.weight_decay,
            )
            self.lr_scheduler = get_scheduler(
                name=self.config.training_params.lr_scheduler_profile,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.training_params.num_warmup_steps,
                num_training_steps=len(self.dataloader)
                * self.config.training_params.num_epochs,
            )

            print("Training Mode.")

        elif isinstance(self.config, InferenceConfig):
            # self.load_weights(saved_run_name) TODO: Fix

            # stats_path = os.path.join("/mnt/droplet/", 'stats.pkl')
            stats_path = os.path.join("saved_weights/default/rs_imle_25p/", "stats.pkl")
            self.stats = np.load(stats_path, allow_pickle=True)

            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((240, 320)),
                    transforms.RandomCrop((216, 288)),
                    transforms.ToTensor(),
                ]
            )

            print("Inference Mode.")

    def load_weights(self, saved_run_name, load_best=True):
        print(saved_run_name)
        if isinstance(self.config, Diffusion):
            print("Loading pretrained weights for diffusion")
            self.ema_nets = copy.deepcopy(self.nets)

            fpath_ema = os.path.join(
                "saved_weights/diffusion_policy_10/", "ema_net_epoch_.pth"
            )
            state_dict_ema = torch.load(fpath_ema, map_location="cuda")
            self.ema_nets.load_state_dict(state_dict_ema)

            if self.precision == torch.float16:
                self.nets.half()
                self.ema_nets.half()

            self.ema = EMAModel(parameters=self.ema_nets.parameters(), power=0.75)

        elif isinstance(self.config, RSIMLE):
            print("Loading pretrained weights for rs_imle")
            fpath = os.path.join("saved_weights//rs_imle_25p/", "net_epoch_last.pth")
            self.nets.load_state_dict(torch.load(fpath, map_location="cuda"))

        print("Pretrained weights loaded.")

    def create_networks(self):
        vision_encoder_side = get_resnet("resnet18")
        vision_encoder_side = replace_bn_with_gn(vision_encoder_side)

        vision_encoder_wrist = get_resnet("resnet18")
        vision_encoder_wrist = replace_bn_with_gn(vision_encoder_wrist)

        if isinstance(self.config.model, Diffusion):
            noise_pred_net = DiffusionConditionalUnet1D(
                input_dim=self.dataset.action_shape,
                global_cond_dim=self.dataset.obs_shape * self.model_config.obs_horizon,
            )
            nets = nn.ModuleDict(
                {
                    "vision_encoder_side": vision_encoder_side,
                    "vision_encoder_wrist": vision_encoder_wrist,
                    "noise_pred_net": noise_pred_net,
                }
            )
        elif isinstance(self.config.model, RSIMLE):
            generator = GeneratorConditionalUnet1D(
                input_dim=self.dataset.action_shape,
                global_cond_dim=self.dataset.obs_shape * self.model_config.obs_horizon,
            )
            nets = nn.ModuleDict(
                {
                    "vision_encoder_side": vision_encoder_side,
                    "vision_encoder_wrist": vision_encoder_wrist,
                    "generator": generator,
                }
            )

        return nets.to(self.model_config.device)
