
from rs_imle_policy.network import get_resnet, replace_bn_with_gn, DiffusionConditionalUnet1D, GeneratorConditionalUnet1D
from rs_imle_policy.utilities import get_config
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from rs_imle_policy.dataset import PolicyDataset, normalize_data, unnormalize_data
import os
import copy
import numpy as np
import torchvision.transforms as transforms
import pdb

class Policy:
    def __init__(self, 
                 config_file=None,
                 saved_run_name=None,
                 mode='train'):
        
        self.config_file = config_file
        self.params = get_config(self.config_file)
        self.mode = mode
        self.method = self.params.method
        self.nets=self.create_networks()

        if self.params.method == 'diffusion':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.params.num_diffusion_iters, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')
        else:
            self.noise_scheduler = None

        self.ema = EMAModel(parameters=self.nets.parameters(), power=0.75)

        self.precision = torch.float32

        self.device = self.params.device


        if mode == 'train':

            # dataset = PushTImageDataset(args.dataset_path, args.pred_horizon, args.obs_horizon, args.action_horizon)
            self.dataset = PolicyDataset(self.params.dataset_path, self.params.pred_horizon, self.params.obs_horizon, self.params.action_horizon)
            # dataset = FilteredDatasetWrapper(dataset, 0.1)
            self.dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                    batch_size=self.params.batch_size, 
                                                    num_workers=self.params.num_workers, 
                                                    shuffle=True, 
                                                    pin_memory=True,
                                                    persistent_workers=True)

            self.optimizer = torch.optim.AdamW(params=self.nets.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
            self.lr_scheduler = get_scheduler(name=self.params.lr_scheduler_profile, 
                                         optimizer=self.optimizer, 
                                         num_warmup_steps=self.params.num_warmup_steps, 
                                         num_training_steps=len(self.dataloader) * self.params.num_epochs)
            
            print('Training Mode.')
            
        
        if mode == 'infer':
            self.load_weights(saved_run_name)

            # stats_path = os.path.join("/mnt/droplet/", 'stats.pkl')
            stats_path = os.path.join("saved_weights/default/rs_imle_25p/", 'stats.pkl')
            self.stats = np.load(stats_path, allow_pickle=True)

            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((240, 320)),
                            transforms.RandomCrop((216, 288)),
                            transforms.ToTensor()])
            
            print('Inference Mode.')


    def load_weights(self, saved_run_name, load_best=True):

        if self.method == 'diffusion':
            print('Loading pretrained weights for diffusion')
            self.ema_nets = copy.deepcopy(self.nets)

            # fpath_ema = os.path.join("/mnt/droplet/", "ema_net.pth")
            # fpath_nets = os.path.join("/mnt/droplet/", "net.pth")

            fpath_ema = os.path.join("saved_weights/diffusion_policy_10/", "ema_net_epoch_1000.pth")
            # fpath_nets = os.path.join("saved_weights/diffusion_policy_1.0/", "net.pth")


            # state_dict_nets = torch.load(fpath_nets, map_location='cuda')
            # self.nets.load_state_dict(state_dict_nets)
            state_dict_ema = torch.load(fpath_ema, map_location='cuda')
            self.ema_nets.load_state_dict(state_dict_ema)

            if self.precision == torch.float16:
                self.nets.half()
                self.ema_nets.half()

            self.ema = EMAModel(parameters=self.ema_nets.parameters(), power=0.75)

        elif self.method == 'rs_imle':
            print('Loading pretrained weights for rs_imle')
            fpath = os.path.join("saved_weights/default/rs_imle_25p/", "net_epoch_1190.pth")
            self.nets.load_state_dict(torch.load(fpath, map_location='cuda'))

        print('Pretrained weights loaded.')

        
    def create_networks(self):
        vision_encoder_side = get_resnet('resnet18')
        vision_encoder_side = replace_bn_with_gn(vision_encoder_side)

        vision_encoder_wrist = get_resnet('resnet18')
        vision_encoder_wrist = replace_bn_with_gn(vision_encoder_wrist)

        if self.method == 'diffusion':
            noise_pred_net = DiffusionConditionalUnet1D(
                input_dim=self.params.action_dim,
                global_cond_dim=self.params.obs_dim*self.params.obs_horizon
            )
            nets = nn.ModuleDict({
                'vision_encoder_side': vision_encoder_side,
                'vision_encoder_wrist': vision_encoder_wrist,
                'noise_pred_net': noise_pred_net
            })
        elif self.method == 'rs_imle':
            generator = GeneratorConditionalUnet1D(
                input_dim=self.params.action_dim,
                global_cond_dim=self.params.obs_dim*self.params.obs_horizon
            )
            nets = nn.ModuleDict({
                'vision_encoder_side': vision_encoder_side,
                'vision_encoder_wrist': vision_encoder_wrist,
                'generator': generator
            })

        return nets.to(self.params.device)

