import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from dataset import PolicyDataset, normalize_data, unnormalize_data
from network import get_resnet, replace_bn_with_gn, DiffusionConditionalUnet1D, GeneratorConditionalUnet1D
import wandb
import numpy as np
from collections import deque
import copy
import pdb
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for robotic pushing")
    parser.add_argument("--dataset_path", type=str, default="data/t_block_1", help="Path to the dataset")
    parser.add_argument("--pred_horizon", type=int, default=16, help="Prediction horizon")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Observation horizon")
    parser.add_argument("--action_horizon", type=int, default=8, help="Action horizon")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=11, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=1200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--num_diffusion_iters", type=int, default=100, help="Number of diffusion iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--eval_interval", type=int, default=1, help="Interval for evaluation during training")
    parser.add_argument("--num_eval_episodes", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--method", type=str, choices=['diffusion', 'rs_imle'], default='diffusion', help="Training method")
    parser.add_argument("--n_samples_per_condition", type=int, default=10, help="Number of samples per condition for RS-IMLE")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Epsilon for RS-IMLE loss")
    parser.add_argument("--use_ema_for_eval", action="store_true", help="Use EMA for evaluation")
    return parser.parse_args()

def create_networks(args, stats):
    vision_encoder_side = get_resnet('resnet18')
    vision_encoder_side = replace_bn_with_gn(vision_encoder_side)

    vision_encoder_wrist = get_resnet('resnet18')
    vision_encoder_wrist = replace_bn_with_gn(vision_encoder_wrist)
    
    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = (vision_feature_dim*2) + lowdim_obs_dim
    action_dim = 2

    if args.method == 'diffusion':
        noise_pred_net = DiffusionConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*args.obs_horizon
        )
        nets = nn.ModuleDict({
            'vision_encoder_side': vision_encoder_side,
            'vision_encoder_wrist': vision_encoder_wrist,
            'noise_pred_net': noise_pred_net
        })
    elif args.method == 'rs_imle':
        generator = GeneratorConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*args.obs_horizon
        )
        nets = nn.ModuleDict({
            'vision_encoder_side': vision_encoder_side,
            'vision_encoder_wrist': vision_encoder_wrist,
            'generator': generator
        })

    return nets.to(args.device)

def rs_imle_loss(real_samples, fake_samples, epsilon=0.1):
    B, T, D = real_samples.shape
    n_samples = fake_samples.shape[1]

    real_flat = real_samples.reshape(B, 1, -1)
    fake_flat = fake_samples.reshape(B, n_samples, -1)

    distances = torch.cdist(real_flat, fake_flat).squeeze(1)
    valid_samples = (distances > epsilon).float()
    wandb.log({"max_distance": distances.max().item(), "min_distance": distances.min().item(), "mean_distance": distances.mean().item(), "epsilon": epsilon})
    min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
    valid_real_samples = (min_distances < distances.max()).float()
    if valid_real_samples.sum() > 0:
        loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
    else:
        loss = torch.tensor(0.0, device=real_samples.device)
    return loss

def train(args, nets, dataloader, noise_scheduler, optimizer, lr_scheduler, ema):
    nets.train()
    for epoch in range(args.num_epochs):
        epoch_loss = []
        start_time = time.time()
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}', leave=False) as tepoch:
            for batch in tepoch:
                
                nimage_side = batch['frames_side'][:,:args.obs_horizon].to(args.device)
                nimage_wrist = batch['frames_wrist'][:,:args.obs_horizon].to(args.device)
                nagent_pos = batch['agent_pos'][:,:args.obs_horizon].to(args.device)
                naction = batch['action'].to(args.device)
                B = nagent_pos.shape[0]

                image_features_side = nets['vision_encoder_side'](nimage_side.flatten(end_dim=1))
                image_features_side = image_features_side.reshape(*nimage_side.shape[:2],-1)

                image_features_wrist = nets['vision_encoder_wrist'](nimage_wrist.flatten(end_dim=1))
                image_features_wrist = image_features_wrist.reshape(*nimage_wrist.shape[:2],-1)

                obs_features = torch.cat([image_features_side, image_features_wrist, nagent_pos], dim=-1)                
                obs_cond = obs_features.flatten(start_dim=1)

                if args.method == 'diffusion':
                    noise = torch.randn(naction.shape, device=args.device)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=args.device).long()
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(noise_pred, noise)
                elif args.method == 'rs_imle':
                    noise = torch.randn(B * args.n_samples_per_condition, *naction.shape[1:], device=args.device)
                    repeated_obs_cond = obs_cond.repeat_interleave(args.n_samples_per_condition, dim=0)

                    fake_actions = nets['generator'](noise, global_cond=repeated_obs_cond)
                    fake_actions = fake_actions.reshape(B, args.n_samples_per_condition, *naction.shape[1:])

                    loss = rs_imle_loss(naction, fake_actions, args.epsilon)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                ema.step(nets.parameters())

                wandb.log({"loss": loss.item()})

                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        ema_nets = copy.deepcopy(nets)
        ema.copy_to(ema_nets.parameters())
        torch.save(nets.state_dict(), f"saved_weights/net.pth")
        torch.save(ema_nets.state_dict(), f"saved_weights/ema_net.pth")

        avg_loss = np.mean(epoch_loss)
        wandb.log({"avg_train_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{args.num_epochs} - Avg. Loss: {avg_loss:.4f} - Time: {time.time()-start_time:.2f}s")

  
    return nets


def main():
    args = parse_args()
    wandb.init(project="real_robot_pusht", config=args)

    # dataset = PushTImageDataset(args.dataset_path, args.pred_horizon, args.obs_horizon, args.action_horizon)
    dataset = PolicyDataset(args.dataset_path, args.pred_horizon, args.obs_horizon, args.action_horizon)
    # dataset = FilteredDatasetWrapper(dataset, 0.1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             num_workers=args.num_workers, 
                                             shuffle=True, 
                                             pin_memory=True,
                                             persistent_workers=True)

    nets = create_networks(args, dataset.stats)
    
    if args.method == 'diffusion':
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_diffusion_iters, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')
    else:
        noise_scheduler = None
    
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(dataloader) * args.num_epochs)

    trained_nets = train(args, nets, dataloader, noise_scheduler, optimizer, lr_scheduler, ema)
    
    # save model ckpt
    ema_nets = copy.deepcopy(nets)
    ema.copy_to(ema_nets.parameters())
    torch.save(nets.state_dict(), f"saved_weights/net.pth")
    torch.save(ema_nets.state_dict(), f"saved_weights/ema_net.pth")

    wandb.finish()

if __name__ == "__main__":
    main()
