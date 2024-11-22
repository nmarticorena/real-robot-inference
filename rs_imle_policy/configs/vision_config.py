
dataset_path = 'data/t_block_1'

# method = 'rs_imle'
method = 'diffusion'
vision_feature_dim = 512
lowdim_obs_dim = 2
obs_dim = (vision_feature_dim*2) + lowdim_obs_dim
action_dim = 2

pred_horizon = 16
action_horizon = 8
obs_horizon = 2

device = 'cuda'

num_diffusion_iters = 100
beta_schedule = 'squaredcos_cap_v2'
clip_sample = True
prediction_type = 'epsilon'

lr = 1e-4
weight_decay = 1e-6
num_epochs = 1200
batch_size = 64
num_workers = 11
lr_scheduler_profile = 'cosine'
num_warmup_steps = 500

