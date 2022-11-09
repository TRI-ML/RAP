from risk_biased.config.paths import (
    log_path,
)

# WandB Project Name
project = "RiskBiased"
entity = "tri"

# Scene Parameters
dt = 0.1
time_scene = 5.0
sample_times = [t * dt for t in range(0, int(time_scene / dt))]
ego_ref_speed = 14.0
ego_length = 4.0
ego_width = 1.75
fast_speed = 2.0
slow_speed = 1.0
p_change_pace = 0.2
proportion_fast = 0.5

# Data Parameters
file_name = "scene_data"
datasets_sizes = {"train": 100000, "val": 10000, "test": 30000}
datasets = list(datasets_sizes.keys())
state_dim = 2
dynamic_state_dim = 2
num_steps = 5
num_steps_future = len(sample_times) - num_steps
ego_speed_init_low = 4.0
ego_speed_init_high = 16.0
ego_acceleration_mean_low = -1.5
ego_acceleration_mean_high = 1.5
ego_acceleration_std = 3.0
perception_noise_std = 0.05
map_state_dim = 0
max_size_lane = 0
num_blocks = 3
interaction_type = None
mcg_dim_expansion = 0
mcg_num_layers = 0
num_attention_heads = 4


# Model Hyperparameters
model_type = "encoder_biased"
condition_on_ego_future = True
latent_dim = 2
hidden_dim = 64
num_vq = 256
latent_distribution = "gaussian"  # "gaussian" or "quantized"
num_hidden_layers = 3
sequence_encoder_type = "MLP"  # one of "MLP", "LSTM", "maskedLSTM"
sequence_decoder_type = "MLP"  # one of "MLP", "LSTM", "maskedLSTM"
is_mlp_residual = True

# Variational Loss Hyperparameters
kl_weight = 0.3
kl_threshold = 0.1
latent_regularization = 0.1

# Risk distribution should be one of the following types :
#  {"type": "uniform", "min": 0, "max": 1},
#  {"type": "normal", "mean": 0, "sigma": 1},
#  {"type": "bernoulli", "p": 0.5, "min": 0, "max": 1},
#  {"type": "beta", "alpha": 2, "beta": 5, "min": 0, "max": 1},
#  {"type": "chi2", "k": 3, "min": 0, "scale": 1},
#  {"type": "log-normal", "mu": 0, "sigma": 1, "min": 0, "scale": 1}
#  {"type": "log-uniform", "min": 0, "max": 1, "scale": 1}
risk_distribution = {"type": "log-uniform", "min": 0, "max": 1, "scale": 3}


# Monte Carlo risk estimator should be one of the following types :
# {"type": "entropic", "eps": 1e-4}
# {"type": "cvar", "eps": 1e-4}

risk_estimator = {"type": "cvar", "eps": 1e-3}
if latent_distribution == "quantized":
    # Number of samples used to estimate the risk from the unbiased distribution
    n_mc_samples_risk = num_vq
    # Number of samples used to estimate the averaged cost of the biased distribution
    n_mc_samples_biased = num_vq
else:
    # Number of samples used to estimate the risk from the unbiased distribution
    n_mc_samples_risk = 512
    # Number of samples used to estimate the averaged cost of the biased distribution
    n_mc_samples_biased = 256


# Risk Loss Hyperparameters
risk_weight = 1
risk_assymetry_factor = 200
use_risk_constraint = True  # For encoder_biased only
risk_constraint_update_every_n_epoch = (
    1  # For encoder_biased only, not used if use_risk_constraint == False
)
risk_constraint_weight_update_factor = (
    1.5  # For encoder_biased only, not used if use_risk_constraint == False
)
risk_constraint_weight_maximum = (
    1e5  # For encoder_biased only, not used if use_risk_constraint == False
)


# Training Hyperparameters
learning_rate = 1e-4
batch_size = 512
num_epochs_cvae = 100
num_epochs_bias = 100
gpus = [0]
seed = 0  # Give an integer value to seed will set seed for pseudo-random number generators in: pytorch, numpy, python.random
early_stopping = False
accumulate_grad_batches = 1

num_workers = 4
log_weights_and_grads = False
num_samples_min_fde = 16
val_check_interval_epoch = 1
plot_interval_epoch = 1
histogram_interval_epoch = 1

# State Cost Hyperparameters
cost_scale = 10
cost_reduce = (
    "mean"  # choose in "discounted_mean", "mean", "min", "max", "now", "final"
)
discount_factor = 0.95  # only used if cost_reduce == "discounted_mean", discounts the cost by this factor at each time step
distance_bandwidth = 2
time_bandwidth = 0.5
min_velocity_diff = 0.03


# List all above parameters that make a difference in the dataset to distringuish datasets once generated
dataset_parameters = {
    "dt": dt,
    "time_scene": time_scene,
    "sample_times": sample_times,
    "ego_ref_speed": ego_ref_speed,
    "ego_speed_init_low": ego_speed_init_low,
    "ego_speed_init_high": ego_speed_init_high,
    "ego_acceleration_mean_low": ego_acceleration_mean_low,
    "ego_acceleration_mean_high": ego_acceleration_mean_high,
    "ego_acceleration_std": ego_acceleration_std,
    "fast_speed": fast_speed,
    "slow_speed": slow_speed,
    "p_change_pace": p_change_pace,
    "proportion_fast": proportion_fast,
    "file_name": file_name,
    "datasets_sizes": datasets_sizes,
    "state_dim": state_dim,
    "num_steps": num_steps,
    "num_steps_future": num_steps_future,
    "perception_noise_std": perception_noise_std,
}

# List files that should be saved as log
files_to_log = ["risk_biased/utils/loss.py"]
