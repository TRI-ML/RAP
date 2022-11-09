from risk_biased.config.paths import (
    data_dir,
    sample_dataset_path,
    val_dataset_path,
    train_dataset_path,
    test_dataset_path,
    log_path,
)

# Data augmentation:
normalize_angle = True
random_rotation = False
angle_std = 3.14 / 4
random_translation = False
translation_distance_std = 0.1
p_exchange_two_first = 0.5

# Data diminution:
min_num_observation = 2
max_size_lane = 50
train_dataset_size_limit = None
val_dataset_size_limit = None
max_num_agents = 50
max_num_objects = 50

# Data caracterization:
time_scene = 9.1
dt = 0.1
num_steps = 11
num_steps_future = 80

# TODO: avoid conditioning on the name of the directory in the path
if data_dir == "interactive_veh_type":
    map_state_dim = 2 + num_steps * 8
    state_dim = 11
    dynamic_state_dim = 5
elif data_dir == "interactive_full":
    map_state_dim = 2
    state_dim = 5
    dynamic_state_dim = 5
else:
    map_state_dim = 2
    state_dim = 2
    dynamic_state_dim = 2

# Variational Loss Hyperparameters
kl_weight = 1.0
kl_threshold = 0.01

# Training Parameters
learning_rate = 3e-4
batch_size = 64
accumulate_grad_batches = 2
num_epochs_cvae = 0
num_epochs_bias = 100
gpus = [1]
seed = 0  # Give an integer value to seed will set seed for pseudo-random number generators in: pytorch, numpy, python.random
num_workers = 8

# Model hyperparameter
model_type = "interaction_biased"
condition_on_ego_future = False
latent_dim = 16
hidden_dim = 128
feature_dim = 16
num_vq = 512
latent_distribution = "gaussian"  # "gaussian" or "quantized"
is_mlp_residual = True
num_hidden_layers = 3
num_blocks = 3
interaction_type = "Attention"  # one of "ContextGating", "Attention", "Hybrid"
## MCG parameters
mcg_dim_expansion = 2
mcg_num_layers = 0
## Attention parameters
num_attention_heads = 4
sequence_encoder_type = "MLP"  # one of "MLP", "LSTM", "maskedLSTM"
sequence_decoder_type = "MLP"  # one of "MLP", "LSTM"


# Risk Loss Hyperparameters
cost_reduce = "discounted_mean"  # choose in "discounted_mean", "mean", "min", "max", "now", "final"
discount_factor = 0.95  # only used if cost_reduce == "discounted_mean", discounts the cost by this factor at each time step
min_velocity_diff = 0.1
n_mc_samples_risk = 32
n_mc_samples_biased = 16
risk_weight = 1
risk_assymetry_factor = 30
use_risk_constraint = True  # For encoder_biased only
risk_constraint_update_every_n_epoch = (
    1  # For encoder_biased only, not used if use_risk_constraint == False
)
risk_constraint_weight_update_factor = (
    1.5  # For encoder_biased only, not used if use_risk_constraint == False
)
risk_constraint_weight_maximum = (
    1000  # For encoder_biased only, not used if use_risk_constraint == False
)

# List files that should be saved as log
files_to_log = [
    "./risk_biased/models/biased_cvae_model.py",
    "./risk_biased/models/latent_distributions.py",
]
