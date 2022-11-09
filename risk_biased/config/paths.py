# Data split path:
base_path = "<set your path to the pre-processed data directory here>"
data_dir = "interactive_veh_type"  # This directory name conditions the model hyperparameters, make sure to set it correctly
sample_dataset_path = base_path + data_dir + "/sample"
val_dataset_path = base_path + data_dir + "/validation"
train_dataset_path = base_path + data_dir + "/training"
test_dataset_path = base_path + data_dir + "/sample"

log_path = "<set your path to any directory where you want the logs to be stored>"
