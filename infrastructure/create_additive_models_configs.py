import itertools
import os
import yaml

# Define options for each attribute
dgp_names = ["additive_model_I", "additive_sparse_smooth", "additive_sparse_jump", "hierarchical-interaction_sparse_jump"]
feature_dims = [30]
n_samples_options = [2000]

# Generate all combinations of the three attributes
combinations = itertools.product(dgp_names, feature_dims, n_samples_options)

# Generate YAML config files
# Define the folder path
folder_path = "experiments/experiment_configs/simulated_data_configs/standard/"
# make sure this folder exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Update the filename line to include the folder path
for dgp_name, feature_dim, n_samples in combinations:
    # Define the configuration dictionary
    config = {
        "dgp_name": dgp_name,
        "feature_dim": feature_dim,
        "n_samples": n_samples
    }
    
    # Create the filename based on the key-value pairs
    filename = f"{folder_path}{dgp_name}_feature_dim_{feature_dim}_n_samples_{n_samples}.yaml"
    
    # Write the config to a YAML file
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Created {filename}")



# Play with feature dim, to observe if signal type is important or dimension
dgp_names = ["additive_model_I", "additive_sparse_smooth", "additive_sparse_jump", "hierarchical-interaction_sparse_jump"]
feature_dims = [5, 10, 100]
n_samples_options = [2000]
combinations_I = itertools.product(dgp_names, feature_dims, n_samples_options)

# Generate YAML config files
# Define the folder path
folder_path = "experiments/experiment_configs/simulated_data_configs/expanded/"
# if not exist, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Update the filename line to include the folder path
for dgp_name, feature_dim, n_samples in combinations_I:
    # Define the configuration dictionary
    config = {
        "dgp_name": dgp_name,
        "feature_dim": feature_dim,
        "n_samples": n_samples
    }
    
    # Create the filename based on the key-value pairs
    filename = f"{folder_path}{dgp_name}_feature_dim_{feature_dim}_n_samples_{n_samples}.yaml"
    
    # Write the config to a YAML file
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Created {filename}")

# Run Time
dgp_names = ["additive_sparse_smooth"]
feature_dims = [100]
n_samples_options = [10000, 20000, 30000]
combinations_II = itertools.product(dgp_names, feature_dims, n_samples_options)

# Generate YAML config files
# Define the folder path
folder_path = "experiments/experiment_configs/simulated_data_configs/runtimes/rf/"
# if not exist, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Update the filename line to include the folder path
for dgp_name, feature_dim, n_samples in combinations_II:
    # Define the configuration dictionary
    config = {
        "dgp_name": dgp_name,
        "feature_dim": feature_dim,
        "n_samples": n_samples
    }
    
    # Create the filename based on the key-value pairs
    filename = f"{folder_path}{dgp_name}_feature_dim_{feature_dim}_n_samples_{n_samples}.yaml"
    
    # Write the config to a YAML file
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Created {filename}")

# Run Time 
dgp_names = ["additive_sparse_smooth"]
feature_dims = [30]
n_samples_options = [10000, 20000] #100000]
combinations_III = itertools.product(dgp_names, feature_dims, n_samples_options)

# Generate YAML config files
# Define the folder path
folder_path = "experiments/experiment_configs/simulated_data_configs/runtimes/single/"
# if not exist, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Update the filename line to include the folder path
for dgp_name, feature_dim, n_samples in combinations_III:
    # Define the configuration dictionary
    config = {
        "dgp_name": dgp_name,
        "feature_dim": feature_dim,
        "n_samples": n_samples
    }
    
    # Create the filename based on the key-value pairs
    filename = f"{folder_path}{dgp_name}_feature_dim_{feature_dim}_n_samples_{n_samples}.yaml"
    
    # Write the config to a YAML file
    with open(filename, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Created {filename}")