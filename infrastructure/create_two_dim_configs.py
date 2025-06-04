import itertools
import os
import yaml

# Define options for each attribute
dgp_names = ["circular", "rectangular", "sine_cosine", "smooth_signal"]

# First combination: feature_dim=5, n_samples=[200, 2000]

combinations_0 = itertools.product(dgp_names, [2, 5], [2000], [None], [0.8])

# Define the folder path for configs
folder_path = "experiments/experiment_configs/simulated_data_configs/standard/"
# make sure this folder exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Generate YAML config files for all combinations
for combinations in [combinations_0]:
    for dgp_name, feature_dim, n_samples, n_ticks, b_p in combinations:
        # Define the configuration dictionary
        config = {
            "dgp_name": dgp_name,
            "n_ticks_per_ax_meshgrid": n_ticks,
            "feature_dim": feature_dim,
            "n_samples": n_samples,
            "bernoulli_p": b_p,
        }

        # Create the filename
        filename = f"{folder_path}{dgp_name}_feature_dim_{feature_dim}_n_samples_{n_samples}_bernoulli_p_{b_p}.yaml"

        # Write the config to a YAML file
        with open(filename, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Created {filename}")

# Define the folder path for configs
folder_path = "experiments/experiment_configs/simulated_data_configs/expanded/"

# Robustness in small sample size
combinations_1 = itertools.product(dgp_names, [5], [200], [None], [0.8])

# Play with feature dim, to observe if signal type is important or dimension
combinations_2 = itertools.product(dgp_names, [10, 30, 100], [2000], [None], [0.8])

# Decrease  noise!
combinations_3 = itertools.product(dgp_names, [30], [2000], [None], [1.0])


# Create directory if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Generate YAML config files for all combinations
for combinations in [combinations_1, combinations_2, combinations_3]:
    for dgp_name, feature_dim, n_samples, n_ticks, b_p in combinations:
        # Define the configuration dictionary
        config = {
            "dgp_name": dgp_name,
            "n_ticks_per_ax_meshgrid": n_ticks,
            "feature_dim": feature_dim,
            "n_samples": n_samples,
            "bernoulli_p": b_p,
        }

        # Create the filename
        filename = f"{folder_path}{dgp_name}_feature_dim_{feature_dim}_n_samples_{n_samples}_bernoulli_p_{b_p}.yaml"

        # Write the config to a YAML file
        with open(filename, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Created {filename}")


### RF
# Define options for each attribute
dgp_names = ["circular", "rectangular", "sine_cosine", "smooth_signal"]

# First combination: feature_dim=5, n_samples=[200, 2000]

combinations_0 = itertools.product(dgp_names, [10], [2000], [None], [0.8])

# Define the folder path for configs
folder_path = "experiments/experiment_configs/simulated_data_configs/rf/standard/"
# make sure this folder exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Generate YAML config files for all combinations
for combinations in [combinations_0]:
    for dgp_name, feature_dim, n_samples, n_ticks, b_p in combinations:
        # Define the configuration dictionary
        config = {
            "dgp_name": dgp_name,
            "n_ticks_per_ax_meshgrid": n_ticks,
            "feature_dim": feature_dim,
            "n_samples": n_samples,
            "bernoulli_p": b_p,
        }

        # Create the filename
        filename = f"{folder_path}{dgp_name}_feature_dim_{feature_dim}_n_samples_{n_samples}_bernoulli_p_{b_p}.yaml"

        # Write the config to a YAML file
        with open(filename, "w") as file:
            yaml.dump(config, file, default_flow_style=False)

        print(f"Created {filename}")
