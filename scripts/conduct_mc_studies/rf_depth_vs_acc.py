OUTPUT_DIR = "experiments/experiment_analysis/rf_es_simulation_study/overview_csvs"
TEST_RUN = False

if TEST_RUN:
    MC_ITER = 1  # default value
    RANDOM_SEED = 7
    MAX_DEPTH = 45  # Maximum depth to try
    FILE_NAME = "scikit_rf_depth_vs_acc_test.csv"
else:
    MC_ITER = 300
    RANDOM_SEED = 7
    MAX_DEPTH = 35  # Maximum depth to try
    FILE_NAME = "scikit_rf_depth_vs_acc.csv"
N_ESTIMATORS = 100

import csv
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import infrastructure.config_reader as config_reader
import infrastructure.dgp.data_generation as data_generation
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

DGP_NAME_MAP = {
    "additive_model_I": "Add. Het.",
    "additive_sparse_jump": "Add. Jump",
    "additive_sparse_smooth": "Add. Smooth",
    "hierarchical-interaction_sparse_jump": "Add. H.I. Jump",
    "circular": "Circular",
    "rectangular": "Rectangular",
    "smooth_signal": "Circular Smooth",
    "sine_cosine": "Sine Cosine",
}


def run_mc_iteration(mc_iteration, dgp_config):
    random_state = RANDOM_SEED + mc_iteration
    X_train, X_test, y_train, y_test, f_train, f_test = (
        data_generation.generate_X_y_f_classification(
            random_state=random_state,
            n_ticks_per_ax_meshgrid=dgp_config.get(
                "n_ticks_per_ax_meshgrid"
            ),  # for 2 dim cases, n_samples
            dgp_name=dgp_config.get("dgp_name"),  # indicating which DGP to use
            bernoulli_p=dgp_config.get("bernoulli_p"),  # for 2 dim cases
            n_samples=dgp_config.get(
                "n_samples"
            ),  # size of dateset (train + test), in additive cases!
            feature_dim=dgp_config.get("feature_dim"),
        )
    )

    train_accuracies_per_depth = np.full(MAX_DEPTH, np.nan)
    test_accuracies_per_depth = np.full(MAX_DEPTH, np.nan)
    actual_depths_per_depth = np.full(MAX_DEPTH, np.nan)  # New array for actual depths

    # Train separate trees for each depth
    for depth in range(1, MAX_DEPTH + 1):
        sklearn_rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=depth,
            max_features="sqrt",
            random_state=RANDOM_SEED,
        )
        sklearn_rf.fit(X_train, y_train)
        train_accuracy = np.mean(sklearn_rf.predict(X_train) == y_train)
        test_accuracy = np.mean(sklearn_rf.predict(X_test) == y_test)
        # Get the maximum depth across all trees in the forest
        actual_depth = max(
            estimator.get_depth() for estimator in sklearn_rf.estimators_
        )

        train_accuracies_per_depth[depth - 1] = train_accuracy
        test_accuracies_per_depth[depth - 1] = test_accuracy
        actual_depths_per_depth[depth - 1] = actual_depth

    return (
        train_accuracies_per_depth,
        test_accuracies_per_depth,
        actual_depths_per_depth,  # Add actual depths to return tuple
    )


def main():
    # Get the absolute path to the config directory relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    dgp_config_dir = os.path.join(
        project_root,
        "experiments",
        "experiment_configs",
        "simulated_data_configs",
        "rf/depth_vs_acc",
    )
    all_results = []

    # Process all yaml files in the directory
    for file in os.listdir(dgp_config_dir):
        if file.endswith(".yaml"):
            dgp_config_path = os.path.join(dgp_config_dir, file)
            print(f"Processing {dgp_config_path}")

            # Set Up data with config file
            dgp_config = config_reader.load_config(dgp_config_path)

            # Initialize arrays to store all MC results
            all_train_accuracies = np.zeros((MC_ITER, MAX_DEPTH))
            all_test_accuracies = np.zeros((MC_ITER, MAX_DEPTH))
            all_actual_depths = np.zeros((MC_ITER, MAX_DEPTH))

            # Run MC iterations in parallel
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(run_mc_iteration, mc_iteration, dgp_config)
                    for mc_iteration in range(MC_ITER)
                ]

                for i, future in enumerate(
                    tqdm(as_completed(futures), total=MC_ITER, desc="MC Iterations")
                ):
                    (
                        train_accuracies_per_depth,
                        test_accuracies_per_depth,
                        actual_depths_per_depth,
                    ) = future.result()

                    # Store results for each iteration
                    all_train_accuracies[i] = train_accuracies_per_depth
                    all_test_accuracies[i] = test_accuracies_per_depth
                    all_actual_depths[i] = actual_depths_per_depth

            # Compute medians across all MC iterations
            mc_train_accuracies_per_depth = np.nanmedian(all_train_accuracies, axis=0)
            mc_test_accuracies_per_depth = np.nanmedian(all_test_accuracies, axis=0)
            mc_actual_depths_per_depth = np.nanmedian(all_actual_depths, axis=0)

            # Prepare results dictionary
            results = {
                "dataset": DGP_NAME_MAP.get(
                    dgp_config.get("dgp_name"), dgp_config.get("dgp_name")
                ),
                "feature_dim": dgp_config.get("feature_dim"),
                "n_samples": dgp_config.get("n_samples"),
                "mc_iterations": MC_ITER,
            }

            # Add metrics for each depth
            for depth in range(1, MAX_DEPTH + 1):
                if not np.isnan(mc_train_accuracies_per_depth[depth - 1]):
                    results.update(
                        {
                            f"Train Accuracy for Depth {depth}": mc_train_accuracies_per_depth[
                                depth - 1
                            ],
                            f"Test Accuracy for Depth {depth}": mc_test_accuracies_per_depth[
                                depth - 1
                            ],
                            f"Depth {depth} Actual Depth": mc_actual_depths_per_depth[
                                depth - 1
                            ],
                        }
                    )

            # Format float values
            results = {
                key: f"{value:.3f}" if isinstance(value, float) else value
                for key, value in results.items()
            }

            all_results.append(results)
    all_results_df = pd.DataFrame(all_results)
    depth_columns = [k for k in all_results_df.keys() if "Actual Depth" in k]
    max_depths = (
        all_results_df[depth_columns].astype(float).max(axis=1).round().astype(int)
    )
    # remove depth columns from dataframe
    all_results_df = all_results_df.drop(columns=depth_columns)
    # Get columns with numbers in their names
    numeric_cols = [
        col for col in all_results_df.columns if any(c.isdigit() for c in col)
    ]

    # For each row, set values to NaN if column number > max_depth
    for idx, max_depth in enumerate(max_depths):
        for col in numeric_cols:
            col_num = int("".join(filter(str.isdigit, col)))
            if col_num > max_depth:
                all_results_df.loc[idx, col] = np.nan

    # Store all results in a single CSV file
    output_dir = os.path.join(
        project_root,
        "experiments",
        "experiment_analysis",
        "rf_es_simulation_study",
        "overview_csvs",
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, FILE_NAME)

    if not all_results_df.empty:
        all_results_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()
