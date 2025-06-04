import os
import pandas as pd
import argparse

# Add flag at the beginning
RATMIR = False


def main(
    directory, folder_name, output_filename, average_type, average_type_provided=False
):
    """
    directory: str - path to the parent directory containing the csv files
    folder_name: str - name of the folder to store the output csv file in
    output_filename: str - name of the output csv file
    average_type: str - type of average to use ("mean" or "median")
    average_type_provided: bool - whether the average_type was explicitly provided
    """
    # List to store the dataframes
    dfs = []
    # Move AVERAGE_TYPE from global to local
    AVERAGE_TYPE = average_type
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".csv"):
                # Construct the full file path
                csv_path = os.path.join(root, filename)
                # Read the csv file into a pandas dataframe
                df = pd.read_csv(csv_path)
                # check if df is empty, if os set a breakpoint
                if df.empty:
                    pass
                else:
                    # Add the dataframe to the list
                    dfs.append(df)

    # Concatenate the dataframes row-wise
    result_df = pd.concat(dfs, ignore_index=True)

    # Filter columns based on AVERAGE_TYPE
    if AVERAGE_TYPE == "mean" and RATMIR:
        result_df = result_df.loc[
            :, ~result_df.columns.str.contains(r"\(median\)", case=False, na=False)
        ]
    elif AVERAGE_TYPE == "median" and RATMIR:
        result_df = result_df.loc[
            :, ~result_df.columns.str.contains(r"\(mean\)", case=False, na=False)
        ]

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
    result_df["dataset"] = result_df["dataset"].map(DGP_NAME_MAP)

    if RATMIR:
        columns_to_keep = [
            "dataset",
            "algo_config",
            "method",
            "feature_dim",
            "n_samples",
            "mc_iterations",
            "max_features",
            "kappa",
            "es_offset",
            "estimate_noise_before_sampling",
        ]
        columns_to_keep_is_in = [
            "fit_duration",
            "train_acc",
            "test_acc",
            "n_leaves",
            "depth",
        ]

        # Filter columns based on the criteria
        kept_columns = []
        for col in result_df.columns:
            # Keep if column name is in columns_to_keep
            if col in columns_to_keep:
                kept_columns.append(col)
            # Keep if any string in columns_to_keep_is_in is in column name
            elif any(keep_str in col for keep_str in columns_to_keep_is_in):
                kept_columns.append(col)

        # Filter the dataframe to keep only the selected columns
        result_df = result_df[kept_columns]

        # Drop columns that contain 'sd'
        result_df = result_df.loc[:, ~result_df.columns.str.contains("sd", case=False)]
        # Drop columns that contain 'acc_' followed by a number
        result_df = result_df.loc[
            :,
            ~(
                result_df.columns.str.contains(r"_test_acc", case=False)
                | result_df.columns.str.contains(r"_train_acc", case=False)
                | result_df.columns.str.contains(r"acc_\d", case=False)
                | result_df.columns.str.contains(r"sd_acc ", case=False)
            ),
        ]

    # Determine file suffix based on whether average_type was explicitly provided
    file_suffix = average_type if average_type_provided else "mean_median"

    os.makedirs(
        f"experiments/experiment_analysis/{folder_name}/overview_csvs",
        exist_ok=True,
    )
    result_df.to_csv(
        f"experiments/experiment_analysis/{folder_name}/overview_csvs/{output_filename}_{file_suffix}.csv",
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate overview csv file from multiple csv files"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="experiments/experiment_raw_results/dt_es_simulation_study",
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--output_folder_name",
        type=str,
        default="dt_es_simulation_study",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="results_df_burr",
    )
    parser.add_argument(
        "--average_type",
        type=str,
        default="median",
        choices=["mean", "median"],
        help="Type of average to use (mean or median)",
    )

    # Track if average_type was explicitly provided
    average_type_provided = False
    for arg in parser._actions:
        if (
            arg.dest == "average_type"
            and arg.default != parser.parse_args().average_type
        ):
            average_type_provided = True
            break

    args = parser.parse_args()
    main(
        args.directory,
        args.output_folder_name,
        args.output_file_name,
        args.average_type,
        average_type_provided,
    )
