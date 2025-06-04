import itertools
import pandas as pd

# Read the CSV file
df = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv"
)
# in the dataset column, rename "Smooth Signal" to "Circular Smooth" if applicable
df.loc[df["dataset"] == "Smooth Signal", "dataset"] = "Circular Smooth"
df = df[df["mc_iterations"] == 300]
# Remove algorithm MD
df = df[df["algorithm"] != "MD"]


def create_pivot_table(
    df,
    dgp_config: str = "standard",
    feature_dim: int | list = None,
    n_train_samples: int | list = None,
    metric_column: str = "N Leaves",
    mean_type: str = "median",
    round_to: int = 2,
) -> pd.DataFrame:
    """
    Creates a pivot table from the input DataFrame with specific filtering conditions.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data
        dgp_config (str): DGP config folder to filter by. Options: "standard", "expanded", "single"
        feature_dim (int | list): Feature dimension(s) to include. For standard config, specifies dimension to exclude
        n_train_samples (int | list): Number of training samples to filter by
        metric_column (str): Metric to analyze. Options:
            - "Test Accuracy"
            - "Test Log Loss"
            - "Test Matthews"
            - "Depth"
            - "N Leaves"
            - "fit_duration"
        mean_type (str): Type of mean to use. Options: "mean", "median"
        round_to (int): Number of decimal places to round to

    Returns:
        pd.DataFrame: Pivot table with datasets as index and algorithms as columns
    """
    # Validate inputs
    valid_dgp_configs = ["standard", "expanded", "single"]
    if dgp_config not in valid_dgp_configs:
        raise ValueError(f"dgp_config must be one of {valid_dgp_configs}")

    valid_metrics = [
        "Test Accuracy",
        "Test Log Loss",
        "Test Matthews",
        "Test F1",
        "Depth",
        "N Leaves",
        "fit_duration",
    ]
    if metric_column not in valid_metrics:
        raise ValueError(f"metric_column must be one of {valid_metrics}")

    valid_mean_types = ["mean", "median"]
    if mean_type not in valid_mean_types:
        raise ValueError(f"mean_type must be one of {valid_mean_types}")

    # Create a deep copy of the DataFrame
    df_copy = df.copy(deep=True)

    # Apply filters based on dgp_config
    df_copy = df_copy[df_copy["dgp_config_folder"] == dgp_config]

    # Apply dimension and sample size filters
    if dgp_config == "standard":
        # For standard config, exclude the specified feature_dim
        df_copy = df_copy[df_copy["feature_dim"] != feature_dim]
    else:
        # For expanded/single config, include only the specified feature_dim
        df_copy = df_copy[df_copy["feature_dim"] == feature_dim]

    # Filter by n_train_samples if specified
    if n_train_samples is not None:
        df_copy = df_copy[df_copy["n_train_samples"] == n_train_samples]

    # Remove specific algorithm combinations
    df_copy = df_copy[
        ~(
            (df_copy["algorithm"].isin(["TS", "TS*"]))
            & (df_copy["full_alpha_range"] == 1.0)
        )
    ]
    df_copy = df_copy[
        ~(
            (df_copy["algorithm"].isin(["CCP", "CCP*"]))
            & (df_copy["full_alpha_range"] == 1.0)
        )
    ]

    # Create metric column name
    metric_column_name = f"{metric_column} ({mean_type})"
    depth_column_name = f"Depth ({mean_type})"

    # Round values appropriately
    if "Depth" in metric_column_name or "N Leaves" in metric_column_name:
        df_copy[metric_column_name] = df_copy[metric_column_name].astype(int)
        df_copy[depth_column_name] = df_copy[depth_column_name].astype(int)
    else:
        df_copy[metric_column_name] = df_copy[metric_column_name].round(round_to)

    # Define desired_order based on metric_column
    if metric_column == "fit_duration":
        desired_order = ["CCP", "ES", "TS"]
    else:
        desired_order = ["CCP", "ES*", "ES", "TS*", "TS"]

    # Create two pivot tables if metric is N Leaves
    if metric_column == "N Leaves":
        leaves_dt = df_copy.pivot(
            index="dataset", columns="algorithm", values=metric_column_name
        )
        depth_dt = df_copy.pivot(
            index="dataset", columns="algorithm", values=depth_column_name
        )

        # Combine the values
        final_dt = leaves_dt.copy()
        for col in final_dt.columns:
            final_dt[col] = (
                leaves_dt[col].astype(str) + "(" + depth_dt[col].astype(str) + ")"
            )
    elif metric_column == "fit_duration":
        # Create separate pivot tables for regular and star versions
        regular_dt = df_copy.pivot(
            index="dataset", columns="algorithm", values=metric_column_name
        )

        # Combine ES/ES* and TS/TS* values
        final_dt = pd.DataFrame(index=regular_dt.index)
        final_dt["CCP"] = regular_dt["CCP"]
        final_dt["ES"] = (
            regular_dt["ES"].astype(str) + "(" + regular_dt["ES*"].astype(str) + ")"
        )
        final_dt["TS"] = (
            regular_dt["TS"].astype(str) + "(" + regular_dt["TS*"].astype(str) + ")"
        )

        # Add speedup comparison columns
        final_dt["ES vs. CCP"] = regular_dt["CCP"] / regular_dt["ES"]
        final_dt["TS vs. CCP"] = regular_dt["CCP"] / regular_dt["TS"]

    else:
        # Handle duplicate entries by keeping row with higher Test Accuracy
        duplicates = (
            df_copy.groupby(["dataset", "algorithm"]).size().reset_index(name="count")
        )
        duplicates = duplicates[duplicates["count"] > 1]
        if not duplicates.empty:
            print("\nRemoving duplicate entries, keeping higher Test Accuracy rows")
            for _, row in duplicates.iterrows():
                dataset, algo = row["dataset"], row["algorithm"]
                dupe_mask = (df_copy["dataset"] == dataset) & (
                    df_copy["algorithm"] == algo
                )
                dupe_rows = df_copy[dupe_mask]

                # Keep the row with higher Test Accuracy
                best_idx = dupe_rows["Test Accuracy (median)"].idxmax()
                df_copy = df_copy.drop(dupe_rows.index[dupe_rows.index != best_idx])

        final_dt = df_copy.pivot(
            index="dataset", columns="algorithm", values=metric_column_name
        )

    # Reorder columns to match desired order
    if metric_column == "fit_duration":
        desired_order = ["CCP", "ES", "TS", "ES vs. CCP", "TS vs. CCP"]
    else:
        desired_order = ["CCP", "ES*", "ES", "TS*", "TS"]
    final_dt = final_dt[desired_order]

    print(
        f"n_train_samples: {df_copy['n_train_samples'].unique()}, d={df_copy['feature_dim'].unique()}"
    )
    print(final_dt)

    # Create LaTeX table with specific formatting
    def create_latex_row(row):
        if metric_column == "N Leaves":
            # For N Leaves, extract the numeric values before parentheses for comparison
            numeric_vals = [int(val.split("(")[0]) for val in row]
            min_val = min(numeric_vals)
            # Format with bold if it's the minimum value
            formatted_vals = [
                (
                    f"\\textbf{{{val.split('(')[0]}}} ({val.split('(')[1]}"
                    if int(val.split("(")[0]) == min_val
                    else f"{val.split('(')[0]} ({val.split('(')[1]}"
                )
                for val in row
            ]
            return " & ".join([row.name] + formatted_vals) + " \\\\"
        elif metric_column == "fit_duration":
            # Extract the numeric values before parentheses for comparison
            numeric_vals = []
            for val in row[:3]:  # Only look at the first three columns (CCP, ES, TS)
                if "(" in str(val):
                    numeric_vals.append(float(str(val).split("(")[0]))
                else:
                    numeric_vals.append(float(val))

            min_val = min(numeric_vals)
            formatted_vals = []

            # Format the timing columns (first three columns)
            for i, val in enumerate(row[:3]):
                if numeric_vals[i] == min_val:
                    if "(" in str(val):
                        base, star = str(val).split("(")
                        formatted_vals.append(
                            f"\\textbf{{{float(base):.{round_to}f}}} ({float(star[:-1]):.{round_to}f})"
                        )
                    else:
                        formatted_vals.append(f"\\textbf{{{float(val):.{round_to}f}}}")
                else:
                    if "(" in str(val):
                        base, star = str(val).split("(")
                        formatted_vals.append(
                            f"{float(base):.{round_to}f} ({float(star[:-1]):.{round_to}f})"
                        )
                    else:
                        formatted_vals.append(f"{float(val):.{round_to}f}")

            # Format the speedup columns (last two columns) with no decimals
            for val in row[3:]:
                formatted_vals.append(f"{int(round(float(val)))}x")

            return " & ".join([row.name] + formatted_vals) + " \\\\"
        elif metric_column in ["Test Log Loss", "fit_duration"]:
            min_val = row.min()  # Use minimum for these metrics
            formatted_vals = [
                (
                    f"\\textbf{{{val:.{round_to}f}}}"
                    if val == min_val
                    else f"{val:.{round_to}f}"
                )
                for val in row
            ]
            return " & ".join([row.name] + formatted_vals) + " \\\\"
        else:
            max_val = row.max()  # Use maximum for other metrics
            formatted_vals = [
                (
                    f"\\textbf{{{val:.{round_to}f}}}"
                    if val == max_val
                    else f"{val:.{round_to}f}"
                )
                for val in row
            ]
            return " & ".join([row.name] + formatted_vals) + " \\\\"

    latex_rows = final_dt.apply(create_latex_row, axis=1)

    # Create the complete LaTeX table
    latex_table = [
        "\\begin{tabular}{l" + "c" * len(final_dt.columns) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(final_dt.columns) + " \\\\",
        "\\midrule",
        *latex_rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]

    print("\nLaTeX Table:")
    print("\n".join(latex_table))
    # Save the LaTeX table to a file
    table_name = f"dt_{metric_column_name.replace(' ', '_')}_n{df_copy['n_train_samples'].unique()[0]}_d{'_'.join(map(str, df_copy['feature_dim'].unique()))}"
    table_path = f"experiments/experiment_analysis/dt_es_simulation_study/latex_tables/{dgp_config}/{table_name}.tex"

    # Create directories if they don't exist
    import os

    os.makedirs(os.path.dirname(table_path), exist_ok=True)

    # Write the table to file
    with open(table_path, "w") as f:
        f.write("\n".join(latex_table))
    return final_dt


# Standard Scenarios
valid_metrics = [
    "Test Log Loss",
    "Test Matthews",
    "N Leaves",
    "fit_duration",
    "Test F1",
]

# Standard scenarios (excluding dim 5)
for metric_column in valid_metrics:
    result = create_pivot_table(
        df,
        dgp_config="standard",
        feature_dim=5,  # dimension to exclude
        metric_column=metric_column,
        mean_type="median",
        round_to=2,
    )

# Expanded Scenarios - Small Sample Size
result = create_pivot_table(
    df,
    dgp_config="expanded",
    feature_dim=5,  # fixed dimension
    n_train_samples=100,
    metric_column="Test Matthews",
    mean_type="median",
    round_to=2,
)

# Expanded Scenarios - Sparsity Analysis
feature_dims = [5, 10, 30, 100]
n_samples = 1000
for dim in feature_dims:
    for metric in ["Test Matthews", "Depth"]:
        print(dim)
        result = create_pivot_table(
            df,
            dgp_config="expanded",
            feature_dim=dim,
            n_train_samples=n_samples,
            metric_column=metric,
            mean_type="median",
            round_to=2,
        )

# Runtime Analysis
feature_dim = 30
sample_sizes = [5000, 10000]
for n_samples in sample_sizes:
    result = create_pivot_table(
        df,
        dgp_config="single",
        feature_dim=feature_dim,
        n_train_samples=n_samples,
        metric_column="fit_duration",
        mean_type="median",
        round_to=2,
    )
