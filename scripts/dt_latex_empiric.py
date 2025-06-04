import pandas as pd
import math

# Flag to control inclusion of MD method in tables
INCLUDE_MD = False

# Load the results
results = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_empirical_study.csv"
)
# rename the value in the column "Dataset" to "Wisc. Cancer"
results.loc[results["dataset"] == "Wisc. Breast Cancer", "dataset"] = "Wisc. Cancer"

# Create a summary DataFrame with one row per dataset
dataset_summary = results.drop_duplicates(subset=["dataset"])[
    ["dataset", "class_1_fraction", "n_samples", "n_features", "estimated_noise"]
]

# Sort by estimated noise in ascending order
dataset_summary = dataset_summary.sort_values(by="estimated_noise").reset_index(
    drop=True
)

# Get the ordered list of datasets to preserve this order in all tables
ordered_datasets = dataset_summary["dataset"].tolist()

# Rename columns for the table
dataset_summary = dataset_summary.rename(
    columns={
        "dataset": "Dataset",
        "class_1_fraction": "Class Ratio",
        "n_samples": "Samples",
        "n_features": "Features",
        "estimated_noise": "\(\hat{\sigma}^2\)",
    }
)

# Create the LaTeX table
latex_table = "\\begin{tabular}{l" + "c" * 4 + "}\n\\toprule\n"

# Table header
header = "Dataset & Class Ratio & Samples & Features & \(\hat{\sigma}^2\)"
latex_table += header + " \\\\\n\\midrule\n"

for _, row in dataset_summary.iterrows():
    sigma_value = row["\(\hat{\sigma}^2\)"]
    row_str = f"{row['Dataset']} & {row['Class Ratio']} & {row['Samples']} & {row['Features']} & {sigma_value}"
    latex_table += row_str + " \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}"

print(latex_table)

# Optionally save to file
with open("latex/thesis/tables/dt_empirical_characteristics.tex", "w") as f:
    f.write(latex_table)


# train_acc_mean, test_acc_mean, test_mcc_mean, test_f1_mean, test_log_loss_mean, depth_mean, n_leaves_mean
# Create a pivot table with test accuracies
test_metric = "test_acc_mean"
methods = ["CCP", "TS", "ES"]
if INCLUDE_MD:
    methods.append("MD")

test_acc_pivot = results.pivot(
    index="dataset",
    columns="method",
    values=test_metric,
)[methods]

# Reorder the pivot table to match the noise-based ordering
test_acc_pivot = test_acc_pivot.reindex(ordered_datasets)

# Create the LaTeX table for test accuracies
latex_acc_table = "\\begin{tabular}{l" + "c" * len(methods) + "}\n\\toprule\n"
latex_acc_table += "Dataset & " + " & ".join(methods) + " \\\\\n\\midrule\n"

# For each dataset, add a row and make the maximum value bold
for dataset, row in test_acc_pivot.iterrows():
    max_value = row.max()
    formatted_values = []

    for method in methods:  # Use the methods list we defined earlier
        value = row[method]
        if value == max_value:
            # Format as bold and round to 3 decimal places
            formatted_values.append(f"\\textbf{{{value:.2f}}}")
        else:
            # Just round to 3 decimal places
            formatted_values.append(f"{value:.2f}")

    latex_acc_table += f"{dataset} & {' & '.join(formatted_values)} \\\\\n"

latex_acc_table += "\\bottomrule\n\\end{tabular}"

print("\nTest Accuracy Table:")
print(latex_acc_table)

# Optionally save to file
with open(f"experiments/experiment_analysis/dt_es_simulation_study/latex_tables/dt_empirical_{test_metric}.tex", "w") as f:
    f.write(latex_acc_table)

print("esel")


# Create a function to generate metric tables
def create_metric_table(results, metric_name, save_to_file=True):
    # Determine which methods to include
    if metric_name == "fit_time_mean":
        methods = ["CCP", "TS", "ES"]
    elif metric_name in ["test_log_loss_mean", "test_acc_mean"]:
        methods = ["CCP", "TS", "ES"]
        if INCLUDE_MD:
            methods.append("MD")
    else:
        methods = ["CCP", "TS", "ES"]

    # Create a pivot table with the specified metric
    metric_pivot = results.pivot(
        index="dataset",
        columns="method",
        values=metric_name,
    )[methods]

    # Reorder according to ordered_datasets to match noise-based ordering
    metric_pivot = metric_pivot.reindex(ordered_datasets)

    # Special case for test_acc_mean - also fetch train_acc_mean
    if metric_name == "test_acc_mean":
        train_pivot = results.pivot(
            index="dataset",
            columns="method",
            values="train_acc_mean",
        )[methods]
        train_pivot = train_pivot.reindex(ordered_datasets)

    # Special case for n_leaves_mean - also fetch depth_mean
    if metric_name == "n_leaves_mean":
        depth_pivot = results.pivot(
            index="dataset",
            columns="method",
            values="depth_mean",
        )[methods]
        depth_pivot = depth_pivot.reindex(ordered_datasets)

    # Create the LaTeX table for the metric
    latex_table = "\\begin{tabular}{l" + "c" * len(methods)

    # Add extra columns for fitting time comparisons
    if "fit_time_mean" in metric_name:
        latex_table += "cc"  # Two additional columns for comparisons

    latex_table += "}\n\\toprule\n"

    # Table header
    header = "Dataset & " + " & ".join(methods)
    if "fit_time_mean" in metric_name:
        header += " & ES vs. CCP & TS vs. CCP"
    latex_table += header + " \\\\\n\\midrule\n"

    # For each dataset, add a row and make the maximum value bold (or minimum for n_leaves_mean)
    for dataset, row in metric_pivot.iterrows():
        # For n_leaves_mean, fit_time_mean or log_loss, highlight the minimum value; for others, highlight the maximum
        best_value = (
            row.min()
            if (
                metric_name == "n_leaves_mean"
                or "fit_time_mean" in metric_name
                or metric_name == "test_log_loss_mean"
            )
            else row.max()
        )
        formatted_values = []

        for method in methods:
            value = row[method]

            if metric_name == "test_acc_mean":
                train_value = train_pivot.loc[dataset, method]
                if value == best_value:
                    # Format test acc (train acc) with test acc bold, both rounded to 2 decimals
                    formatted_values.append(
                        f"\\textbf{{{value:.2f}}} ({train_value:.2f})"
                    )
                else:
                    # Format test acc (train acc), both rounded to 2 decimals
                    formatted_values.append(f"{value:.2f} ({train_value:.2f})")
            elif metric_name == "n_leaves_mean":
                depth_value = depth_pivot.loc[dataset, method]
                if value == best_value:
                    # Format leaves (depth) with leaves bold, both rounded to 1 decimal
                    formatted_values.append(
                        f"\\textbf{{{value:.1f}}} ({depth_value:.1f})"
                    )
                else:
                    # Format leaves (depth), both rounded to 1 decimal
                    formatted_values.append(f"{value:.1f} ({depth_value:.1f})")
            elif "fit_time_mean" in metric_name:
                if value == best_value:
                    # Format as bold and round to 3 decimal places for fitting time
                    formatted_values.append(f"\\textbf{{{value:.3f}}}")
                else:
                    # Just round to 3 decimal places for fitting time
                    formatted_values.append(f"{value:.3f}")
            else:
                if value == best_value:
                    # Format as bold and round to 2 decimal places
                    formatted_values.append(f"\\textbf{{{value:.2f}}}")
                else:
                    # Just round to 2 decimal places
                    formatted_values.append(f"{value:.2f}")

        # Build the row string
        row_str = f"{dataset} & {' & '.join(formatted_values)}"

        # Add comparison columns for fitting time
        if "fit_time_mean" in metric_name:
            # Calculate speedup ratios (how many times faster)
            es_vs_ccp = row["CCP"] / row["ES"] if row["ES"] > 0 else float("inf")
            ts_vs_ccp = row["CCP"] / row["TS"] if row["TS"] > 0 else float("inf")

            # Format the ratios with one decimal place, rounded up
            es_vs_ccp_rounded = math.ceil(es_vs_ccp * 10) / 10
            ts_vs_ccp_rounded = math.ceil(ts_vs_ccp * 10) / 10

            # Show as integer if decimal part is 0
            es_format = (
                f"{int(es_vs_ccp_rounded)}x"
                if es_vs_ccp_rounded == int(es_vs_ccp_rounded)
                else f"{es_vs_ccp_rounded:.1f}x"
            )
            ts_format = (
                f"{int(ts_vs_ccp_rounded)}x"
                if ts_vs_ccp_rounded == int(ts_vs_ccp_rounded)
                else f"{ts_vs_ccp_rounded:.1f}x"
            )

            row_str += f" & {es_format} & {ts_format}"

        latex_table += row_str + " \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}"

    # Optionally save to file
    if save_to_file:
        with open(f"experiments/experiment_analysis/dt_es_simulation_study/latex_tables/dt_empirical_{metric_name}.tex", "w") as f:
            f.write(latex_table)

    return latex_table


# Define metrics to iterate over
metrics = [
    "test_acc_mean",
    "test_mcc_mean",
    "test_f1_mean",
    "test_log_loss_mean",
    # "depth_mean",
    "n_leaves_mean",
    "fit_time_mean",
]

# Generate tables for each metric
for metric in metrics:
    print(f"\n{metric.replace('_', ' ').title()} Table:")
    latex_table = create_metric_table(results, metric)
    print(latex_table)

