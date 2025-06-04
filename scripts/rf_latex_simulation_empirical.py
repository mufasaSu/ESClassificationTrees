import pandas as pd


def create_metric_table(
    results,
    metric_name,
    aggregate_type="median",
    save_to_file=True,
    empiric=False,
):
    """
    Creates a LaTeX table for the specified metric with appropriate formatting.
    """
    # Create the full column name with aggregate type
    full_metric_name = f"{metric_name} ({aggregate_type})"

    # Define algorithms based on empirical vs simulation case
    algorithms = ["MD_custom", "IGES*", "ILES*", "URES*", "UGES*"]
    if empiric:
        algorithms = ["MD_custom", "IGES", "ILES", "URES", "UGES"]

    # Create a pivot table with the specified metric
    metric_pivot = results.pivot(
        index="dataset",
        columns="algorithm_name",
        values=full_metric_name,
    )[algorithms]

    # For empirical data, reorder according to the fixed order
    if empiric:
        valid_ordered_datasets = [
            d for d in empirical_dataset_order if d in metric_pivot.index
        ]
        metric_pivot = metric_pivot.loc[valid_ordered_datasets]

    # Special case for test_acc - also fetch train_acc
    if metric_name == "test_acc":
        train_metric_name = f"train_acc ({aggregate_type})"
        train_pivot = results.pivot(
            index="dataset",
            columns="algorithm_name",
            values=train_metric_name,
        )[algorithms]

    # Special case for n_leaves - also fetch depth information
    if metric_name == "median_n_leaves":
        depth_metric_name = f"median_depth ({aggregate_type})"
        depth_pivot = results.pivot(
            index="dataset",
            columns="algorithm_name",
            values=depth_metric_name,
        )[algorithms].astype(int)

    # Create the LaTeX table structure
    latex_table = "\\begin{tabular}{l" + "c" * len(algorithms) + "}\n\\toprule\n"

    # Table header - simplify MD display names
    display_algorithms = [
        algo if not algo.startswith("MD_") else "MD" for algo in algorithms
    ]
    header = "Dataset & " + " & ".join(display_algorithms)
    latex_table += header + " \\\\\n\\midrule\n"

    # For each dataset, add a row and make the best value bold
    for dataset, row in metric_pivot.iterrows():
        # Determine best value based on metric (minimizing or maximizing)
        if metric_name in ["median_n_leaves", "log_loss_test"]:
            best_value = row.min()
            # Find all values that are tied for the minimum
            is_best = row == best_value
        else:
            best_value = row.max()
            # Find all values that are tied for the maximum
            is_best = row == best_value

        formatted_values = []
        for algo in algorithms:
            value = row[algo]
            # Check if this value is among the best (tied or not)
            is_this_best = is_best[algo]

            if metric_name == "test_acc":
                train_value = train_pivot.loc[dataset, algo]
                formatted_values.append(
                    f"\\textbf{{{value:.2f}}} ({train_value:.2f})"
                    if is_this_best
                    else f"{value:.2f} ({train_value:.2f})"
                )
            elif metric_name == "median_n_leaves":
                depth_value = depth_pivot.loc[dataset, algo]
                formatted_values.append(
                    f"\\textbf{{{int(value)}}} ({int(depth_value)})"
                    if is_this_best
                    else f"{int(value)} ({int(depth_value)})"
                )
            else:
                formatted_values.append(
                    f"\\textbf{{{value:.2f}}}" if is_this_best else f"{value:.2f}"
                )

        latex_table += f"{dataset} & {' & '.join(formatted_values)} \\\\\n"

    latex_table += "\\bottomrule\n\\end{tabular}"

    # Save to file if requested
    if save_to_file:
        empiric_suffix = "_empiric" if empiric else ""
        filename = f"experiments/experiment_analysis/rf_es_simulation_study/latex_tables/rf_{metric_name}_{aggregate_type}{empiric_suffix}.tex"
        with open(filename, "w") as f:
            f.write(latex_table)
        print(f"Table saved to {filename}")

    return latex_table


# Define the fixed order for empirical datasets
empirical_dataset_order = [
    "Banknote",
    "Wisc. Cancer",
    "Ozone",
    "Spam",
    "Haberman",
    "Pima Indians",
    "SA Heart",
]

# Define metrics
metrics = [
    "test_acc",
    "mcc_test",
    "f1_test",
    "log_loss_test",
    "median_n_leaves",
]

# Load and process simulation results
results = pd.read_csv(
    "experiments/experiment_analysis/rf_es_simulation_study/overview_csvs/rf_dgps_results_mean_median.csv"
)
results = results[
    (results["dgp_config_folder"] == "standard") & (results["mc_iterations"] == 300)
]
results = results[~results["algorithm_name"].isin(["UES* (time)", "ICCP", "MD_scikit"])]
results["algorithm_name"] = results["algorithm_name"].replace("UES*", "URES*")
# Round numerical columns to 2 decimal places
results = results.round(2)

# Generate simulation tables
for metric in metrics:
    print(f"\n{metric.replace('_', ' ').title()} Table (Median):")
    latex_table = create_metric_table(
        results,
        metric,
        aggregate_type="median",
        save_to_file=True,
        empiric=False,
    )
    print(latex_table)

# Load and process empirical results
empiric_results = pd.read_csv(
    "experiments/experiment_analysis/rf_es_simulation_study/overview_csvs/rf_empirical_results.csv"
)
empiric_results["dataset"] = empiric_results["dataset"].replace(
    "Wisc. Breast Cancer", "Wisc. Cancer"
)
empiric_results["algorithm_name"] = empiric_results["algorithm_name"].replace(
    {
        "FMSE": "UGES",
        "UES": "URES",
    }
)
empiric_results = empiric_results.round(2)

# Generate empirical tables
for metric in metrics:
    print(f"\n{metric.replace('_', ' ').title()} Table (Median):")
    latex_table = create_metric_table(
        empiric_results,
        metric,
        aggregate_type="median",
        save_to_file=True,
        empiric=True,
    )
    print(latex_table)
