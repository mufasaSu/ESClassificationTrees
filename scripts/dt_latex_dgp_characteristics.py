import itertools
import pandas as pd

# Read the CSV file
df = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv"
)
# in the dataset column, rename "Smooth Signal" to "Circular Smooth" if applicable
df.loc[df["dataset"] == "Smooth Signal", "dataset"] = "Circular Smooth"
df = df[df["mc_iterations"] == 300]
df = df[df["dgp_config_folder"] == "standard"]
df = df[df["feature_dim"] != 5]
# Remove algorithm MD
df = df[df["algorithm"].isin(["ES", "TS"])]
df = df[df["full_alpha_range"] != 0]


columns_to_keep = [
    "dataset",
    "algorithm",
    "feature_dim",
    "n_train_samples",
    "Train Noise (median)",
    "Train Noise Est. Diff. (median)",
    "Class 1 Fraction (median)",
]


df = df[columns_to_keep].round(2)
print(df)
df["Noise Level normalized"] = df["Train Noise (median)"] / 0.25

# multiply the Estimation difference column by -1
df["Train Noise Est. Diff. (median)"] = df["Train Noise Est. Diff. (median)"] * -1

# Create LaTeX table
latex_table = df.pivot_table(
    index="dataset",
    values=[
        "Class 1 Fraction (median)",
        "Train Noise (median)",
        "Train Noise Est. Diff. (median)",
    ],
    aggfunc="first",
).round(2)

# Rename columns to match desired order
latex_table.columns = ["Class Ratio", "Noise Level", "Estimation Difference"]

# Replace -0.00 with 0.00 and then 0.00 with 0
latex_table = latex_table.replace(-0.00, 0.00)
latex_table = latex_table.replace(0.00, 0)

# Convert to LaTeX format with 2 decimal places
latex_output = (
    latex_table.to_latex(float_format="%.2f", index_names=False, header=False)
    .replace("\\begin{tabular}{lrrr}", "\\begin{tabular}{lccc}")
    .replace(
        "\\toprule\n",
        "\\toprule\nDataset & Class Ratio & Noise Level & Estimation Difference \\\\\n",
    )
    .replace("}{", "} {")
)

# Save to file
with open(
    "experiments/experiment_analysis/dt_es_simulation_study/latex_tables/dt_dgp_characteristics.tex",
    "w",
) as f:
    f.write(latex_output)

print(latex_output)
