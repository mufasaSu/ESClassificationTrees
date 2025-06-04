import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv"
)
# in the dataset column, rename "Smooth Signal" to "Circular Smooth" if applicable
df.loc[df["dataset"] == "Smooth Signal", "dataset"] = "Circular Smooth"
df = df[df["mc_iterations"] == 300]
df = df[df["n_train_samples"] == 1000]
# Remove algorithm MD
df = df[df["algorithm"] == "MD"]
# Group by dataset and feature dimension, and get the first value if multiple exist
df = df.groupby(["dataset", "algorithm", "feature_dim"])["Depth (median)"].unique()
print(df)

df = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv"
)
# in the dataset column, rename "Smooth Signal" to "Circular Smooth" if applicable
df.loc[df["dataset"] == "Smooth Signal", "dataset"] = "Circular Smooth"
df = df[df["mc_iterations"] == 300]
df = df[df["dataset"] == "Add. Smooth"]
df = df[df["feature_dim"] == 30]
# Remove algorithm MD
df = df[df["algorithm"] == "ES"]
# Group by dataset and feature dimension, and get the first value if multiple exist
df = (
    df.groupby(["dataset", "algorithm", "n_train_samples"])[
        "Train Noise Est. Diff. (median)"
    ].unique()
    # .apply(lambda x: x[0])
)
print(df)
df = df.reset_index()


df = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv"
)
# in the dataset column, rename "Smooth Signal" to "Circular Smooth" if applicable
df.loc[df["dataset"] == "Smooth Signal", "dataset"] = "Circular Smooth"
df = df[df["mc_iterations"] == 300]
df = df[df["dataset"] == "Add. Smooth"]
df = df[df["feature_dim"] == 30]
# Remove algorithm MD
df = df[df["algorithm"].isin(["ES", "TS", "CCP"])]
# Group by dataset and feature dimension, and get the first value if multiple exist
df = (
    df.groupby(["dataset", "algorithm", "n_train_samples"])["fit_duration (median)"]
    .unique()
    .apply(lambda x: x[0])
)
print(df)
df = df.reset_index()

# Set the font to serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Create the lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="n_train_samples",
    y="fit_duration (median)",
    hue="algorithm",
    marker="o",
)

plt.xlabel("Number of Training Samples", fontsize=14, labelpad=10)
plt.ylabel("Training Duration (seconds)", fontsize=14, labelpad=10)
plt.xticks(fontsize=12)  # Increase x-axis tick label size
plt.yticks(fontsize=12)  # Increase y-axis tick label size
plt.xlim(0, 11000)
plt.grid(True, alpha=0.3)
plt.legend(
    bbox_to_anchor=(1.05, 0.5),
    loc="center left",
    title="Method",
    fontsize=12,
    title_fontsize=12,
)
plt.tight_layout()
plt.savefig(
    "experiments/experiment_analysis/dt_es_simulation_study/plots/fitting_time_vs_n_samples.png",
    bbox_inches="tight",
    dpi=600,
)
plt.close()


# Look at n impact on Add. Sparse Smooth
# Read the CSV file
df = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv"
)
# in the dataset column, rename "Smooth Signal" to "Circular Smooth" if applicable
df.loc[df["dataset"] == "Smooth Signal", "dataset"] = "Circular Smooth"
df = df[df["mc_iterations"] == 300]
df = df[df["dataset"] == "Add. Smooth"]
df = df[df["feature_dim"] == 30]
# Remove algorithm MD
df = df[df["algorithm"].isin(["ES"])]
df["Train Noise Est. Diff. (median)"] = df["Train Noise Est. Diff. (median)"] * (-1)

# Group by dataset and feature dimension, and get the first value if multiple exist
df = (
    df.groupby(["dataset", "n_train_samples"])["Train Noise Est. Diff. (median)"]
    .unique()
    .apply(lambda x: x[0])
)
print(df)
df = df.reset_index()

# Set the font to serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Create the lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="n_train_samples",
    y="Train Noise Est. Diff. (median)",
    marker="o",
)

plt.ylabel(r"$\hat{\sigma}^2 - \sigma^2$", fontsize=14, labelpad=10)
plt.xlabel("Number of Training Samples", fontsize=14, labelpad=10)
plt.ylabel(r"$\hat{\sigma}^2 - \sigma^2$", fontsize=14, labelpad=10)
plt.ylim(0, 0.16)
plt.xticks(fontsize=12)  # Increase x-axis tick label size
plt.yticks(fontsize=12)  # Increase y-axis tick label size
plt.grid(True, alpha=0.3)
plt.tight_layout()  # Adjust layout to prevent legend cutoff
plt.savefig(
    "experiments/experiment_analysis/dt_es_simulation_study/plots/noise_est_vs_n_samples.png",
    bbox_inches="tight",
    dpi=600,
)
plt.close()


# Read the CSV file
df = pd.read_csv(
    "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs/dt_mc_results.csv"
)
# in the dataset column, rename "Smooth Signal" to "Circular Smooth" if applicable
df.loc[df["dataset"] == "Smooth Signal", "dataset"] = "Circular Smooth"
df = df[df["mc_iterations"] == 300]
df = df[df["n_train_samples"] == 1000]
# Remove algorithm MD
df = df[df["algorithm"].isin(["ES"])]
df["Train Noise Est. Diff. (median)"] = df["Train Noise Est. Diff. (median)"] * (-1)

# Group by dataset and feature dimension, and get the first value if multiple exist
df = (
    df.groupby(["dataset", "feature_dim"])["Train Noise Est. Diff. (median)"]
    .unique()
    .apply(lambda x: x[0])
)

# Reset index to convert Series to DataFrame for plotting
df = df.reset_index()

# Set the font to serif
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Create the lineplot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="feature_dim",
    y="Train Noise Est. Diff. (median)",
    hue="dataset",
    marker="o",
)

plt.xlabel("Feature Dimension", fontsize=14, labelpad=10)
plt.ylabel(r"$\hat{\sigma}^2 - \sigma^2$", fontsize=14, labelpad=10)
plt.xticks(fontsize=12)  # Increase x-axis tick label size
plt.yticks(fontsize=12)  # Increase y-axis tick label size
plt.grid(True, alpha=0.3)
plt.legend(
    bbox_to_anchor=(1.05, 0.5),
    loc="center left",
    title="DGP",
    fontsize=12,
    title_fontsize=12,
)
plt.tight_layout()  # Adjust layout to prevent legend cutoff

# Save the figure instead of showing it
plt.savefig(
    "experiments/experiment_analysis/dt_es_simulation_study/plots/noise_est_vs_d.png",
    bbox_inches="tight",
    dpi=600,
)
plt.close()
