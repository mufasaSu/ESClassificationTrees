# Residual-Based Early Stopping for Classification Trees

This repository contains the code for a master's thesis on a residual-based early stopping method for classification trees.

## Abstract

Decision trees are a popular machine learning algorithm, but they are prone to overfitting. Pruning methods are commonly used to counteract this. This work introduces a novel early stopping method for classification trees based on residuals, which aims to find a simpler and well-performing tree structure.

## Getting Started

### Prerequisites

- Python 3.x
- R (for plotting scripts)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ESClassificationTrees.git
    cd ESClassificationTrees
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

The project is organized as follows:

```
.
├── data/
│   └── (Contains datasets for plotting)
├── experiments/
│   ├── experiment_analysis/
│   │   ├── (Plots, final overview CSVs, and LaTeX tables)
│   ├── experiment_cmds/
│   │   ├── (Shell scripts to run experiments)
│   ├── experiment_configs/
│   │   ├── (Configuration files for experiments)
│   ├── experiment_raw_results/
│   │   └── (Raw CSV results from individual experiment runs)
├── infrastructure/
│   ├── algorithms/
│   │   └── (Implementations of classification tree algorithms)
│   ├── dgp/
│   │   └── (Data Generation Process scripts)
│   └── (Other utility scripts)
├── r_scripts/
│   └── (R scripts for generating plots from results)
├── scripts/
│   ├── conduct_mc_studies/
│   │   └── (Scripts for Monte Carlo simulation studies)
│   └── (Other scripts for analysis, animations, and tables)
├── tests/
│   └── (Tests for the classification tree implementation)
├── README.md
└── requirements.txt
```

-   `data/`: Contains datasets used for plotting and visualization.
-   `experiments/`: A parent directory for all experiment-related files.
    -   `experiment_analysis/`: Stores the final outputs of the analysis, such as plots, concatenated CSV results, and LaTeX tables.
    -   `experiment_cmds/`: Contains shell scripts (`.sh`) for running various experiment configurations.
    -   `experiment_configs/`: Holds `.yaml` configuration files that define hyperparameters and settings for different experiments.
    -   `experiment_raw_results/`: This is where the raw CSV files from each individual experiment run are saved.
-   `infrastructure/`: Core code for the experiments.
    -   `algorithms/`: Contains the Python implementations of the classification tree algorithms, including the proposed residual-based early stopping method.
    -   `dgp/`: Scripts for generating the synthetic datasets used in the simulations.
-   `r_scripts/`: R scripts used to create plots and visualizations from the experiment results.
-   `scripts/`: Contains Python scripts for conducting the Monte Carlo studies, generating LaTeX tables, creating animations, and other analysis tasks.
-   `tests/`: Unit tests for the classification tree implementation.

## Running the Experiments

The experiments are managed through a combination of Python scripts and shell scripts.

### Running a Single Simulation Experiment

To run a single simulation experiment, you can use the `dt_mc_study.py` script. You need to provide paths to the data generation config, the algorithm config, and the environment config.

**Example:**

```bash
python scripts/conduct_mc_studies/dt_mc_study.py \
    --dgp_config experiments/experiment_configs/simulated_data_configs/standard/circular_feature_dim_2_n_samples_2000_bernoulli_p_0.8.yaml \
    --algorithm_config experiments/experiment_configs/algorithm_configs/dt_simulation_study/max_depth.yaml \
    --env_config experiments/experiment_configs/env_setting_configs/dt_experiments/test.yaml
```

### Running a Group of Simulation Experiments

You can run a group of simulation experiments using the provided shell scripts in `experiments/experiment_cmds/`.

First, make the script executable:

```bash
chmod +x experiments/experiment_cmds/dt_simulations/full/dt_standard.sh
```

Then, run the script:

```bash
./experiments/experiment_cmds/dt_simulations/full/dt_standard.sh
```

### Running Empirical Experiments

```bash
python scripts/conduct_mc_studies/dt_empirical_study.py
```


## Analyzing the Results

### Concatenating Raw Results

After running multiple simulations, the individual result CSVs can be concatenated into a single overview file. When groups of experiments are run via the provided shell scripts, this will happen automatically.

**Example:**

```bash
python infrastructure/concatenate_result_csvs.py \
    --directory experiments/experiment_raw_results/dt_es_simulation_study \
    --output_folder_name dt_es_simulation_study \
    --output_file_name dt_mc_results
```
The concatenated results will be saved in the `experiments/experiment_analysis` directory.

### Generating Plots

Plots are generated using the R scripts in the `r_scripts/` directory. These scripts will typically read the concatenated CSV files from `experiments/experiment_analysis`.

### Generating LaTeX Tables

LaTeX tables can be generated by running the corresponding Python scripts in the `scripts/` folder (e.g., `scripts/dt_latex_empiric.py`). The results will also be saved in the `experiments/experiment_analysis` directory.


### Additional Information
The provided examples are for the single decision tree. The structure for the random forest experiments is nearly identical.

