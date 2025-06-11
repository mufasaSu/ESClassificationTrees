# Residual-Based Early Stopping Classification Trees
Code for my master thesis

## Create virtual environment

## Install prequisites
pip install requirements.txt

## File Structure
- data: two datasets only used for plotting
- experiments: -- experiment analysis (plots, final overview csv containing all experiment results, latex tables generated from script)
-              -- experiment_cmds (sh scripts that contain the commands to run the experiments
-              -- experiment_configs (configuration files for the settings of the experiments, e.g. hyperparameters)
-              -- experiment_raw_results (containing the results for single experiment runs as csvs; basis for concatenating into one overview csv file containing all runs)
- infrastructure algorithms, dgp scripts and other stuff needed for the experiments
- r_scripts -- scripts for creating the plots
- scripts -- scripts containing the experiments for the mc runs, latex table creation, animations etc.
- tests -- test for the single classification tree         

## Run Experiment, Create Overview CSV, Create Plots, Create Tables
make executable
### Example of Decision tree:
Make MC runs executable:
chmod +x experiments/experiment_cmds/dt_simulations/full/dt_standard.sh

Run Experiments in Terminal
Run single simulation mc experiment:
python scripts/conduct_mc_studies/dt_mc_study.py \
            --dgp_config experiments/experiment_configs/simulated_data_configs/standard/circular_feature_dim_2_n_samples_2000_bernoulli_p_0.8.yaml \
            --algorithm_config experiments/experiment_configs/algorithm_configs/dt_simulation_study/max_depth.yaml \
            --env_config experiments/experiment_configs/env_setting_configs/dt_experiments/60_mc_runs.yaml

Concatenate Results of multiple simulations into one csv:
python infrastructure/concatenate_result_csvs.py \
    --directory experiments/experiment_raw_results/dt_es_simulation_study \
    --output_folder_name dt_es_simulation_study \
    --output_file_name dt_mc_results

Run group of experiments:
experiments/experiment_cmds/dt_simulations/full/dt_standard.sh

Run empirical simulation mc experiments:
python scripts/conduct_mc_studies/dt_empirical_study.py

Scripts for creating the plots:
r_scripts

Scripts for creating latex tables:
in folder scripts, filename contains latex, e.g. scripts/dt_latex_empiric.py

Results (Plots, concatenated csv and latex tables mainly stored in experiments/experiment_analysis


