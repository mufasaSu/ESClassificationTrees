#!/bin/bash

# caffeinate -i -s &

# Define arrays for DGP, algorithm, and environment configs
dgp_configs=(
    experiments/experiment_configs/simulated_data_configs/runtimes/single/*.yaml
)

# algo_configs=(
#     experiments/experiment_configs/algorithm_configs/dt_simulation_study/*.yaml
# )
algo_configs=(
    experiments/experiment_configs/algorithm_configs/dt_simulation_study/*.yaml
)


env_config="experiments/experiment_configs/env_setting_configs/dt_experiments/300_mc_runs.yaml"

# Loop through the configurations
for dgp_config in "${dgp_configs[@]}"; do
    for algo_config in "${algo_configs[@]}"; do
        echo "Running simulation study for $dgp_config and $algo_config"
        python scripts/conduct_mc_studies/dt_mc_study.py \
            --dgp_config "$dgp_config" \
            --algorithm_config "$algo_config" \
            --env_config "$env_config"
    done
done


python infrastructure/concatenate_result_csvs.py \
    --directory experiments/experiment_raw_results/dt_es_simulation_study \
    --output_folder_name dt_es_simulation_study \
    --output_file_name dt_mc_results