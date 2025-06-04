#!/bin/bash

# Loop through all .yaml files in the algorithm_configs/ directory
# for dgp_config in simulated_data_configs/*.yaml
ENV_CONFIG="experiments/experiment_configs/env_setting_configs/rf_experiments/test.yaml"

dgp_configs=(
    experiments/experiment_configs/simulated_data_configs/rf/standard/additive_sparse_jump_feature_dim_30_n_samples_2000.yaml
)
algo_configs=(
    experiments/experiment_configs/algorithm_configs/es_rf_simulation_study/*.yaml
)

for dgp_config in "${dgp_configs[@]}"
do
    for algo_config in "${algo_configs[@]}"
    do
        echo "Running simulation study for $dgp_config and $algo_config"
        python scripts/conduct_mc_studies/rf_mc_study.py --dgp_config "$dgp_config" --algorithm_config "$algo_config" --env_config "$ENV_CONFIG"
    done
done

# Concatenate results
python infrastructure/concatenate_result_csvs.py \
    --directory experiments/experiment_raw_results/rf_es_simulation_study \
    --output_folder_name rf_es_simulation_study \
    --output_file_name rf_dgps_results  

