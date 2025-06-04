ENV_CONFIG="experiments/experiment_configs/env_setting_configs/rf_experiments/n_estimators_100.yaml" #test.yaml" # n_estimators_100.yaml

dgp_configs=(
    experiments/experiment_configs/simulated_data_configs/rf/standard/*.yaml
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


dgp_configs=(
    experiments/experiment_configs/simulated_data_configs/runtimes/rf/*.yaml
)
algo_configs=(
    experiments/experiment_configs/algorithm_configs/es_rf_simulation_study/rf/new_structure/UES_RF_star_time.yaml
    experiments/experiment_configs/algorithm_configs/es_rf_simulation_study/rf/new_structure/UES_RF_star.yaml
    experiments/experiment_configs/algorithm_configs/es_rf_simulation_study/rf/new_structure/MD_RF_custom.yaml
    experiments/experiment_configs/algorithm_configs/es_rf_simulation_study/rf/new_structure/MD_RF_scikit.yaml
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