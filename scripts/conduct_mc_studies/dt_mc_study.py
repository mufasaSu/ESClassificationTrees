# Default values for debug run
ALGO_PATH = r"experiments/experiment_configs/algorithm_configs/dt_simulation_study/post_pruned.yaml"
DGP_PATH = "experiments/experiment_configs/simulated_data_configs/standard/circular_feature_dim_2_n_samples_2000_bernoulli_p_0.8.yaml"
ENV_PATH = "experiments/experiment_configs/env_setting_configs/dt_experiments/test.yaml"

import argparse
import csv
import os
import sys

from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
)
from sklearn.inspection import permutation_importance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import infrastructure.config_reader as config_reader
import infrastructure.dgp.data_generation as data_generation
from infrastructure.algorithms.clean_dt import DecisionTreeLevelWise
import infrastructure.noise_level_estimator as noise_est
from infrastructure.model_builder import build_post_pruned_dt_clf

from concurrent.futures import ProcessPoolExecutor, as_completed
import time


# for mc_iteration in range(MC_ITERATIONS):
def run_mc_iteration(mc_iteration, dgp_config, algo_config, env_config):
    # print(f"MC iteration {mc_iteration}")
    random_state = env_config.get("random_state") + mc_iteration
    X_train, X_test, y_train, y_test, f_train, f_test = (
        data_generation.generate_X_y_f_classification(
            random_state=random_state,
            n_ticks_per_ax_meshgrid=dgp_config.get(
                "n_ticks_per_ax_meshgrid"
            ),  # for 2 dim cases, n_samples
            dgp_name=dgp_config.get("dgp_name"),  # indicating which DGP to use
            bernoulli_p=dgp_config.get("bernoulli_p"),  # for 2 dim cases
            n_samples=dgp_config.get(
                "n_samples"
            ),  # size of dateset (train + test), in additive cases!
            feature_dim=dgp_config.get("feature_dim"),
        )
    )

    if (
        algo_config.get("algorithm") == "es"
        or algo_config.get("algorithm") == "two_step"
    ):
        kappa = algo_config.get("kappa")
        noise_estimator = noise_est.Estimator(X_train, y_train)
        noise_learning_start_time = time.time()
        if kappa == "1nn":
            mean_estimated_train_noise = noise_estimator.estimate_1NN()
        elif kappa == "lasso":
            mean_estimated_train_noise = noise_estimator.estimate_LASSO()
        elif kappa == "mean_var":
            mean_estimated_train_noise = np.mean(f_train * (1 - f_train))
        else:
            raise ValueError("Kappa not recognized")
        noise_learning_end_time = time.time()
        noise_learning_duration = noise_learning_end_time - noise_learning_start_time

        dt = DecisionTreeLevelWise(
            max_depth=algo_config.get("max_depth"),
            min_samples_split=algo_config.get("min_samples_split"),
            max_features=algo_config.get("max_n_features"),
            kappa=mean_estimated_train_noise,
            random_state=random_state,
        )
        dt.fit(X_train, y_train)
        es_dt_depth = max(dt.get_depth(), 1)
        # Fit scikit-learn decision tree with same max_depth
        dt = DecisionTreeClassifier(
            max_depth=es_dt_depth,
            min_samples_split=2,
            max_features=None,
            random_state=random_state,
        )
        start_time = time.time()  # Record start time
        dt.fit(X_train, y_train)
        end_time = time.time()  # Record end time

        if algo_config.get("algorithm") == "two_step":
            if kappa != "mean_var":
                es_dt_depth += 1  # account for upward bias
            start_time = time.time()  # Record start time
            dt = build_post_pruned_dt_clf(
                X_train=X_train,
                y_train=y_train,
                max_depth=es_dt_depth,
                random_state=random_state,
                n_cv_alpha=5,
                full_alpha_range=algo_config.get("full_alpha_range"),
            )
            end_time = time.time()  # Record end time
    elif algo_config.get("algorithm") == "ccp":
        start_time = time.time()  # Record start time
        dt = build_post_pruned_dt_clf(
            X_train=X_train,
            y_train=y_train,
            max_depth=None,
            random_state=random_state,
            n_cv_alpha=5,
            full_alpha_range=algo_config.get("full_alpha_range"),
        )
        end_time = time.time()  # Record end time
        mean_estimated_train_noise = np.nan
    elif algo_config.get("algorithm") == "max depth":
        dt = DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            max_features=None,
            random_state=random_state,
        )
        start_time = time.time()  # Record start time
        dt.fit(X_train, y_train)
        end_time = time.time()  # Record end time
        mean_estimated_train_noise = np.nan

    fit_duration = end_time - start_time  # Calculate duration of fit
    if (
        algo_config.get("algorithm") == "es"
        or algo_config.get("algorithm") == "two_step"
    ):
        fit_duration += noise_learning_duration

    train_accuracy = np.mean(dt.predict(X_train) == y_train)
    test_accuracy = np.mean(dt.predict(X_test) == y_test)

    # After calculating test_accuracy, add these metrics:
    y_pred_test = dt.predict(X_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_matthews = matthews_corrcoef(y_test, y_pred_test)

    y_pred_proba = dt.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    test_log_loss = log_loss(y_test, y_pred_proba)

    # Track Tree Structure
    tree_depth = dt.get_depth()
    n_leaves = dt.get_n_leaves()

    # Track Noise Related Stuff
    true_whole_set_noise_level = np.mean(f_train * (1 - f_train))

    # Noise Estimation related Stuff
    whole_set_noise_true_vs_estimate = (
        true_whole_set_noise_level - mean_estimated_train_noise
    )

    # Get feature importances from the final decision tree
    feature_importances = dt.feature_importances_

    noise_learning_duration = (
        0  # Default value for methods that don't use noise learning
    )

    # Inside your Monte Carlo loop where you have access to dt, X_test, y_test
    perm_importance = permutation_importance(
        dt, X_test, y_test, n_repeats=5, random_state=42
    )
    mc_perm_importance_means = perm_importance.importances_mean
    mc_perm_importance_stds = perm_importance.importances_std

    # Calculate fraction of class 1 (positive class)
    all_y = np.concatenate([y_train, y_test])
    positive_class_fraction = np.mean(all_y)

    return (
        train_accuracy,
        test_accuracy,
        tree_depth,
        n_leaves,
        true_whole_set_noise_level,
        mean_estimated_train_noise,
        whole_set_noise_true_vs_estimate,
        fit_duration,
        test_precision,
        test_recall,
        test_f1,
        test_auc,
        feature_importances,
        noise_learning_duration,
        mc_perm_importance_means,
        mc_perm_importance_stds,
        test_log_loss,
        test_matthews,
        positive_class_fraction,  # Add positive class fraction to return tuple
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm_config",
        "-alg_cfg",
        type=str,
        default=ALGO_PATH,
        help="Path to a yaml file containing the hyperparameters for the algorithm",
        required=False,
    )
    parser.add_argument(
        "--dgp_config",
        "-dgp_cfg",
        type=str,
        default=DGP_PATH,
        help="Path to a yaml file containing the hyperparameters for the DGP",
        required=False,
    )
    parser.add_argument(
        "--env_config",
        "-env_cfg",
        type=str,
        default=ENV_PATH,
        help="Path to yaml where random state and mc_iterations are stored",
        required=False,
    )
    args = parser.parse_args()
    env_config = config_reader.load_config(args.env_config)
    MC_ITERATIONS = env_config.get("mc_iter")

    # Set Up data with config file
    dgp_config = config_reader.load_config(args.dgp_config)
    algo_config = config_reader.load_config(args.algorithm_config)
    # Get algo_config filename at the end of path
    algo_config_filename = os.path.basename(args.algorithm_config)

    # Add lists to store all values for median calculation
    all_train_accuracies = []
    all_test_accuracies = []
    all_depths = []
    all_n_leaves = []
    all_true_noise_levels = []
    all_estimated_noise = []
    all_noise_differences = []
    all_fit_durations = []
    all_test_precisions = []
    all_test_recalls = []
    all_test_f1s = []
    all_test_aucs = []
    all_feature_importances = []
    all_noise_learning_durations = []
    mc_perm_importance_means = []
    mc_perm_importance_stds = []
    all_test_log_losses = []
    all_test_matthews = []
    all_positive_class_fractions = []  # Add new list to store positive class fractions

    # Run MC iterations in parallel
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_mc_iteration, mc_iteration, dgp_config, algo_config, env_config
            )
            for mc_iteration in range(MC_ITERATIONS)
        ]

        for future in tqdm(
            as_completed(futures), total=MC_ITERATIONS, desc="MC Iterations"
        ):
            (
                train_accuracy,
                test_accuracy,
                tree_depth,
                n_leaves,
                true_whole_set_noise_level,
                mean_estimated_train_noise,
                whole_set_noise_true_vs_estimate,
                fit_duration,
                test_precision,
                test_recall,
                test_f1,
                test_auc,
                feature_importances,
                noise_learning_duration,
                mc_perm_importance_means_iter,
                mc_perm_importance_stds_iter,
                test_log_loss,
                test_matthews,
                positive_class_fraction,  # Unpack positive class fraction
            ) = future.result()

            # Store values for mean and median calculation
            all_train_accuracies.append(train_accuracy)
            all_test_accuracies.append(test_accuracy)
            all_depths.append(tree_depth)
            all_n_leaves.append(n_leaves)
            all_true_noise_levels.append(true_whole_set_noise_level)
            all_estimated_noise.append(mean_estimated_train_noise)
            all_noise_differences.append(whole_set_noise_true_vs_estimate)
            all_fit_durations.append(fit_duration)
            all_test_precisions.append(test_precision)
            all_test_recalls.append(test_recall)
            all_test_f1s.append(test_f1)
            all_test_aucs.append(test_auc)
            all_feature_importances.append(feature_importances)
            all_noise_learning_durations.append(
                noise_learning_duration if not np.isnan(noise_learning_duration) else 0
            )
            mc_perm_importance_means.append(mc_perm_importance_means_iter)
            mc_perm_importance_stds.append(mc_perm_importance_stds_iter)
            all_test_log_losses.append(test_log_loss)
            all_test_matthews.append(test_matthews)
            all_positive_class_fractions.append(
                positive_class_fraction
            )  # Store positive class fraction

    # Calculate means and medians
    mc_train_accuracy = np.mean(all_train_accuracies)
    mc_test_accuracy = np.mean(all_test_accuracies)
    mc_avg_depth = np.mean(all_depths)
    mc_avg_n_leaves = np.mean(all_n_leaves)
    mc_true_whole_set_noise_level = np.mean(all_true_noise_levels)
    mc_mean_estimated_train_noise = np.mean(all_estimated_noise)
    mc_whole_set_noise_true_vs_estimate = np.mean(all_noise_differences)
    mc_fit_duration = np.mean(all_fit_durations)
    mc_test_precision = np.mean(all_test_precisions)
    mc_test_recall = np.mean(all_test_recalls)
    mc_test_f1 = np.mean(all_test_f1s)
    mc_test_auc = np.mean(all_test_aucs)
    mc_noise_learning_duration = np.mean(all_noise_learning_durations)

    mc_train_accuracy_median = np.median(all_train_accuracies)
    mc_test_accuracy_median = np.median(all_test_accuracies)
    mc_avg_depth_median = np.median(all_depths)
    mc_avg_n_leaves_median = np.median(all_n_leaves)
    mc_true_whole_set_noise_level_median = np.median(all_true_noise_levels)
    mc_mean_estimated_train_noise_median = np.median(all_estimated_noise)
    mc_whole_set_noise_true_vs_estimate_median = np.median(all_noise_differences)
    mc_fit_duration_median = np.median(all_fit_durations)
    mc_test_precision_median = np.median(all_test_precisions)
    mc_test_recall_median = np.median(all_test_recalls)
    mc_test_f1_median = np.median(all_test_f1s)
    mc_test_auc_median = np.median(all_test_aucs)
    mc_noise_learning_duration_median = np.median(all_noise_learning_durations)

    # Calculate median feature importances
    median_feature_importances = np.median(all_feature_importances, axis=0)

    # After the Monte Carlo loop, calculate medians
    median_perm_importance_means = np.median(mc_perm_importance_means, axis=0)
    median_perm_importance_stds = np.median(mc_perm_importance_stds, axis=0)
    # Store results in a table

    ALGO_NAME_MAP = {
        "es": "ES" + "*" if algo_config.get("kappa") == "mean_var" else "ES",
        "two_step": "TS" + "*" if algo_config.get("kappa") == "mean_var" else "TS",
        "ccp": "CCP",
        "max depth": "MD",
    }
    results = {
        "algo_config": algo_config_filename,
        "dgp_config_folder": args.dgp_config.split("/")[-2],
        "algorithm": ALGO_NAME_MAP.get(algo_config.get("algorithm")),
        "full_alpha_range": algo_config.get("full_alpha_range"),
        "dataset": dgp_config.get("dgp_name"),
        "feature_dim": dgp_config.get("feature_dim"),
        "n_train_samples": int(dgp_config.get("n_samples") / 2),
        "n_test_samples": int(dgp_config.get("n_samples") / 2),
        "mc_iterations": MC_ITERATIONS,
        "max_depth": algo_config.get("max_depth"),
        "min_samples_split": algo_config.get("min_samples_split"),
        "kappa": algo_config.get("kappa"),
        "max_features": algo_config.get("max_n_features"),
        "fit_duration (mean)": mc_fit_duration,
        "fit_duration (median)": mc_fit_duration_median,
        "noise_learning_duration (mean)": (
            mc_noise_learning_duration
            if not np.isnan(mc_noise_learning_duration)
            else "N/A"
        ),
        "noise_learning_duration (median)": (
            mc_noise_learning_duration_median
            if not np.isnan(mc_noise_learning_duration_median)
            else "N/A"
        ),
        "permutation_importance_means": ",".join(
            [f"{round(x, 2)}" for x in median_perm_importance_means]
        ),
        "permutation_importance_stds": ",".join(
            [f"{round(x, 2)}" for x in median_perm_importance_stds]
        ),
    }
    # Store feature importances as a single string with comma-separated values
    results["feature_importances"] = ",".join(
        [f"{round(x, 2)}" for x in median_feature_importances]
    )

    metrics = {
        "Train Noise (mean)": mc_true_whole_set_noise_level,
        "Train Noise (median)": mc_true_whole_set_noise_level_median,
        "Est. Train Noise (mean)": (
            mc_mean_estimated_train_noise
            if not np.isnan(mc_mean_estimated_train_noise)
            else "N/A"
        ),
        "Est. Train Noise (median)": (
            mc_mean_estimated_train_noise_median
            if not np.isnan(mc_mean_estimated_train_noise_median)
            else "N/A"
        ),
        "Train Noise Est. Diff. (mean)": (
            mc_whole_set_noise_true_vs_estimate
            if not np.isnan(mc_whole_set_noise_true_vs_estimate)
            else "N/A"
        ),
        "Train Noise Est. Diff. (median)": (
            mc_whole_set_noise_true_vs_estimate_median
            if not np.isnan(mc_whole_set_noise_true_vs_estimate_median)
            else "N/A"
        ),
        "Train Accuracy (mean)": mc_train_accuracy,
        "Train Accuracy (median)": mc_train_accuracy_median,
        "Test Accuracy (mean)": mc_test_accuracy,
        "Test Accuracy (median)": mc_test_accuracy_median,
        "Test Precision (mean)": mc_test_precision,
        "Test Precision (median)": mc_test_precision_median,
        "Test Recall (mean)": mc_test_recall,
        "Test Recall (median)": mc_test_recall_median,
        "Test F1 (mean)": mc_test_f1,
        "Test F1 (median)": mc_test_f1_median,
        "Depth (mean)": mc_avg_depth,
        "Depth (median)": mc_avg_depth_median,
        "N Leaves (mean)": mc_avg_n_leaves,
        "N Leaves (median)": mc_avg_n_leaves_median,
        "Test AUC (mean)": mc_test_auc,
        "Test AUC (median)": mc_test_auc_median,
        "Test Log Loss (mean)": np.mean(all_test_log_losses),
        "Test Log Loss (median)": np.median(all_test_log_losses),
        "Test Matthews (mean)": np.mean(all_test_matthews),
        "Test Matthews (median)": np.median(all_test_matthews),
        "Class 1 Fraction (mean)": np.mean(all_positive_class_fractions),
        "Class 1 Fraction (median)": np.median(all_positive_class_fractions),
    }
    results.update(metrics)

    results = {
        key: f"{round(value, 3)}" if isinstance(value, float) else value
        for key, value in results.items()
    }

    ## Store Results
    # Check if directory "experiment_results" exists, if not create it
    folder_name = "_".join(
        [
            f"{key}_{value}" if key != "dgp_name" else f"{value}"
            for key, value in dgp_config.items()
        ]
    )  # e.g. "dgp_name_additive_model_feature_dim_5_n_samples_1000"
    path_name = (
        "experiments/experiment_raw_results/dt_es_simulation_study/"
        + dgp_config.get("dgp_name")
        + "/"
        + folder_name
    )
    if not os.path.exists(path_name):
        os.makedirs(path_name)

    file_name = (
        "_".join(
            [
                f"{key}_{value}" if key != "algorithm" else f"{value}"
                for key, value in algo_config.items()
            ]
        )
        + f"_mc_iters_{MC_ITERATIONS}"
        + ".csv"
    )

    file_path = os.path.join(path_name, file_name)
    # Save results in a csv file
    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results.keys())
        # Write the header (keys of the dictionary)
        writer.writeheader()
        # Write the row (values of the dictionary)
        writer.writerow(results)


if __name__ == "__main__":
    main()
