import os
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
RANDOM_SEED = 7
MC_ITERATIONS = 60
OUTPUT_PATH = "experiments/experiment_analysis/dt_es_simulation_study/overview_csvs"

# Circular DGP parameters (from circular.yaml)
DGP_PARAMS = {
    "n_ticks_per_ax_meshgrid": 128,
    "n_samples": 4096 * 2,
}


# Functions for post-pruning approaches
def original_post_pruning(X_train, y_train):
    # First create a max depth tree
    max_depth_dt = DecisionTreeClassifier(random_state=7).fit(X_train, y_train)

    # Get pruning path
    path = max_depth_dt.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    # Grid search
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=7),
        param_grid={"ccp_alpha": ccp_alphas},
        cv=5,
        n_jobs=1,
    )
    grid_search.fit(X_train, y_train)

    # Create final tree with best alpha
    best_alpha = grid_search.best_params_["ccp_alpha"]
    final_dt = DecisionTreeClassifier(random_state=7, ccp_alpha=best_alpha)
    final_dt.fit(X_train, y_train)

    return final_dt


def new_post_pruning(X_train, y_train):
    # Base parameters
    base_params = {
        "max_depth": None,
        "min_samples_split": 2,
        "max_features": None,
        "random_state": 7,
    }

    # Get pruning path once
    path = DecisionTreeClassifier(**base_params).cost_complexity_pruning_path(
        X_train, y_train
    )
    ccp_alphas = path.ccp_alphas

    # Filter alphas
    if len(ccp_alphas) > 0:
        ccp_alphas = ccp_alphas[ccp_alphas > 0]

    # If no valid alphas, return unpruned tree
    if len(ccp_alphas) == 0:
        return DecisionTreeClassifier(**base_params).fit(X_train, y_train)

    # Grid search
    grid_search = GridSearchCV(
        DecisionTreeClassifier(**base_params),
        param_grid={"ccp_alpha": ccp_alphas},
        cv=5,
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def generate_two_dim_X_meshgrid(n_ticks_per_ax_meshgrid=128):
    # n_ticks_per_ax_meshgrid Number of points in each dimension for meshgrid
    X1, X2 = np.linspace(0, 1, n_ticks_per_ax_meshgrid), np.linspace(
        0, 1, n_ticks_per_ax_meshgrid
    )

    X1_train, X2_train = X1[::2], X2[::2]  # 64 for train and test each
    X1_train, X2_train = np.meshgrid(X1_train, X2_train)  # 64 x 64
    X_train = np.c_[
        X1_train.ravel(), X2_train.ravel()
    ]  # 64 ** 2 = 4096, 2 ist der shape

    X1_test, X2_test = X1[1::2], X2[1::2]  # same shape as above
    X1_test, X2_test = np.meshgrid(X1_test, X2_test)
    X_test = np.c_[X1_test.ravel(), X2_test.ravel()]
    return X_train, X_test


def generate_circular_classification(X, p=0.8):
    """Draw X.shape[0] Y from a Bernoulli distribution with probability p if X is inside the circle and probability 0.2 else.

    Return: y = Bernoullie draws,
            f = p = f(X) = true probabilities of resp. DGP/Ber"""
    X_in_circular = np.sqrt((X[:, 0] - 1 / 2) ** 2 + (X[:, 1] - 1 / 2) ** 2) <= 1 / 4
    f = 0.2 + X_in_circular.astype(int) * (p - 0.2)
    y = np.random.binomial(1, f, X.shape[0])
    return y


# Circular data generation
def generate_circular_data(n_samples, is_uniform=False):
    if is_uniform:
        # Uniform distribution in unit square
        X_train = np.random.random((int(n_samples * 0.5), 2))
        X_test = np.random.random((int(n_samples * 0.5), 2))
    else:
        X_train, X_test = generate_two_dim_X_meshgrid(n_ticks_per_ax_meshgrid=128)

    # Generate circular boundary labels
    y_train = generate_circular_classification(X_train)
    y_test = generate_circular_classification(X_test)

    return X_train, X_test, y_train, y_test


# MC iteration function
def run_mc_iteration(mc_iteration, is_uniform=False):
    np.random.seed(7 + mc_iteration)

    # Generate data
    X_train, X_test, y_train, y_test = generate_circular_data(
        DGP_PARAMS["n_samples"], is_uniform
    )

    # Test original approach
    original_dt = original_post_pruning(X_train, y_train)
    original_test_accuracy = np.mean(original_dt.predict(X_test) == y_test)

    # Test new approach
    new_dt = new_post_pruning(X_train, y_train)
    new_test_accuracy = np.mean(new_dt.predict(X_test) == y_test)

    return original_test_accuracy, new_test_accuracy


# Main function
def main():
    np.random.seed(RANDOM_SEED)

    # Tracking variables
    grid_original_acc = 0
    grid_new_acc = 0
    uniform_original_acc = 0
    uniform_new_acc = 0

    print("Running grid-based simulations...")
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_mc_iteration, mc_iteration, False)
            for mc_iteration in range(MC_ITERATIONS)
        ]

        for future in tqdm(as_completed(futures), total=MC_ITERATIONS):
            orig_acc, new_acc = future.result()
            grid_original_acc += orig_acc
            grid_new_acc += new_acc

    print("Running uniform-based simulations...")
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_mc_iteration, mc_iteration, True)
            for mc_iteration in range(MC_ITERATIONS)
        ]

        for future in tqdm(as_completed(futures), total=MC_ITERATIONS):
            orig_acc, new_acc = future.result()
            uniform_original_acc += orig_acc
            uniform_new_acc += new_acc

    # Calculate averages
    grid_original_acc /= MC_ITERATIONS
    grid_new_acc /= MC_ITERATIONS
    uniform_original_acc /= MC_ITERATIONS
    uniform_new_acc /= MC_ITERATIONS

    # Results table
    results = [
        {
            "Dataset": "circular",
            "X Distribution": "Grid",
            "Original PostPrune Accuracy": f"{grid_original_acc:.3f}",
            "New PostPrune Accuracy": f"{grid_new_acc:.3f}",
        },
        {
            "Dataset": "circular",
            "X Distribution": "Uniform",
            "Original PostPrune Accuracy": f"{uniform_original_acc:.3f}",
            "New PostPrune Accuracy": f"{uniform_new_acc:.3f}",
        },
    ]

    # Save results
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    file_path = os.path.join(OUTPUT_PATH, "post_prune_comparison.csv")

    with open(file_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {file_path}")


if __name__ == "__main__":
    main()
