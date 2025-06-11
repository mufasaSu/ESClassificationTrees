SHOW_PLOTS = False

from collections import deque
import os
import sys

from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from infrastructure.algorithms.clean_dt import DecisionTreeLevelWise as custom_tree
import infrastructure.config_reader as config_reader
from infrastructure.dgp.data_generation import create_test_sample_data


def test_initialization():
    # Test initialization and default attribute values
    tree = custom_tree()
    assert tree.leaf_count == 0, "Leaf count should initialize at 0"
    assert tree.max_depth_reached == -1, "Max depth reached should initialize at 0"


def test_fitting_tree():
    X, y = create_test_sample_data()
    tree = custom_tree(max_depth=5, min_samples_split=2, kappa=None)
    tree.fit(X, y)

    # Check that the tree depth and leaf count are as expected
    assert tree.get_n_leaves() >= 1, "Tree should have at least one leaf after fitting"
    assert tree.get_depth() >= 1, "Max depth reached should be at least 1 after fitting"
    assert tree.get_depth() <= 5, "Max depth should not exceed specified max_depth"


def test_depth_tracking():
    X, y = create_test_sample_data(case="chess")
    tree = custom_tree(max_depth=2, min_samples_split=2, kappa=None)
    tree.fit(X, y)

    # Check depth tracking during and after fitting
    # tree.plot_splits(X, y)
    assert (
        tree.get_depth() == 2
    ), "Max depth reached should not exceed the specified depth limit"


def test_leaf_count():
    X, y = create_test_sample_data(case="chess")
    tree = custom_tree(max_depth=1, min_samples_split=2, kappa=None)
    tree.fit(X, y)
    # Check the leaf count
    # tree.plot_splits(X, y)
    assert tree.get_n_leaves() == 2
    tree = custom_tree(max_depth=2, min_samples_split=2, kappa=None)
    tree.fit(X, y)
    # Check the leaf count
    assert tree.get_n_leaves() == 4
    X, y = create_test_sample_data(case="rectangular_top_right")
    tree = custom_tree(max_depth=4, min_samples_split=4, kappa=None)
    tree.fit(X, y)
    # Check the leaf count
    if SHOW_PLOTS:
        tree.plot_splits(X, y)
    assert tree.get_n_leaves() == 3


def test_max_depth():
    X, y = create_test_sample_data(case="chess")
    tree = custom_tree(max_depth=2, min_samples_split=2, kappa=None)
    tree.fit(X, y)
    assert (
        tree.get_depth() == 2
    ), "Max depth reached should not exceed the specified depth limit"


def test_min_samples_split():
    X, y = create_test_sample_data(case="chess")
    tree = custom_tree(max_depth=4, min_samples_split=5, kappa=None)
    tree.fit(X, y)
    assert tree.get_depth() == 3
    assert tree.get_n_leaves() == 5


def test_immediate_es():
    X, y = create_test_sample_data(case="chess")
    tree = custom_tree(max_depth=4, min_samples_split=2, kappa=1)
    # root node ist unter kappa
    tree.fit(X, y)
    assert tree.get_depth() == 0
    assert tree.get_n_leaves() == 1


def test_depth_two_es():
    X, y = create_test_sample_data(case="mixed_blobs")
    tree = custom_tree(max_depth=None, min_samples_split=2, kappa=0.2)
    # print a new line
    print("\n")
    tree.fit(X, y)
    assert tree.get_depth() == 1
    assert tree.get_n_leaves() == 2


def test_prediction_formats():
    X, y = create_test_sample_data()
    tree = custom_tree(max_depth=3, min_samples_split=2, kappa=None)
    tree.fit(X, y)

    # Make predictions on the same dataset
    predictions = tree.predict(X)
    assert len(predictions) == len(
        y
    ), "Number of predictions should match number of samples in X"
    predictions_proba = tree.predict_proba(X)
    assert predictions_proba.shape[0] == len(
        y
    ), "Number of predictions should match number of samples in X"
    assert predictions_proba.shape[1] == 2, "Predict_proba should return two columns"


def test_traverse_logic():
    X, y = create_test_sample_data(case="one_X_cut")
    tree = custom_tree(max_depth=3, min_samples_split=2, kappa=None)
    tree.fit(X, y)
    X_probe = np.array([0.1, 0.1])
    assert tree.predict_proba(X_probe, depth=1)[0, 1] == tree.root.node_prediction

    first_split_tresh = tree.root.split_threshold
    if X_probe[tree.root.feature] <= first_split_tresh:
        assert (
            tree.predict_proba(X_probe, depth=2)[0, 1]
            == tree.root.left_child.node_prediction
        )
        assert tree.predict(X_probe, depth=2) == (
            tree.root.right_child.node_prediction < 0.5
        )
    else:
        assert (
            tree.predict_proba(X_probe, depth=2)[0, 1]
            == tree.root.right_child.node_prediction
        )
        assert tree.predict(X_probe, depth=2) == (
            tree.root.right_child.node_prediction >= 0.5
        )
    X, y = create_test_sample_data(case="chess_simple")
    tree = custom_tree(max_depth=None, min_samples_split=3, kappa=None)
    tree.fit(X, y)
    assert tree.predict_proba(X_probe)[0, 1] == 0.5  # equals a mixed node


def test_mtry_sqrt():
    X, y = create_test_sample_data(case="mixed_blobs")
    tree = custom_tree(
        max_depth=None, min_samples_split=2, kappa=None, max_features="sqrt"
    )
    tree.fit(X, y)

    count_feature_one = 0
    for _ in range(tree.get_n_leaves()):
        feature_indices = np.random.choice(
            range(X.shape[1]),  # to choose from
            int(np.sqrt(X.shape[1])),  # number of features to choose
            replace=False,
        )  # creates array of indices
        count_feature_one += feature_indices[0] == 1
    assert 0.40 <= count_feature_one / tree.get_n_leaves() <= 0.60


def test_best_split():
    X, y = create_test_sample_data(case="mixed_blobs")
    tree = custom_tree(
        max_depth=None, min_samples_split=2, kappa=None, max_features="all"
    )
    tree._best_split(X, y, np.arange(X.shape[0]))


# def bfs_attributes(root):
#     queue = deque([root])  # Initialize the queue with the root node

#     while queue:
#         current_node = queue.popleft()  # Get the next node in the queue

#         # Print or process the attributes you need
#         print(current_node.feature, current_node.split_threshold)


#         # Add children to the queue if they exist
#         if current_node.left_child:
#             queue.append(current_node.left_child)
#         if current_node.right_child:
#             queue.append(current_node.right_child)
def test_es_offset():
    X, y = create_test_sample_data(case="mixed_blobs")
    tree = custom_tree(
        max_depth=None, min_samples_split=2, kappa=0.2, max_features="all", es_offset=0
    )
    tree.fit(X, y)
    print(tree.get_depth())
    X, y = create_test_sample_data(case="mixed_blobs")
    tree = custom_tree(
        max_depth=None, min_samples_split=2, kappa=0.2, max_features="all", es_offset=1
    )
    tree.fit(X, y)
    print(tree.get_depth())


def test_rf_train_mse():
    X, y = create_test_sample_data(case="mixed_blobs")

    # sample with replacement from X and y
    X_sample, y_sample = resample(X, y, replace=True)
    tree = custom_tree(
        max_depth=3,
        min_samples_split=2,
        kappa=0.01,
        max_features="sqrt",
        rf_train_mse=True,
        random_state=7,
    )
    tree.fit(X_sample, y_sample, X, y)
    tree_rf = custom_tree(
        max_depth=3,
        min_samples_split=2,
        kappa=0.01,
        max_features="sqrt",
        rf_train_mse=False,
        random_state=7,
    )
    tree_rf.fit(X_sample, y_sample, X, y)
    print(tree.get_depth())
    print(tree_rf.get_depth())
    assert tree.get_depth() == tree_rf.get_depth()


def compare_to_scikit_tree():
    X, y = create_test_sample_data(case="mixed_blobs")
    depth = 11
    tree = custom_tree(
        max_depth=depth,
        min_samples_split=2,
        kappa=None,
        max_features="all",
        random_state=7,
    )
    tree.fit(X, y)
    scikit_tree = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_split=2,
        max_features=None,
        random_state=9,  # Mit nem anderen random state, bricht es hier --> tie breaking 999, no breaking 7
    )
    scikit_tree.fit(X, y)
    print(f"Custom Depth: {tree.get_depth()} == {scikit_tree.get_depth()}")
    print(f"Custom Leaves: {tree.get_n_leaves()} == {scikit_tree.get_n_leaves()}")
    assert (tree.get_depth()) == scikit_tree.get_depth()
    assert tree.get_n_leaves() == scikit_tree.get_n_leaves()


def run_tests():
    if SHOW_PLOTS:
        # For each case in create_test_sample_data, create a scatter plot and summarize to one figure
        cases = [
            "one_X_cut",
            "two_X_cuts",
            "chess_simple",
            "chess",
            "rectangular_top_right",
            "mixed_blobs",
        ]
        fig, axes = plt.subplots(
            2, (len(cases) + 1) // 2, figsize=(15, 10)
        )  # Create 2 rows of subplots

        for i, case in enumerate(cases):
            X, y = create_test_sample_data(case)
            row = i // ((len(cases) + 1) // 2)  # Determine row index
            col = i % ((len(cases) + 1) // 2)  # Determine column index

            ax = axes[row, col]
            ax.scatter(
                X[y == 0, 0],
                X[y == 0, 1],
                label="Class 0",
                marker="o",
                s=50,
                edgecolor="k",
            )
            ax.scatter(
                X[y == 1, 0],
                X[y == 1, 1],
                label="Class 1",
                marker="^",
                s=50,
                edgecolor="k",
            )
            ax.set_title(f"Case: {case}")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.legend()

        # Hide any unused subplots (if len(cases) is odd)
        for j in range(i + 1, 2 * ((len(cases) + 1) // 2)):
            fig.delaxes(axes[j // ((len(cases) + 1) // 2), j % ((len(cases) + 1) // 2)])

        plt.tight_layout()
        plt.show()
    ##############################################################################################
    test_rf_train_mse()
    test_initialization()
    test_fitting_tree()
    test_depth_tracking()
    test_leaf_count()
    test_max_depth()
    test_min_samples_split()
    test_immediate_es()
    test_depth_two_es()
    test_prediction_formats()
    test_traverse_logic()
    test_mtry_sqrt()
    test_best_split()
    test_es_offset()
    compare_to_scikit_tree()
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
