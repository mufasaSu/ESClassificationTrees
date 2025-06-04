BOUNDARY_COLORS = ["#FF7F7F", "#00CED1"]  # Default boundary colors
SCATTER_COLORS = ["red", "blue"]  # Default scatter colors
SCATTER_ALPHA = 0.6
CONTOUR_ALPHA = 0.7
SHOW_TEXT_BOXES = True  # Default setting for text box visibility
SAMPLE_SIZE = 500  # vorher 250, 1000

# import decision tree classifier from scikit learn
from sklearn.tree import DecisionTreeClassifier

# import numpy
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt

# import seaborn
import seaborn as sns
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from infrastructure.dgp.data_generation import (
    generate_circular_classification,
    generate_rectangular_classification,
)
import infrastructure.noise_level_estimator as noise_est

# Move this to the beginning of the script, after imports and constants
output_folder = (
    "experiments/experiment_analysis/dt_es_simulation_study/plots/growing_images_w_boxes"
    if SHOW_TEXT_BOXES
    else "experiments/experiment_analysis/dt_es_simulation_study/plots/growing_images"
)

# Add at the beginning of the file, after other constants
MAX_LEAVES_BESTFS = 8  # Maximum leaves for best-first search with iris data
MAX_LEAVES_RECTANGULAR = 125  # Maximum leaves for rectangular data best-first search
MAX_DEPTH_RECTANGULAR = 14  # Maximum depth for rectangular data breadth-first search


def plot_decision_boundaries(
    X,
    y,
    max_depth=1,
    max_leaf_nodes=None,
    f_train=None,
    classifier=None,
    boundary_colors=BOUNDARY_COLORS,
    scatter_colors=SCATTER_COLORS,
    scatter_alpha=1.0,
    contour_alpha=CONTOUR_ALPHA,
    show_text_boxes=SHOW_TEXT_BOXES,
):
    """
    Plot the decision boundaries of a classifier for 2D data, showing all split edges.

    Parameters:
    -----------
    X : array-like of shape (n_samples, 2)
        The input features
    y : array-like of shape (n_samples,)
        The target values
    max_depth : int, optional
        Maximum depth of the tree
    max_leaf_nodes : int, optional
        Maximum number of leaf nodes
    f_train : array-like, optional
        Training data for noise level calculation
    classifier : sklearn classifier object, optional
        Pre-trained classifier
    boundary_colors : list of str
        Two colors for decision boundaries [class0_color, class1_color]
    scatter_colors : list of str
        Two colors for scatter points [class0_color, class1_color]
    show_text_boxes : bool
        Whether to display the text boxes with MSE and depth information
    """

    if classifier is None:
        classifier = DecisionTreeClassifier(
            max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42
        )
        classifier.fit(X, y)
    if f_train is None:
        noise_estimator = noise_est.Estimator(X, y)
        mean_estimated_noise_lvl = noise_estimator.estimate_1NN()
    else:
        mean_estimated_noise_lvl = np.mean(f_train * (1 - f_train))

    # Get tree structure
    tree = classifier.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    n_samples = tree.n_node_samples
    values = tree.value

    # Calculate MSE for each leaf node
    leaf_mses = []
    leaf_weights = []

    for node_id in range(n_nodes):
        # Check if node is leaf
        if children_left[node_id] == children_right[node_id]:
            # Get probability distribution at leaf
            probs = values[node_id]
            p_hat = probs.max()
            # Calculate MSE at leaf as p*(1-p) where p is prob of majority class
            leaf_mse = p_hat * (1 - p_hat)
            leaf_mses.append(leaf_mse)
            leaf_weights.append(n_samples[node_id])

    # Calculate weighted MSE and critical value, rounded to 2 decimals
    weighted_mse = round(np.average(leaf_mses, weights=leaf_weights), 2)

    if f_train is None:
        noise_estimator = noise_est.Estimator(X, y)
        critical_value = round(noise_estimator.estimate_1NN(), 2)
        noise_text = f"${critical_value} = \hat{{\sigma}}^2$"
    else:
        critical_value = round(np.mean(f_train * (1 - f_train)), 2)
        noise_text = f"${critical_value} = \sigma^2$"

    # Create a mesh grid with even higher resolution
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Make predictions for each point in the mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(10, 10))  # vorher 10, 8

    # Create colormaps using the provided colors
    boundary_cmap = plt.cm.colors.ListedColormap(boundary_colors)
    scatter_cmap = plt.cm.colors.ListedColormap(scatter_colors)

    # Plot decision boundaries with new colors
    plt.contourf(xx, yy, Z, cmap=boundary_cmap, alpha=contour_alpha)

    # Plot scatter points with original colors and save to variable
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=scatter_cmap,
        alpha=scatter_alpha,
        s=25,
        zorder=2,
        edgecolors="none",
    )

    # Get the tree structure
    n_nodes = classifier.tree_.node_count
    children_left = classifier.tree_.children_left
    children_right = classifier.tree_.children_right
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold

    # Plot all split lines
    def plot_split_line(node_idx, x_min, x_max, y_min, y_max):
        if node_idx < n_nodes and children_left[node_idx] != children_right[node_idx]:
            if feature[node_idx] == 0:  # vertical line
                plt.plot(
                    [threshold[node_idx], threshold[node_idx]],
                    [y_min, y_max],
                    "k-",
                    linewidth=2,
                )
                plot_split_line(
                    children_left[node_idx], x_min, threshold[node_idx], y_min, y_max
                )
                plot_split_line(
                    children_right[node_idx], threshold[node_idx], x_max, y_min, y_max
                )
            else:  # horizontal line
                plt.plot(
                    [x_min, x_max],
                    [threshold[node_idx], threshold[node_idx]],
                    "k-",
                    linewidth=2,
                )
                plot_split_line(
                    children_left[node_idx], x_min, x_max, y_min, threshold[node_idx]
                )
                plot_split_line(
                    children_right[node_idx], x_min, x_max, threshold[node_idx], y_max
                )

    # Set style to grey grid
    plt.style.use("ggplot")
    plt.grid(
        True, alpha=0.4, zorder=1
    )  # Add explicit grid with higher alpha and lower zorder

    # Plot all split lines recursively
    plot_split_line(0, x_min, x_max, y_min, y_max)

    # Set labels
    plt.xlabel(
        "X1",
        fontsize=22,
        labelpad=15,
        fontfamily="Times New Roman",
    )
    plt.ylabel(
        "X2",
        fontsize=22,
        labelpad=15,
        fontfamily="Times New Roman",
    )

    # Increase tick size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if max_leaf_nodes is not None:
        growing_type = "Best First Search"
    else:
        growing_type = "Breadth First Search"

    # Simplified text box logic
    if show_text_boxes:
        # Add depth and leaves info with larger font size
        info_text = (
            f"Depth: {classifier.get_depth()}\nLeaves: {classifier.get_n_leaves()}"
        )
        plt.text(
            0.5,
            0.9,
            info_text,
            transform=plt.gca().transAxes,
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                zorder=5,
                pad=0.5,  # Add padding
                boxstyle="round,pad=0.5",  # Rounded corners and padding
            ),
            horizontalalignment="center",
            fontsize=16,  # Increased font size
        )

        mse_text = f"MSE = {weighted_mse}"

        # Check if this is the last iteration (both parameters are None)
        is_last_iteration = max_depth is None and max_leaf_nodes is None

        # Compare rounded values
        if is_last_iteration:
            # Last iteration - show max depth/leaves reached
            text = f"{mse_text} $\leq$ {noise_text}"
            color = "black"
            extra_text = "\nMax depth reached"
        elif weighted_mse > critical_value:
            # MSE > critical value - continue splitting
            text = f"{mse_text} $>$ {noise_text}"
            color = "black"
            extra_text = ""
            global early_stop_shown
            early_stop_shown = False
        elif weighted_mse <= critical_value and not early_stop_shown:
            # First time MSE <= critical value - show early stop
            text = f"{mse_text} $\leq$ {noise_text}"
            color = "red"
            extra_text = "\nEARLY STOP"
            early_stop_shown = True
        else:
            # MSE <= critical value and not first occurrence - show learning noise
            text = f"{mse_text} $\leq$ {noise_text}"
            color = "black"
            extra_text = "\nLearning noise"

        plt.text(
            0.5,
            0.8,
            text + extra_text,
            transform=plt.gca().transAxes,
            bbox=dict(
                facecolor="white",
                alpha=0.9,
                zorder=5,
                pad=0.5,  # Add padding
                boxstyle="round,pad=0.5",  # Rounded corners and padding
            ),
            horizontalalignment="center",
            color=color,
            fontweight="bold" if extra_text else "normal",
            fontsize=16,  # Increased font size
        )

    if f_train is None:
        plt.ylim(1.8, 4.5)
        plt.xlim(4, 8)
    else:
        plt.ylim(0, 1)
        plt.xlim(0, 1)
    # plt.show()

    return plt, classifier


def plot_initial_state(
    X,
    y,
    f_train=None,
    scatter_colors=SCATTER_COLORS,
    scatter_alpha=1.0,
    show_text_boxes=SHOW_TEXT_BOXES,
):
    """
    Plot the initial state (depth=0) showing only the scatter plot with textboxes.
    """
    # Create the plot
    plt.figure(figsize=(10, 10))  # vorher 10, 8

    # Create a custom colormap for scatter points
    scatter_cmap = plt.cm.colors.ListedColormap(scatter_colors)

    # Set style to grey grid
    plt.style.use("ggplot")
    plt.grid(True, alpha=0.4, zorder=1)

    # Plot the training points
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=scatter_cmap,
        s=25,
        zorder=2,
        alpha=scatter_alpha,
        edgecolors="none",
    )

    # Set labels
    plt.xlabel(
        "X1",
        fontsize=22,
        labelpad=15,
        fontfamily="Times New Roman",
    )
    plt.ylabel(
        "X2",
        fontsize=22,
        labelpad=15,
        fontfamily="Times New Roman",
    )

    # Increase tick size
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Modify text box additions to be conditional
    if show_text_boxes:
        # Add textbox with depth and leaves information
        info_text = "Depth: 0\nLeaves: 1"
        plt.text(
            0.5,
            0.9,
            info_text,
            transform=plt.gca().transAxes,
            bbox=dict(
                facecolor="white",
                alpha=1,
                zorder=5,
                pad=0.5,  # Add padding
                boxstyle="round,pad=0.5",  # Rounded corners and padding
            ),
            horizontalalignment="center",
            fontsize=16,  # Increased font size
        )

        # Calculate initial MSE
        unique, counts = np.unique(y, return_counts=True)
        prob_majority = np.max(counts) / len(y)
        weighted_mse = round(prob_majority * (1 - prob_majority), 2)

        # Get critical value
        if f_train is None:
            noise_estimator = noise_est.Estimator(X, y)
            critical_value = round(noise_estimator.estimate_1NN(), 2)
            noise_text = f"${critical_value} = \hat{{\sigma}}^2$"
        else:
            critical_value = round(np.mean(f_train * (1 - f_train)), 2)
            noise_text = f"${critical_value} = \sigma^2$"

        mse_text = f"MSE = {weighted_mse}"

        # Compare rounded values
        if weighted_mse > critical_value:
            text = f"{mse_text} $>$ {noise_text}"
            color = "black"
            extra_text = ""
        elif weighted_mse == critical_value:
            text = f"{mse_text} $=$ {noise_text}"
            color = "red"
            extra_text = "\nEARLY STOP"
        else:
            text = f"{mse_text} $<$ {noise_text}"
            color = "black"
            extra_text = "\nLearning noise"

        plt.text(
            0.5,
            0.8,
            text + extra_text,
            transform=plt.gca().transAxes,
            bbox=dict(
                facecolor="white",
                alpha=1,
                zorder=5,
                pad=0.5,  # Add padding
                boxstyle="round,pad=0.5",  # Rounded corners and padding
            ),
            horizontalalignment="center",
            color=color,
            fontweight="bold" if extra_text else "normal",
            fontsize=16,  # Increased font size
        )

    if f_train is None:
        plt.ylim(1.8, 4.5)
        plt.xlim(4, 8)
    else:
        plt.ylim(0, 1)
        plt.xlim(0, 1)

    return plt


# Example usage:
# load iris dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
y[y == 2] = 1
np.random.seed(7)
X_train = np.random.rand(SAMPLE_SIZE, 2)
y_train, f_train = generate_rectangular_classification(X_train, 0.9)

import imageio

# Create a list to store the filenames of the generated plots
filenames = []

################################################################################
# BEST FIRST SEARCH ############################################################
################################################################################
# First, generate the initial state plot for best-first search
plt = plot_initial_state(X, y)
# make dir "plots/growing_images/single_images/max_depth_bestfs/"
os.makedirs(f"{output_folder}/single_images/max_depth_bestfs/", exist_ok=True)
filename = (
    f"{output_folder}/single_images/max_depth_bestfs/decision_boundary_bestfs_0.png"
)
plt.savefig(filename, bbox_inches="tight", dpi=300)
filenames.append(filename)
plt.close()

# For best-first search with iris data
early_stop_shown = False
for max_leaf_nodes in list(range(2, 8)) + [None]:
    plt, classifier = plot_decision_boundaries(
        X,
        y,
        max_depth=(
            None if max_leaf_nodes is None else None
        ),  # Both None for last iteration
        max_leaf_nodes=max_leaf_nodes,
        show_text_boxes=SHOW_TEXT_BOXES,
    )
    leaf_num = MAX_LEAVES_BESTFS if max_leaf_nodes is None else max_leaf_nodes
    filename = f"{output_folder}/single_images/max_depth_bestfs/decision_boundary_bestfs_{leaf_num}.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    filenames.append(filename)
    plt.close()

# Create gif for best-first search
with imageio.get_writer(
    f"{output_folder}/iris_max_depth_bestfs.gif", mode="I", fps=0.5
) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

################################################################################
# RECTANGULAR DATA BEST FIRST SEARCH ###############################################
################################################################################
filenames = []

# First, generate the initial state plot
plt = plot_initial_state(X_train, y_train, f_train=f_train)
# make dir "plots/growing_images/single_images/rectangular_bestfs/"
os.makedirs(f"{output_folder}/single_images/rectangular_bestfs/", exist_ok=True)
filename = f"{output_folder}/single_images/rectangular_bestfs/rectangular_decision_boundary_bestfs_1.png"
plt.savefig(filename, bbox_inches="tight", dpi=300)
filenames.append(filename)
plt.close()

early_stop_shown = False
for max_leaf_nodes in list(range(2, 18)) + [None]:
    plt, classifier = plot_decision_boundaries(
        X_train,
        y_train,
        max_depth=(
            None if max_leaf_nodes is None else None
        ),  # Both None for last iteration
        max_leaf_nodes=max_leaf_nodes,
        f_train=f_train,
        show_text_boxes=SHOW_TEXT_BOXES,
    )
    filename = f"{output_folder}/single_images/rectangular_bestfs/rectangular_decision_boundary_bestfs_{max_leaf_nodes if max_leaf_nodes is not None else 18}.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    filenames.append(filename)
    plt.close()

# Create gif for best-first search
with imageio.get_writer(
    f"{output_folder}/rectangular_decision_boundaries_bestfs.gif", mode="I", fps=0.5
) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

################################################################################
# RECTANGULAR DATA BREADTH FIRST SEARCH ############################################
################################################################################
filenames = []

# First, generate the initial state plot
plt = plot_initial_state(X_train, y_train, f_train=f_train)
# make dir "plots/growing_images/single_images/rectangular_breadthfs/"
os.makedirs(f"{output_folder}/single_images/rectangular_breadthfs/", exist_ok=True)
filename = f"{output_folder}/single_images/rectangular_breadthfs/rectangular_decision_boundary_breadthfs_0.png"
plt.savefig(filename, bbox_inches="tight", dpi=300)
filenames.append(filename)
plt.close()

early_stop_shown = False
for max_depth in list(range(1, 14)) + [None]:
    plt, classifier = plot_decision_boundaries(
        X_train,
        y_train,
        max_depth=max_depth,
        max_leaf_nodes=(
            None if max_depth is None else None
        ),  # Both None for last iteration
        f_train=f_train,
        show_text_boxes=SHOW_TEXT_BOXES,
    )
    depth_str = MAX_DEPTH_RECTANGULAR if max_depth is None else str(max_depth)
    filename = f"{output_folder}/single_images/rectangular_breadthfs/rectangular_decision_boundary_breadthfs_{depth_str}.png"
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    filenames.append(filename)
    plt.close()
# Create gif for breadth-first search
with imageio.get_writer(
    f"{output_folder}/rectangular_decision_boundaries_breadthfs.gif", mode="I", fps=0.5
) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
