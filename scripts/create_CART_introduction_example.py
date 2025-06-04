from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import infrastructure.noise_level_estimator as noise_est

BOUNDARY_COLORS = ["#FF7F7F", "#00CED1"]  # Default boundary colors
SCATTER_COLORS = ["red", "blue"]  # Default scatter colors
SCATTER_ALPHA = 0.6
CONTOUR_ALPHA = 0.7


def plot_decision_boundaries(
    X,
    y,
    max_depth=1,
    max_leaf_nodes=None,
    f_train=None,
    classifier=None,
    text_box_type=1,
    boundary_colors=BOUNDARY_COLORS,
    scatter_colors=SCATTER_COLORS,
    scatter_alpha=1.0,
    contour_alpha=CONTOUR_ALPHA,
    show_text_boxes=True,
    show_title=False,
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
    text_box_type : int
        1: Shows MSE ≥ noise level
        2: Shows MSE < noise level with early stop message in red
        3: Shows MSE < noise level with "Learning noise" in black
    boundary_colors : list of str
        Two colors for decision boundaries [class0_color, class1_color]
    scatter_colors : list of str
        Two colors for scatter points [class0_color, class1_color]
    show_text_boxes : bool, optional (default=True)
        Whether to show the text boxes with depth, leaves, and MSE information
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

    # Calculate weighted MSE
    weighted_mse = np.average(leaf_mses, weights=leaf_weights)
    # Create a mesh grid with even higher resolution
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Make predictions for each point in the mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

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
        fontsize=14,
        labelpad=15,
        fontfamily="Times New Roman",
    )
    plt.ylabel(
        "X2",
        fontsize=14,
        labelpad=15,
        fontfamily="Times New Roman",
    )

    if max_leaf_nodes is not None:
        growing_type = "Best First Search"
    else:
        growing_type = "Breadth First Search"
    if show_title:
        plt.title(
            f"CART Decision Boundaries ({growing_type})",
            fontsize=18,
            pad=20,
            fontfamily="Times New Roman",
            fontweight="bold",
        )
    # Add legend for the colors with adaptive positioning
    if show_text_boxes:
        # Original position when text boxes are shown
        bbox_to_anchor = (1.01, 0.11)
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=["Class 0", "Class 1"],
            bbox_to_anchor=bbox_to_anchor,
            loc="upper left",
        )

    # Add textbox with depth and leaves information - only if show_text_boxes is True
    if show_text_boxes:
        info_text = (
            f"Depth: {classifier.get_depth()}\nLeaves: {classifier.get_n_leaves()}"
        )
        plt.text(
            0.5,
            0.9,
            info_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.9, zorder=5),
            horizontalalignment="center",
        )

    # Add textbox with mean estimated noise level (indicate it in latex via sigma hat) compared to the weighted MSE, indicate also which is higher
    # set a breakpoint here if mean_estimated_noise_lvl and weighted_mse are the same up to 2 decimal places
    if f_train is None:
        noise_text = f"${mean_estimated_noise_lvl:.2f} = \hat{{\sigma}}^2$"
    else:
        noise_text = f"${np.mean(f_train * (1 - f_train)):.2f} = \sigma^2$"
    mse_text = f"MSE = {weighted_mse:.2f}"

    # Replace the if-else block for text box with new logic
    if show_text_boxes:
        if text_box_type == 1:
            plt.text(
                0.5,
                0.8,
                f"{mse_text} $\geq$ {noise_text}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.9, zorder=5),
                horizontalalignment="center",
            )
        elif text_box_type == 2:
            plt.text(
                0.5,
                0.8,
                f"{mse_text} $<$ {noise_text}\nEARLY STOP",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.9, zorder=5),
                horizontalalignment="center",
                color="red",
                fontweight="bold",
            )
        elif text_box_type == 3:
            plt.text(
                0.5,
                0.8,
                f"{mse_text} $<$ {noise_text}\nLearning noise",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.9, zorder=5),
                horizontalalignment="center",
                color="black",
                fontweight="bold",
            )
        elif text_box_type == 4:
            plt.text(
                0.5,
                0.8,
                f"{mse_text} $<$ {noise_text}\nMax Depth Reached",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.9, zorder=5),
                horizontalalignment="center",
                color="black",
                fontweight="bold",
            )
        elif text_box_type == 5:
            plt.text(
                0.5,
                0.8,
                f"{mse_text} $\leq$ {noise_text}\nMax Depth Reached",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.9, zorder=5),
                horizontalalignment="center",
                color="black",
                fontweight="bold",
            )

    if f_train is None:
        plt.ylim(1.8, 4.5)
        plt.xlim(4, 8)
    else:
        plt.ylim(0, 1)
        plt.xlim(0, 1)
    # plt.show()

    return plt, classifier


iris = load_iris()
X = iris.data[:, :2]
y = iris.target

y[y == 2] = 1

fig, clf = plot_decision_boundaries(
    X,
    y,
    max_depth=None,
    max_leaf_nodes=4,
    show_text_boxes=False,
)

# save the figure
fig.savefig(
    "experiments/experiment_analysis/dt_es_simulation_study/plots/CART_example_boundaries.png",
    bbox_inches="tight",
    dpi=300,
)

# plot the tree structure
plt.figure(figsize=(10, 10), dpi=300)
plot_tree(clf, impurity=False, rounded=True, filled=True)
plt.savefig(
    "experiments/experiment_analysis/dt_es_simulation_study/plots/CART_example_tree.png"
)
print("done")
