import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import infrastructure.dgp.data_generation as data_generation

X_train, X_test, y_train, y_test, f_train, f_test = (
    data_generation.generate_X_y_f_classification(
        random_state=7,
        dgp_name="additive_sparse_jump",
        n_samples=2000,
        feature_dim=30,
    )
)


# plot scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette="Set2")
plt.title("Scatter Plot of Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Class")
plt.show()


X_train, X_test, y_train, y_test, f_train, f_test = (
    data_generation.generate_X_y_f_classification(
        random_state=7,
        dgp_name="additive_sparse_smooth",
        n_samples=2000,
        feature_dim=2,
    )
)


# plot scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette="Set2")
plt.title("Scatter Plot of Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Class")
plt.show()
print(np.mean(f_train * (1 - f_train)))
