import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sgtlearn import SGTClassifier, plot_tree, make_plus

# 3x3 "Plus Sign" dataset from the README: 
X, y = make_plus(n_samples=1500, grid=3, margin=0.07, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# num_partitions=2 -> binary outer branches; inner_max_depth=2 lets each node's
# shape function carve one feature into up to 4 bins
model = SGTClassifier(
    max_depth=2,
    inner_max_depth=2,
    random_state=42,
)
model.fit(X_train, y_train)

print(f"train accuracy: {model.score(X_train, y_train):.3f}")
print(f"test accuracy:  {model.score(X_test, y_test):.3f}")


# Visualize the dataset and the learned SGT
fig, (ax_data, ax_tree) = plt.subplots(1, 2, figsize=(18, 7))


# right: the learned SGT (pass X to get sample routing histograms)
plot_tree(model, X=X_train, ax=ax_tree)
ax_tree.set_title("Learned SGT")


# left: the checkerboard
ax_data.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", s=12, edgecolors="none"
)
for k in range(1,3):
    ax_data.axvline(k, color="k", lw=0.5, alpha=0.3)
    ax_data.axhline(k, color="k", lw=0.5, alpha=0.3)
ax_data.set_xlabel("$x_1$")
ax_data.set_ylabel("$x_2$")
ax_data.set_title("Plus Sign dataset")


plt.tight_layout()
plt.savefig("tree.png", dpi=150)
plt.show()
