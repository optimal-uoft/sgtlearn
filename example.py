from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sgtlearn

X, y = make_classification(
    n_samples=500,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = sgtlearn.SGTClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

print(f"train accuracy: {model.score(X_train, y_train):.3f}")
print(f"test accuracy:  {model.score(X_test, y_test):.3f}")
