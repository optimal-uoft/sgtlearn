"""Out-of-the-box fitting and optional-dependency behavior."""

import subprocess
import sys

import numpy as np

from sgtlearn import SGTClassifier, SGTRegressor


def test_default_trees_fit_both_tasks_and_return_original_regression_units():
    X = np.arange(300.0)[:, None]
    labels = np.where(X[:, 0] < 150, "low", "high")
    classifier = SGTClassifier().fit(X, labels)
    assert classifier.score(X, labels) > 0.9
    targets = 1000 + 3 * X[:, 0]
    regressor = SGTRegressor().fit(X, targets)
    prediction = regressor.predict(X)
    assert prediction.shape == targets.shape
    assert regressor.score(X, targets) > 0.9
    assert prediction.min() > 900


def test_regular_import_does_not_require_or_load_optuna():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from sgtlearn.tune import recommended_params; "
                "from sgtlearn import SGTClassifier; "
                "SGTClassifier(**recommended_params(use_tao=True, use_bivariate=True)); "
                "assert 'optuna' not in sys.modules"
            ),
        ],
        check=True,
    )
