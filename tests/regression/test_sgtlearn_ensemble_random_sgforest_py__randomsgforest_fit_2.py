from sgtlearn import RandomSGForestClassifier, RandomSGForestRegressor
import numpy as np
import pytest


def _clf_data(n_samples: int = 40, n_features: int = 3, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _reg_data(n_samples: int = 40, n_features: int = 3, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, n_features))
    y = 2.0 * X[:, 0] - X[:, 1]
    return X, y


def _clf(**kwargs) -> RandomSGForestClassifier:
    params = dict(n_estimators=2, tao_n_runs=0, random_state=0)
    params.update(kwargs)
    return RandomSGForestClassifier(**params)


def _reg(**kwargs) -> RandomSGForestRegressor:
    params = dict(n_estimators=2, tao_n_runs=0, random_state=0)
    params.update(kwargs)
    return RandomSGForestRegressor(**params)


@pytest.mark.parametrize("n_estimators", [0, -1])
def test_n_estimators_below_one_raises_value_error(n_estimators: int) -> None:
    X, y = _clf_data()

    with pytest.raises(ValueError):
        _clf(n_estimators=n_estimators).fit(X, y)


def test_max_samples_with_bootstrap_false_raises_value_error() -> None:
    X, y = _reg_data()

    with pytest.raises(ValueError):
        _reg(bootstrap=False, max_samples=0.5).fit(X, y)


def test_negative_max_samples_raises_value_error() -> None:
    X, y = _reg_data()

    with pytest.raises(ValueError):
        _reg(bootstrap=True, max_samples=-0.5).fit(X, y)


def test_infinite_value_in_X_raises_value_error() -> None:
    X, y = _clf_data()
    X = X.copy()
    X[5, 0] = np.inf

    with pytest.raises(ValueError):
        _clf().fit(X, y)


def test_negative_infinite_value_in_X_raises_value_error() -> None:
    X, y = _clf_data()
    X = X.copy()
    X[5, 2] = -np.inf

    with pytest.raises(ValueError):
        _clf().fit(X, y)


def test_value_overflowing_float32_raises_value_error() -> None:
    X, y = _clf_data()
    X = X.copy()
    X[7, 1] = 1e39

    with pytest.raises(ValueError):
        _clf().fit(X, y)


def test_every_row_is_validated_even_with_small_bootstrap() -> None:
    X, y = _reg_data(n_samples=200)
    X = X.copy()
    X[-1, 0] = np.inf

    with pytest.raises(ValueError):
        _reg(n_estimators=1, bootstrap=True, max_samples=0.01).fit(X, y)


def test_single_class_labels_raise_value_error() -> None:
    X, _ = _clf_data()
    y = np.zeros(X.shape[0], dtype=int)

    with pytest.raises(ValueError):
        _clf().fit(X, y)


def test_mismatched_X_and_y_lengths_raise() -> None:
    X, y = _clf_data()

    with pytest.raises((ValueError, TypeError)):
        _clf().fit(X, y[:-1])


def test_negative_sample_weight_raises() -> None:
    X, y = _clf_data()
    sw = np.ones(X.shape[0])
    sw[0] = -1.0

    with pytest.raises((ValueError, TypeError)):
        _clf().fit(X, y, sw)


def test_sample_weight_wrong_length_raises() -> None:
    X, y = _reg_data()

    with pytest.raises((ValueError, TypeError)):
        _reg().fit(X, y, np.ones(X.shape[0] - 3))


def test_invalid_class_weight_raises() -> None:
    X, y = _clf_data()

    with pytest.raises((ValueError, TypeError)):
        _clf(class_weight="not_a_valid_class_weight").fit(X, y)


def test_feature_dict_with_unknown_column_raises() -> None:
    X, y = _clf_data()

    with pytest.raises((ValueError, TypeError)):
        _clf().fit(X, y, feature_dict={"a": [0], "b": [99]})


def test_invalid_processed_features_raises() -> None:
    X, y = _clf_data()

    with pytest.raises((ValueError, TypeError)):
        _clf().fit(X, y, processed_features="not processed features")


@pytest.mark.parametrize("n_jobs", [0, "many"])
def test_invalid_n_jobs_raises(n_jobs) -> None:
    X, y = _clf_data()

    with pytest.raises((ValueError, TypeError)):
        _clf(n_jobs=n_jobs).fit(X, y)


def test_base_tree_rejects_negative_min_impurity_decrease() -> None:
    X, y = _clf_data()

    with pytest.raises((ValueError, TypeError)):
        _clf(min_impurity_decrease=-1.0).fit(X, y)


def test_base_tree_rejects_invalid_tao_n_runs() -> None:
    X, y = _reg_data()

    with pytest.raises((ValueError, TypeError)):
        _reg(tao_n_runs=-1).fit(X, y)
