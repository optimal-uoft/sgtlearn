import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError

import sgtlearn.tao as m

try:
    from sgtlearn import SGTClassifier, SGTRegressor
except ImportError:  # pragma: no cover
    SGTClassifier = m.SGTClassifier
    SGTRegressor = m.SGTRegressor

N_FEATURES = 3


def _data(n=40, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, N_FEATURES)
    return X


@pytest.fixture(scope="module")
def clf():
    X = _data()
    y = (X[:, 0] > 0.5).astype(int)
    return SGTClassifier().fit(X, y)


@pytest.fixture(scope="module")
def reg():
    X = _data()
    y = X[:, 0] * 2.0 + X[:, 1]
    return SGTRegressor().fit(X, y)


def _cls_y(n=10):
    return np.array([0, 1] * (n // 2))


def test_classifier_valid_input_returns_float64_arrays(clf):
    X = _data(10, seed=1)
    Xo, yo = m._validate_X_y(clf, X, _cls_y(), check_input=True)
    assert isinstance(Xo, np.ndarray) and isinstance(yo, np.ndarray)
    assert Xo.dtype == np.float64
    assert Xo.shape == (10, N_FEATURES)
    assert Xo.shape[0] == yo.shape[0]
    np.testing.assert_allclose(Xo, X)


def test_list_input_converted_to_float64_when_checking(clf):
    X = [[1, 2, 3], [4, 5, 6]]
    Xo, yo = m._validate_X_y(clf, X, [0, 1], check_input=True)
    assert Xo.dtype == np.float64
    assert isinstance(yo, np.ndarray)
    np.testing.assert_array_equal(Xo, np.array(X, dtype=float))


def test_classifier_string_labels_returned_as_ndarray(clf):
    X = _data(4, seed=2)
    Xo, yo = m._validate_X_y(clf, X, ["a", "b", "a", "b"], check_input=True)
    assert isinstance(yo, np.ndarray)
    assert yo.shape[0] == 4


def test_nan_in_X_allowed_when_checking(clf):
    X = _data(10, seed=3)
    X[2, 1] = np.nan
    Xo, _ = m._validate_X_y(clf, X, _cls_y(), check_input=True)
    assert np.isnan(Xo[2, 1])
    assert Xo.dtype == np.float64


def test_inf_in_X_rejected_when_checking(clf):
    X = _data(10, seed=4)
    X[0, 0] = np.inf
    with pytest.raises(ValueError):
        m._validate_X_y(clf, X, _cls_y(), check_input=True)


def test_non_numeric_X_rejected_when_checking(clf):
    X = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=object)
    with pytest.raises(ValueError):
        m._validate_X_y(clf, X, [0, 1], check_input=True)


@pytest.mark.parametrize("check_input", [True, False])
def test_1d_X_rejected(clf, check_input):
    with pytest.raises(ValueError):
        m._validate_X_y(clf, np.arange(3.0), np.array([0, 1, 0]), check_input=check_input)


@pytest.mark.parametrize("check_input", [True, False])
def test_3d_X_rejected(clf, check_input):
    X = np.zeros((4, N_FEATURES, 2))
    with pytest.raises(ValueError):
        m._validate_X_y(clf, X, _cls_y(4), check_input=check_input)


@pytest.mark.parametrize("check_input", [True, False])
def test_sparse_X_rejected(clf, check_input):
    X = sp.csr_matrix(_data(10, seed=5))
    with pytest.raises(ValueError):
        m._validate_X_y(clf, X, _cls_y(), check_input=check_input)


@pytest.mark.parametrize("check_input", [True, False])
def test_empty_X_rejected(clf, check_input):
    X = np.empty((0, N_FEATURES))
    with pytest.raises(ValueError):
        m._validate_X_y(clf, X, np.array([], dtype=int), check_input=check_input)


@pytest.mark.parametrize("check_input", [True, False])
def test_feature_count_mismatch_message_names_counts_and_class(clf, check_input):
    X = np.random.RandomState(6).rand(10, N_FEATURES + 2)
    with pytest.raises(ValueError) as exc:
        m._validate_X_y(clf, X, _cls_y(), check_input=check_input)
    msg = str(exc.value)
    assert str(N_FEATURES) in msg
    assert str(N_FEATURES + 2) in msg
    assert type(clf).__name__ in msg


@pytest.mark.parametrize("check_input", [True, False])
def test_scalar_y_rejected(clf, check_input):
    with pytest.raises(ValueError):
        m._validate_X_y(clf, _data(1, seed=7), np.array(1), check_input=check_input)


@pytest.mark.parametrize("check_input", [True, False])
def test_sample_count_mismatch_rejected_classifier(clf, check_input):
    with pytest.raises(ValueError):
        m._validate_X_y(clf, _data(10, seed=8), _cls_y(8), check_input=check_input)


@pytest.mark.parametrize("check_input", [True, False])
def test_sample_count_mismatch_rejected_regressor(reg, check_input):
    with pytest.raises(ValueError):
        m._validate_X_y(reg, _data(10, seed=8), np.arange(7.0), check_input=check_input)


def test_regressor_valid_single_output(reg):
    X = _data(10, seed=9)
    y = np.linspace(0, 1, 10)
    Xo, yo = m._validate_X_y(reg, X, y, check_input=True)
    assert Xo.dtype == np.float64
    assert yo.shape[0] == 10
    np.testing.assert_allclose(np.ravel(yo), y)


def test_regressor_multi_output_accepted():
    X = _data(10, seed=10)
    y = np.random.RandomState(10).rand(10, 2)
    reg2 = SGTRegressor().fit(_data(), np.random.RandomState(0).rand(40, 2))
    Xo, yo = m._validate_X_y(reg2, X, y, check_input=True)
    assert Xo.shape[0] == yo.shape[0] == 10
    assert yo.shape == (10, 2)


def test_regressor_nan_y_rejected_when_checking(reg):
    y = np.linspace(0, 1, 10)
    y[3] = np.nan
    with pytest.raises(ValueError):
        m._validate_X_y(reg, _data(10, seed=11), y, check_input=True)


def test_regressor_inf_y_rejected_when_checking(reg):
    y = np.linspace(0, 1, 10)
    y[3] = np.inf
    with pytest.raises(ValueError):
        m._validate_X_y(reg, _data(10, seed=12), y, check_input=True)


def test_regressor_non_numeric_y_rejected_when_checking(reg):
    y = np.array(["a", "b"] * 5, dtype=object)
    with pytest.raises(ValueError):
        m._validate_X_y(reg, _data(10, seed=13), y, check_input=True)


def test_regressor_inf_X_rejected_when_checking(reg):
    X = _data(10, seed=14)
    X[1, 2] = -np.inf
    with pytest.raises(ValueError):
        m._validate_X_y(reg, X, np.linspace(0, 1, 10), check_input=True)


def test_no_dtype_coercion_without_check(clf):
    X = np.arange(6, dtype=np.int64).reshape(2, 3)
    Xo, yo = m._validate_X_y(clf, X, np.array([0, 1]), check_input=False)
    assert isinstance(Xo, np.ndarray)
    assert Xo.dtype == np.int64
    np.testing.assert_array_equal(Xo, X)


def test_lists_converted_to_ndarrays_without_check(clf):
    Xo, yo = m._validate_X_y(clf, [[1, 2, 3], [4, 5, 6]], [0, 1], check_input=False)
    assert isinstance(Xo, np.ndarray) and isinstance(yo, np.ndarray)
    assert Xo.shape == (2, 3)
    assert yo.shape[0] == 2


def test_regressor_nan_y_not_checked_without_check(reg):
    y = np.linspace(0, 1, 10)
    y[0] = np.nan
    Xo, yo = m._validate_X_y(reg, _data(10, seed=15), y, check_input=False)
    assert isinstance(yo, np.ndarray)
    assert Xo.shape[0] == yo.shape[0]


@pytest.mark.parametrize("check_input", [True, False])
def test_unfitted_model_raises_not_fitted(check_input):
    with pytest.raises(NotFittedError):
        m._validate_X_y(SGTClassifier(), _data(10), _cls_y(), check_input=check_input)


@pytest.mark.parametrize("check_input", [True, False])
def test_unfitted_regressor_raises_not_fitted(check_input):
    with pytest.raises(NotFittedError):
        m._validate_X_y(SGTRegressor(), _data(10), np.linspace(0, 1, 10), check_input=check_input)


def test_none_n_features_in_raises_not_fitted():
    model = SGTClassifier()
    model.n_features_in_ = None
    with pytest.raises(NotFittedError):
        m._validate_X_y(model, _data(10), _cls_y(), check_input=True)


@pytest.mark.parametrize("check_input", [True, False])
def test_not_fitted_takes_priority_over_bad_shapes(check_input):
    with pytest.raises(NotFittedError):
        m._validate_X_y(SGTClassifier(), np.arange(3.0), np.array(1), check_input=check_input)


def test_inputs_not_modified(clf):
    X = _data(10, seed=16).astype(np.float32)
    X[0, 0] = np.nan
    y = _cls_y()
    X_before, y_before = X.copy(), y.copy()
    m._validate_X_y(clf, X, y, check_input=True)
    np.testing.assert_array_equal(X, X_before)
    assert X.dtype == np.float32
    np.testing.assert_array_equal(y, y_before)


def test_model_not_modified(clf):
    before = clf.n_features_in_
    m._validate_X_y(clf, _data(10, seed=17), _cls_y(), check_input=True)
    assert clf.n_features_in_ == before
