from sgtlearn import RandomSGForestRegressor
import numpy as np
import pytest


def _make_estimator() -> RandomSGForestRegressor:
    return RandomSGForestRegressor(n_estimators=2, tao_n_runs=0, random_state=0)


def _make_data(n_samples: int = 6, n_features: int = 3) -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(n_samples * n_features, dtype=np.float64).reshape(n_samples, n_features) * 0.5
    y = np.arange(n_samples, dtype=np.float64) * 0.25
    return X, y


# --------------------------------------------------------------------------
# Successful normalization
# --------------------------------------------------------------------------


def test_x_out_is_c_contiguous_float32_with_same_shape() -> None:
    X, y = _make_data()
    X_out, _, _ = _make_estimator()._check_X_y(X, y, None)

    assert isinstance(X_out, np.ndarray)
    assert X_out.dtype == np.float32
    assert X_out.flags["C_CONTIGUOUS"]
    assert X_out.shape == X.shape


def test_x_out_values_match_input() -> None:
    X, y = _make_data()
    X_out, _, _ = _make_estimator()._check_X_y(X, y, None)

    np.testing.assert_array_equal(X_out, X.astype(np.float32))


def test_fortran_ordered_x_becomes_c_contiguous() -> None:
    X, y = _make_data()
    X_f = np.asfortranarray(X)
    X_out, _, _ = _make_estimator()._check_X_y(X_f, y, None)

    assert X_out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(X_out, X.astype(np.float32))


def test_list_inputs_are_accepted_as_array_like() -> None:
    X = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    y = [1.0, 2.0, 3.0]
    X_out, y_out, _ = _make_estimator()._check_X_y(X, y, None)

    assert X_out.dtype == np.float32
    assert X_out.shape == (3, 2)
    assert y_out.dtype == np.float32
    assert y_out.shape == (3,)


def test_nan_in_x_is_preserved() -> None:
    X, y = _make_data()
    X[1, 2] = np.nan
    X[4, 0] = np.nan
    X_out, _, _ = _make_estimator()._check_X_y(X, y, None)

    assert np.isnan(X_out[1, 2])
    assert np.isnan(X_out[4, 0])
    np.testing.assert_array_equal(np.isnan(X_out), np.isnan(X))
    np.testing.assert_array_equal(X_out, X.astype(np.float32))  # NaN-aware equality


def test_one_dimensional_y_stays_one_dimensional_float32() -> None:
    X, y = _make_data()
    _, y_out, _ = _make_estimator()._check_X_y(X, y, None)

    assert y_out.dtype == np.float32
    assert y_out.flags["C_CONTIGUOUS"]
    assert y_out.ndim == 1
    assert y_out.shape == y.shape
    np.testing.assert_array_equal(y_out, y.astype(np.float32))


def test_column_vector_y_stays_two_dimensional() -> None:
    X, y = _make_data()
    y_col = y.reshape(-1, 1)
    _, y_out, _ = _make_estimator()._check_X_y(X, y_col, None)

    assert y_out.dtype == np.float32
    assert y_out.flags["C_CONTIGUOUS"]
    assert y_out.shape == (X.shape[0], 1)
    np.testing.assert_array_equal(y_out, y_col.astype(np.float32))


def test_multi_output_y_keeps_its_shape() -> None:
    X, y = _make_data()
    y2 = np.asfortranarray(np.column_stack([y, y * 2.0, y + 1.0]))
    _, y_out, _ = _make_estimator()._check_X_y(X, y2, None)

    assert y_out.dtype == np.float32
    assert y_out.flags["C_CONTIGUOUS"]
    assert y_out.shape == (X.shape[0], 3)
    np.testing.assert_array_equal(y_out, y2.astype(np.float32))


def test_integer_y_is_converted_to_float32() -> None:
    X, _ = _make_data()
    y_int = np.arange(X.shape[0], dtype=np.int64)
    _, y_out, _ = _make_estimator()._check_X_y(X, y_int, None)

    assert y_out.dtype == np.float32
    np.testing.assert_array_equal(y_out, y_int.astype(np.float32))


def test_none_sample_weight_returns_none() -> None:
    X, y = _make_data()
    _, _, w_out = _make_estimator()._check_X_y(X, y, None)

    assert w_out is None


def test_sample_weight_is_returned_as_c_contiguous_float32() -> None:
    X, y = _make_data()
    w = np.array([1.0, 0.5, 0.0, 2.0, 1.0, 3.0], dtype=np.float64)
    _, _, w_out = _make_estimator()._check_X_y(X, y, w)

    assert isinstance(w_out, np.ndarray)
    assert w_out.dtype == np.float32
    assert w_out.flags["C_CONTIGUOUS"]
    assert w_out.shape == (X.shape[0],)
    np.testing.assert_array_equal(w_out, w.astype(np.float32))


def test_sample_weight_with_single_positive_value_is_accepted() -> None:
    X, y = _make_data()
    w = np.zeros(X.shape[0])
    w[3] = 1.0
    _, _, w_out = _make_estimator()._check_X_y(X, y, w)

    assert w_out.shape == (X.shape[0],)
    np.testing.assert_array_equal(w_out, w.astype(np.float32))


def test_single_sample_is_valid_input() -> None:
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([4.0])
    w = np.array([1.0])
    X_out, y_out, w_out = _make_estimator()._check_X_y(X, y, w)

    assert X_out.shape == (1, 3)
    assert y_out.shape == (1,)
    assert w_out.shape == (1,)


def test_values_at_float32_max_are_accepted() -> None:
    f32_max = float(np.finfo(np.float32).max)
    X = np.array([[f32_max, -f32_max], [0.0, 1.0]])
    y = np.array([f32_max, -f32_max])
    w = np.array([f32_max, 1.0])
    X_out, y_out, w_out = _make_estimator()._check_X_y(X, y, w)

    assert np.all(np.isfinite(X_out))
    assert np.all(np.isfinite(y_out))
    assert np.all(np.isfinite(w_out))


# --------------------------------------------------------------------------
# Purity
# --------------------------------------------------------------------------


def test_check_x_y_does_not_set_attributes_on_unfitted_estimator() -> None:
    est = _make_estimator()
    before = dict(vars(est))
    X, y = _make_data()
    est._check_X_y(X, y, np.ones(X.shape[0]))

    after = dict(vars(est))
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert after[key] is value


def test_check_x_y_does_not_set_attributes_on_fitted_estimator() -> None:
    est = _make_estimator()
    X, y = _make_data(n_samples=20)
    est.fit(X, y)
    before = dict(vars(est))

    X_new = np.ones((5, 7))
    y_new = np.ones((5, 2))
    est._check_X_y(X_new, y_new, None)

    after = dict(vars(est))
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert after[key] is value


def test_check_x_y_does_not_set_attributes_when_raising() -> None:
    est = _make_estimator()
    before = dict(vars(est))
    X, y = _make_data()
    y[0] = np.nan

    with pytest.raises(ValueError):
        est._check_X_y(X, y, None)

    after = dict(vars(est))
    assert after.keys() == before.keys()
    for key, value in before.items():
        assert after[key] is value


def test_failed_refit_leaves_previous_forest_unchanged() -> None:
    est = _make_estimator()
    X, y = _make_data(n_samples=20)
    est.fit(X, y)
    estimators_before = est.estimators_
    estimators_list_before = list(est.estimators_)
    n_features_before = est.n_features_in_
    n_outputs_before = est.n_outputs_
    pred_before = est.predict(X)

    X_bad = np.ones((8, 5))
    X_bad[2, 1] = np.inf
    with pytest.raises(ValueError):
        est.fit(X_bad, np.ones(8))

    assert est.estimators_ is estimators_before
    assert list(est.estimators_) == estimators_list_before
    assert all(a is b for a, b in zip(est.estimators_, estimators_list_before))
    assert est.n_features_in_ == n_features_before
    assert est.n_outputs_ == n_outputs_before
    np.testing.assert_array_equal(est.predict(X), pred_before)


def test_fit_rejects_overflowing_row_even_if_bootstrap_would_skip_it() -> None:
    est = RandomSGForestRegressor(
        n_estimators=2, tao_n_runs=0, random_state=0, bootstrap=True, max_samples=0.1
    )
    X, y = _make_data(n_samples=50)
    X[-1, 0] = 1e39

    with pytest.raises(ValueError, match="float32"):
        est.fit(X, y)


# --------------------------------------------------------------------------
# X errors
# --------------------------------------------------------------------------


def test_sparse_x_is_rejected() -> None:
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _make_data()
    X_sparse = sparse.csr_matrix(X)

    with pytest.raises((TypeError, ValueError)):
        _make_estimator()._check_X_y(X_sparse, y, None)


def test_one_dimensional_x_is_rejected() -> None:
    X = np.arange(6, dtype=np.float64)
    y = np.arange(6, dtype=np.float64)

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_three_dimensional_x_is_rejected() -> None:
    X = np.zeros((6, 2, 2))
    y = np.arange(6, dtype=np.float64)

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_positive_infinity_in_x_is_rejected() -> None:
    X, y = _make_data()
    X[2, 1] = np.inf

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_negative_infinity_in_x_is_rejected() -> None:
    X, y = _make_data()
    X[0, 0] = -np.inf

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_infinity_in_x_alongside_nan_is_rejected() -> None:
    X, y = _make_data()
    X[0, 0] = np.nan
    X[3, 2] = np.inf

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_positive_float32_overflow_in_x_is_rejected_naming_float32() -> None:
    X, y = _make_data()
    X[1, 1] = 1e39

    with pytest.raises(ValueError, match="float32"):
        _make_estimator()._check_X_y(X, y, None)


def test_negative_float32_overflow_in_x_is_rejected_naming_float32() -> None:
    X, y = _make_data()
    X[5, 2] = -1e39

    with pytest.raises(ValueError, match="float32"):
        _make_estimator()._check_X_y(X, y, None)


# --------------------------------------------------------------------------
# Sample count errors
# --------------------------------------------------------------------------


def test_fewer_targets_than_samples_is_rejected() -> None:
    X, y = _make_data()

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y[:-1], None)


def test_more_targets_than_samples_is_rejected() -> None:
    X, y = _make_data()
    y_long = np.concatenate([y, [1.0]])

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y_long, None)


def test_two_dimensional_y_with_mismatched_rows_is_rejected() -> None:
    X, y = _make_data()
    y2 = np.column_stack([y, y])[:-2]

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y2, None)


def test_zero_samples_is_rejected() -> None:
    X = np.empty((0, 3))
    y = np.empty((0,))

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


# --------------------------------------------------------------------------
# y errors
# --------------------------------------------------------------------------


def test_three_dimensional_y_is_rejected() -> None:
    X, _ = _make_data()
    y = np.zeros((X.shape[0], 2, 2))

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_non_numeric_y_is_rejected() -> None:
    X, _ = _make_data()
    y = np.array(["a", "b", "c", "d", "e", "f"], dtype=object)

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_nan_in_y_is_rejected() -> None:
    X, y = _make_data()
    y[2] = np.nan

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_nan_in_two_dimensional_y_is_rejected() -> None:
    X, y = _make_data()
    y2 = np.column_stack([y, y])
    y2[4, 1] = np.nan

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y2, None)


def test_positive_infinity_in_y_is_rejected() -> None:
    X, y = _make_data()
    y[0] = np.inf

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_negative_infinity_in_y_is_rejected() -> None:
    X, y = _make_data()
    y[-1] = -np.inf

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, None)


def test_float32_overflow_in_y_is_rejected_naming_float32_and_y() -> None:
    X, y = _make_data()
    y[3] = 1e39

    with pytest.raises(ValueError, match="float32") as exc_info:
        _make_estimator()._check_X_y(X, y, None)
    assert "y" in str(exc_info.value)


def test_negative_float32_overflow_in_y_is_rejected_naming_float32_and_y() -> None:
    X, y = _make_data()
    y[1] = -1e39

    with pytest.raises(ValueError, match="float32") as exc_info:
        _make_estimator()._check_X_y(X, y, None)
    assert "y" in str(exc_info.value)


# --------------------------------------------------------------------------
# sample_weight errors
# --------------------------------------------------------------------------


def test_sample_weight_too_short_is_rejected() -> None:
    X, y = _make_data()

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, np.ones(X.shape[0] - 1))


def test_sample_weight_too_long_is_rejected() -> None:
    X, y = _make_data()

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, np.ones(X.shape[0] + 1))


def test_two_dimensional_sample_weight_is_rejected() -> None:
    X, y = _make_data()

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, np.ones((X.shape[0], 2)))


def test_nan_sample_weight_is_rejected() -> None:
    X, y = _make_data()
    w = np.ones(X.shape[0])
    w[2] = np.nan

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, w)


def test_infinite_sample_weight_is_rejected() -> None:
    X, y = _make_data()
    w = np.ones(X.shape[0])
    w[1] = np.inf

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, w)


def test_negative_sample_weight_is_rejected() -> None:
    X, y = _make_data()
    w = np.ones(X.shape[0])
    w[0] = -0.5

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, w)


def test_all_zero_sample_weight_is_rejected() -> None:
    X, y = _make_data()

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, np.zeros(X.shape[0]))


def test_float32_overflow_in_sample_weight_is_rejected() -> None:
    X, y = _make_data()
    w = np.ones(X.shape[0])
    w[4] = 1e39

    with pytest.raises(ValueError):
        _make_estimator()._check_X_y(X, y, w)
