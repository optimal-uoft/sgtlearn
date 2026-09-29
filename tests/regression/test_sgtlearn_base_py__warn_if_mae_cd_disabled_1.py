import contextvars
import warnings

import numpy as np
import pytest

import sgtlearn.base as m

ENV = "SGTLEARN_MAE_CD"


def _user_warnings(criterion):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = m._warn_if_mae_cd_disabled(criterion)
    assert result is None
    return [w for w in rec if issubclass(w.category, UserWarning)]


@pytest.fixture
def env_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)


@pytest.mark.parametrize("criterion", ["absolute_error", "mae"])
def test_mae_alias_warns_when_env_unset(env_unset, criterion):
    assert len(_user_warnings(criterion)) == 1


@pytest.mark.parametrize(
    "criterion", ["MAE", "  mae  ", "Absolute_Error", "\tABSOLUTE_ERROR\n"]
)
def test_mae_alias_matched_case_insensitively_after_trimming(env_unset, criterion):
    assert len(_user_warnings(criterion)) == 1


@pytest.mark.parametrize(
    "criterion", ["squared_error", "gini", "entropy", "", "ma e", "absolute", "mae_x"]
)
def test_non_mae_criterion_does_not_warn(env_unset, criterion):
    assert _user_warnings(criterion) == []


def test_none_criterion_does_not_warn(env_unset):
    assert _user_warnings(None) == []


def test_non_string_non_mae_criterion_does_not_warn(env_unset):
    assert _user_warnings(123) == []
    assert _user_warnings(np.float64(1.5)) == []


def test_non_string_criterion_uses_string_form(env_unset):
    class Crit:
        def __str__(self):
            return " MAE "

    assert len(_user_warnings(Crit())) == 1


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "YES", " yes ", "yes"])
def test_env_enabled_values_suppress_warning(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert _user_warnings("mae") == []
    assert _user_warnings("absolute_error") == []


@pytest.mark.parametrize("value", ["", "0", "false", "no", "on", "2", "y", "enabled"])
def test_env_non_enabling_values_still_warn(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert len(_user_warnings("mae")) == 1


def test_env_enabled_with_non_mae_does_not_warn(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert _user_warnings("squared_error") == []


def test_warning_emitted_once_per_call(env_unset):
    assert len(_user_warnings("mae")) == 1
    assert len(_user_warnings("mae")) == 1


def test_warning_message_nonempty(env_unset):
    ws = _user_warnings("mae")
    assert len(ws) == 1
    assert str(ws[0].message).strip() != ""


def test_same_message_for_both_aliases(env_unset):
    a = _user_warnings("mae")
    b = _user_warnings("absolute_error")
    assert str(a[0].message) == str(b[0].message)


def test_never_raises_for_odd_criteria(env_unset):
    for crit in [None, 0, 1.0, [], {}, object(), b"mae", np.array([1, 2])]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert m._warn_if_mae_cd_disabled(crit) is None


def _find_suppress_flag():
    for name in dir(m):
        low = name.lower()
        if "suppress" in low or "silence" in low or "quiet" in low:
            obj = getattr(m, name)
            if isinstance(obj, (contextvars.ContextVar, bool)):
                return name, obj
    return None, None


def test_suppression_flag_prevents_warning(env_unset, monkeypatch):
    name, obj = _find_suppress_flag()
    if name is None:
        pytest.skip("suppression flag not discoverable")
    if isinstance(obj, contextvars.ContextVar):
        token = obj.set(True)
        try:
            assert _user_warnings("mae") == []
        finally:
            obj.reset(token)
    else:
        monkeypatch.setattr(m, name, True)
        assert _user_warnings("mae") == []


def test_falsy_suppression_flag_allows_warning(env_unset, monkeypatch):
    name, obj = _find_suppress_flag()
    if name is None:
        pytest.skip("suppression flag not discoverable")
    if isinstance(obj, contextvars.ContextVar):
        token = obj.set(False)
        try:
            assert len(_user_warnings("mae")) == 1
        finally:
            obj.reset(token)
    else:
        monkeypatch.setattr(m, name, False)
        assert len(_user_warnings("mae")) == 1
