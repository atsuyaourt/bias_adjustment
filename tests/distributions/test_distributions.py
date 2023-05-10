from typing import Union, get_args

import numpy as np
import pytest
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats._distn_infrastructure import rv_continuous_frozen, rv_discrete_frozen

from bias_adjustment.distributions import Distributions
from bias_adjustment.distributions.distributions import _fit_dist, _fit_hist
from bias_adjustment.utils import is_array_like
from tests.data import obs

DistType = Union[rv_continuous, rv_discrete, rv_continuous_frozen, rv_discrete_frozen]


@pytest.mark.parametrize(
    "params, error",
    [
        ({"data": obs}, None),
        ({}, TypeError),
        ({"data": [1, 2, 3]}, TypeError),
        ({"data": np.array(["1", "2", "3"])}, TypeError),
        ({"data": np.array([1.1, 2.4, 3.0])}, ValueError),
        ({"data": obs, "bins": 300}, None),
        ({"data": obs, "bins": "30"}, TypeError),
    ],
    ids=[
        "default",
        "no input",
        "data: wrong type 1",
        "data: wrong type 2",
        "data: wrong length",
        "bins: valid value",
        "bins: wrong type",
    ],
)
def test_fit_hist(params, error):
    if error is None:
        assert isinstance(_fit_hist(**params), get_args(DistType))
    else:
        with pytest.raises(error):
            _fit_hist(**params)


@pytest.mark.parametrize(
    "params, error",
    [
        ({"data": obs}, None),
        ({}, TypeError),
        ({"data": [1, 2, 3]}, TypeError),
        ({"data": np.array(["1", "2", "3"])}, TypeError),
        ({"data": np.array([1.1, 2.4, 3.0])}, ValueError),
        ({"data": obs, "dist_type": "norm"}, None),
        ({"data": obs, "dist_type": 30}, TypeError),
        ({"data": obs, "dist_type": "invalid_dist_name"}, ValueError),
    ],
    ids=[
        "default",
        "no input",
        "data: wrong type 1",
        "data: wrong type 2",
        "data: wrong length",
        "dist_type: valid value",
        "dist_type: wrong type",
        "dist_type: invalid value",
    ],
)
def test_fit_dist(params, error):
    if error is None:
        assert isinstance(_fit_dist(**params), get_args(DistType))
    else:
        with pytest.raises(error):
            _fit_dist(**params)


class TestDistributions:
    @pytest.mark.parametrize(
        "params, error",
        [
            ({"data": obs}, None),
            ({"data": "obs"}, TypeError),
            ({"data": np.ones(6)}, ValueError),
            ({"data": obs, "min_len": 12}, None),
            ({"data": obs, "min_len": "12"}, TypeError),
        ],
        ids=[
            "default",
            "data: wrong type",
            "data: wrong length",
            "min_len: valid value",
            "min_len: wrong type",
        ],
    )
    def test_init(self, params, error):
        if error is None:
            obj = Distributions(**params)
            for k, v in params.items():
                if is_array_like(v):
                    assert (getattr(obj, k) == v).all()
                else:
                    assert getattr(obj, k) == v
        else:
            with pytest.raises(error):
                Distributions(**params)

    @pytest.mark.parametrize(
        "params, error",
        [
            ({}, None),
            ({"dist_type": "gamma"}, None),
            ({"dist_type": 5}, TypeError),
            ({"dist_type": "not_valid_dist"}, ValueError),
            ({"bins": 50}, None),
            ({"bins": "50"}, TypeError),
        ],
        ids=[
            "default",
            "dist_type: valid value",
            "dist_type: wrong type",
            "dist_type: invalid value",
            "bins: valid value",
            "bins: wrong type",
        ],
    )
    def test_method_fit(self, params, error):
        obj = Distributions(obs)
        assert hasattr(obj, "fit")
        if error is None:
            assert isinstance(obj.fit(**params), get_args(DistType))
        else:
            with pytest.raises(error):
                obj.fit(**params)

    @pytest.mark.parametrize(
        "params, error, warning",
        [
            ({}, None, None),
            ({"distributions": ["gamma"]}, None, None),
            ({"distributions": "gamma"}, TypeError, None),
            ({"distributions": [1, 2, 3]}, TypeError, None),
            ({"distributions": ["not_valid_dist", "invalid_dist"]}, None, UserWarning),
            ({"bins": 50}, None, None),
            ({"bins": "50"}, TypeError, None),
        ],
        ids=[
            "default",
            "distributions: valid value",
            "distributions: wrong type 1",
            "distributions: wrong type 2",
            "distributions: invalid values",
            "bins: valid value",
            "bins: wrong type",
        ],
    )
    def test_method_best_fit(self, params, error, warning):
        obj = Distributions(obs)
        assert hasattr(obj, "best_fit")
        if warning is not None:
            with pytest.warns(warning):
                obj.best_fit(**params)
        elif error is None:
            assert isinstance(obj.best_fit(**params), get_args(DistType))
        else:
            with pytest.raises(error):
                obj.best_fit(**params)
