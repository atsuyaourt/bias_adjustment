import numpy as np
import pytest

from bias_adjustment.quantile_mapping.qm import QuantileMapping
from bias_adjustment.utils import is_array_like, is_float_ndarray
from tests.data import max_cdf, modf, modh, obs


class TestQuantileMapping:
    @pytest.mark.parametrize(
        "params, error",
        [
            ({"obs": obs, "mod": modh, "data": modf}, None),
            ({"obs": "obs", "mod": "modh", "data": "modf"}, TypeError),
            ({"obs": obs, "mod": "modh", "data": modf}, TypeError),
            ({"obs": obs, "mod": modh, "data": "modf"}, TypeError),
            ({"obs": np.ones(6), "mod": modh, "data": modf}, ValueError),
            ({"obs": obs, "mod": np.ones(6), "data": modf}, ValueError),
            ({"obs": obs, "mod": modh, "data": np.ones(6)}, ValueError),
            ({"obs": obs, "mod": modh, "data": modf, "max_cdf": max_cdf}, None),
            ({"obs": obs, "mod": modh, "data": modf, "max_cdf": "0.88"}, TypeError),
            ({"obs": obs, "mod": modh, "data": modf, "max_cdf": 1.2}, ValueError),
            ({"obs": obs, "mod": modh, "data": modf, "max_cdf": -0.5}, ValueError),
        ],
        ids=[
            "default",
            "obs: wrong type",
            "mod: wrong type",
            "data: wrong type",
            "obs: invalid input length",
            "mod: invalid input length",
            "data: invalid input length",
            "max_cdf: default",
            "max_cdf: wrong type",
            "max_cdf: should be < 1",
            "max_cdf: should be >= 0.5",
        ],
    )
    def test_init(self, params, error):
        if error is None:
            obj = QuantileMapping(**params)
            for k, v in params.items():
                if is_array_like(v):
                    assert (getattr(obj, k) == v).all()
                else:
                    assert getattr(obj, k) == v
        else:
            with pytest.raises(error):
                QuantileMapping(**params)

    @pytest.mark.parametrize(
        "params, error",
        [
            ({}, None),
            ({"dist_type": "gamma"}, None),
            ({"dist_type": 5}, TypeError),
            ({"dist_type": "not_valid_dist"}, ValueError),
        ],
        ids=[
            "default",
            "dist_type: valid value",
            "dist_type: wrong type",
            "dist_type: invalid value",
        ],
    )
    def test_method_compute(self, params, error):
        obj = QuantileMapping(obs, modh, modf)
        assert hasattr(obj, "compute")
        if error is None:
            assert is_float_ndarray(obj.compute(**params))
        else:
            with pytest.raises(error):
                obj.compute(**params)
