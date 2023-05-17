import numpy as np
import pytest

from bias_adjustment import BiasAdjustment
from bias_adjustment.bias_adjustment import _get_ba_mode
from bias_adjustment.quantile_mapping import (
    DetrendedQuantileMapping,
    QuantileDeltaMapping,
    QuantileMapping,
)
from bias_adjustment.utils import is_array_like, is_float_ndarray
from tests.data import max_cdf, modf, modh, obs


class TestBiasAdjustment:
    @pytest.mark.parametrize(
        "params, error",
        [
            ({"obs": obs, "mod": modh}, None),
            ({"obs": "obs", "mod": "modh"}, TypeError),
            ({"obs": obs, "mod": "modh"}, TypeError),
            ({"obs": np.ones(6), "mod": modh}, ValueError),
            ({"obs": obs, "mod": np.ones(6)}, ValueError),
            ({"obs": obs, "mod": modh, "max_cdf": max_cdf}, None),
            ({"obs": obs, "mod": modh, "max_cdf": "0.88"}, TypeError),
            ({"obs": obs, "mod": modh, "max_cdf": 1.2}, ValueError),
            ({"obs": obs, "mod": modh, "max_cdf": -0.5}, ValueError),
        ],
        ids=[
            "default",
            "obs: wrong type",
            "mod: wrong type",
            "obs: invalid input length",
            "mod: invalid input length",
            "max_cdf: default",
            "max_cdf: wrong type",
            "max_cdf: should be < 1",
            "max_cdf: should be >= 0.5",
        ],
    )
    def test_init(self, params, error):
        if error is None:
            obj = BiasAdjustment(**params)
            for k, v in params.items():
                if is_array_like(v):
                    assert (getattr(obj, k) == v).all()
                else:
                    assert getattr(obj, k) == v
        else:
            with pytest.raises(error):
                BiasAdjustment(**params)

    @pytest.mark.parametrize(
        "params, error",
        [
            ({"data": modf}, None),
            ({}, TypeError),
            ({"data": "1,2,3"}, TypeError),
            ({"data": modf, "method": "qm"}, None),
            ({"data": modf, "method": "dqm"}, None),
            ({"data": modf, "method": "dqm.rel"}, None),
            ({"data": modf, "method": "dqm.abs"}, None),
            ({"data": modf, "method": "qdm.invalid_mode"}, ValueError),
            ({"data": modf, "method": "qdm"}, None),
            ({"data": modf, "method": "qdm.rel"}, None),
            ({"data": modf, "method": "qdm.abs"}, None),
            ({"data": modf, "method": "dqm.invalid_mode"}, ValueError),
        ],
        ids=[
            "default",
            "no input data",
            "data: wrong type",
            "method: qm",
            "method: dqm default mode",
            "method: dqm.rel",
            "method: dqm.abs",
            "method: dqm invalid mode",
            "method: qdm default mode",
            "method: qdm.rel",
            "method: qdm.abs",
            "method: qdm invalid mode",
        ],
    )
    def test_method_adjust(self, mocker, params: dict, error):
        obj = BiasAdjustment(obs, modh)
        assert hasattr(obj, "adjust")
        if error is None:
            if "method" in params:
                method: str = params.get("method")
                if method == "qm":
                    mock_compute = mocker.patch.object(QuantileMapping, "compute")
                    obj = BiasAdjustment(obs, modh)
                    obj.adjust(**params)
                    mock_compute.assert_called_once()
                elif method.startswith("dqm"):
                    mode = _get_ba_mode(method)
                    mock_compute = mocker.patch.object(
                        DetrendedQuantileMapping, "compute"
                    )
                    obj = BiasAdjustment(obs, modh)
                    obj.adjust(**params)
                    if mode is not None:
                        mock_compute.assert_called_once_with(
                            mode=mode, dist_type="hist", ignore_trace=False
                        )
                    else:
                        mock_compute.assert_called_once_with(
                            dist_type="hist", ignore_trace=False
                        )
                elif method.startswith("qdm"):
                    mode = _get_ba_mode(method)
                    mock_compute = mocker.patch.object(QuantileDeltaMapping, "compute")
                    obj = BiasAdjustment(obs, modh)
                    obj.adjust(**params)
                    if mode is not None:
                        mock_compute.assert_called_once_with(
                            mode=mode, dist_type="hist", ignore_trace=False
                        )
                    else:
                        mock_compute.assert_called_once_with(
                            dist_type="hist", ignore_trace=False
                        )
            else:
                assert is_float_ndarray(obj.adjust(**params))
                mock_compute = mocker.patch.object(QuantileMapping, "compute")
                obj = BiasAdjustment(obs, modh)
                obj.adjust(**params)
                mock_compute.assert_called_once()
        else:
            with pytest.raises(error):
                obj.adjust(**params)
