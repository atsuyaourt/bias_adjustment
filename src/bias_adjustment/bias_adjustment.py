from dataclasses import dataclass

from bias_adjustment.const import MAX_CDF
from bias_adjustment.quantile_mapping import (
    DetrendedQuantileMapping,
    QuantileDeltaMapping,
    QuantileMapping,
)
from bias_adjustment.utils import FloatNDArray, is_float_ndarray


def _get_ba_mode(method: str):
    args = method.split(".")
    if len(args) == 2:
        return args[1]
    return


@dataclass
class BiasAdjustment:
    obs: FloatNDArray
    mod: FloatNDArray
    max_cdf: float = MAX_CDF

    def __post_init__(self):
        for name in ["obs", "mod"]:
            attr = getattr(self, name)
            if not is_float_ndarray(attr):
                raise TypeError(f"`{name}` is not a numpy array.")
            min_len = 10
            if len(attr) < min_len:
                raise ValueError(f"Length of `{name}` must be greater than {min_len}.")

        for name in ["max_cdf"]:
            attr = getattr(self, name)
            if not isinstance(attr, float):
                raise TypeError(f"`{name}` is not a float.")
            if name == "max_cdf":
                if attr < 0.5 or attr >= 1:
                    raise ValueError((f"`{name}` should be [0.5, 1)."))

    def adjust(
        self,
        data: FloatNDArray,
        method="qm",
        dist_type="hist",
    ):
        if method == "qm":
            return QuantileMapping(
                self.obs, self.mod, data, max_cdf=self.max_cdf
            ).compute(dist_type=dist_type)
        elif method.startswith("dqm"):
            mode = _get_ba_mode(method)
            qm = DetrendedQuantileMapping(
                self.obs, self.mod, data, max_cdf=self.max_cdf
            )
            if mode is not None:
                return qm.compute(
                    mode=mode,
                    dist_type=dist_type,
                )
            return qm.compute(
                dist_type=dist_type,
            )
        elif method.startswith("qdm"):
            mode = _get_ba_mode(method)
            qm = QuantileDeltaMapping(self.obs, self.mod, data, max_cdf=self.max_cdf)
            if mode is not None:
                return qm.compute(
                    mode=mode,
                    dist_type=dist_type,
                )
            return qm.compute(
                dist_type=dist_type,
            )
        return QuantileMapping(self.obs, self.mod, data, max_cdf=self.max_cdf).compute(
            dist_type=dist_type
        )
