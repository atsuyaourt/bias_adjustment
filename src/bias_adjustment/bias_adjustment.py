from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bias_adjustment.quantile_mapping import (
    DetrendedQuantileMapping,
    QuantileDeltaMapping,
    QuantileMapping,
)

FloatNDArray = npt.NDArray[np.float64]


@dataclass
class BiasAdjustment:
    obs: FloatNDArray
    mod: FloatNDArray
    max_cdf: float = 0.99999

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
            mode = method.split(".")[1]
            return DetrendedQuantileMapping(
                self.obs, self.mod, data, max_cdf=self.max_cdf
            ).compute(
                mode=mode,
                dist_type=dist_type,
            )
        elif method.startswith("qdm"):
            mode = method.split(".")[1]
            return QuantileDeltaMapping(
                self.obs, self.mod, data, max_cdf=self.max_cdf
            ).compute(
                mode=mode,
                dist_type=dist_type,
            )
