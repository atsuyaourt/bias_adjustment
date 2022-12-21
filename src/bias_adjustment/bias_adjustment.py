from dataclasses import dataclass

import numpy as np

from bias_adjustment.quantile_mapping import (
    DetrendedQuantileMapping,
    QuantileDeltaMapping,
    QuantileMapping,
)


@dataclass
class BiasAdjustment:
    obs: np.ndarray
    mod: np.ndarray
    max_cdf: float = 0.99999

    def adjust(
        self,
        data: np.ndarray,
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
