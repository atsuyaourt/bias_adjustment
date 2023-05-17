from dataclasses import dataclass
from typing import get_args

import numpy as np

from bias_adjustment.quantile_mapping.qm import QuantileMapping
from bias_adjustment.utils import BAMode, FloatNDArray


@dataclass
class QuantileDeltaMapping(QuantileMapping):
    def compute(
        self,
        mode: BAMode = "rel",
        dist_type="hist",
        ignore_trace: bool = False,
    ) -> FloatNDArray:
        if mode not in get_args(BAMode):
            raise ValueError(f"Length of `mode` must be {get_args(BAMode)}.")

        o_dist = self.generate_distribution(
            self.obs, dist_type, ignore_trace, self.trace_val
        )
        mh_dist = self.generate_distribution(
            self.mod, dist_type, ignore_trace, self.trace_val
        )
        mf_dist = self.generate_distribution(
            self.data, dist_type, ignore_trace, self.trace_val
        )

        mf_cdf = np.minimum(self.max_cdf, mf_dist.cdf(self.data))

        if mode == "rel":  # Relative
            return o_dist.ppf(mf_cdf) * (self.data / mh_dist.ppf(mf_cdf))
        elif mode == "abs":  # Absolute
            return o_dist.ppf(mf_cdf) + self.data - mh_dist.ppf(mf_cdf)
        return o_dist.ppf(mf_cdf)
