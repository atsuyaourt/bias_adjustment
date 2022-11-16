from dataclasses import dataclass
from typing import Literal
import numpy as np

from bias_adjustment.quantile_mapping.qm import QuantileMapping


@dataclass
class QuantileDeltaMapping(QuantileMapping):
    def compute(
        self,
        mode: Literal["rel", "abs"] = "rel",
        dist_type="hist",
    ) -> np.ndarray:
        o_dist = self.generate_distribution(self.obs, dist_type)
        mh_dist = self.generate_distribution(self.mod, dist_type)
        mf_dist = self.generate_distribution(self.data, dist_type)

        mf_cdf = np.minimum(self.max_cdf, mf_dist.cdf(self.data))

        if mode == "rel":  # Relative
            return o_dist.ppf(mf_cdf) * (self.data / mh_dist.ppf(mf_cdf))
        elif mode == "abs":  # Absolute
            return o_dist.ppf(mf_cdf) + self.data - mh_dist.ppf(mf_cdf)
        else:
            return None
