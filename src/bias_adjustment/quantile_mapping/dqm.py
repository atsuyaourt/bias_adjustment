from dataclasses import dataclass
from typing import get_args

import numpy as np

from bias_adjustment.quantile_mapping.qm import QuantileMapping
from bias_adjustment.utils import BAMode, FloatNDArray


@dataclass
class DetrendedQuantileMapping(QuantileMapping):
    def delta(self, mode: BAMode = "rel"):
        mod_mean = np.nanmean(self.mod)
        dat_mean = np.nanmean(self.data)
        if mode == "rel":
            return dat_mean / mod_mean
        elif mode == "abs":
            return dat_mean - mod_mean
        return None

    def compute(
        self,
        mode: BAMode = "rel",
        dist_type="hist",
    ) -> FloatNDArray:
        if mode not in get_args(BAMode):
            raise ValueError(f"Length of `mode` must be {get_args(BAMode)}.")

        o_dist = self.generate_distribution(self.obs, dist_type)
        m_dist = self.generate_distribution(self.mod, dist_type)

        delta = self.delta(mode)
        if mode == "rel":
            m_cdf = np.minimum(self.max_cdf, m_dist.cdf(self.data / delta))
            return o_dist.ppf(m_cdf) * delta
        elif mode == "abs":
            m_cdf = np.minimum(self.max_cdf, m_dist.cdf(self.data - delta))
            return o_dist.ppf(m_cdf) + delta
        return o_dist.ppf(m_cdf)
