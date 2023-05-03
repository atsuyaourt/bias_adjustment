from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from bias_adjustment.quantile_mapping.qm import QuantileMapping

FloatNDArray = npt.NDArray[np.float64]


@dataclass
class DetrendedQuantileMapping(QuantileMapping):
    def delta(self, mode: Literal["rel", "abs"] = "rel"):
        mod_mean = np.nanmean(self.mod)
        dat_mean = np.nanmean(self.data)
        if mode == "rel":
            return dat_mean / mod_mean
        elif mode == "abs":
            return dat_mean - mod_mean
        return None

    def compute(
        self,
        mode: Literal["rel", "abs"] = "rel",
        dist_type="hist",
    ) -> FloatNDArray:
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
