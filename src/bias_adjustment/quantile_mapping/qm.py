from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bias_adjustment.distributions import Distributions

FloatNDArray = npt.NDArray[np.float64]


@dataclass
class QuantileMapping:
    obs: FloatNDArray
    mod: FloatNDArray
    data: FloatNDArray
    max_cdf: float = 0.99999

    @staticmethod
    def generate_distribution(dat: FloatNDArray, dist_type="hist"):
        return Distributions(dat).fit(dist_type)

    def compute(
        self,
        dist_type="hist",
    ) -> FloatNDArray:
        o_dist = self.generate_distribution(self.obs, dist_type)
        m_dist = self.generate_distribution(self.mod, dist_type)

        m_cdf = np.minimum(self.max_cdf, m_dist.cdf(self.data))
        return o_dist.ppf(m_cdf)
