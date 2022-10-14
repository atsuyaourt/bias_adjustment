from dataclasses import dataclass
import numpy as np

from distributions import Distributions


@dataclass
class QuantileMapping:
    obs: np.ndarray
    mod: np.ndarray
    data: np.ndarray
    max_cdf: float = 0.99999

    def generate_distribution(self, dat: np.ndarray, dist_type="hist"):
        return Distributions(dat).fit(dist_type)

    def compute(
        self,
        dist_type="hist",
    ) -> np.ndarray:
        o_dist = self.generate_distribution(self.obs, dist_type)
        m_dist = self.generate_distribution(self.mod, dist_type)

        mod_max = self.mod.max()

        adjusted = self.data.copy()
        m_cdf = np.minimum(self.max_cdf, m_dist.cdf(self.data[self.data <= mod_max]))
        adjusted[adjusted <= mod_max] = o_dist.ppf(m_cdf)

        delta = o_dist.ppf(self.max_cdf) - m_dist.ppf(self.max_cdf)
        adjusted[adjusted > mod_max] = adjusted[adjusted > mod_max] + delta

        return adjusted
