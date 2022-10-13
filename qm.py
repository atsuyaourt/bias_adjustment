from dataclasses import dataclass
import numpy as np

from distributions import Distributions


@dataclass
class QuantileMapping:
    obs: np.ndarray
    mod: np.ndarray
    data: np.ndarray
    min_val: float = None
    max_cdf: float = 0.99999

    def filter_data(self, dat: np.ndarray):
        _dat = dat.copy()
        if self.min_val is not None:
            _dat = _dat[_dat >= self.min_val]
        return _dat

    def generate_distribution(self, dat: np.ndarray, dist_type="hist"):
        _dat = self.filter_data(dat)
        return Distributions(_dat).fit(dist_type)

    def compute(
        self,
        dist_type="hist",
    ) -> np.ndarray:
        o_dist = self.generate_distribution(self.obs, dist_type)
        m_dist = self.generate_distribution(self.mod, dist_type)
        m_cdf = np.minimum(self.max_cdf, m_dist.cdf(self.data))

        return o_dist.ppf(m_cdf)
