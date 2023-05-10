from dataclasses import dataclass

import numpy as np

from bias_adjustment.distributions import Distributions
from bias_adjustment.utils import FloatNDArray, is_float_ndarray


@dataclass
class QuantileMapping:
    obs: FloatNDArray
    mod: FloatNDArray
    data: FloatNDArray
    max_cdf: float = 0.99999

    def __post_init__(self):
        for name in ["obs", "mod", "data"]:
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

    @staticmethod
    def generate_distribution(data: FloatNDArray, dist_type="hist"):
        """Generate Distribution
        Args:
            data (FloatNDArray): Input data.
            dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "hist".
        """
        return Distributions(data).fit(dist_type)

    def compute(
        self,
        dist_type="hist",
    ) -> FloatNDArray:
        """Adjust the bias

        Args:
            dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "hist".

        Returns:
            FloatNDArray: The adjusted values.
        """
        o_dist = self.generate_distribution(self.obs, dist_type)
        m_dist = self.generate_distribution(self.mod, dist_type)

        m_cdf = np.minimum(self.max_cdf, m_dist.cdf(self.data))
        return o_dist.ppf(m_cdf)
