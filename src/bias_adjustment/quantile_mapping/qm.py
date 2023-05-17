from dataclasses import dataclass

import numpy as np

from bias_adjustment.const import MAX_CDF, TRACE_VAL
from bias_adjustment.distributions import Distributions
from bias_adjustment.utils import FloatNDArray, is_float_ndarray, rand_trace


@dataclass
class QuantileMapping:
    obs: FloatNDArray
    mod: FloatNDArray
    data: FloatNDArray
    max_cdf: float = MAX_CDF
    trace_val: float = TRACE_VAL

    def __post_init__(self):
        for name in ["obs", "mod", "data"]:
            attr = getattr(self, name)
            if not is_float_ndarray(attr):
                raise TypeError(f"`{name}` is not a numpy array.")
            min_len = 10
            if len(attr) < min_len:
                raise ValueError(f"Length of `{name}` must be greater than {min_len}.")

        for name in ["max_cdf", "trace_val"]:
            attr = getattr(self, name)
            if not isinstance(attr, float):
                raise TypeError(f"`{name}` is not a float.")
            if name == "max_cdf":
                if attr < 0.5 or attr >= 1:
                    raise ValueError((f"`{name}` should be [0.5, 1)."))

    @staticmethod
    def generate_distribution(
        data: FloatNDArray,
        dist_type="hist",
        ignore_trace: bool = False,
        trace_val: float = TRACE_VAL,
    ):
        f"""Generate Distribution
        Args:
            data (FloatNDArray): Input data.
            dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "hist".
            ignore_trace (bool, optional): Ignore trace values? Defaults to "False".
            trace_val (float, optional): Trace value. Ignored when `ignore_trace` = False Defaults to `{TRACE_VAL}`.
        """
        if ignore_trace:
            _data = data[data > 0].copy()  # ignore zeroes
            _data[_data < trace_val] = rand_trace(
                trace_val
            )  # replace trace with random values (0, trace_val]
        else:
            _data = data.copy()
        return Distributions(_data).fit(dist_type)

    def compute(
        self,
        dist_type="hist",
        ignore_trace: bool = False,
    ) -> FloatNDArray:
        """Adjust the bias

        Args:
            dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "hist".
            ignore_trace (bool, optional): Ignore trace values? Defaults to "False".

        Returns:
            FloatNDArray: The adjusted values.
        """
        o_dist = self.generate_distribution(
            self.obs, dist_type, ignore_trace, self.trace_val
        )
        m_dist = self.generate_distribution(
            self.mod, dist_type, ignore_trace, self.trace_val
        )

        m_cdf = np.minimum(self.max_cdf, m_dist.cdf(self.data))
        return o_dist.ppf(m_cdf)
