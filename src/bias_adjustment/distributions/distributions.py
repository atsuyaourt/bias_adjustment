import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import stats as st

from bias_adjustment.utils import FloatNDArray, is_float_ndarray


def _fit_hist(data: FloatNDArray, bins=200):
    """Generate a distribution from a histogram

    Args:
        data (FloatNDArray): Input data
        bins (int, optional): Number of bins. Defaults to 200.
    """

    for v in [data]:
        if not is_float_ndarray(v):
            raise TypeError(f"`{v}` is not a float numpy array.")
        min_len = 10
        if len(v) < min_len:
            raise ValueError(f"Length of `{v}` must be greater than {min_len}.")

    for v in [bins]:
        if not isinstance(v, int):
            raise TypeError(f"`{v}` is not an integer.")

    h = np.histogram(data[~np.isnan(data)], bins=bins)
    return st.rv_histogram(h)


def _fit_dist(data: FloatNDArray, dist_type="gamma"):
    """Generate a distribution given a distribution type

    Args:
        data (FloatNDArray): Input data.
        dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "gamma".
    """

    for v in [data]:
        if not is_float_ndarray(v):
            raise TypeError(f"`{v}` is not a float numpy array.")
        min_len = 10
        if len(v) < min_len:
            raise ValueError(f"Length of `{v}` must be greater than {min_len}.")

    if not (
        hasattr(st, dist_type) and isinstance(getattr(st, dist_type), st.rv_continuous)
    ):
        raise ValueError("`dist_type` must be a valid scipy.stats distribution name.")

    dist: st.rv_continuous = getattr(st, dist_type)
    params = dist.fit(data[~np.isnan(data)])
    return dist(*params)


@dataclass
class Distributions:
    data: FloatNDArray
    min_len: int = 10

    def __post_init__(self):
        for name in ["data"]:
            attr = getattr(self, name)
            if not is_float_ndarray(attr):
                raise TypeError(f"`{name}` is not a float numpy array.")
            if len(attr) < self.min_len:
                raise ValueError(
                    f"Length of `{name}` must be greater than {self.min_len}."
                )

        for name in ["min_len"]:
            attr = getattr(self, name)
            if not isinstance(attr, int):
                raise TypeError(f"`{name}` is not an integer.")

    def fit(self, dist_type="hist", bins=200):
        """Generate distribution

        Args:
            dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "hist".
            bins (int, optional): Number of bins. Defaults to 200.
        """

        if not isinstance(dist_type, str):
            raise TypeError(
                "`dist_type` must be a valid scipy.stats distribution name."
            )

        if not isinstance(bins, int):
            raise TypeError("`bins` must be an integer.")

        if dist_type == "hist":
            return _fit_hist(self.data, bins)
        elif hasattr(st, dist_type) and isinstance(
            getattr(st, dist_type), st.rv_continuous
        ):
            return _fit_dist(self.data, dist_type)
        else:
            raise ValueError(
                "`dist_type` must be a valid scipy.stats distribution name."
            )

    def best_fit(self, distributions: List[str] = None, bins=100):
        """Find best fit distribution to data

        Args:
            distributions (List[str], optional): List of distribution names to test.
                Defaults to ["norm", "lognorm", "gamma"].
            bins (int, optional): Number of bins. Defaults to 100.
        """

        if distributions is None:
            distributions = ["norm", "lognorm", "gamma"]
        elif not (
            isinstance(distributions, list)
            and all(isinstance(elem, str) for elem in distributions)
        ):
            raise TypeError(
                "`distributions` must be a valid scipy.stats distribution name."
            )

        if not isinstance(bins, int):
            raise TypeError("`bins` must be an integer.")

        # Get histogram of original data
        _data = self.data[~np.isnan(self.data)]
        y, x = np.histogram(_data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Best holders
        best_dist: st.rv_continuous = getattr(st, "norm")
        best_sse = np.inf

        for dist_name in distributions:
            # Try to fit the distribution
            # print(dist_name)
            try:
                # fit dist to data
                dist = self.fit(dist_type=dist_name, bins=bins)

                # Calculate fitted PDF and error with fit in distribution
                pdf = dist.pdf(x)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_dist = dist
                    best_sse = sse
            except Exception as e:
                warnings.warn(str(e))
                continue
            # print(dist, params)

        return best_dist
