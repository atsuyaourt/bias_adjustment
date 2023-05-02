from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
from scipy import stats as st

Vector = npt.NDArray[np.float64]


def fit_hist(data: Vector, bins=200):
    """Generate a distribution from a histogram

    Args:
        data (Vector): Input data
        bins (int, optional): Number of bins. Defaults to 200.

    """
    h = np.histogram(data[~np.isnan(data)], bins=bins)
    return st.rv_histogram(h)


def fit_dist(data: Vector, dist_type="gamma"):
    """Generate a distribution given a distribution type

    Args:
        data (Vector): Input data.
        dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "gamma".
    """
    dist: st.rv_continuous = getattr(st, dist_type)
    params = dist.fit(data[~np.isnan(data)])
    return dist(*params)


@dataclass
class Distributions:
    data: Vector

    def fit(self, dist_type="hist", bins=200):
        """Generate distribution

        Args:
            dist_type (str, optional): Valid scipy.stats distribution name. Defaults to "hist".
            bins (int, optional): Number of bins. Defaults to 200.
        """

        if dist_type == "hist":
            return fit_hist(self.data, bins)
        else:
            return fit_dist(self.data, dist_type)

    def best_fit(
        self, distributions: List[str] = ["norm", "lognorm", "gamma"], bins=100
    ):
        """Find best fit distribution to data

        Args:
            distributions (List[str], optional): List of distribution names to test.
                Defaults to ["norm", "lognorm", "gamma"].
            bins (int, optional): Number of bins. Defaults to 100.
        """
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
            except Exception:
                continue
            # print(dist, params)

        return best_dist
