from dataclasses import dataclass

import numpy as np
from scipy import stats as st


def fit_hist(data, bins=200):
    h = np.histogram(data[~np.isnan(data)], bins=bins)
    return st.rv_histogram(h)


def fit_dist(data, dist_type="gamma"):
    dist = getattr(st, dist_type)
    params = dist.fit(data[~np.isnan(data)])
    return dist(*params)


@dataclass
class Distributions:
    data: np.ndarray

    def fit(self, dist_type="hist", bins=200):
        """Generate distribution"""

        if dist_type == "hist":
            return fit_hist(self.data, bins)
        else:
            return fit_dist(self.data, dist_type)

    def best_fit(self, distributions=None, bins=100):
        """Model data by finding best fit distribution to data"""
        # Get histogram of original data
        _data = self.data[~np.isnan(self.data)]
        y, x = np.histogram(_data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        if distributions is None:
            distributions = ["norm", "lognorm", "gamma"]

        # Best holders
        best_dist = st.norm
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
