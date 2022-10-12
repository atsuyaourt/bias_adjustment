from dataclasses import dataclass

import numpy as np
from scipy import stats as st


@dataclass
class Distributions:
    data: np.ndarray

    def fit(self, dist_type="hist", bins=100):
        """Generate distribution"""

        _data = self.data[~np.isnan(self.data)]

        if dist_type == "hist":
            h = np.histogram(_data, bins=bins, density=True)
            return st.rv_histogram(h)
        else:
            # fit dist to data
            dist = getattr(st, dist_type)
            params = dist.fit(_data)
            return dist(*params)

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
