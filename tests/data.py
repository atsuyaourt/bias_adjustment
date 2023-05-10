from scipy.stats import gamma

seed = 1
N = 10_000
obs = gamma.rvs(4, scale=7.5, size=N, random_state=seed)
modh = gamma.rvs(8.15, scale=3.68, size=N, random_state=seed)
modf = gamma.rvs(16, scale=2.63, size=N, random_state=seed)
max_cdf = 0.9999
