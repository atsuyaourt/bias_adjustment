import numpy as np
from scipy.stats import gamma

import matplotlib.pyplot as plt
import seaborn as sns

from bias_adjustment import BiasAdjustment

# sns.set_context('talk')
sns.set_context("paper")
sns.set_style("ticks")

plt_args = {
    "obs": {"name": "Obs", "color": "black", "linetype": "-"},
    "modh": {"name": "Model hist", "color": "red", "linetype": "-"},
    "modh_adj_qm": {"name": "Model hist QM", "color": "red", "linetype": "--"},
    "modh_adj_qdm.rel": {
        "name": "Model hist QDM",
        "color": "orange",
        "linetype": "--",
    },
    "modf": {"name": "Model future", "color": "blue", "linetype": "-"},
    "modf_adj_qm": {"name": "Model future QM", "color": "blue", "linetype": "--"},
    "modf_adj_qdm.rel": {
        "name": "Model future QDM",
        "color": "teal",
        "linetype": "--",
    },
}


def generate_test_data(size=1000, random_state=None):
    dats = {
        "obs": {"k": 4, "loc": 0, "scale": 7.5},
        "modh": {"k": 8.15, "loc": 0, "scale": 3.68},
        "modf": {"k": 16, "loc": 0, "scale": 2.63},
    }
    for dat_name, dat_info in dats.items():
        dats[dat_name]["dat"] = gamma.rvs(
            dat_info["k"], scale=dat_info["scale"], size=size, random_state=random_state
        )
    return dats


dats = generate_test_data(size=10000, random_state=1)

# region process data
dist_type = "gamma"
dat_types = ["modh", "modf"]
adj_types = ["qm", "qdm.rel"]

for dat_type in dat_types:
    for adj_type in adj_types:
        dat_name = f"{dat_type}_adj_{adj_type}"
        print(dat_name)
        dats[dat_name] = dict()
        dats[dat_name]["dat"] = BiasAdjustment(
            dats["obs"]["dat"], dats["modh"]["dat"]
        ).adjust(dats[dat_type]["dat"], method=adj_type, dist_type=dist_type)
        k, loc, scale = gamma.fit(dats[dat_name]["dat"])
        dats[dat_name]["k"] = k
        dats[dat_name]["loc"] = loc
        dats[dat_name]["scale"] = scale
# endregion process data


# region plot pdf gamma
x = np.linspace(0, 100, 101)
fig, ax = plt.subplots(figsize=(8, 5.5))
for dat_name, dat_info in dats.items():
    mu = dat_info["dat"].mean().round(1)
    std = dat_info["dat"].std().round(1)
    y = gamma.pdf(x, dat_info["k"], loc=dat_info["loc"], scale=dat_info["scale"])
    label = f"{plt_args[dat_name]['name']}; $\mu$={mu}, sd={std}"
    ax.plot(
        x,
        y,
        color=plt_args[dat_name]["color"],
        linestyle=plt_args[dat_name]["linetype"],
        label=label,
    )
ax.legend()
ax.set_xlabel("Value")
ax.set_ylabel("Density")
plt.tight_layout()
# endregion plot pdf gamma

# region plot compare delta
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="black", linestyle=":")
q = [0.25, 0.5, 0.75, 0.95, 0.99]
p = np.multiply(100, q)
c = np.percentile(dats["modh"]["dat"], p)
x = (np.percentile(dats["modf"]["dat"], p) - c) / c
adj_types = ["qm", "qdm.rel"]
for adj_type in adj_types:
    dat_name = "modh_adj_" + adj_type
    c = np.percentile(dats[dat_name]["dat"], p)
    dat_name = "modf_adj_" + adj_type
    y = (np.percentile(dats[dat_name]["dat"], p) - c) / c
    ax.plot(
        x,
        y,
        color=plt_args[dat_name]["color"],
        linestyle=plt_args[dat_name]["linetype"],
        label=plt_args[dat_name]["name"],
    )
ax.legend()
ax.set_xlabel("Model relative change")
ax.set_ylabel("Bias adjusted relative change")
ax.set_xlim(0, 1)
plt.tight_layout()
# endregion plot compare delta
