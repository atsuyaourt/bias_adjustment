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

# region process data
N = 10000
seed = 1

dats = {
    "obs": {"k": 4, "loc": 0, "scale": 7.5},
    "modh": {"k": 8.15, "loc": 0, "scale": 3.68},
    "modf": {"k": 16, "loc": 0, "scale": 2.63},
}

for dat_name, dat_info in dats.items():
    dats[dat_name]["dat"] = gamma.rvs(
        dat_info["k"], scale=dat_info["scale"], size=N, random_state=seed
    )
    dats[dat_name]["mu"], dats[dat_name]["var"] = gamma.stats(
        dat_info["k"], scale=dat_info["scale"]
    )

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
        dats[dat_name]["mu"], dats[dat_name]["var"] = gamma.stats(
            dats[dat_name]["k"],
            loc=dats[dat_name]["loc"],
            scale=dats[dat_name]["scale"],
        )
# endregion process data


# region plot pdf gamma
x = np.linspace(0, 100, 101)
fig, ax = plt.subplots(figsize=(8, 5.5))
for dat_name, dat_info in dats.items():
    mu = dat_info["mu"]
    sd = np.sqrt(dat_info["var"])
    y = gamma.pdf(x, dat_info["k"], loc=dat_info["loc"], scale=dat_info["scale"])
    label = f"{plt_args[dat_name]['name']}; $\mu$={mu:2.1f}, sd={sd:2.1f}"
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
c = gamma.ppf(
    q, dats["modh"]["k"], loc=dats["modh"]["loc"], scale=dats["modh"]["scale"]
)
x = (
    gamma.ppf(
        q, dats["modf"]["k"], loc=dats["modf"]["loc"], scale=dats["modf"]["scale"]
    )
    - c
) / c
adj_types = ["qm", "qdm.rel"]
for adj_type in adj_types:
    dat_name = "modh_adj_" + adj_type
    c = gamma.ppf(
        q,
        dats[dat_name]["k"],
        loc=dats[dat_name]["loc"],
        scale=dats[dat_name]["scale"],
    )
    dat_name = "modf_adj_" + adj_type
    y = (
        gamma.ppf(
            q,
            dats[dat_name]["k"],
            loc=dats[dat_name]["loc"],
            scale=dats[dat_name]["scale"],
        )
        - c
    ) / c
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
