import numpy as np
from scipy.stats import gamma, rv_histogram
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from qm import do_qmap

# sns.set_context('talk')
sns.set_context('paper')
sns.set_style('ticks')

plt_args = {
    'obs': {
        'name': 'Obs',
        'color': 'black',
        'linetype': '-'
    },
    'c_mod': {
        'name': 'Model hist',
        'color': 'red',
        'linetype': '-'
    },
    'c_mod_adj': {
        'name': 'Model hist CDF',
        'color': 'darkred',
        'linetype': '--'
    },
    'p_mod': {
        'name': 'Model future',
        'color': 'blue',
        'linetype': '-'
    },
    'p_mod_adj_cdf': {
        'name': 'Model future CDF',
        'color': 'teal',
        'linetype': '--'
    },
    'p_mod_adj_edcdf': {
        'name': 'Model future EDCDF',
        'color': 'deepskyblue',
        'linetype': '--'
    },
    'p_mod_adj_dqm': {
        'name': 'Model future DQM',
        'color': 'teal',
        'linetype': ':'
    },
    'p_mod_adj_qdm': {
        'name': 'Model future QDM',
        'color': 'darkgreen',
        'linetype': '-'
    }
}

# region process data
N = 1000

dats = {
    'obs': {
        'k': 4,
        'loc': 0,
        'scale': 7.5
    },
    'c_mod': {
        'k': 8.15,
        'loc': 0,
        'scale': 3.68
    },
    'p_mod': {
        'k': 16,
        'loc': 0,
        'scale': 2.63
    }
}

for dat_name, dat_info in dats.items():
    dats[dat_name]['dat'] = gamma.rvs(dat_info['k'], scale=dat_info['scale'], size=N)

dat_name = 'c_mod_adj'
dats[dat_name] = dict()
dats[dat_name]['dat'] = do_qmap(dats['obs']['dat'], dats['c_mod']['dat'])
k, loc, scale = gamma.fit(dats[dat_name]['dat'])
dats[dat_name]['k'] = k.round(1)
dats[dat_name]['loc'] = loc.round(1)
dats[dat_name]['scale'] = scale.round(1)

adj_types = ['cdf', 'edcdf', 'dqm', 'qdm']
for adj_type in adj_types:
    dat_name = 'p_mod_adj_' + adj_type
    print(adj_type)
    dats[dat_name] = dict()
    _c_mod_adj, dats[dat_name]['dat'] = do_qmap(dats['obs']['dat'], dats['c_mod']['dat'], dats['p_mod']['dat'], proj_adj_type=adj_type)
    k, loc, scale = gamma.fit(dats[dat_name]['dat'])
    dats[dat_name]['k'] = k.round(1)
    dats[dat_name]['loc'] = loc.round(1)
    dats[dat_name]['scale'] = scale.round(1)
# endregion process data


# region plot pdf gamma
x = np.linspace(0, 100, 101)
fig, ax = plt.subplots(figsize=(8, 5.5))
for dat_name, dat_info in dats.items():
    mu, var = gamma.stats(dat_info['k'], loc=dat_info['loc'], scale=dat_info['scale'])
    sd = np.sqrt(var)
    y = gamma.pdf(x, dat_info['k'], loc=dat_info['loc'], scale=dat_info['scale'])
    label = '{}; $\mu$={:2.1f}, sd={:2.1f}'.format(plt_args[dat_name]['name'], mu, sd)
    # ax.plot(
    #     x, y,
    #     color=plt_args[dat_name]['color'],
    #     linestyle=plt_args[dat_name]['linetype'],
    #     label=label)
    sns.distplot(
        dat_info['dat'],
        kde=False, hist=False, fit=gamma,
        ax=ax,
        fit_kws={'color': plt_args[dat_name]['color'], 'linestyle': plt_args[dat_name]['linetype']},
        label=label)
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Density')
plt.tight_layout()
# endregion plot pdf gamma

# region plot dist kde
fig, ax = plt.subplots(figsize=(8, 5.5))
for dat_name, dat_info in dats.items():
    mu = dat_info['dat'].mean()
    sd = dat_info['dat'].std()
    label = '{}; mean={:2.1f}, sd={:2.1f}'.format(plt_args[dat_name]['name'], mu, sd)
    sns.distplot(
        dat_info['dat'],
        kde=True, hist=False,
        ax=ax,
        kde_kws={'color': plt_args[dat_name]['color'], 'linestyle': plt_args[dat_name]['linetype']},
        label=label)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
plt.tight_layout()
# endregion plot dist kde

# region plot compare delta (gamma)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(
    [0, 1], [0, 1],
    color='black',
    linestyle=':')
q = [0.25, 0.5, 0.75, 0.95, 0.99]
c = gamma.ppf(q, dats['c_mod']['k'], loc=dats['c_mod']['loc'], scale=dats['c_mod']['scale'])
x = (gamma.ppf(q, dats['p_mod']['k'], loc=dats['p_mod']['loc'], scale=dats['p_mod']['scale']) - c) / c
adj_types = ['cdf', 'edcdf', 'dqm', 'qdm']
for adj_type in adj_types:
    dat_name = 'p_mod_adj_' + adj_type
    c = gamma.ppf(q, dats['c_mod_adj']['k'], loc=dats['c_mod_adj']['loc'], scale=dats['c_mod_adj']['scale'])
    y = (gamma.ppf(q, dats[dat_name]['k'], loc=dats[dat_name]['loc'], scale=dats[dat_name]['scale']) - c) / c
    ax.plot(
        x, y,
        color=plt_args[dat_name]['color'],
        linestyle=plt_args[dat_name]['linetype'],
        label=plt_args[dat_name]['name'])
ax.legend()
ax.set_xlabel('Model relative change')
ax.set_ylabel('Bias adjusted relative change')
ax.set_xlim(0.1, 0.6)
plt.tight_layout()
# endregion plot compare delta (gamma)

# region plot compare delta
def gen_hist_dist(dat, bins=100):
    h = np.histogram(dat, bins=bins)
    return rv_histogram(h)


fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(
    [0, 1], [0, 1],
    color='black',
    linestyle=':')
q = [0.25, 0.5, 0.75, 0.95, 0.99]
c = gen_hist_dist(dats['c_mod']['dat']).ppf(q)
x = (gen_hist_dist(dats['p_mod']['dat']).ppf(q) - c) / c
adj_types = ['cdf', 'edcdf', 'dqm', 'qdm']
for adj_type in adj_types:
    dat_name = 'p_mod_adj_' + adj_type
    c = gen_hist_dist(dats['c_mod_adj']['dat']).ppf(q)
    y = (gen_hist_dist(dats[dat_name]['dat']).ppf(q) - c) / c
    ax.plot(
        x, y,
        color=plt_args[dat_name]['color'],
        linestyle=plt_args[dat_name]['linetype'],
        label=plt_args[dat_name]['name'])
ax.legend()
ax.set_xlabel('Model relative change')
ax.set_ylabel('Bias adjusted relative change')
ax.set_xlim(0.1, 0.6)
plt.tight_layout()
# endregion plot compare delta
