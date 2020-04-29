import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rv_histogram

from qm import do_qmap

# sns.set_context('talk')
sns.set_context('paper')
sns.set_style('ticks')

rf_yrs = [1986, 2005]
proj_yrs = [2046, 2065]

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

index_col = ['year', 'month', 'day']

obs_df = pd.read_csv('test/input/obs.csv', index_col=index_col)
obs_df = obs_df.sort_index(level=index_col)
mod_df = pd.read_csv('test/input/mod.csv', index_col=index_col)
mod_df = mod_df.sort_index(level=index_col)

dats = {
    'obs': {
        'dat': obs_df.loc[slice(*rf_yrs), ]
    },
    'c_mod': {
        'dat': mod_df.loc[slice(*rf_yrs), ]
    },
    'p_mod': {
        'dat': mod_df.loc[slice(*proj_yrs), ]
    }
}

# region default
c_mod_adj_df = []
for m in range(1, 13):
    _obs_df = dats['obs']['dat'].loc[(slice(None), m), ]
    _c_mod_df = dats['c_mod']['dat'].loc[(slice(None), m), ].copy()
    _c_mod_adj_df = _c_mod_df.copy()
    _c_mod_adj_df['val'] = do_qmap(_obs_df['val'].to_numpy(), _c_mod_df['val'].to_numpy(), transform='log')
    c_mod_adj_df.append(_c_mod_adj_df)

dats['c_mod_adj'] = dict()
dats['c_mod_adj']['dat'] = pd.concat(c_mod_adj_df, sort=True)

adj_types = ['cdf', 'edcdf', 'dqm', 'qdm']
for adj_type in adj_types:
    dat_name = 'p_mod_adj_' + adj_type
    print(adj_type)
    dats[dat_name] = dict()
    p_mod_adj_df = []
    for m in range(1, 13):
        _obs_df = dats['obs']['dat'].loc[(slice(None), m), ]
        _c_mod_df = dats['c_mod']['dat'].loc[(slice(None), m), ].copy()
        _p_mod_df = dats['p_mod']['dat'].loc[(slice(None), m), ].copy()
        _p_mod_adj_df = _p_mod_df.copy()
        _, _p_mod_adj_df['val'] = do_qmap(
                                    _obs_df['val'].to_numpy(),
                                    _c_mod_df['val'].to_numpy(), _p_mod_df['val'].to_numpy(),
                                    proj_adj_type=adj_type, transform='log')
        p_mod_adj_df.append(_p_mod_adj_df)

    dats[dat_name]['dat'] = pd.concat(p_mod_adj_df, sort=True)
# endregion default

# region plot dist kde
m = 6
m = 1
fig, ax = plt.subplots(figsize=(8, 5.5))
for dat_name, dat_info in dats.items():
    dat = dat_info['dat'].loc[(slice(None), m), 'val'].dropna().to_numpy()
    mu = dat.mean()
    sd = dat.std()
    label = '{}; mean={:2.1f}, sd={:2.1f}'.format(plt_args[dat_name]['name'], mu, sd)
    sns.distplot(
        dat,
        kde=True, hist=False,
        ax=ax,
        kde_kws={'color': plt_args[dat_name]['color'], 'linestyle': plt_args[dat_name]['linetype']},
        label=label)
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_xlim(0,25)
plt.tight_layout()
# endregion plot dist kde


# region plot compare delta
def gen_hist_dist(dat, bins=100):
    h = np.histogram(dat, bins=bins)
    return rv_histogram(h)

m = 6
m = 1
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(
    [-1, 1], [-1, 1],
    color='black',
    linestyle=':')
q = [0.25, 0.5, 0.75, 0.95, 0.99]
c_dat = dats['c_mod']['dat'].loc[(slice(None), m), 'val'].dropna().to_numpy()
p_dat = dats['p_mod']['dat'].loc[(slice(None), m), 'val'].dropna().to_numpy()
c = gen_hist_dist(c_dat).ppf(q)
x = (gen_hist_dist(p_dat).ppf(q) - c) / c
adj_types = ['cdf', 'edcdf', 'dqm', 'qdm']
for adj_type in adj_types:
    dat_name = 'p_mod_adj_' + adj_type
    c_dat = dats['c_mod_adj']['dat'].loc[(slice(None), m), 'val'].dropna().to_numpy()
    p_dat = dats[dat_name]['dat'].loc[(slice(None), m), 'val'].dropna().to_numpy()
    c = gen_hist_dist(c_dat).ppf(q)
    y = (gen_hist_dist(p_dat).ppf(q) - c) / c
    ax.plot(
        x, y,
        color=plt_args[dat_name]['color'],
        linestyle=plt_args[dat_name]['linetype'],
        label=plt_args[dat_name]['name'])
ax.legend()
ax.set_xlabel('Model relative change')
ax.set_ylabel('Bias adjusted relative change')
# ax.set_xlim(-0.25, 0.1)
# ax.set_ylim(-0.5, 0.3)
ax.set_xlim(-0.5, 1.2)
ax.set_ylim(-0.5, 2.2)
plt.tight_layout()
# endregion plot compare delta
