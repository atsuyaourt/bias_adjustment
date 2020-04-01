import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from qm import do_qmap

sns.set_context('talk')
sns.set_style('ticks')


rf_yrs = [1986, 2005]
proj_yrs = [2046, 2065]

index_col = ['year', 'month', 'day']

obs_df = pd.read_csv('test/input/obs.csv', index_col=index_col)
obs_df = obs_df.sort_index(level=index_col)
mod_df = pd.read_csv('test/input/mod.csv', index_col=index_col)
mod_df = mod_df.sort_index(level=index_col)

obs_df = obs_df.loc[slice(*rf_yrs), ]
c_mod_df = mod_df.loc[slice(*rf_yrs), ]
p_mod_df = mod_df.loc[slice(*proj_yrs), ]

# region wet day
c_mod_adj_df = []
p_mod_adj_df = []
wet_day = True
for m in range(1, 13):
    _obs_df = obs_df.loc[(slice(None), m), ]
    _c_mod_df = c_mod_df.loc[(slice(None), m), ].copy()
    _c_mod_adj_df = _c_mod_df.copy()
    _c_mod_adj_df['val'] = do_qmap(_obs_df['val'].to_numpy(), _c_mod_df['val'].to_numpy(), wet_day=wet_day)
    c_mod_adj_df.append(_c_mod_adj_df)
    _p_mod_df = p_mod_df.loc[(slice(None), m), ].copy()
    _p_mod_adj_df = _p_mod_df.copy()
    _, _p_mod_adj_df['val'] = do_qmap(
        _obs_df['val'].to_numpy(), _c_mod_df['val'].to_numpy(), _p_mod_df['val'].to_numpy(),
        proj_adj_type='edcdf', wet_day=wet_day)
    p_mod_adj_df.append(_p_mod_adj_df)

c_mod_adj_df = pd.concat(c_mod_adj_df, sort=True)
p_mod_adj_df = pd.concat(p_mod_adj_df, sort=True)
# endregion wet day


# region plot hist
plt_dict = {
    'obs': {
        'dat': obs_df,
        'name': 'APHRODITE',
        'color': 'gray',
        'linetype': '-'
    },
    'mod': {
        'dat': c_mod_df,
        'name': 'RCM',
        'color': 'blue',
        'linetype': '-'
    },
    'mod_edcdf': {
        'dat': c_mod_adj_df,
        'name': 'RCM CDF',
        'color': 'blue',
        'linetype': '--'
    }
}

fig, ax = plt.subplots(figsize=(14, 10))

for plt_name, plt_info in plt_dict.items():
    sns.distplot(
        plt_info['dat'].dropna(),
        kde=True, hist=False,
        ax=ax,
        kde_kws={'color': plt_info['color'], 'linestyle': plt_info['linetype']},
        label=plt_info['name'])

# endregion plot hist