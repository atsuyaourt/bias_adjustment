import pandas as pd

from qm import do_qmap


rf_yrs = [1986, 2005]
proj_yrs = [2046, 2065]

obs_df = pd.read_csv('test/input/obs.csv')
mod_df = pd.read_csv('test/input/mod.csv')

obs_df = obs_df.loc[obs_df['year'].between(*rf_yrs)]
c_mod_df = mod_df.loc[mod_df['year'].between(*rf_yrs)]
p_mod_df = mod_df.loc[mod_df['year'].between(*proj_yrs)]

c_mod_adj = do_qmap(obs_df['val'].to_numpy(), c_mod_df['val'].to_numpy())

c_mod_adj_df = []
p_mod_adj_df = []
for m in range(1, 13):
    _obs_df = obs_df.loc[obs_df['month']==m, ]
    _c_mod_df = c_mod_df.loc[c_mod_df['month']==m, ].copy()
    _c_mod_adj_df = _c_mod_df.copy()
    _c_mod_adj_df['val'] = do_qmap(_obs_df['val'].to_numpy(), _c_mod_df['val'].to_numpy())
    c_mod_adj_df.append(_c_mod_adj_df)
    _p_mod_df = p_mod_df.loc[p_mod_df['month']==m, ].copy()
    _p_mod_adj_df = _p_mod_df.copy()
    _, _p_mod_adj_df['val'] = do_qmap(_obs_df['val'].to_numpy(), _c_mod_df['val'].to_numpy(), _p_mod_df['val'].to_numpy(), proj_adj_type='edcdf')
    p_mod_adj_df.append(_p_mod_adj_df)
