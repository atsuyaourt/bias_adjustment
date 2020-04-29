import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

r_qmap = importr('qmap')
pandas2ri.activate()


def do_qmap(obs, c_mod, p_mod=None, proj_adj_type='cdf', wet_day=False,
            transform=None, verbose=True):
    """ Quantile mapping

    Arguments:
        obs {ndarray} -- observed time series
        c_mod {ndarray} -- model data for the reference period

    Keyword Arguments:
        p_mod {ndarray} -- model data for the future scenario (default: {None})
        proj_adj_type {str} -- type of adjustment to be applied to the future scenario
                            -- 'cdf' quantile mapping (default)
                            -- 'edcdf' Equidistant CDF; Li et al. (2010)
                            -- 'dqm' Detrended QM; Cannon et al. (2015)
                            -- 'qdm' Quantile Delta Mapping; Cannon et al. (2015)
        wet_day {bool} -- indicating whether to perform wet day correction or not
                {float} -- threshold below which all values are set to zero
        transform {str} -- apply data transform (default: {None})
                        -- 'log' log transform

    Returns:
        [type] -- [description]
    """
    decimals = 2

    if (obs is None) | (c_mod is None):
        return None

    obs_mask = ~np.isnan(obs)
    c_mod_mask = ~np.isnan(c_mod)

    if transform == 'log':
        obs_mask &= np.greater(obs, 0, where=obs_mask)
        c_mod_mask &= np.greater(c_mod, 0, where=c_mod_mask)
        _obs = np.log(obs[obs_mask]).round(decimals)
        _c_mod = np.log(c_mod[c_mod_mask]).round(decimals)
    else:
        _obs = obs[obs_mask].round(decimals)
        _c_mod = c_mod[c_mod_mask].round(decimals)

    if len(_c_mod) == 0:
        if verbose:
            print('No available historical model data, cancelling...')
        return None

    if verbose:
        print('Historical model data available, performing bias adjustment...')
    c_mod_adj = c_mod.copy()
    fit1 = r_qmap.fitQmapQUANT(_obs, _c_mod, wet_day=wet_day)
    if transform == 'log':
        c_mod_adj[c_mod_mask] = np.exp(r_qmap.doQmapQUANT(_c_mod, fit1))
    else:
        c_mod_adj[c_mod_mask] = r_qmap.doQmapQUANT(_c_mod, fit1)
    if verbose:
        print('Bias adjustment done.')

    _p_mod = []
    if p_mod is not None:
        p_mod_mask = ~np.isnan(p_mod)
        if transform == 'log':
            p_mod_mask &= np.greater(p_mod, 0, where=p_mod_mask)
            _p_mod = np.log(p_mod[p_mod_mask]).round(decimals)
        else:
            _p_mod = p_mod[p_mod_mask].round(decimals) 
    if len(_p_mod) > 0:
        if verbose:
            print('Projection model data available, performing bias adjustment...')
        p_mod_adj = p_mod.copy()
        # fit1 = r_qmap.fitQmapQUANT(_obs, _p_mod, wet_day=wet_day)
        fit2 = r_qmap.fitQmapQUANT(_c_mod, _p_mod, wet_day=wet_day)
        if proj_adj_type == 'edcdf':
            if verbose:
                print('Method: EDCDF')
            _p_mod_adj = _p_mod + r_qmap.doQmapQUANT(_p_mod, fit1) - r_qmap.doQmapQUANT(_p_mod, fit2)
        elif proj_adj_type == 'dqm':
            if verbose:
                print('Method: DQM')
            scl_fct = _c_mod.mean() / _p_mod.mean()
            _p_mod_adj = r_qmap.doQmapQUANT(scl_fct * _p_mod, fit1) / scl_fct
        elif proj_adj_type == 'qdm':
            if verbose:
                print('Method: QDM')
            fit1 = r_qmap.fitQmapQUANT(_obs, _p_mod, wet_day=wet_day)
            _p_mod_adj = _p_mod * r_qmap.doQmapQUANT(_p_mod, fit1) / r_qmap.doQmapQUANT(_p_mod, fit2)
        else:
            if verbose:
                print('Method: CDF')
            _p_mod_adj = r_qmap.doQmapQUANT(_p_mod, fit1)
        if transform == 'log':
            p_mod_adj[p_mod_mask] = np.exp(_p_mod_adj)
        else:
            p_mod_adj[p_mod_mask] = _p_mod_adj
        if verbose:
            print('Bias adjustment done.')
        return (c_mod_adj, p_mod_adj)
    return c_mod_adj
