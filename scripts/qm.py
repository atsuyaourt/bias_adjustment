import numpy as np

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

r_qmap = importr('qmap')
pandas2ri.activate()


def do_qmap(obs, c_mod, p_mod=None, proj_adj_type='cdf', wet_day=False, verbose=True):
    """ Quantile mapping
    
    Arguments:
        obs {ndarray} -- observed time series
        c_mod {ndarray} -- model data for the reference period
    
    Keyword Arguments:
        p_mod {ndarray} -- model data for the future scenario (default: {None})
        proj_adj_type {str} -- type of adjustment to be applied to the future scenario
                            -- 'cdf' or 'edcdf' (default: {'cdf'})
        wet_day {bool} -- indicating whether to perform wet day correction or not
                {float} -- threshold below which all values are set to zero
    
    Returns:
        [type] -- [description]
    """
    decimals = 2

    if (obs is None) | (c_mod is None):
        return None
    
    _obs = obs[~np.isnan(obs)].round(decimals)
    _c_mod = c_mod[~np.isnan(c_mod)].round(decimals)
    
    if len(_c_mod) == 0:
        if verbose:
            print('No available historical model data, cancelling...')
        return None
    
    if verbose:
        print('Historical model data available, performing bias adjustment...')
    c_mod_adj = c_mod.copy()
    fit1 = r_qmap.fitQmapQUANT(_obs, _c_mod, wet_day=wet_day)
    c_mod_adj[~np.isnan(c_mod_adj)] = r_qmap.doQmapQUANT(_c_mod , fit1)
    if verbose:
        print('Bias adjustment done.')

    _p_mod = []
    if p_mod is not None:
       _p_mod = p_mod[~np.isnan(p_mod)].round(decimals) 
    if len(_p_mod) > 0:
        if verbose:
            print('Projection model data available, performing bias adjustment...')
        p_mod_adj = p_mod.copy()
        fit1 = r_qmap.fitQmapQUANT(_obs, _p_mod, wet_day=wet_day)
        fit2 = r_qmap.fitQmapQUANT(_c_mod, _p_mod, wet_day=wet_day)
        if proj_adj_type == 'edcdf':
            if verbose:
                print('Method: EDCDF')
            p_mod_adj[~np.isnan(p_mod_adj)] += r_qmap.doQmapQUANT(_p_mod , fit1) - r_qmap.doQmapQUANT(_p_mod, fit2)
        else:
            if verbose:
                print('Method: CDF')
            p_mod_adj[~np.isnan(p_mod_adj)] =  r_qmap.doQmapQUANT(_p_mod , fit1)
        if verbose:
            print('Bias adjustment done.')
        return (c_mod_adj, p_mod_adj)
    return c_mod_adj
