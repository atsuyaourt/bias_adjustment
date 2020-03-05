import numpy as np

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

r_qmap = importr('qmap')
pandas2ri.activate()


def do_qmap(obs, c_mod, p_mod=None, proj_adj_type='cdf'):
    """ Quantile mapping
    
    Arguments:
        obs {ndarray} -- observed time series
        c_mod {ndarray} -- model data for the reference period
    
    Keyword Arguments:
        p_mod {ndarray} -- model data for the future scenario (default: {None})
        proj_adj_type {str} -- type of adjustment to be applied to the future scenario
                            -- 'cdf' or 'edcdf' (default: {'cdf'})
    
    Returns:
        [type] -- [description]
    """
    decimals = 2

    if (obs is None) | (c_mod is None):
        return None
    
    _obs = obs[~np.isnan(obs)].round(decimals)
    _c_mod = c_mod[~np.isnan(c_mod)].round(decimals)

    _p_mod = []
    if p_mod is not None:
       _p_mod = p_mod[~np.isnan(p_mod)].round(decimals) 
    
    if len(_c_mod) == 0:
        return None
    
    c_mod_adj = c_mod.copy()
    o_c_fit = r_qmap.fitQmapQUANT(_obs, _c_mod)
    c_mod_adj[~np.isnan(c_mod_adj)] = r_qmap.doQmapQUANT(_c_mod , o_c_fit)

    if len(_p_mod) > 0:
        p_mod_adj = p_mod.copy()
        c_p_fit = r_qmap.fitQmapQUANT(_c_mod, _p_mod)
        __c_mod = r_qmap.doQmapQUANT(_p_mod, c_p_fit)
        if proj_adj_type == 'edcdf':
            p_mod_adj[~np.isnan(p_mod_adj)] += r_qmap.doQmapQUANT(_p_mod , o_c_fit) - __c_mod
            p_mod_adj[p_mod_adj < 0] = 0
        else:
            p_mod_adj[~np.isnan(p_mod_adj)] = r_qmap.doQmapQUANT(__c_mod, o_c_fit)
        return (c_mod_adj, p_mod_adj)
    return c_mod_adj
