import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.preprocessing import PowerTransformer

r_qmap = importr("qmap")
pandas2ri.activate()

transform_method_enum = ["box-cox", "yeo-johnson"]
adj_type_enum = ["cdf", "dqm", "edcdf", "qdm"]


def do_qmap(
    obs,
    ref_mod,
    proj_mod=None,
    adj_type="cdf",
    wet_day=False,
    transform_method=None,
    verbose=True,
):
    """ Quantile mapping

    Arguments:
        obs {ndarray} -- observed time series
        ref_mod {ndarray} -- model data for the reference period

    Keyword Arguments:
        proj_mod {ndarray} -- model data for the future scenario (default: {None})
        adj_type {str} -- type of adjustment to be applied
                            -- 'cdf' quantile mapping (default)
                            -- 'edcdf' Equidistant CDF; Li et al. (2010)
                            -- 'dqm' Detrended QM; Cannon et al. (2015)
                            -- 'qdm' Quantile Delta Mapping; Cannon et al. (2015)
        wet_day {bool} -- indicating whether to perform wet day correction or not
                {float} -- threshold below which all values are set to zero
        transform_method {str} -- apply data transform (default: {None})
                        -- implements sklean PowerTransform to transform data using
                        -- 'box-cox' or 'yeo-johnson' method

    Returns:
        [type] -- [description]
    """
    decimals = 2

    if (obs is None) | (ref_mod is None):
        return None

    mod = ref_mod.copy()
    if proj_mod is not None:
        mod = np.concatenate((mod, proj_mod))

    obs_mask = ~np.isnan(obs)
    ref_mod_mask = ~np.isnan(ref_mod)
    mod_mask = ~np.isnan(mod)

    if transform_method in transform_method_enum:
        if transform_method == "box-cox":
            obs_mask &= np.greater(obs, 0, where=obs_mask)
            ref_mod_mask &= np.greater(ref_mod, 0, where=ref_mod_mask)
            mod_mask &= np.greater(mod, 0, where=mod_mask)
        pt = PowerTransformer(method=transform_method, standardize=True)
        obs_pt = pt.fit(obs[obs_mask].reshape(-1, 1))
        _obs = obs_pt.transform(obs[obs_mask].reshape(-1, 1)).round(decimals).flatten()
        ref_mod_pt = pt.fit(ref_mod[ref_mod_mask].reshape(-1, 1))
        _ref_mod = (
            ref_mod_pt.transform(ref_mod[ref_mod_mask].reshape(-1, 1))
            .round(decimals)
            .flatten()
        )
        mod_pt = pt.fit(mod[mod_mask].reshape(-1, 1))
        _mod = mod_pt.transform(mod[mod_mask].reshape(-1, 1)).round(decimals).flatten()
    else:
        _obs = obs[obs_mask].round(decimals)
        _ref_mod = ref_mod[ref_mod_mask].round(decimals)
        _mod = mod[mod_mask].round(decimals)

    if _mod.size > 0:
        if verbose:
            print("Performing bias adjustment...")
        mod_adj = mod.copy()
        # if adj_type == "qdm":
        fit1 = r_qmap.fitQmapQUANT(_obs, _mod, wet_day=wet_day)
        fit2 = r_qmap.fitQmapQUANT(_ref_mod, _mod, wet_day=wet_day)
        scl_fct = 1.0
        if adj_type == "dqm":
            if verbose:
                print("Method: DQM")
            if transform_method not in transform_method_enum:
                scl_fct = _ref_mod.mean() / _mod.mean()

        p1 = r_qmap.doQmapQUANT(scl_fct * _mod, fit1) / scl_fct
        if transform_method is not None:
            p1 = obs_pt.inverse_transform(p1.reshape(-1, 1)).round(decimals).flatten()
        if adj_type in ["edcdf", "qdm"]:
            p2 = r_qmap.doQmapQUANT(_mod, fit2)
            p0 = _mod
            if transform_method is not None:
                p2 = (
                    ref_mod_pt.inverse_transform(p2.reshape(-1, 1))
                    .round(decimals)
                    .flatten()
                )
                p0 = (
                    mod_pt.inverse_transform(_mod.reshape(-1, 1))
                    .round(decimals)
                    .flatten()
                )
            if adj_type == "edcdf":
                if verbose:
                    print("Method: EDCDF")
                _mod_adj = p0 + p1 - p2
            elif adj_type == "qdm":
                if verbose:
                    print("Method: QDM")
                _mod_adj = p0 * p1 / p2
        else:
            _mod_adj = p1

        mod_adj[mod_mask] = _mod_adj
        if verbose:
            print("Bias adjustment done.")
        if mod_adj.size == ref_mod.size:
            return mod_adj
        else:
            return (mod_adj[0 : ref_mod.size], mod_adj[ref_mod.size :])
    else:
        if verbose:
            print("No available model data, cancelling...")
        return None
