import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

r_qmap = importr("qmap")
pandas2ri.activate()

adj_type_enum = ["cdf", "dqm", "edcdf", "qdm"]


def do_qmap(
    train, data, adj_type="cdf", wet_day=False, verbose=True,
):
    """ Quantile mapping

    Arguments:
        train {ndarray, 2D} -- training data
        data {ndarray} -- data to apply quantile mapping

    Keyword Arguments:
        adj_type {str} -- type of adjustment to be applied
                            -- 'cdf' quantile mapping (default)
                            -- 'edcdf' Equidistant CDF; Li et al. (2010)
                            -- 'dqm' Detrended QM; Cannon et al. (2015)
                            -- 'qdm' Quantile Delta Mapping; Cannon et al. (2015)
        wet_day {bool} -- indicating whether to perform wet day correction or not
                {float} -- threshold below which all values are set to zero

    Returns:
        [type] -- [description]
    """
    decimals = 2

    if (train is None) | (data is None):
        return None

    train_mask = ~np.isnan(train)
    data_mask = ~np.isnan(data)

    _train_obs = train[:, 0][train_mask[:, 0]].round(decimals)
    _train_mod = train[:, 1][train_mask[:, 1]].round(decimals)
    _data = data[data_mask].round(decimals)

    if _train_mod.size > 0:
        if verbose:
            print("Performing bias adjustment...")
        data_adj = data.copy()
        # if adj_type == "qdm":
        fit1 = r_qmap.fitQmapQUANT(_train_obs, _data, wet_day=wet_day)
        fit2 = r_qmap.fitQmapQUANT(_train_mod, _data, wet_day=wet_day)
        scl_fct = 1.0
        if adj_type == "dqm":
            if verbose:
                print("Method: DQM")
            scl_fct = _train_mod.mean() / _data.mean()

        p1 = r_qmap.doQmapQUANT(scl_fct * _data, fit1) / scl_fct
        if adj_type in ["edcdf", "qdm"]:
            p2 = r_qmap.doQmapQUANT(_data, fit2)
            p0 = _data
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

        data_adj[data_mask] = _mod_adj
        if verbose:
            print("Bias adjustment done.")
        return data_adj
    else:
        if verbose:
            print("No available model data, cancelling...")
        return None
