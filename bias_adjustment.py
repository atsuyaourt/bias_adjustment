from dataclasses import dataclass
from typing import Literal
import numpy as np

from distributions import Distributions


def quantile_mapping(
    obs: np.ndarray,
    mod: np.ndarray,
    data: np.ndarray,
    dist_type="hist",
    min_val=None,
    max_cdf=0.99999,
) -> np.ndarray:
    _obs, _mod, _dat = [d.copy() for d in [obs, mod, data]]
    if min_val is not None:
        _obs, _mod, _dat = [d[d >= min_val] for d in [_obs, _mod, _dat]]

    o_dist = Distributions(_obs).fit(dist_type)
    m_dist = Distributions(_mod).fit(dist_type)
    m_cdf = np.minimum(max_cdf, m_dist.cdf(_dat))

    if min_val is not None:
        adj = data.copy()
        adj[adj >= min_val] = o_dist.ppf(m_cdf)
    else:
        adj = o_dist.ppf(m_cdf)
    return adj


def quantile_delta_mapping(
    obs: np.ndarray,
    mod: np.ndarray,
    data: np.ndarray,
    qdm_mode: Literal["rel", "abs"] = "rel",
    dist_type="hist",
    min_val=None,
    max_cdf=0.99999,
) -> np.ndarray:
    _obs, _mod, _dat = [d.copy() for d in [obs, mod, data]]
    if min_val is not None:
        _obs, _mod, _dat = [d[d >= min_val] for d in [_obs, _mod, _dat]]

    o_dist = Distributions(_obs).fit(dist_type)
    mh_dist = Distributions(_mod).fit(dist_type)
    mf_dist = Distributions(_dat).fit(dist_type)

    mf_cdf = np.minimum(max_cdf, mf_dist.cdf(_dat))

    if qdm_mode == "rel":  # Relative
        _adj = o_dist.ppf(mf_cdf) * (_dat / mh_dist.ppf(mf_cdf))
    elif qdm_mode == "abs":  # Absolute
        _adj = o_dist.ppf(mf_cdf) + _dat - mh_dist.ppf(mf_cdf)
    else:
        _adj = data.copy()
    if min_val is not None:
        adj = data.copy()
        adj[adj >= min_val] = _adj
    else:
        adj = _adj
    return adj


@dataclass
class BiasAdjustment:
    obs: np.ndarray
    mod: np.ndarray

    def adjust(
        self,
        data: np.ndarray,
        method="qm",
        dist_type="hist",
        min_val=None,
        max_cdf=0.99999,
    ):
        if method == "qm":
            return quantile_mapping(
                self.obs,
                self.mod,
                data,
                dist_type=dist_type,
                min_val=min_val,
                max_cdf=max_cdf,
            )
        elif method.startswith("qdm"):
            qdm_mode = method.split(".")[1]
            return quantile_delta_mapping(
                self.obs,
                self.mod,
                data,
                qdm_mode=qdm_mode,
                dist_type=dist_type,
                min_val=min_val,
                max_cdf=max_cdf,
            )
