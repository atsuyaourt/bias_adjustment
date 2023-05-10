from typing import Literal, Union, get_args

import numpy as np
import numpy.typing as npt

FloatNDArray = npt.NDArray[np.float64]

ARRAY_LIKE = Union[list, np.ndarray]

BAMode = Literal["rel", "abs"]


def is_float_ndarray(v):
    return isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)


def is_array_like(v):
    return isinstance(v, get_args(ARRAY_LIKE))
