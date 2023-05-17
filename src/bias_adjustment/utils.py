import math
import random
from typing import Literal, Union, get_args

import numpy as np
import numpy.typing as npt

from bias_adjustment.const import TRACE_VAL

FloatNDArray = npt.NDArray[np.float64]

ARRAY_LIKE = Union[list, np.ndarray]

BAMode = Literal["rel", "abs"]


def is_float_ndarray(v):
    return isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)


def is_array_like(v):
    return isinstance(v, get_args(ARRAY_LIKE))


def rand_trace(trace_val: float = TRACE_VAL, exponent: int = 5):
    min_val = math.pow(10, -exponent)
    max_val = trace_val - min_val
    return round(random.uniform(min_val, max_val), exponent)
