import pytest

from bias_adjustment.utils import rand_trace


def test_rand_trace():
    assert isinstance(rand_trace(), float)
    trace_val = 0.05
    assert rand_trace(trace_val) < trace_val
    with pytest.raises(TypeError):
        rand_trace("0.8")
