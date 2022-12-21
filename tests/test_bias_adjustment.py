import numpy as np
import pytest
from scipy.stats import gamma

from bias_adjustment import BiasAdjustment


@pytest.fixture
def random_state():
    return 1


@pytest.fixture
def sample_size():
    return 10000


@pytest.fixture
def obs(sample_size, random_state):
    return gamma.rvs(4, scale=7.5, size=sample_size, random_state=random_state)


@pytest.fixture
def modh(sample_size, random_state):
    return gamma.rvs(8.15, scale=3.68, size=sample_size, random_state=random_state)


@pytest.fixture
def modf(sample_size, random_state):
    return gamma.rvs(16, scale=2.63, size=sample_size, random_state=random_state)


class TestBiasAdjustment:
    @pytest.fixture
    def obj(self, obs, modh):
        return BiasAdjustment(obs, modh)

    def test_params(self, obj, obs, modh):
        assert (obj.obs == obs).all()
        assert (obj.mod == modh).all()

    def test_method_adjust(self, obj, modf):
        print(obj.adjust(modf))
        assert isinstance(obj.adjust(modf), np.ndarray)
