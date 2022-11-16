# read version from installed package
from importlib.metadata import version
from bias_adjustment.bias_adjustment import BiasAdjustment

__version__ = version("bias_adjustment")
__all__ = [BiasAdjustment]
