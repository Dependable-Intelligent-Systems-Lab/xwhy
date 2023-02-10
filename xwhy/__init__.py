import warnings
import sys

__version__ = '0.41.0'

# check python version
if (sys.version_info < (3, 0)):
    warnings.warn("As of version 0.29.0 shap only supports Python 3 (not 2)!")
