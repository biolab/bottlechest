from __future__ import absolute_import

# Supported dtypes
dtypes = ['int8', 'int32', 'int64', 'float32', 'float64']

from . import slow
from . import func

# If you bork the build (e.g. by messing around with the templates),
# you still want to be able to import Bottleneck so that you can
# rebuild using the templates. So try to import the compiled Bottleneck
# functions to the top level, but move on if not successful.
try:
    from .allnan import allnan
    from .nansum import nansum
    from .nanmax import nanmax
    from .nanmin import nanmin
    from .nanmean import nanmean
    from .nanstd import nanstd
    from .nanvar import nanvar
    from .median import (median, nanmedian)
    from .nanargmin import nanargmin
    from .nanargmax import nanargmax
    from .rankdata import (rankdata, nanrankdata)
    from .ss import ss
    from .nn import nn
    from .replace import replace
    from .anynan import anynan
    from .allnan import allnan
    from .nanequal import nanequal
    from .bincount import bincount
    from .valuecount import valuecount
    from .countnans import countnans
    from .contingency import contingency
    from .stats import stats
    from .partsort import partsort
    from .argpartsort import argpartsort
except IOError:
    pass


from bottleneck.version import __version__
from bottleneck.benchmark.bench import bench

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No Bottleneck unit testing available.")
