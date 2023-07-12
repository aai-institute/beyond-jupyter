import sys

from . import aggregation
from . import cache
from . import cache_mysql
from .dtype import dict2OrderedTuples
from .sequences import concatSequences
from .helper import *
from .logging import LogTime


def _backwardCompatibility():
    # module 'counter' was moved to 'aggregation'
    sys.modules[f"{__name__}.counter"] = aggregation


_backwardCompatibility()
