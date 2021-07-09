"""Custom, optimized copying functions used thoughout dantro"""

import copy
import pickle as _pickle
from typing import Any

# -----------------------------------------------------------------------------

_shallowcopy = copy.copy


def _deepcopy(obj: Any) -> Any:
    """A pickle based deep-copy overload, that uses ``copy.deepcopy`` only as a
    fallback option if serialization was not possible.

    The pickling approach being based on a C implementation, this can easily
    be many times faster than the pure-Python-based ``copy.deepcopy``.
    """
    try:
        return _pickle.loads(_pickle.dumps(obj))
    except:
        return copy.deepcopy(obj)
