"""Custom, optimized copying functions used thoughout dantro"""

import copy
import pickle as _pickle
from typing import Any

# -----------------------------------------------------------------------------

_shallowcopy = copy.copy
"""An alias for a shallow copy function used throughout dantro, currently
pointing to :py:func:`copy.copy`.
"""


def _deepcopy(obj: Any) -> Any:
    """A pickle-based deep-copy overload, that uses :py:func:`copy.deepcopy`
    only as a fallback option if serialization was not possible.

    Calls :py:func:`pickle.loads` on the output of :py:func:`pickle.dumps` of
    the given object.

    The pickling approach being based on a C implementation, this can easily
    be many times faster than the pure-Python-based :py:func:`copy.deepcopy`.
    """
    try:
        return _pickle.loads(_pickle.dumps(obj))
    except:
        return copy.deepcopy(obj)
