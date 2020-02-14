"""Defines a loader mixin to load numpy dumps"""

import numpy as np

from ..containers import NumpyDataContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------

class NumpyLoaderMixin:
    """Supplies functionality to load numpy binary dumps into numpy objects"""

    @add_loader(TargetCls=NumpyDataContainer)
    def _load_numpy_binary(filepath: str, *, TargetCls: type,
                           **load_kwargs) -> NumpyDataContainer:
        """Loads the output of ``numpy.save`` back into a
        :py:class:`~dantro.containers.numeric.NumpyDataContainer`.
        
        Args:
            filepath (str): Where the ``*.npy`` file is located
            TargetCls (type): The class constructor
            **load_kwargs: Passed on to ``numpy.load``, see there for kwargs
        
        Returns:
            NumpyDataContainer: The reconstructed NumpyDataContainer
        """
        return TargetCls(data=np.load(filepath, **load_kwargs),
                         attrs=dict(filepath=filepath))

    # Also make it available under plain 'numpy'
    _load_numpy = _load_numpy_binary
