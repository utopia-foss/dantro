"""Defines a loader mixin to load numpy objects"""

import numpy as np

from ..containers import NumpyDataContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------

class NumpyLoaderMixin:
    """Supplies functionality to load numpy objects"""

    @add_loader(TargetCls=NumpyDataContainer)
    def _load_numpy_binary(filepath: str, *, TargetCls: type,
                           **load_kwargs) -> NumpyDataContainer:
        """Loads the output of numpy.save back into a NumpyDataContainer.
        
        Args:
            filepath (str): Where the *.npy file is located
            TargetCls (type): The class constructor
            **load_kwargs: Passed on to numpy.load, see there for kwargs
        
        Returns:
            NumpyDataContainer: The reconstructed NumpyDataContainer
        """
        return TargetCls(data=np.load(filepath, **load_kwargs),
                         attrs=dict(filepath=filepath))

    # Also make it available under plain 'numpy'
    _load_numpy = _load_numpy_binary
