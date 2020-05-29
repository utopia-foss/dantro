"""Defines a data loader for Python pickles."""

import pickle as pkl

from ..containers import ObjectContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------

class PickleLoaderMixin:
    """Supplies functionality to load pickled python objects

    Attributes:
        _PICKLE_LOAD_FUNC (Callable): Which pickle loading function to use.
            By default this is ``pickle.load`` from the Python pickle module.
    """

    # Define the load function such that it can be set in a flexible way
    _PICKLE_LOAD_FUNC = pkl.load

    @add_loader(TargetCls=ObjectContainer, omit_self=False)
    def _load_pickle(self, filepath: str, *, TargetCls: type,
                     **pkl_kwargs) -> ObjectContainer:
        """Load a pickled object.

        This uses the load function defined under the _PICKLE_LOAD_FUNC class
        variable, which defaults to the pickle.load function.

        Args:
            filepath (str): Where the pickle-dumped file is located
            TargetCls (type): The class constructor
            **pkl_kwargs: Passed on to the load function

        Returns:
            ObjectContainer: The unpickled file, stored in a dantro container
        """
        # Open file in binary mode and unpickle with the given load function
        with open(filepath, mode='rb') as f:
            obj = self._PICKLE_LOAD_FUNC(f, **pkl_kwargs)

        # Populate the target container with the object
        return TargetCls(data=obj, attrs=dict(filepath=filepath))

    # Also make available under `pkl`
    _load_pkl = _load_pickle
