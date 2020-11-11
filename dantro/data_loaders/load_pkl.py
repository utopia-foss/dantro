"""Defines a data loader for Python pickles."""

import dill

from ..containers import ObjectContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------


class PickleLoaderMixin:
    """Supplies a load function for pickled python objects.

    For unpickling, the dill package is used.
    """

    @add_loader(TargetCls=ObjectContainer, omit_self=False)
    def _load_pickle(
        self, filepath: str, *, TargetCls: type, **pkl_kwargs
    ) -> ObjectContainer:
        """Load a pickled object using ``dill.load``.

        Args:
            filepath (str): Where the pickle-dumped file is located
            TargetCls (type): The class constructor
            **pkl_kwargs: Passed on to the load function

        Returns:
            ObjectContainer: The unpickled object, stored in a dantro container
        """
        # Open file in binary mode and unpickle with the given load function
        with open(filepath, mode="rb") as f:
            obj = dill.load(f, **pkl_kwargs)

        # Populate the target container with the object
        return TargetCls(data=obj, attrs=dict(filepath=filepath))

    # Also make available under `pkl`
    _load_pkl = _load_pickle
