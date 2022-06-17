"""Defines a data loader for Python pickles."""

from .._import_tools import LazyLoader
from ..containers import ObjectContainer
from ._tools import add_loader

pkl = LazyLoader("dill")

# -----------------------------------------------------------------------------


class PickleLoaderMixin:
    """Supplies a load function for pickled python objects.

    For unpickling, the ``dill`` package is used.
    """

    @add_loader(TargetCls=ObjectContainer, omit_self=False)
    def _load_pickle(
        self, filepath: str, *, TargetCls: type, **pkl_kwargs
    ) -> ObjectContainer:
        """Load a pickled object using :py:func:`dill._dill.load`.

        Args:
            filepath (str): Where the pickle-dumped file is located
            TargetCls (type): The class constructor
            **pkl_kwargs: Passed on to :py:func:`dill._dill.load`

        Returns:
            ObjectContainer: The unpickled object, stored in a dantro container
        """
        with open(filepath, mode="rb") as f:
            obj = pkl.load(f, **pkl_kwargs)

        # Populate the target container with the object
        return TargetCls(data=obj, attrs=dict(filepath=filepath))

    # Also make available under `pkl`
    _load_pkl = _load_pickle
