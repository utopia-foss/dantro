"""This module implements loaders mixin classes for use with the
:py:class:`~dantro.data_mngr.DataManager`.

All these mixin classes should follow the following signature:

.. testcode::

    from dantro.data_loaders import add_loader
    from dantro.base import BaseDataContainer

    class TheTargetContainerClass(BaseDataContainer):
        pass

    class LoadernameLoaderMixin:

        @add_loader(TargetCls=TheTargetContainerClass)
        def _load_loadername(filepath: str, *, TargetCls: type):
            # ...
            return TargetCls(...)

As ensured by the :py:func:`~dantro.data_loaders._tools.add_loader` decorator
(implemented in :py:mod:`dantro.data_loaders._tools` module), each
``_load_loadername`` method gets supplied with the path to a file and the
``TargetCls`` argument, which can be called to create an object of the correct
type and name.

By default, and to decouple the loader from the container, it should be
considered to be a static method; in other words: the first positional argument
should ideally *not* be ``self``!
If ``self`` is required for some reason, set the ``omit_self`` option of the
decorator to ``False``, making it a regular (instead of a static) method.
"""

from ._tools import add_loader
from .hdf5 import Hdf5LoaderMixin
from .numpy import NumpyLoaderMixin
from .pickle import PickleLoaderMixin
from .text import TextLoaderMixin
from .xarray import XarrayLoaderMixin
from .yaml import YamlLoaderMixin


class AllAvailableLoadersMixin(
    TextLoaderMixin,
    YamlLoaderMixin,
    PickleLoaderMixin,
    Hdf5LoaderMixin,
    XarrayLoaderMixin,
    NumpyLoaderMixin,
):
    """A mixin bundling all data loaders that are available in dantro.

    This is useful for a more convenient import in a downstream
    :py:class:`~dantro.data_mngr.DataManager`.

    See the individual mixins for a more detailed documentation.
    """

    pass


# fmt: off

LOADER_BY_FILE_EXT = {
    "txt":      "text",
    "log":      "text",

    "yml":      "yml",
    "yaml":     "yaml",

    "pickle":   "pickle",
    "pkl":      "pkl",

    "hdf5":     "hdf5",
    "h5":       "hdf5",

    "nc":       "xr_dataarray",
    "netcdf":   "xr_dataarray",
    "nc_da":    "xr_dataarray",
    "xrdc":     "xr_dataarray",
    "nc_ds":    "xr_dataset",

    "npy":      "numpy_binary",
}
"""A map of file extensions to preferred loader names"""
