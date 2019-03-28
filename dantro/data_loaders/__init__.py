"""This module implements loaders mixin classes for use with the DataManager.

All these mixin classes should follow the following pattern:

class LoadernameLoaderMixin:

    @add_loader(TargetCls=TheTargetContainerClass)
    def _load_loadername(filepath: str, *, TargetCls):
        # ...
        return TargetCls(...)

As ensured by the add_loader decorator (see _tools module), each
`_load_loadername` method gets supplied with path to a file and the
`TargetCls` argument, which can be called to create an object of the correct
type and name.

By default, and to decouple the loader from the container, it should be
considered to be a staticmethod; in other words: the first positional argument
should _not_ be `self`!
If `self` is required, `omit_self=False` may be given to the decorator.
"""

from .load_yaml import YamlLoaderMixin
from .load_pkl import PickleLoaderMixin
from .load_hdf5 import Hdf5LoaderMixin