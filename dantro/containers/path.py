"""This module implements a container that holds the path to a file as data"""

import logging
import pathlib
from typing import Union

from ..base import BaseDataContainer
from ..mixins import ForwardAttrsToDataMixin

log = logging.getLogger(__name__)


class PathContainer(ForwardAttrsToDataMixin, BaseDataContainer):
    """A container that maps to a file system path.

    It uses :py:class:`pathlib.Path` to represent the given path and allow
    easy access and manipulation.
    To have direct access to the underlying :py:class:`pathlib.Path` object,
    use the :py:meth:`.fs_path` property.

    .. note::

        The paths can also be paths to directories. However, it's worth
        considering using a :py:class:`~dantro.groups.dirpath.DirectoryGroup`
        if it is desired to carry over the directory tree structure into the
        data tree.

        Unlike in :py:class:`~dantro.groups.dirpath.DirectoryGroup`, the local
        file system path is set via the ``data`` argument during initialization
        and not via a data attribute.
    """

    def __init__(self, *args, data: Union[str, pathlib.Path], **kwargs):
        """Sets up a container that holds a filesystem path as data.

        Args:
            *args: Passed to parent
            data (Union[str, pathlib.Path]): The filesystem path this object
                is meant to represent. The filesystem path need not
                necessarily exist and it also need not be equivalent to the
                path within the data tree.
            **kwargs: Passed to parent
        """
        data = pathlib.Path(data)
        super().__init__(*args, data=data, **kwargs)

    @property
    def fs_path(self) -> pathlib.Path:
        """Returns the filesystem path associated with this container.
        This is equivalent to the ``data`` property.
        """
        return self.data

    def __delitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass
