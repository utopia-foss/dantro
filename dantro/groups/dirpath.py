"""A data group that represents a directory path and holds the corresponding
file paths as members"""

import logging
import pathlib
from typing import Optional

from ..containers import PathContainer
from ..exceptions import ItemAccessError
from . import OrderedDataGroup, is_group

log = logging.getLogger(__name__)


@is_group
class DirectoryGroup(OrderedDataGroup):
    """A group that maps to a directory path in the file system.

    .. note::

        The local file system path is extracted from the data attributes of
        this group. It is *not* given during initialization.
    """

    __fs_path: Optional[pathlib.Path] = None
    """The attribute holding the cached Path object.

    This is set upon first invocation of :py:meth:`.fs_path`.
    """

    _DIRPATH_ATTR: str = "fs_path"
    """The data attribute to use to extract the file system directory path."""

    _NEW_CONTAINER_CLS: type = PathContainer

    @property
    def fs_path(self) -> pathlib.Path:
        """Returns the filesystem path associated with this group by reading
        the corresponding data attribute (by default: ``fs_path``).

        .. note::

            The path this object represents may or may not exist.
        """
        if self.__fs_path is None:
            try:
                self.__fs_path = pathlib.Path(self.attrs[self._DIRPATH_ATTR])
            except KeyError as err:
                _key = self._DIRPATH_ATTR
                raise ItemAccessError(
                    self.attrs,
                    key=_key,
                    show_hints=True,
                    suffix=(
                        "Make sure to store the directory path as data "
                        f"attribute '{_key}' of {self.logstr}. "
                        "If you do not have the filesystem path available, "
                        "consider using a different group class instead."
                    ),
                ) from err

        return self.__fs_path
