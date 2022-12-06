"""A data group that represents a directory path and holds the corresponding
file paths as members"""

import logging
import pathlib
from typing import Optional, Union

from ..containers import PathContainer
from ..exceptions import ItemAccessError
from . import OrderedDataGroup, is_group

log = logging.getLogger(__name__)


@is_group
class DirectoryGroup(OrderedDataGroup):
    """A group that maps to a directory path in the file system."""

    _NEW_CONTAINER_CLS: type = PathContainer
    _ALLOWED_CONT_TYPES: tuple = ("self", PathContainer)

    def __init__(
        self,
        *args,
        name: str,
        dirpath: Union[str, pathlib.Path] = None,
        strict: bool = True,
        parent: "DirectoryGroup" = None,
        **kwargs,
    ):
        """Sets up a DirectoryGroup instance that holds the path to a certain
        directory.

        By default, this group only allows members to be
        :py:class:`~dantro.containers.path.PathContainer` or
        :py:class:`~dantro.groups.dirpath.DirectoryGroup` instances in order to
        maintain an association to the filesystem directory represented by this
        group.

        .. note::

            It is not actually checked whether the path points to a directory.

        Args:
            *args: Passed to parent init
            dirpath (Union[str, pathlib.Path], optional): A path compatible to
                :py:class:`pathlib.Path`. This path need not exist in the file
                system but it should point to a directory.
                If not given at initialization, it should be set afterwards.
            strict (bool, optional): If not True, will allow members of this
                group to be of *any* kind of dantro object.
            **kwargs: Passed to parent init
        """
        super().__init__(*args, name=name, parent=parent, **kwargs)

        if dirpath is None:
            if parent is not None and isinstance(parent, DirectoryGroup):
                dirpath = parent.fs_path.joinpath(name)
            else:
                raise TypeError(
                    f"If not supplying `dirpath` argument to {self.logstr}, "
                    "need a parent that is a DirectoryGroup and that can be "
                    "used to deduce a directory path. "
                    f"Parent was:  {parent}"
                )

        self._fs_path = pathlib.Path(dirpath)

        if not strict:
            self._ALLOWED_CONT_TYPES = None

    @property
    def fs_path(self) -> pathlib.Path:
        """Returns the filesystem path associated with this group by reading
        the corresponding data attribute (by default: ``fs_path``).

        .. note::

            The path this object represents may or may not exist.
        """
        return self._fs_path
