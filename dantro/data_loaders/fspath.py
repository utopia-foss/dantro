"""A data loader that loads a directory tree into the data tree"""

import glob
import os
from typing import List, Union

from ..containers import PathContainer
from ..groups import DirectoryGroup
from ..tools import glob_paths
from ._registry import add_loader

# -----------------------------------------------------------------------------


class FSPathLoaderMixin:
    """A mixin for :py:class:`~dantro.data_mngr.DataManager` that can load a
    file system directory tree into the data tree.

    The mixin supplies two load functions:

    - The ``fspath`` loader (:py:meth:`._load_fspath`) loads individual file
      paths into the data tree, representing them as
      :py:class:`~dantro.containers.path.PathContainer`.
      This is useful to generate a flat structure from a potentially nested
      filesystem structure, i.e. all paths will (by default) be in one group.
    - The ``fstree`` loader (:py:meth:`._load_fstree`) will load a file system
      tree into the data tree, retaining the tree structure.
      This is useful if a representation of some file system structure in the
      data tree is desired.
    """

    @add_loader(TargetCls=PathContainer)
    def _load_fspath(
        fspath: str,
        *,
        TargetCls: type,
        glob: str = None,
        recursive: bool = False,
    ) -> PathContainer:
        """Creates a representation of a filesystem path using the
        :py:class:`~dantro.containers.path.PathContainer`.

        Args:
            fspath (str): Filesystem path to a file or directory
            TargetCls (type): The class constructor

        Returns:
            PathContainer:
                The container representing the file or directory path
        """
        return TargetCls(data=fspath)

    @add_loader(TargetCls=DirectoryGroup)
    def _load_fstree(
        dirpath: str,
        *,
        TargetCls: type,
        tree_glob: Union[str, dict] = "**/*",
        directories_first: bool = True,
    ) -> DirectoryGroup:
        """Loads a directory tree into the data tree using
        :py:class:`~dantro.groups.dirpath.DirectoryGroup` to represent
        directories and :py:class:`~dantro.containers.path.PathContainer` to
        represent files.

        Args:
            dirpath (str): The base *directory* path to start the search from.
            TargetCls (type): The class constructor
            tree_glob (Union[str, dict], optional): The globbing parameters,
                passed to :py:func:`~dantro.tools.glob_paths`. By default,
                all paths of files *and* directories are matched.
            directories_first (bool, optional): If True, will first add the
                directories to the data tree, such that they appear on top.

        Returns:
            DirectoryGroup: The group representing the root of the data tree
                that was to be loaded, i.e. anchored at ``dirpath``.
        """
        # Prepare arguments
        if not isinstance(tree_glob, dict):
            tree_glob = dict(glob_str=tree_glob)

        tree_glob["base_path"] = tree_glob.get("base_path", dirpath)
        tree_glob["sort"] = tree_glob.get("sort", True)

        # Get all the paths that are to be added (recursion done by glob)
        all_paths = glob_paths(**tree_glob)

        if directories_first:
            all_paths.sort(key=lambda p: not os.path.isdir(p))

        # Add them to the root DirectoryGroup
        root = TargetCls(dirpath=dirpath)

        for path in all_paths:
            relpath = os.path.relpath(path, start=dirpath)
            if os.path.isdir(path):
                root.new_group(relpath, Cls=DirectoryGroup, dirpath=path)
            else:
                root.new_container(relpath, data=path, Cls=PathContainer)

        return root
