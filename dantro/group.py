"""In this module, BaseDataContainer specialisations that group data containers are implemented."""

import logging
from typing import Union

from dantro.base import BaseDataGroup, BaseDataContainer, PATH_JOIN_CHAR

log = logging.getLogger(__name__)

# Local variables

# -----------------------------------------------------------------------------

class DataGroup(BaseDataGroup):
    """The DataGroup class manages groups of data containers.

    It uses an OrderedDict to associate containers with this group.
    """
    
    # .........................................................................
    # Recursive item access via a path

    def __getitem__(self, key: str):
        """Returns the container in this group with the given name.
        
        Args:
            key (str): The object to retrieve. If this is a path, will recurse
                down until at the end.
        
        Returns:
            The object at `key`
        """
        if not isinstance(key, list):
            # Assuming this is a string ...
            key = key.split(PATH_JOIN_CHAR)

        # Can be sure that this is a list now
        # If there is more than one entry, need to call this recursively
        if len(key) > 1:
            return self.data[key[0]][key[1:]]
        # else: end of recursion
        return self.data[key[0]]

    def __setitem__(self, key: str, val) -> None:
        """Sets an attribute at `key`.
        
        Args:
            key (str): The key to which to set the value. If this is a path,
                will recurse down to the lowest level. Note that all inter-
                mediate keys need to be present.
            val: The value to set
        
        """
        if not isinstance(key, list):
            key = key.split(PATH_JOIN_CHAR)

        # Depending on length of the key sequence, start recursion or not
        if len(key) > 1:
            self.data[key[0]][key[1:]] = val
        # else: end of recursion, set the value
        self.data[key[0]] = val

    # .........................................................................

    def __len__(self) -> int:
        """The length of the data."""
        return len(self.data)

    def __contains__(self, cont: Union[str, BaseDataContainer]) -> bool:
        """Whether the given container is in this group or not.
        
        Args:
            cont (Union[str, BaseDataContainer]): The name of the container or 
                an object reference. 
        
        Returns:
            bool: Whether the given container is in this group.
        """
        if isinstance(cont, BaseDataContainer):
            return bool(cont in self.values())
        elif not isinstance(cont, list):
            # assume it is a string
            key_seq = cont.split(PATH_JOIN_CHAR)
        else:
            key_seq = cont

        # is a list of keys, might have to check recursively
        if len(key_seq) > 1:
            return bool(key_seq[1:] in self[key_seq[0]])
        return bool(key_seq[0] in self.keys())

    def keys(self):
        """Returns an iterator over the container names in this group."""
        return self.data.keys()

    def values(self):
        """Returns an iterator over the containers in this group."""
        return self.data.values()

    def items(self):
        """Returns an iterator over the (name, data container) tuple of this group."""
        return self.data.items()

    def get(self, key, default=None):
        """Return the container at `key`, or `default` if container with name `key` is not available."""
        return self.data.get(key, default)

    def setdefault(self, key, default=None):
        """If `key` is in the dictionary, return its value. If not, insert `key` with a value of `default` and return `default`. `default` defaults to None."""
        if key in self:
            return self[key]
        # else: not available
        self.data[key] = default
        return default

    # .........................................................................
    # Formatting

    def _format_info(self) -> str:
        """A __format__ helper function: returns an info string that is used to characterise this object. Does NOT include name and classname!"""
        return str(len(self)) + " members"

    def _format_tree(self) -> str:
        """Returns a multi-line string tree representation of this group."""
        raise NotImplementedError
