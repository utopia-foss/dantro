"""In this module, BaseDataContainer specialisations that group data containers are implemented."""

import collections
import logging
from typing import Union

from paramspace import ParamSpace

from dantro.base import BaseDataGroup

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class OrderedDataGroup(BaseDataGroup, collections.abc.MutableMapping):
    """The OrderedDataGroup class manages groups of data containers, preserving
    the order in which they were added to this group.

    It uses an OrderedDict to associate containers with this group.
    """
    
    def __init__(self, *, name: str, containers: list=None, **kwargs):
        """Initialize a OrderedDataGroup from the list of given containers.
        
        Args:
            name (str): The name of this group
            containers (list, optional): A list of containers to add
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("OrderedDataGroup.__init__ called.")

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers,
                         StorageCls=collections.OrderedDict, **kwargs)

        # Done.
        log.debug("OrderedDataGroup.__init__ finished.")

# -----------------------------------------------------------------------------
# ParamSpaceGroup and associated classes

class ParamSpaceStateGroup(OrderedDataGroup):
    """A ParamSpaceStateGroup is meant as the member of the ParamSpaceGroup."""

    # The child class should not be of the same type as this class.
    _NEW_GROUP_CLS = OrderedDataGroup
    
    def __init__(self, *, name: str, containers: list=None, **kwargs):
        """Initialize a ParamSpaceStateGroup from the list of given containers.
        
        Args:
            name (str): The name of this group, which needs to be convertible
                to an integer.
            containers (list, optional): A list of containers to add
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("ParamSpaceStateGroup.__init__ called.")

        # Assert that the name is valid, i.e. convertible to an integer
        try:
            int(name)
        except ValueError as err:
            raise ValueError("Only names that are representible as integers "
                             "are possible for {}!".format(self.classname)
                             ) from err

        # ... and not negative
        if int(name) < 0:
            raise ValueError("Name for {} needs to be positive when converted "
                             "to integer, was: {}".format(self.classname,name))

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # Done.
        log.debug("ParamSpaceStateGroup.__init__ finished.")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class ParamSpaceGroup(OrderedDataGroup):
    """The ParamSpaceGroup is associated with a ParamSpace object and the
    loaded results of an iteration over this parameter space.

    Thus, the groups that are stored in the ParamSpaceGroup need all relate to
    a state of the parameter space, identified by a zero-padded string name.
    In fact, this group allows no other kinds of groups stored inside.

    To make access to a specific state easier, it allows accessing a state by
    its state number as integer.
    """

    # Class variables that define some of the behaviour
    # Define which .attrs entry to return from the `pspace` property
    _PSPGRP_PSPACE_ATTR_NAME = 'pspace'

    # Define the class to use for the direct members of this group
    _NEW_GROUP_CLS = ParamSpaceStateGroup

    # Define allowed container types
    _ALLOWED_CONT_TYPES = (ParamSpaceStateGroup,)

    # .........................................................................

    def __init__(self, *, name: str, containers: list=None, **kwargs):
        """Initialize a OrderedDataGroup from the list of given containers.
        
        Args:
            name (str): The name of this group.
            containers (list, optional): A list of containers to add, which
                need to be ParamSpaceStateGroup objects.
            **kwargs: Further initialisation kwargs, e.g. `attrs` ...
        """

        log.debug("ParamSpaceGroup.__init__ called.")

        # Initialize with parent method, which will call .add(*containers)
        super().__init__(name=name, containers=containers, **kwargs)

        # Private attribute needed for state access via strings
        self._num_digs = 0

        # Done.
        log.debug("ParamSpaceGroup.__init__ finished.")


    # Properties ..............................................................

    @property
    def pspace(self) -> ParamSpace:
        """Reads the entry named _PSPGRP_PSPACE_ATTR_NAME in .attrs and
        returns a ParamSpace object, if available there.
        """
        return self.attrs[self._PSPGRP_PSPACE_ATTR_NAME]


    # Item access .............................................................

    def _check_cont(self, cont: ParamSpaceStateGroup) -> None:
        """Asserts that only containers with valid names are added.
        
        Args:
            cont (ParamSpaceStateGroup): The state group to add
        
        Returns:
            None
        
        Raises:
            ValueError: For a state name that has an invalid length
        """
        # Check if this is the first container to be added. This also
        # determines the number of possible digits the state number can have
        if not len(self) or self._num_digs == 0:
            self._num_digs = len(cont.name)
            log.debug("Set _num_digs to %d.", self._num_digs)
            return

        # else: check the name against the already set number of digits
        if len(cont.name) != self._num_digs:
            raise ValueError("Containers added to {} need names that have a "
                             "string representation of same length: for this "
                             "instance, a zero-padded integer of width {}. "
                             "Got: {}".format(self.logstr, self._num_digs,
                                              cont.name))

        # TODO could also check against .pspace.max_state_no ...
        # Everything ok. No return value needed.

    def __getitem__(self, key: Union[str, int]):
        """Adjusts the parent method to allow integer item access"""
        if isinstance(key, int):
            # Generate a padded string to access the state
            key = self._padded_id_from_int(key)

        # Use the parent method to return the value
        return super().__getitem__(key)

    def __contains__(self, key: Union[str, int]) -> bool:
        """Adjusts the parent method to allow checking for integers"""
        if isinstance(key, int):
            # Generate a string from the given integer
            key = self._padded_id_from_int(key)

        return super().__contains__(key)

    def _padded_id_from_int(self, state_no: int) -> str:
        """This generates a zero-padded state number string from the given int.

        Note that the ParamSpaceGroup only allows its members to have keys of
        the same length.
        """
        # If the number of digits is not yet known, but there are already
        # entries present, find out and set the ._num_digs attribute
        if self._num_digs == 0 and len(self) > 0:
            key_lengths = [len(k) for k in self.keys()]

            # Assert they are all the same; which should be ensured by the .add
            # method anyway ...
            if not (key_lengths.count(key_lengths[0]) == len(key_lengths)):
                raise RuntimeError("Key lengths were not of the same length. "
                                   "This should not have happened! Did you "
                                   "add entries via the private API?")

            self._num_digs = key_lengths[0]

        # Now, the number of digits is known. Check the requested state number
        if state_no < 0:
            raise KeyError("State numbers cannot be negative! {} is negative."
                           "".format(state_no))

        elif state_no > 10**self._num_digs - 1:
            raise KeyError("State numbers for {} cannot be larger than {}! "
                           "Requested state number: {}"
                           "".format(self.logstr,
                                     10**self._num_digs - 1, state_no))

        # Everything ok. Generate the zero-padded string.
        return "{sno:0{digs:d}d}".format(sno=state_no, digs=self._num_digs)
