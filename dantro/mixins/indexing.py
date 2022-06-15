"""This module implement mixin classes that provide indexing capabilities"""

import logging
from typing import Union

from ..abc import AbstractDataContainer, AbstractDataGroup
from .base import ItemAccessMixin

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class IntegerItemAccessMixin:
    """This mixin allows accessing items via integer keys and also supports
    calling the ``__contains__`` magic method with integer keys. It is meant to
    be used to add features to an AbstractDataGroup-derived class, although
    this is not enforced.

    .. note::

        The ``__setitem__`` method is not covered by this!

    .. note::

        The class using this mixin has to implement index access methods and
        the ``__contains__`` magic method independently from this mixin!
    """

    def _parse_key(self, key: Union[str, int]) -> str:
        """Makes sure a key is a string"""
        if isinstance(key, int):
            return str(key)
        return key

    def __getitem__(self, key: Union[str, int]):
        """Adjusts the parent method to allow integer key item access"""
        return super().__getitem__(self._parse_key(key))
        # NOTE Don't need two-argument super() here because the parents are
        #      deduced when the class is assembled, i.e. when mixins and other
        #      inheritances are all combined.

    def __setitem__(self, key: Union[str, int]):
        """Adjusts the parent method to allow item setting by integer key"""
        return super().__setitem__(self._parse_key(key))

    def __delitem__(self, key: Union[str, int]):
        """Adjusts the parent method to allow item deletion by integer key"""
        return super().__delitem__(self._parse_key(key))

    def __contains__(self, key: Union[str, int]) -> bool:
        """Adjusts the parent method to allow checking for integers"""
        return super().__contains__(self._parse_key(key))


class PaddedIntegerItemAccessMixin(IntegerItemAccessMixin):
    """This mixin allows accessing items via integer keys that map to members
    that have a zero-padded integer name. It can only be used as mixin for
    :py:class:`~dantro.abc.AbstractDataGroup`-derived classes!

    The ``__contains__`` magic method is also supported in this mixin.

    .. note::

        The class using this mixin has to implement index access methods and
        the ``__contains__`` magic method independently from this mixin!
    """

    _PADDED_INT_KEY_WIDTH: int = None
    """The number of digits of the padded string representing the integer"""

    _PADDED_INT_FSTR: str = None
    """The format string to generate a padded integer; deduced upon first call
    """

    _PADDED_INT_STRICT_CHECKING: bool = True
    """Whether to use strict checking when parsing keys, i.e. check that the
    range of keys is valid and an error is thrown when an integer key was
    given that cannot be represented consistently by a padded string of the
    determined key width."""

    _PADDED_INT_MAX_VAL: int = None
    """The allowed maximum value of an integer key; checked only in strict mode
    """

    # .........................................................................

    @property
    def padded_int_key_width(self) -> Union[int, None]:
        """Returns the width of the zero-padded integer key or None, if it is
        not already specified.
        """
        return self._PADDED_INT_KEY_WIDTH

    @padded_int_key_width.setter
    def padded_int_key_width(self, key_width: int):
        """Method to manually provide an integer key width

        Args:
            key_width (int): The

        Raises:
            ValueError: Description
        """
        if self._PADDED_INT_FSTR:
            raise ValueError(
                f"Padded integer key width is already set for {self.logstr}; "
                "cannot set it again!"
            )

        elif key_width <= 0:
            raise ValueError(
                "Argument `key_width` to padded_int_key_width setter property "
                f"of {self.logstr} needs to be positive, was '{key_width}'!"
            )

        # Deduce the key width by going over all member names
        self._PADDED_INT_KEY_WIDTH = key_width

        # Assemble the format string to something like {:05d}
        self._PADDED_INT_FSTR = "{:0" + str(self._PADDED_INT_KEY_WIDTH) + "d}"

        # Compute the maximum value that is fully representable by the string
        self._PADDED_INT_MAX_VAL = 10**self._PADDED_INT_KEY_WIDTH - 1

    # .........................................................................

    def _parse_key(self, key: Union[str, int]) -> str:
        """Parse a potentially integer key to a zero-padded string"""
        # Check if it even is an integer
        if not isinstance(key, int):
            return key

        # Also, cannot work properly if no format string was generated, i.e. no
        # key width was provided. This can be the case if there are no members
        # added to the group using this mixin yet
        if not self._PADDED_INT_FSTR:
            return str(key)

        # Optionally, make an additional check for integer key values
        if self._PADDED_INT_STRICT_CHECKING:
            if key < 0 or key > self._PADDED_INT_MAX_VAL:
                raise IndexError(
                    "Integer index {} out of range [0, {}] for {}!".format(
                        key, self._PADDED_INT_MAX_VAL, self.logstr
                    )
                )

        # Generate the key string from the format string
        return self._PADDED_INT_FSTR.format(key)

    def _check_cont(self, cont: AbstractDataContainer) -> None:
        """This method is invoked when adding a member to a group and makes
        sure the name of the added group is correctly zero-padded.

        Also, upon first call, communicates the zero padded integer key width,
        i.e.: the length of the container name, to the
        PaddedIntegerItemAccessMixin.

        Args:
            cont: The member container to add

        Returns
            None: No return value needed
        """
        if self.padded_int_key_width is None:
            self.padded_int_key_width = len(cont.name)

        if len(cont.name) != self.padded_int_key_width:
            raise ValueError(
                f"All containers that are to be added to {self.logstr} need "
                f"names of the same length ({self.padded_int_key_width}), "
                f"but {cont.logstr} had a name of length {len(cont.name)}!"
            )

        # Still invoke the potentially existing parent method
        super()._check_cont(cont)
