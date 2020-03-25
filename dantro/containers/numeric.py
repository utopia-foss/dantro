"""This module implements specializations of the BaseDataContainer class
that focus on holding numerical, array-like data"""

import logging

import numpy as np

from ..base import BaseDataContainer
from ..mixins import ItemAccessMixin, CheckDataMixin
from ..mixins import ForwardAttrsToDataMixin, NumbersMixin, ComparisonMixin

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class NumpyDataContainer(ForwardAttrsToDataMixin, NumbersMixin,
                         ComparisonMixin, CheckDataMixin, ItemAccessMixin,
                         BaseDataContainer):
    """The NumpyDataContainer stores numerical array-shaped data.

    Specifically: it is made for use with the np.ndarray class.
    """

    # Specify expected data types for this container class
    DATA_EXPECTED_TYPES = (np.ndarray,)
    DATA_ALLOW_PROXY = False
    DATA_UNEXPECTED_ACTION = 'raise'

    def __init__(self, *, name: str, data: np.ndarray, **dc_kwargs):
        """Initialize a NumpyDataContainer, storing data that is ndarray-like.

        Arguments:
            name (str): The name of this container
            data (np.ndarray): The numpy data to store
            **dc_kwargs: Additional arguments for container initialization,
                passed on to parent method
        """
        # To be a bit more tolerant, allow lists as data argument
        if isinstance(data, list):
            log.debug("Received a list as `data` argument to %s '%s'. "
                      "Calling np.array on it ...", self.classname, name)
            data = np.array(data)

        # Initialize with parent method
        super().__init__(name=name, data=data, **dc_kwargs)

        # Done.

    def _format_info(self) -> str:
        """A __format__ helper function: returns info about the item

        In this case, the dtype and shape of the stored data is returned. Note
        that this relies on the ForwardAttrsToDataMixin.
        """
        return "{}, shape {}, {}".format(self.dtype, self.shape,
                                         super()._format_info())

    def copy(self):
        """Return a copy of this NumpyDataContainer.

        NOTE that this will create copies of the stored data.
        """
        log.debug("Creating copy of %s ...", self.logstr)
        return self.__class__(name=self.name + "_copy",
                              data=self.data.copy(),
                              attrs={k: v for k, v in self.attrs.items()})

    def save(self, path: str, **save_kwargs):
        """Saves the NumpyDataContainer to a file by invoking the np.save
        function on the underlying data.

        The file extension should be ``.npy``, which is compatible with the
        numpy-based data loader. If another file extension is given, the numpy
        method will _append_ ``.npy``!

        .. warning::

            This does NOT store container attributes!

        Args:
            path (str): The path to save the file at
            **save_kwargs: Passed to the np.save method
        """
        np.save(path, self.data, **save_kwargs)
