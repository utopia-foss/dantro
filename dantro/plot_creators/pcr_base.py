"""This module implements the base PlotCreator class.

Classes derived from this class create plots for single files.

The interface is defined as an abstract base class and partly implemented by
the BasePlotCreator (which still remains abstract).
"""

import os
import copy
import logging

import dantro.abc
import dantro.tools as tools
from dantro.data_mngr import DataManager

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class BasePlotCreator(dantro.abc.AbstractPlotCreator):
    """The base class for PlotCreators
    
    Note that the `_plot` method remains abstract, thus this class needs to be
    subclassed and the method implemented!
    
    Attributes:
        DEFAULT_EXT (str): The class variable to use for default extension.
        default_ext (str): The property-managed actual value for the default
            extension to use. This value is needed by the PlotManager in order
            to generate an out_path. It can be changed during runtime, but
            not by passing arguments to __call__, as at that point the out_path
            already needs to be fixed.
        DEFAULT_EXT_REQUIRED (bool): Whether a default extension is required
            or not. If True and the default_ext property evaluates to False,
            an error will be raised.
        EXTENSIONS (tuple): The supported extensions. If 'all', no checks for
            the extensions are performed
        POSTPONE_PATH_PREPARATION (bool): Whether to prepare paths in the base
            class's __call__ method or not. If the derived class wants to
            take care of this on their own, this should be set to True and the
            _prepare_path method, adjusted or not, should be called at another
            point of the plot execution.
    """
    EXTENSIONS = 'all'
    DEFAULT_EXT = None
    DEFAULT_EXT_REQUIRED = True
    POSTPONE_PATH_PREPARATION = False

    def __init__(self, name: str, *, dm: DataManager, default_ext: str=None, **plot_cfg):
        """Create a PlotCreator instance for a plot with the given `name`.
        
        Args:
            name (str): The name of this plot
            dm (DataManager): The data manager that contains the data to plot
            default_ext (str, optional): The default extension to use; needs
                to be in EXTENSIONS, if that class variable is not set to
                'all'. The value given here is needed by the PlotManager to
                build the output path.
            **plot_cfg: The default plot configuration
        """
        # Store arguments as private attributes
        self._name = name
        self._dm = dm
        self._plot_cfg = plot_cfg

        # Initialise property-managed attributes
        self._logstr = None
        self._default_ext = None

        # And others via their property setters
        # Set the default extension, first from argument, then default.
        if default_ext is not None:
            self.default_ext = default_ext
        
        elif self.DEFAULT_EXT is not None:
            self.default_ext = self.DEFAULT_EXT

        # Check that it was set correctly
        if self.DEFAULT_EXT_REQUIRED and not self.default_ext:
            raise ValueError("{} requires a default extension, but neither "
                             "the argument ('{}') nor the DEFAULT_EXT class "
                             "variable ('{}') evaluated to True."
                             "".format(self.logstr, default_ext,
                                       self.DEFAULT_EXT))

        log.debug("%s initialised.", self.logstr)

    # .........................................................................
    # Properties

    @property
    def name(self) -> str:
        """Returns this creator's name"""
        return self._name

    @property
    def classname(self) -> str:
        """Returns this creator's class name"""
        return self.__class__.__name__
    
    @property
    def logstr(self) -> str:
        """Returns the classname and name of this object; a combination often
        used in logging..."""
        if not self._logstr:
            self._logstr = "{} for '{}'".format(self.classname, self.name)
        return self._logstr

    @property
    def dm(self) -> DataManager:
        """Return the DataManager"""
        return self._dm

    @property
    def plot_cfg(self) -> dict:
        """Returns a deepcopy of the plot configuration, assuring that plot
        configurations are completely independent of each other.
        """
        return copy.deepcopy(self._plot_cfg)

    @property
    def default_ext(self) -> str:
        """Returns the default extension to use for the plots"""
        return self._default_ext

    @default_ext.setter
    def default_ext(self, val: str) -> None:
        """Sets the default extension. Needs to be in EXTENSIONS"""
        if self.EXTENSIONS != 'all' and val not in self.EXTENSIONS:
            raise ValueError("Extension '{}' not supported in {}. Supported "
                             "extensions are: {}"
                             "".format(val, self.logstr, self.EXTENSIONS))

        self._default_ext = val

    # .........................................................................
    # Main API functions

    def __call__(self, *, out_path: str, **update_plot_cfg):
        """Perform the plot, updating the configuration passed to __init__
        with the given values and then calling _plot.
        
        Args:
            out_path (str): The full output path to store the plot at
            **update_plot_cfg: Keys with which to update the default plot
                configuration
        
        Returns:
            The return value of the _plot function
        """
        # TODO add logging messages

        # Get (a deep copy of) the initial plot config
        cfg = self.plot_cfg

        # Check if a recursive update needs to take place
        if update_plot_cfg:
            cfg = tools.recursive_update(cfg, update_plot_cfg)

        # Prepare the output path
        if not self.POSTPONE_PATH_PREPARATION:
            self._prepare_path(out_path)

        # Now call the plottig function with these arguments
        return self._plot(out_path=out_path, **cfg)

    # .........................................................................
    # Helpers

    def get_ext(self) -> str:
        """Returns the extension to use for the upcoming plot by checking
        the supported extensions and can be subclassed to have different
        behaviour.
        """
        return self.default_ext

    def _prepare_path(self, out_path: str) -> None:
        """Prepares the output path, creating directories if needed, then
        returning the full absolute path.
        
        This is called from __call__ and is meant to postpone directory
        creation as far as possible.
        
        Args:
            out_path (str): The absolute output path to start with
        """
        # Check that the file path does not already exist:
        if os.path.exists(out_path):
            raise FileExistsError("There already exists a file at the desired "
                                  "output path for {} at: {}"
                                  "".format(self.logstr, out_path))

        # Ensure that all necessary directories exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Nothing more to do here, at least in the base class
