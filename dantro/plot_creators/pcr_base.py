"""This module implements the base PlotCreator class.

Classes derived from this class create plots for single files.

The interface is defined as an abstract base class and partly implemented by
the BasePlotCreator (which still remains abstract).
"""

import os
import copy
import logging
from typing import Union, Tuple, Sequence

from paramspace import ParamSpace

from ..abc import AbstractPlotCreator
from ..tools import recursive_update
from ..data_mngr import DataManager
from ..dag import TransformationDAG

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class BasePlotCreator(AbstractPlotCreator):
    """The base class for PlotCreators
    
    Note that the ``plot`` method remains abstract, thus this class needs to be
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
        SUPPORTS_DAG (bool): Whether the data selection and transformation
            interface is supported by this PlotCreator. If False, the related
            methods will not be called.
    """
    EXTENSIONS = 'all'
    DEFAULT_EXT = None
    DEFAULT_EXT_REQUIRED = True
    POSTPONE_PATH_PREPARATION = False
    SUPPORTS_DAG = False

    def __init__(self, name: str, *, dm: DataManager, default_ext: str=None,
                 **plot_cfg):
        """Create a PlotCreator instance for a plot with the given `name`.
        
        Args:
            name (str): The name of this plot
            dm (DataManager): The data manager that contains the data to plot
            default_ext (str, optional): The default extension to use; needs
                to be in EXTENSIONS, if that class variable is not set to
                'all'. The value given here is needed by the PlotManager to
                build the output path.
            **plot_cfg: The default plot configuration
        
        Raises:
            ValueError: On bad `default_ext` argument
        """
        self._name = name
        self._dm = dm
        self._plot_cfg = plot_cfg

        # Initialize property-managed attributes
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
    # Main API functions, required by PlotManager

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
            cfg = recursive_update(cfg, update_plot_cfg)

        # Prepare the output path
        if not self.POSTPONE_PATH_PREPARATION:
            self._prepare_path(out_path)

        # Perform data selection and transformation, if the plot creator class
        # supports it. 
        # Even if the creator supports it, it might be disabled in the config;
        # in that case, the method below behaves like a passthrough of the cfg,
        # only filtering out the `use_dag` key.
        if self.SUPPORTS_DAG:
            cfg = self._perform_data_selection(**cfg)

        # Now call the plottig function with these arguments
        return self.plot(out_path=out_path, **cfg)

    def get_ext(self) -> str:
        """Returns the extension to use for the upcoming plot by checking
        the supported extensions and can be subclassed to have different
        behaviour.
        """
        return self.default_ext

    def prepare_cfg(self, *, plot_cfg: dict,
                    pspace: Union[ParamSpace, dict]) -> tuple:
        """Prepares the plot configuration for the PlotManager.

        This function is called by the plot manager before the first plot
        is created.
        
        The base implementation just passes the given arguments through.
        However, it can be re-implemented by derived classes to change the
        behaviour of the plot manager, e.g. by converting a plot configuration
        to a parameter space.
        """
        return plot_cfg, pspace

    def can_plot(self, creator_name: str, **plot_cfg) -> bool:
        """Whether this plot creator is able to make a plot for the given plot
        configuration. By default, this always returns false.
        
        Args:
            creator_name (str): The name for this creator used within the
                PlotManager.
            **plot_cfg: The plot configuration with which to decide this.
        
        Returns:
            bool: Whether this creator can be used for the given plot config
        """
        return False

    # .........................................................................
    # Helpers

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
    
    # .........................................................................
    # Data selection interface, using TransformationDAG

    def _perform_data_selection(self, *, use_dag: bool=False, **cfg) -> dict:
        """If this plot creator supports data selection and transformation, it
        is carried out in this method.
        
        This method uses a number of other private methods to carry out the
        setup of the DAG, computing it and combining its results with the
        remaining plot configuration.
        
        .. note::
        
            For specializing the behaviour of the data selection and transform,
            it is best to specialize *NOT* this method, but the more granular
            DAG-related private methods.
        
        Args:
            use_dag (bool, optional): The main toggle for whether the DAG
                should be used or not. This is passed as default value to
                another method, which takes the final decision on whether the
                DAG is used or not.
            **cfg: The full plot configuration, including DAG-related arguments
        
        Returns:
            dict: The plot configuration that can be passed on to the main
                ``plot`` method.
        """
        if not self._should_use_dag(value_in_cfg=use_dag):
            # Passthrough everything except the `use_dag` argument.
            return dict(**cfg)

        # Prepare DAG parameters
        dag_params, plot_cfg = self._prepare_dag_params(**cfg)

        # Create the DAG
        dag = self._create_dag(**dag_params['init'])

        # Compute the DAG
        dag_results = self._compute_dag(dag, **dag_params['compute'])

        # Prepare the parameters passed back to __call__ and on to self.plot
        return self._combine_dag_results_and_plot_cfg(dag=dag,
                                                      dag_results=dag_results,
                                                      plot_cfg=plot_cfg)

    def _should_use_dag(self, *, value_in_cfg: bool) -> bool:
        """Whether the DAG should actually be used.
        
        This method can be subclassed to evaluate this choice in a detailed
        fashion. The base class merely returns the value passed to it.
        
        Args:
            value_in_cfg (bool): The user-specified value from the plot config
        
        Returns:
            bool: Whether the data selection and transform should be used
        """
        return value_in_cfg

    def _prepare_dag_params(self, *, select: dict=None,
                            transform: Sequence[dict]=None,
                            file_cache_defaults: dict=None,
                            compute_only: Sequence[str]=None,
                            **plot_cfg) -> Tuple[dict, dict]:
        """Filters out parameters needed for DAG initialization and compute"""
        init_kwargs = dict(select=select, transform=transform,
                           file_cache_defaults=file_cache_defaults)
        compute_kwargs = dict(compute_only=compute_only)

        return dict(init=init_kwargs, compute=compute_kwargs), plot_cfg

    def _create_dag(self, **dag_params) -> TransformationDAG:
        """Creates the actual DAG object"""
        return TransformationDAG(dm=self.dm, **dag_params)

    def _compute_dag(self, dag: TransformationDAG, **compute_kwargs) -> dict:
        """Compute the dag results"""
        return dag.compute(**compute_kwargs)

    def _combine_dag_results_and_plot_cfg(self, *, dag: TransformationDAG,
                                          dag_results: dict, plot_cfg: dict
                                          ) -> dict:
        """Combine the results of the DAG computation with the plotting
        parameters. This is then passed to the .plot method.


        This method can be subclassed to combine the results and the config in
        a specific way. By default, the results are passed as `dag_results` and
        the dag is passed along as `dag`.
        """
        return dict(dag=dag, dag_results=dag_results, **plot_cfg)
