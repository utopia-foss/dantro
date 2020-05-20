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
        DAG_INVOKE_IN_BASE (bool): Whether the DAG should be created and
            computed here (in the base class). If False, the base class does
            nothing to create or compute it and the derived classes have to
            take care of it on their own.
        DAG_SUPPORTED (bool): Whether the data selection and transformation
            interface is supported by this PlotCreator. If False, the related
            methods will not be called.
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
        OUT_PATH_EXIST_OK (bool): Whether a warning should be shown (instead
            of an error, when a plot file already exists at the specified
            output path
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
    OUT_PATH_EXIST_OK = False
    DAG_SUPPORTED = False
    DAG_INVOKE_IN_BASE = True

    def __init__(self, name: str, *, dm: DataManager,
                 default_ext: str=None,
                 exist_ok: bool=None,
                 raise_exc: bool=None,
                 **plot_cfg):
        """Create a PlotCreator instance for a plot with the given ``name``.

        Typically, a creator has not be instantiated separately, but the
        :py:class:`~dantro.plot_mngr.PlotManager` takes care of it.

        Args:
            name (str): The name of this plot
            dm (DataManager): The data manager that contains the data to plot
            default_ext (str, optional): The default extension to use; needs
                to be in ``EXTENSIONS``, if that class variable is not set to
                'all'. The value given here is needed by the PlotManager to
                build the output path.
            exist_ok (bool, optional): If True, no error will be raised when
                a plot already exists at the specified output path. If None,
                the value specified in the ``OUT_PATH_EXIST_OK`` class variable
                will be used to determine this behaviour.
            raise_exc (bool, optional): Whether to raise exceptions during the
                plot procedure. This does not pertain to *all* exceptions, but
                only to those that would *unnecessarily* stop plotting.
                Furthermore, whether this setting is used or not depends on the
                used creator specialization.
            **plot_cfg: The default configuration for the plot(s) that this
                creator is supposed to create.

        Raises:
            ValueError: On bad `default_ext` argument
        """
        self._name = name
        self._dm = dm
        self._plot_cfg = plot_cfg
        self._exist_ok = (self.OUT_PATH_EXIST_OK if exist_ok is None
                          else exist_ok)
        self.raise_exc = raise_exc

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
            self._logstr = f"{self.classname} for '{self.name}'"
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
            The return value of the
            :py:meth:`~dantro.plot_creators.pcr_base.BasePlotCreator.plot`
            method, which is an abstract method in
            :py:class:`~dantro.plot_creators.pcr_base.BasePlotCreator`.
        """
        # TODO add logging messages

        # Get (a deep copy of) the initial plot config
        cfg = self.plot_cfg

        # Check if a recursive update needs to take place
        if update_plot_cfg:
            cfg = recursive_update(cfg, copy.deepcopy(update_plot_cfg))

        # Find out if it's ok if out_path already exists, then prepare the path
        exist_ok = self._exist_ok
        if 'exist_ok' in cfg:
            exist_ok = cfg.pop('exist_ok')

        if not self.POSTPONE_PATH_PREPARATION:
            self._prepare_path(out_path, exist_ok=exist_ok)

        # Perform data selection and transformation, if the plot creator class
        # supports it.
        # Even if the creator supports it, it might be disabled in the config;
        # in that case, the method below behaves like a passthrough of the cfg,
        # filtering out all transformation-related arguments.
        if self.DAG_SUPPORTED and self.DAG_INVOKE_IN_BASE:
            use_dag = cfg.pop('use_dag', None)
            _, cfg = self._perform_data_selection(use_dag=use_dag,
                                                  plot_kwargs=cfg)

        # Now call the plottig function with these arguments
        return self.plot(out_path=out_path, **cfg)

    def get_ext(self) -> str:
        """Returns the extension to use for the upcoming plot by checking
        the supported extensions and can be subclassed to have different
        behaviour.
        """
        return self.default_ext

    def prepare_cfg(self, *, plot_cfg: dict, pspace: Union[ParamSpace, dict]
                    ) -> Tuple[dict, ParamSpace]:
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

    def _prepare_path(self, out_path: str, *, exist_ok: bool) -> None:
        """Prepares the output path, creating directories if needed, then
        returning the full absolute path.

        This is called from __call__ and is meant to postpone directory
        creation as far as possible.

        Args:
            out_path (str): The absolute output path to start with
            exist_ok (bool): If True, will emit a warning instead of an error

        Raises:
            FileExistsError: Raised on already existing out path and exist_ok
                being False.
        """
        # Check that the file path does not already exist:
        if os.path.exists(out_path):
            msg = ("There already exists a file at the specified output path "
                   "for {}:\n  {}".format(self.logstr, out_path))
            if not exist_ok:
                raise FileExistsError(msg)
            log.warning(msg)

        # Ensure that all necessary directories exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Nothing more to do here, at least in the base class

    # .........................................................................
    # Data selection interface, using TransformationDAG

    def _perform_data_selection(self, *, use_dag: bool=None, plot_kwargs: dict,
                                **shared_kwargs) -> Tuple[bool, dict]:
        """If this plot creator supports data selection and transformation, it
        is carried out in this method.

        This method uses a number of other private methods to carry out the
        setup of the DAG, computing it and combining its results with the
        remaining plot configuration. Those methods have access to a subset of
        the whole configuration, thus allowing to parse the parameters that
        they need.

        .. note::

            For specializing the behaviour of the data selection and transform,
            it is best to specialize *NOT* this method, but the more granular
            DAG-related private methods.

        Args:
            use_dag (bool, optional): The main toggle for whether the DAG
                should be used or not. This is passed as default value to
                another method, which takes the final decision on whether the
                DAG is used or not. If None, will NOT use the DAG.
            plot_kwargs (dict): The plot configuration
            **shared_kwargs: Shared keyword arguments that are passed through
                to the helper methods ``_use_dag`` and ``_get_dag_params``

        Returns:
            Tuple[bool, dict]: Whether data selection was used and the plot
                configuration that can be passed on to the main ``plot``
                method.
        """
        # Determine whether the DAG framework should be used or not
        if not self._use_dag(use_dag=use_dag, plot_kwargs=plot_kwargs,
                             **shared_kwargs):
            # Only return the plot configuration, without DAG-related keys
            return False, plot_kwargs

        # Extract DAG-related parameters from the plot configuration. These are
        # not available in the plotting function.
        dag_params, plot_kwargs = self._get_dag_params(**plot_kwargs,
                                                       **shared_kwargs)

        # else: DAG should be used -> Create and compute it.
        dag = self._create_dag(**dag_params['init'])
        dag_results = self._compute_dag(dag, **dag_params['compute'])

        # Prepare the parameters passed back to __call__ and on to self.plot
        kws = self._combine_dag_results_and_plot_cfg(dag=dag,
                                                     dag_results=dag_results,
                                                     dag_params=dag_params,
                                                     plot_kwargs=plot_kwargs)
        return True, kws

    def _get_dag_params(self, *,
                        select: dict=None, transform: Sequence[dict]=None,
                        compute_only: Sequence[str]=None,
                        dag_options: dict=None,
                        **plot_kwargs) -> Tuple[dict, dict]:
        """Filters out parameters needed for DAG initialization and compute

        Args:
            select (dict, optional): DAG selection
            transform (Sequence[dict], optional): DAG transformation
            compute_only (Sequence[str], optional): DAG tags to be computed
            dag_options (dict, optional): Other DAG options for initialization
            **plot_kwargs: The full plot configuration

        Returns:
            Tuple[dict, dict]: Tuple of DAG parameters and plot kwargs
        """
        # Top-level arguments
        init_kwargs = dict(select=select, transform=transform)
        compute_kwargs = dict(compute_only=compute_only)

        # Options. Only add those, if available
        if dag_options:
            init_kwargs = dict(**init_kwargs, **dag_options)

        return dict(init=init_kwargs, compute=compute_kwargs), plot_kwargs

    def _use_dag(self, *, use_dag: bool, plot_kwargs: dict, **_kws) -> bool:
        """Whether the data transformation framework should be used.

        Args:
            use_dag (bool): The value from the plot configuration
            plot_kwargs (dict): The plot configuration
            **_kws: Any further kwargs that can be used to assess whether the
                DAG should be used or not. Ignored here.

        Returns:
            bool: Whether the DAG should be used or not
        """
        return (use_dag if use_dag is not None else False)

    def _create_dag(self, **dag_params) -> TransformationDAG:
        """Creates the actual DAG object"""
        return TransformationDAG(dm=self.dm, **dag_params)

    def _compute_dag(self, dag: TransformationDAG, **compute_kwargs) -> dict:
        """Compute the dag results"""
        return dag.compute(**compute_kwargs)

    def _combine_dag_results_and_plot_cfg(self, *, dag: TransformationDAG,
                                          dag_results: dict, dag_params: dict,
                                          plot_kwargs: dict) -> dict:
        """Combines DAG reuslts and plot configuration into one dict. The
        returned dict is then passed along to the ``plot`` method.

        The base class method ditches the ``dag_params`` and only retains the
        results, the DAG object itself, and (of course) all the remaining plot
        configuration.

        .. note::

            When subclassing, this is the method to overwrite or extend in
            order to affect which data gets passed on.
        """
        # Build a dict, delibaretly excluding the dag_params
        return dict(dag=dag, dag_results=dag_results, **plot_kwargs)
