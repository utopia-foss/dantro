"""This module implements :py:class:`.BasePlotCreator`, the base class for plot
creators.

The interface is defined as an abstract base class and partly implemented by
the :py:class:`.BasePlotCreator`, which is no longer abstract but has only the
functionality that is general enough for all derived creators to profit from.
"""

import contextlib
import copy
import gc
import importlib
import importlib.util
import logging
import os
import time
from typing import Callable, Dict, Sequence, Tuple, Union

from paramspace import ParamSpace

from ..._copy import _deepcopy
from ..._dag_utils import resolve_placeholders as _resolve_placeholders
from ..._hash import _hash
from ...abc import AbstractPlotCreator
from ...dag import TransformationDAG
from ...data_mngr import DataManager
from ...exceptions import PlotCreatorError, SkipPlot
from ...tools import format_time as _format_time
from ...tools import recursive_update

log = logging.getLogger(__name__)

_fmt_time = lambda t: _format_time(t, ms_precision=1)

_DAG_OBJECT_CACHE: Dict[str, TransformationDAG] = dict()
"""A dict holding the previously-used :py:class:`~dantro.dag.TransformationDAG`
objects to allow re-using them in another plot function.
The keys are hashes of the configuration used in creating the DAG."""


# -----------------------------------------------------------------------------


class BasePlotCreator(AbstractPlotCreator):
    """The base class for plot creators.

    It provides the following functionality:

    - Resolving a plot function, which can be a directly given callable, an
      importable module and name, or a path to a Python file that is to be
      imported.
    - Parsing plot configuration arguments.
    - Optionally, performing data selection from the associated
      :py:class:`~dantro.data_mngr.DataManager` using the
      :ref:`data transformation framework <dag_framework>`.
    - Invoking the plot function.

    As such, the this base class is agnostic to the exact way of how plot
    output is generated; the plot function is responsible for that.
    """

    EXTENSIONS: Union[Tuple[str], str] = "all"
    """A tuple of supported file extensions.
    If ``all``, no checks for the extensions are performed."""

    DEFAULT_EXT = None
    """The default file extension to use; is only used if no default extension
    is specified during initialization"""

    DEFAULT_EXT_REQUIRED: bool = True
    """Whether a default extension is required or not. If True and the
    ``default_ext`` property evaluates to False, an error will be raised."""

    POSTPONE_PATH_PREPARATION: bool = False
    """Whether to prepare paths in the base class's
    :py:meth:`~dantro.plot.creators.base.BasePlotCreator.__call__` method
    or not. If the derived class wants to take care of this on their own, this
    should be set to True and the
    :py:meth:`~dantro.plot.creators.base.BasePlotCreator._prepare_path`
    method, adjusted or not, should be called at another point of the plot
    execution."""

    OUT_PATH_EXIST_OK: bool = False
    """Whether a warning should be shown (instead of an error), when a plot
    file already exists at the specified output path"""

    DAG_USE_DEFAULT: bool = False
    """Whether the :ref:`data transformation framework <pcr_base_dag_support>`
    is enabled *by default*; this can still be controlled by the ``use_dag``
    argument of the plot configuration.
    """

    DAG_RESOLVE_PLACEHOLDERS: bool = True
    """Whether placeholders in the plot config,
    :py:class:`~dantro._dag_utils.ResultPlaceholder` objects, should be
    replaced with results from the data transformations."""

    DAG_TO_KWARG_MAPPING: Dict[str, str] = {
        "results_dict": "data",
        "dag_object": "dag",
    }
    """The keyword argument names by which to pass the data transformation
    results (``results_dict``) or the :py:class:`~dantro.dag.TransformationDAG`
    object itself (``dag_object``) to the plot function.
    """

    # .........................................................................

    def __init__(
        self,
        name: str,
        *,
        dm: DataManager,
        plot_func: Callable,
        default_ext: str = None,
        exist_ok: Union[bool, str] = None,
        raise_exc: bool = None,
        **plot_cfg,
    ):
        """Create a plot creator instance for a plot with the given ``name``.

        Typically, a creator has not be instantiated separately, but the
        :py:class:`~dantro.plot_mngr.PlotManager` takes care of it.

        Args:
            name (str): The name of this plot
            dm (DataManager): The data manager that contains the data to plot
            default_ext (str, optional): The default extension to use; needs
                to be in ``EXTENSIONS``, if that class variable is not set to
                'all'. The value given here is needed by the PlotManager to
                build the output path.
            exist_ok (Union[bool, str], optional): If True, no error will be
                raised when a plot already exists at the specified output path.
                If None, the value specified in the ``OUT_PATH_EXIST_OK`` class
                variable will be used to determine this behaviour.
                If ``skip``, will skip the plot, allowing other plots to be
                carried out; see :ref:`plot_mngr_skipping_plots`.
            raise_exc (bool, optional): Whether to raise exceptions during the
                plot procedure. This does not pertain to *all* exceptions, but
                only to those that would *unnecessarily* stop plotting.
                Furthermore, whether this setting is used or not depends on the
                used creator specialization.
            **plot_cfg: The default configuration for the plot(s) that this
                creator is supposed to create.

        Raises:
            ValueError: On bad ``base_module_file_dir`` or ``default_ext``
        """
        self._name = name
        self._dm = dm
        self._plot_cfg = plot_cfg
        self._plot_func = plot_func
        self._using_dag = None
        self._dag_obj_cache = _DAG_OBJECT_CACHE
        self._out_path = None
        self._exist_ok = (
            self.OUT_PATH_EXIST_OK if exist_ok is None else exist_ok
        )
        self.raise_exc = raise_exc

        # Property-managed attributes
        self._logstr = None
        self._default_ext = None
        self._dag = None

        # DAG visualization
        self._dag_vis_kwargs = None
        self._dag_vis_done_for = []

        # Set the default extension, first from argument, then default.
        # Then check that it was set correctly
        if default_ext is not None:
            self.default_ext = default_ext

        elif self.DEFAULT_EXT is not None:
            self.default_ext = self.DEFAULT_EXT

        if self.DEFAULT_EXT_REQUIRED and not self.default_ext:
            raise ValueError(
                f"{self.logstr} requires a default extension, but neither "
                f"the argument ('{default_ext}') nor the DEFAULT_EXT class "
                f"variable ('{self.DEFAULT_EXT}') was set."
            )

    # .. Properties ...........................................................

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
    def plot_func(self) -> Callable:
        """Returns the plot function"""
        return self._plot_func

    @property
    def plot_func_name(self) -> str:
        """Returns a readable name of the plot function"""
        pf = self.plot_func
        return getattr(pf, "name", pf.__name__)

    @property
    def plot_cfg(self) -> Dict[str, dict]:
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
        """Sets the default extension.

        Unless :py:attr:`.EXTENSIONS` is set to ``all``, needs to be a valid
        extension.
        """
        if self.EXTENSIONS != "all" and val not in self.EXTENSIONS:
            raise ValueError(
                f"Extension '{val}' not supported in {self.logstr}. "
                f"Supported extensions are: {', '.join(self.EXTENSIONS)}"
            )

        self._default_ext = val

    @property
    def dag(self) -> TransformationDAG:
        """The associated TransformationDAG object. If not set up, raises."""
        if self._dag is not None:
            return self._dag
        raise ValueError(
            f"{self.logstr} has no TransformationDAG associated (yet)!"
        )

    # .. Main API functions, required by PlotManager ..........................

    def __call__(self, *, out_path: str, **update_plot_cfg):
        """Perform the plot, updating the configuration passed to __init__
        with the given values and then calling :py:meth:`.plot`.

        Args:
            out_path (str): The full output path to store the plot at
            **update_plot_cfg: Keys with which to update the default plot
                configuration

        Returns:
            The return value of the :py:meth:`.plot` method, which is an
            abstract method in
            :py:class:`~dantro.plot.creators.base.BasePlotCreator`.
        """
        # Get (a deep copy of) the initial plot config, update it with new args
        cfg = self.plot_cfg
        if update_plot_cfg:
            cfg = recursive_update(cfg, copy.deepcopy(update_plot_cfg))

        # Allow derived creators to check whether this plot should be skipped
        self._check_skipping(plot_kwargs=cfg)

        # Find out if it's ok if out_path already exists, then prepare the path
        exist_ok = self._exist_ok
        if "exist_ok" in cfg:
            exist_ok = cfg.pop("exist_ok")

        if not self.POSTPONE_PATH_PREPARATION:
            self._prepare_path(out_path, exist_ok=exist_ok)

        # Now call the plotting function with these arguments
        return self.plot(out_path=out_path, **cfg)

    def plot(
        self,
        *,
        out_path: str,
        use_dag: bool = None,
        **func_kwargs,
    ):
        """Prepares argument for the plot function and invokes it.

        Args:
            out_path (str): The output path for the resulting file
            use_dag (bool, optional): Whether to use the :ref:`dag_framework`
                to select and transform data that can be used in the plotting
                function. If not given, will query the plot function attributes
                for whether the DAG should be used. If not, the data selection
                has to occur separately inside the plot function. Note that
                this may require a different plot function signature.
            **func_kwargs: Passed on to the plot function
        """
        # Store the output path, needed by downstream methods
        self._out_path = out_path

        # Prepare arguments, also performing plot data selection
        args, kwargs = self._prepare_plot_func_args(
            use_dag=use_dag, out_path=out_path, **func_kwargs
        )

        # Call the plot function, optionally generating a DAG visualization
        self._invoke_plot_func(*args, **kwargs)

    def get_ext(self) -> str:
        """Returns the extension to use for the upcoming plot by checking
        the supported extensions and can be subclassed to have different
        behaviour.
        """
        return self.default_ext

    def prepare_cfg(
        self, *, plot_cfg: dict, pspace: Union[ParamSpace, dict]
    ) -> Tuple[dict, ParamSpace]:
        """Prepares the plot configuration for the
        :py:class:`~dantro.plot_mngr.PlotManager`. This function is called by
        the manager before the first plot is to be created.

        The base implementation just passes the given arguments through.
        However, it can be re-implemented by derived classes to change the
        behaviour of the plot manager, e.g. by converting a plot configuration
        to a parameter space.
        """
        return plot_cfg, pspace

    # .........................................................................

    def _prepare_path(
        self, out_path: str, *, exist_ok: Union[bool, str]
    ) -> None:
        """Prepares the output path, creating directories if needed, then
        returning the full absolute path.

        This is called from :py:meth:`.__call__` and is meant to postpone
        directory creation as far as possible.

        Args:
            out_path (str): The absolute output path to start with
            exist_ok (Union[bool, str]): If False, will raise if a file of that
                name already exists; if True, will emit a warning instead.
                If ``'skip'``, will initiate skipping of this plot.

        Raises:
            FileExistsError: Raised on already existing out path and exist_ok
                being False.
        """
        if os.path.exists(out_path):
            msg = (
                "There already exists a file at the specified output path "
                f"for {self.logstr}:\n  {out_path}"
            )
            if not exist_ok:
                raise FileExistsError(msg)
            elif exist_ok == "skip":
                raise SkipPlot(f"Plot output already exists:\n  {out_path}")
            else:
                log.warning(msg)

        # Ensure that all necessary directories exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _check_skipping(self, *, plot_kwargs: dict):
        """A method that can be specialized by derived plot creators to check
        whether a plot should be skipped.
        Is invoked from the :py:meth:`.__call__` method, *after*
        :py:meth:`._perform_data_selection` (for plots with activated data
        selection via DAG), and *prior to* :py:meth:`._prepare_path`
        (such that path creation can be avoided).

        In cases where this plot is to be skipped, the custom exception
        :py:class:`~dantro.exceptions.SkipPlot` should be raised,
        the error message allowing to specify a reason for skipping the plot.

        .. note::

            While the base class method may be a no-op, it should still be
            called via ``super()._check_skipping`` from the derived classes.

        Args:
            plot_kwargs (dict): The full plot configuration
        """
        pass

    # .........................................................................
    # Plot argument preparation and function invocation

    def _prepare_plot_func_args(
        self, *, use_dag: bool = None, **kwargs
    ) -> Tuple[tuple, dict]:
        """Prepares the arguments passed to the plot function.

        The passed keyword arguments are carried over; no positional arguments
        are possible.
        Subsequently, possible signatures look as follows:

        - When using the data transformation framework, there are *no*
          positional arguments.
        - When *not* using the data transformation framework, the *only*
          positional argument is the :py:class:`~dantro.data_mngr.DataManager`
          instance that is associated with this plot.

        .. note::

            When subclassing this function, the parent method (this one) should
            still be called to maintain base functionality.

        Args:
            use_dag (bool, optional): Whether to use the data transformation
                framework
            **kwargs: Additional kwargs

        Returns:
            Tuple[tuple, dict]: an (empty) tuple of positional arguments and a
                dict of keyword arguments.
        """
        # Perform data selection and transformation, if the plot creator class
        # supports it.
        # Even if the creator supports it, it might be disabled in the config;
        # in that case, the method below behaves like a passthrough of the cfg,
        # filtering out all transformation-related arguments.
        # The returned kwargs are the adjusted plot function keyword arguments.
        using_dag, kwargs = self._perform_data_selection(
            use_dag=use_dag, plot_kwargs=kwargs
        )
        self._using_dag = using_dag

        # Aggregate as (args, kwargs), passed on to plot function. When using
        # the DAG, the DataManager is NOT passed along, as it is accessible via
        # the tags of the DAG.
        if not using_dag:
            return ((self.dm,), kwargs)
        return ((), kwargs)

    def _invoke_plot_func(self, *args, **kwargs):
        """Method that invokes the plot function with the prepared arguments.

        This additionally allows to generate a DAG visualization in case the
        plotting failed or succeeded.
        """
        log.info("Now calling plotting function '%s' ...", self.plot_func_name)
        try:
            self.plot_func(*args, **kwargs)

        except:
            if self._using_dag:
                self._generate_DAG_vis(
                    scenario="plot_error", **self._dag_vis_kwargs
                )
            raise

        else:
            if self._using_dag:
                self._generate_DAG_vis(
                    scenario="plot_success", **self._dag_vis_kwargs
                )

        log.note("Plotting function '%s' returned.", self.plot_func_name)

    # .........................................................................
    # Data selection interface, using TransformationDAG

    def _perform_data_selection(
        self, *, use_dag: bool = None, plot_kwargs: dict, **shared_kwargs
    ) -> Tuple[bool, dict]:
        """If this plot creator supports data selection and transformation, it
        is carried out in this method.

        This method uses a number of other private methods to carry out the
        setup of the DAG, computing it and combining its results with the
        remaining plot configuration. Those methods have access to a subset of
        the whole configuration, thus allowing to parse the parameters that
        they need.

        This method also sets the ``_dag`` attribute, making the created
        :py:class:`~dantro.dag.TransformationDAG` object available for further
        processing downstream.

        Furthermore, this method invokes placeholder resolution by applying
        :py:func:`~dantro._dag_utils.resolve_placeholders` on the plot config.

        .. note::

            For specializing the behaviour of the data selection and transform,
            it is best to specialize *NOT* this method, but the more granular
            DAG-related private methods.

        .. warning::

            If subclassing this method, make sure to either invoke this parent
            method or set the ``_dag`` attribute in the subclass's method.
            Also note that, when subclassing, the ability to resolve the
            placeholders gets lost / has to be re-implemented in the subclass.

        Args:
            use_dag (bool, optional): The main toggle for whether the DAG
                should be used or not. This is passed as default value to
                another method, which takes the final decision on whether the
                DAG is used or not. If None, will first inspect whether the
                plot function declared that the DAG is to be used.
                If still None, will NOT use the DAG.
            plot_kwargs (dict): The plot configuration
            **shared_kwargs: Shared keyword arguments that are passed through
                to the helper methods :py:meth:`._use_dag` and
                :py:meth:`._get_dag_params`.

        Returns:
            Tuple[bool, dict]: Whether data selection was used and the plot
                configuration that can be passed on to the main ``plot``
                method.
        """
        # Determine whether the DAG framework should be used or not
        if not self._use_dag(
            use_dag=use_dag, plot_kwargs=plot_kwargs, **shared_kwargs
        ):
            # Only return the plot configuration, without DAG-related keys
            return False, plot_kwargs
        # else: DAG should be used
        # Extract DAG-related parameters from the plot configuration. These are
        # not available in the plotting function.
        dag_params, plot_kwargs = self._get_dag_params(
            **plot_kwargs, **shared_kwargs
        )

        # Create the DAG object, optionally reading from and/or writing to the
        # DAG object cache. Then make available to other parts.
        dag = self._setup_dag(dag_params["init"], **dag_params["cache"])
        self._dag = dag

        # Then compute results
        dag_results = self._compute_dag(dag, **dag_params["compute"])

        # If enabled, perform placeholder resolution in plot_kwargs
        if self.DAG_RESOLVE_PLACEHOLDERS:
            plot_kwargs = _resolve_placeholders(plot_kwargs, dag=dag)

        # Prepare the parameters passed back to __call__ and on to self.plot
        kws = self._combine_dag_results_and_plot_cfg(
            dag=dag,
            dag_results=dag_results,
            dag_params=dag_params,
            plot_kwargs=plot_kwargs,
        )
        return True, kws

    def _use_dag(self, *, use_dag: bool, plot_kwargs: dict) -> bool:
        """Whether the DAG should be used or not. This method extends that of
        the base class by additionally checking the plot function attributes
        for any information regarding the DAG.

        This relies on the
        :py:class:`~dantro.plot.utils.plot_func.is_plot_func`
        decorator to set a number of function attributes.
        """
        # If None was given, check the plot function attributes
        if use_dag is None:
            use_dag = getattr(self.plot_func, "use_dag", None)

        # If still None, default to the class variable default
        if use_dag is None:
            use_dag = self.DAG_USE_DEFAULT

        # Complain, if tags were required, but DAG usage was disabled
        if not use_dag and getattr(self.plot_func, "required_dag_tags", None):
            raise ValueError(
                f"The plot function {self.plot_func} requires DAG tags to be "
                "computed, but DAG usage was disabled."
            )

        return use_dag

    def _get_dag_params(
        self,
        *,
        select: dict = None,
        transform: Sequence[dict] = None,
        compute_only: Sequence[str] = None,
        dag_options: dict = None,
        dag_object_cache: dict = None,
        dag_visualization: dict = None,
        invocation_options: dict = None,
        **plot_kwargs,
    ) -> Tuple[dict, dict]:
        """Filters out and parses parameters that are needed for initialization
        of the :py:class:`~dantro.dag.TransformationDAG` in
        :py:meth:`._setup_dag` and computation in :py:meth:`_compute_dag`.

        Args:
            select (dict, optional): DAG selection
            transform (Sequence[dict], optional): DAG transformation
            compute_only (Sequence[str], optional): DAG tags to be computed
            dag_options (dict, optional): Other DAG options for initialization
            dag_object_cache (dict, optional): Cache options for the DAG object
                itself. Expected keys are ``read``, ``write``, ``clear``.
            dag_visualization (dict, optional): If given, controls whether the
                DAG used for data transformations should also be plotted, e.g.
                to make debugging easier.
            invocation_options (dict, optional): Controls whether to pass
                certain objects on to the plot functio or not. Supported keys
                are ``pass_dag_object_along`` and ``unpack_dag_results``, which
                take precedence over the plot function attributes of the same
                name which are set by the plot function decorator
                :py:class:`~dantro.plot.utils.plot_func.is_plot_func`.
            **plot_kwargs: The remaining plot configuration

        Returns:
            Tuple[dict, dict]: Tuple of DAG parameter dict and plot kwargs
        """
        # Top-level arguments
        init_kwargs = dict(select=select, transform=transform)
        compute_kwargs = dict(compute_only=compute_only)
        cache_kwargs = dag_object_cache if dag_object_cache else {}
        vis_kwargs = dag_visualization if dag_visualization else {}

        # Options. Only add those, if available
        dag_options = dag_options if dag_options else {}
        if dag_options:
            init_kwargs = dict(**init_kwargs, **dag_options)

        dag_params = dict(
            init=init_kwargs,
            compute=compute_kwargs,
            cache=cache_kwargs,
            visualization=vis_kwargs,
        )

        # Also store visualization kwargs as attribute
        self._dag_vis_kwargs = vis_kwargs
        self._dag_vis_done_for = []

        # Determine whether the DAG object should be passed along to the func
        invocation_options = invocation_options if invocation_options else {}
        _pass_dag = invocation_options.get("pass_dag_object_along")
        if _pass_dag is None:
            _pass_dag = getattr(self.plot_func, "pass_dag_object_along", False)
        dag_params["pass_dag_object_along"] = _pass_dag

        # Determine whether the DAG results should be unpacked when passing
        # them to the plot function
        _unpack = invocation_options.get("unpack_dag_results")
        if _unpack is None:
            _unpack = getattr(self.plot_func, "unpack_dag_results", False)
        dag_params["unpack_dag_results"] = _unpack

        return dag_params, plot_kwargs

    def _setup_dag(
        self,
        init_params: dict,
        *,
        read: bool = False,
        write: bool = False,
        clear: bool = False,
        collect_garbage: bool = None,
        use_copy: bool = True,
    ) -> TransformationDAG:
        """Creates a :py:class:`~dantro.dag.TransformationDAG` object from the
        given initialization parameters.
        Optionally, will use a hash of the initialization parameters to reuse
        a deep copy of a cached object.

        In case no cached version was available or caching was disabled, uses
        :py:meth:`._create_dag` to create the object.

        Args:
            init_params (dict): Initialization parameters, passed on to the
                ``_create_dag`` method.
            read (bool, optional): Whether to read from memory cache
            write (bool, optional): Whether to write to memory cache
            clear (bool, optional): Whether to clear the whole memory cache,
                can be useful if many objects were stored and memory runs low.
                Afterwards, garbage collection may be required to actually free
                the memory; see ``collect_garbage``.
            collect_garbage (bool, optional): If True, will invoke garbage
                collection; this may be required after clearing the cache but
                may also be useful to invoke separately from that.
                If None, will invoke garbage collection automatically if the
                cache was set to be cleared.
            use_copy (bool, optional): Whether to work on a (deep) copy of the
                cached DAG object. This reduces memory footprint, but may not
                bring a noticeable speedup.
        """
        t0 = time.time()

        log.note("Setting up data transformation framework ...")

        dag = None
        cp_func = _deepcopy if use_copy else lambda d: d

        # Compute the cache key only once and only if needed
        if read or write:
            cache_key = _hash(repr(init_params))

        if read:
            dag = cp_func(self._dag_obj_cache.get(cache_key))

        if dag is not None:
            log.remark(
                "Loaded TransformationDAG from memory cache. "
                "(copy? %s, cache size: %d)",
                use_copy,
                len(self._dag_obj_cache),
            )

        else:
            dag = self._create_dag(**init_params)

            if write and cache_key not in self._dag_obj_cache:
                self._dag_obj_cache[cache_key] = cp_func(dag)
                log.remark(
                    "Stored TransformationDAG in memory cache. "
                    "(copy? %s, cache size: %d)",
                    use_copy,
                    len(self._dag_obj_cache),
                )

        if clear:
            self._dag_obj_cache.clear()
            log.remark("TransformationDAG memory cache cleared.")

        if collect_garbage or (collect_garbage is None and clear):
            log.remark(
                "Invoking garbage collection ... (this may take a while)"
            )
            gc.collect()
            log.remark("Garbage collected.")

        elif collect_garbage is False:
            log.remark("NOT invoking garbage collection.")

        log.note(
            "TransformationDAG with %d nodes set up in %s.",
            len(dag.nodes),
            _fmt_time(time.time() - t0),
        )
        return dag

    def _create_dag(self, **dag_params) -> TransformationDAG:
        """Creates the actual DAG object"""
        return TransformationDAG(dm=self.dm, **dag_params)

    def _compute_dag(
        self,
        dag: TransformationDAG,
        *,
        compute_only: Sequence[str],
        **compute_kwargs,
    ) -> dict:
        """Compute the dag results.

        This checks whether all required tags (set by the
        :py:class:`~dantro.plot.utils.plot_func.is_plot_func` decorator)
        are set to be computed.
        """
        # Extract the required tags from the plot function attributes
        required_tags = getattr(self.plot_func, "required_dag_tags", None)

        # Make sure that all required tags are actually defined
        if required_tags:
            missing_tags = [t for t in required_tags if t not in dag.tags]

            if missing_tags:
                raise ValueError(
                    "Plot function {} required tags that were "
                    "not specified in the DAG: {}. Available "
                    "tags: {}. Please adjust the DAG "
                    "specification accordingly."
                    "".format(
                        self.plot_func_name,
                        ", ".join(missing_tags),
                        ", ".join(dag.tags),
                    )
                )

        # If the compute_only argument was not explicitly given, determine
        # whether to compute only the required tags
        if (
            compute_only is None
            and required_tags is not None
            and getattr(
                self.plot_func, "compute_only_required_dag_tags", False
            )
        ):
            log.remark(
                "Tags that are required by the plot function:  %s",
                ", ".join(required_tags),
            )
            compute_only = required_tags

        # Make sure the compute_only argument contains all the required tags
        elif compute_only is not None and required_tags is not None:
            missing_tags = [t for t in required_tags if t not in compute_only]

            if missing_tags:
                raise ValueError(
                    "Plot function {} required tags that were "
                    "not set to be computed by the DAG: {}. Make "
                    "sure to set the `compute_only` argument "
                    "such that results for all required tags "
                    "({}) will actually be computed.\n"
                    "Available tags:  {}\n"
                    "compute_only:    {}"
                    "".format(
                        self.plot_func_name,
                        ", ".join(missing_tags),
                        ", ".join(required_tags),
                        ", ".join(dag.tags),
                        ", ".join(compute_only),
                    )
                )

        # Now compute the results
        log.info("Computing data transformation results ...")
        try:
            results = dag.compute(compute_only=compute_only, **compute_kwargs)

        except:
            self._generate_DAG_vis(
                scenario="compute_error", **self._dag_vis_kwargs
            )
            raise

        else:
            log.remark("Finished computing data transformation results.")
            self._generate_DAG_vis(
                scenario="compute_success", **self._dag_vis_kwargs
            )

        return results

    def _combine_dag_results_and_plot_cfg(
        self,
        *,
        dag: TransformationDAG,
        dag_results: dict,
        dag_params: dict,
        plot_kwargs: dict,
    ) -> dict:
        """Returns a dict of plot configuration and ``data``, where all the
        DAG results are stored in.
        In case where the DAG results are to be unpacked, the DAG results will
        be made available as separate keyword arguments instead of as the
        single ``data`` keyword argument.

        Furthermore, if the plot function specified in its attributes that the
        DAG object is to be passed along, this is the place where it is
        included or excluded from the arguments.
        """
        if dag_params["unpack_dag_results"]:
            # Unpack the results such that they can be specified in the plot
            # function signature
            try:
                cfg = dict(**dag_results, **plot_kwargs)

            except TypeError as err:
                raise TypeError(
                    "Failed unpacking DAG results! There were arguments of "
                    "the same names as some DAG tags given in the plot "
                    "configuration. Make sure they have unique names or "
                    "disable unpacking of the DAG results.\n"
                    "  Keys in DAG results: {}\n"
                    "  Keys in plot config: {}\n"
                    "".format(
                        ", ".join(dag_results.keys()),
                        ", ".join(plot_kwargs.keys()),
                    )
                ) from err

        else:
            # Make the DAG results available as `data` kwarg
            cfg = dict(**plot_kwargs)
            cfg[self.DAG_TO_KWARG_MAPPING["results_dict"]] = dag_results

        # Add the `dag` kwarg, if configured to do so.
        if dag_params["pass_dag_object_along"]:
            cfg[self.DAG_TO_KWARG_MAPPING["dag_object"]] = dag

        return cfg

    # .........................................................................
    # DAG Visualization

    def _generate_DAG_vis(
        self,
        *,
        scenario: str,
        enabled: bool = True,
        plot_enabled: bool = True,
        export_enabled: bool = True,
        raise_exc: bool = None,
        when: dict = None,
        output: dict = None,
        export: dict = None,
        generation: dict = None,
        **plot_kwargs,
    ) -> Union["networkx.DiGraph", None]:
        """Generates a DAG representation according to certain criteria and
        using :py:meth:`~dantro.dag.TransformationDAG.generate_nx_graph`,
        then invokes :py:meth:`~dantro.dag.TransformationDAG.visualize` to
        create the actual visualization output.

        This method also allows to export the DAG representation using
        :py:func:`~dantro.utils.nx.export_graph`, which can then be used for
        externally working with the DAG representation.

        Also see :ref:`plot_creator_dag_vis` and :ref:`dag_graph_vis`.

        Args:
            scenario (str): The scenario in which the generation is invoked;
                this is used to describe the context in which this method was
                invoked and also becomes part of the output file name.
                See ``when`` for scenarios with certain names. If you want to
                use a different name, make sure to set ``when.always``, such
                that no scenario lookup occurs.
            enabled (bool, optional): If False, will return None.
            plot_enabled (bool, optional): Whether plotting is enabled. The
                result of the ``when`` evaluation overrules this.
            export_enabled (bool, optional): Whether exporting is enabled. The
                result of the ``when`` evaluation overrules this.
            raise_exc (bool, optional): Whether to raise exceptions if anything
                goes wrong within this method. If None, will behave in the same
                way as the creator does. For example, if set to False, an
                error in generating a DAG representation will *not* lead to an
                error being raised.
            when (dict, optional): A dict that specifies in which situations
                the generation should actually be carried out. May contain the
                following keys:

                    - ``always``: Will always generate output.
                    - ``only_once``: If True, will only generate output from
                      one scenario, skipping further invocations.
                    - ``on_compute_error``, ``on_compute_success``, and
                      ``on_plot_error``, ``on_plot_success``: Will generate
                      output only in certain named **scenarios**.
                      These can be a boolean toggle or ``'debug'`` in which
                      case the **creator's** debug flag decides whether output
                      is generated for that scenario.
                      Note that the ``raise_exc`` *argument* does not play a
                      role for that!

            output (dict, optional): A dict specifying where the DAG plot and
                potential exported files are written to. Allowed keys are:

                    - ``plot_dir``: If None, will write output aside the
                      plot output itself. Can also be an absolute path.
                    - ``path_fstr``: A format string that specifies the actual
                      output path and should/can contain the keys ``plot_dir``,
                      ``name``, and ``scenario``.

            export (dict, optional): Export specification, using networkx's
                write methods. Possible keys:

                    - ``manipulate_attrs``: Dict that controls manipulation of
                      node or edge attributes, sometimes necessary for export.
                      These are passed to
                      :py:func:`~dantro.utils.nx.manipulate_attributes`.
                    - any further keyword arguments define the output formats
                      that are to be used.
                      They can be of type ``Dict[str, Union[bool, dict]]``,
                      where the string should correspond to the name of a
                      networkx writer method. The boolean is used to enable
                      or disable a writer. If a dict is given, its content is
                      passed to the writer method.
                      Also see :py:func:`~dantro.utils.nx.export_graph`, where
                      this is implemented.

            generation (dict, optional): Graph generation arguments passed to
                :py:meth:`~dantro.dag.TransformationDAG.generate_nx_graph`.
            **plot_kwargs: Plotting-related arguments, passed on to
                :py:meth:`~dantro.dag.TransformationDAG.visualize`.

        Returns:
            Union[networkx.DiGraph, None]: Either the generated graph object
                or None, if not enabled or ``when`` was evaluated to not
                generating a DAG representation.
        """

        def should_plot(
            scenario: str,
            *,
            enabled: bool,
            always: bool = False,
            only_once: bool = False,
            on_compute_error: Union[bool, str] = True,
            on_compute_success: Union[bool, str] = False,
            on_plot_error: Union[bool, str] = False,
            on_plot_success: Union[bool, str] = False,
        ) -> bool:
            """Decides whether a DAG visualization should be created in a
            certain scenario.
            """
            if not enabled:
                return False

            if always:
                return True

            if only_once and scenario in self._dag_vis_done_for:
                return False

            scenarios = dict(
                compute_error=on_compute_error,
                compute_success=on_compute_success,
                plot_error=on_plot_error,
                plot_success=on_plot_success,
            )
            if scenarios[scenario] == "debug" and self.raise_exc:
                return True
            elif scenarios[scenario] is True:
                return True
            return False

        def parse_output_path(
            scenario: str,
            *,
            plot_dir: str = None,
            path_fstr: str = "{plot_dir:}/{name:}_dag_{scenario:}.pdf",
            **fstr_kwargs,
        ) -> str:
            """Prepares the output path for the DAG visualization"""
            if plot_dir is None and "plot_dir" in path_fstr:
                if self._out_path is None:
                    raise ValueError(
                        "Missing plot output path from which to extract the "
                        "`plot_dir` argument for the DAG visualization output "
                        "path! This should not have happened; make sure your "
                        "plot creator sets the _out_path attribute before "
                        "DAG visualization is invoked."
                    )
                plot_dir = os.path.dirname(self._out_path)

            p = path_fstr.format(
                plot_dir=plot_dir,
                name=self.name,
                scenario=scenario,
                **fstr_kwargs,
            )
            return os.path.expanduser(p)

        @contextlib.contextmanager
        def exception_handling(desc: str):
            """Exception handler for parts of the DAG representation routine"""
            try:
                yield

            except Exception as exc:
                msg = f"Failed {desc}!"
                _raise = raise_exc if raise_exc is not None else self.raise_exc
                if not _raise:
                    log.warning(msg)
                    log.note("Enable debug mode to show traceback.")
                    return

                raise PlotCreatorError(
                    f"{msg}\n"
                    "Inspect the chained traceback for more information "
                    "or disable debug mode to ignore this error message.\n\n"
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        # .....................................................................

        when = when if when else {}
        output = output if output else {}
        export = export if export else {}
        generation = generation if generation else {}

        # Decide whether to plot
        if not should_plot(scenario, enabled=enabled, **when):
            log.debug("Not creating DAG visualization.")
            return

        log.note("Creating DAG visualization (scenario: '%s') ...", scenario)

        # Create the graph object.
        # If this fails and we are not allowed to raise, we have no other
        # option but to return None; would not make sense to continue.
        g = None
        with exception_handling("generating DAG representation"):
            g = self._dag.generate_nx_graph(**generation)

        if g is None:
            return

        # Generate the output path (for the plot)
        out_path = parse_output_path(scenario, **output)

        # Set some parameter defaults
        title_fstr = "DAG @ scenario '{}'"
        default_plot_kwargs = dict(
            annotate_kwargs=dict(
                title=title_fstr.format(scenario.replace("_", " ")),
                add_legend=True,
            )
        )
        plot_kwargs = recursive_update(
            default_plot_kwargs, copy.deepcopy(plot_kwargs)
        )

        # Export the graph object
        if export_enabled:
            from ...utils.nx import export_graph

            with exception_handling("exporting DAG representation"):
                export_graph(g, out_path=out_path, **export)

        # Plot it
        if plot_enabled:
            with exception_handling("plotting DAG representation"):
                self._dag.visualize(g=g, out_path=out_path, **plot_kwargs)

                if "error" in scenario:
                    log.caution(
                        "Created DAG visualization for scenario '%s'. "
                        "For debugging, inspecting the generated plot and the "
                        "traceback information may be helpful:\n  %s",
                        scenario,
                        out_path,
                    )

        # All done
        self._dag_vis_done_for.append(scenario)

        return g
