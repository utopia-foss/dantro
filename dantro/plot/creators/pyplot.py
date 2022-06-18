"""This module implements the :py:class:`.PyPlotCreator` class, which
specializes on creating :py:mod:`matplotlib.pyplot`-based plots."""

import copy
import logging
import os
from typing import Callable, List, Sequence, Tuple, Union

from ..._import_tools import LazyLoader
from ...dag import TransformationDAG
from ...tools import DoNothingContext, load_yml, recursive_update
from ..plot_helper import (
    EnterAnimationMode,
    ExitAnimationMode,
    PlotHelper,
    PlotHelperError,
    PlotHelperErrors,
)
from ..utils import figure_leak_prevention
from .base import BasePlotCreator, _resolve_placeholders

log = logging.getLogger(__name__)

mpl = LazyLoader("matplotlib")
plt = LazyLoader("matplotlib.pyplot")

# -----------------------------------------------------------------------------


class PyPlotCreator(BasePlotCreator):
    """A plot creator that is specialized on creating plots using
    :py:mod:`matplotlib.pyplot`. On top of the capabilities of
    :py:class:`~dantro.plot.creators.base.BasePlotCreator`, this class
    contains specializations for the matplotlib-based plotting backend:

    - The :py:class:`~dantro.plot.plot_helper.PlotHelper` provides an interface
      to a wide range of the :py:mod:`matplotlib.pyplot` interface, allowing to
      let the plot function itself focus on generating a visual representation
      of the data and removing boilerplate code; see :ref:`plot_helper`.
    - There are so-called "style contexts" that a plot can be generated in,
      allowing to have consistent and easily adjsutable aesthetics; see
      :ref:`pcr_pyplot_style`.
    - By including the :py:mod:`matplotlib.animation` framework, allows to
      easily implement plot functions that generate animation output.

    For more information, refer to :ref:`the user manual <pcr_pyplot>`.
    """

    # Settings that are inherited from the BasePlotCreator ....................

    EXTENSIONS = "all"
    """Allowed file extensions; ``all`` means that every extension is allowed
    and that there are no checks performed."""

    DEFAULT_EXT = None
    """The default file extension"""

    DEFAULT_EXT_REQUIRED = False
    """Whether a default extension needs to be specified"""

    DAG_SUPPORTED = True
    """Whether this creator supports :ref:`dag_framework`"""

    DAG_INVOKE_IN_BASE = False
    """Whether DAG invocation should happen in the base class methods; if
    False, can invoke the functions separately in the desired place inside the
    derived class.
    """

    # Newly introduced class variables ........................................

    PLOT_HELPER_CLS: type = PlotHelper
    """Which :py:class:`~dantro.plot.plot_helper.PlotHelper` class to use"""

    # .........................................................................
    # Main API functions, required by PlotManager

    def __init__(
        self,
        name: str,
        *,
        style: dict = None,
        **parent_kwargs,
    ):
        """Initialize a creator for :py:mod:`matplotlib.pyplot`-based plots.

        Args:
            name (str): The name of this plot
            style (dict, optional): The *default* style context defintion to
                enter before calling the plot function. This can be used to
                specify the aesthetics of a plot. It is evaluated here once,
                stored as attribute, and can be updated when the plot method
                is actually called.
            **parent_kwargs: Passed to the parent's
                :py:meth:`~dantro.plot.creators.base.BasePlotCreator.__init__`.
        """
        super().__init__(name, **parent_kwargs)

        # Default style and RC parameters
        self._default_rc_params = None

        if style is not None:
            self._default_rc_params = self._prepare_style_context(**style)

    def plot(
        self,
        *,
        out_path: str,
        plot_func: Union[str, Callable],
        module: str = None,
        module_file: str = None,
        style: dict = None,
        helpers: dict = None,
        animation: dict = None,
        use_dag: bool = None,
        **func_kwargs,
    ):
        """Performs the plot operation by calling a specified plot function.

        The plot function is specified by its name, which is interpreted as a
        full module string, or by directly passing a callable.

        Alternatively, the base module can be loaded from a file path.

        Args:
            out_path (str): The output path for the resulting file
            plot_func (Union[str, Callable]): The plot function or a name or
                module string under which it can be imported.
            module (str, optional): If plot_func was the name of the plot
                function, this needs to be the name of the module to import
            module_file (str, optional): Path to the file to load and look for
                the ``plot_func`` in. If ``base_module_file_dir`` is given,
                this can also be a path relative to that directory.
            style (dict, optional): Parameters that determine the aesthetics of
                the created plot; basically matplotlib rcParams. From them, a
                style context is entered before calling the plot function.
                Valid keys:

                    base_style (str, List[str], optional):
                        names of valid matplotlib styles
                    rc_file (str, optional):
                        path to a YAML RC parameter file that is used to
                        update the base style
                    ignore_defaults (bool, optional):
                        Whether to ignore the default style passed to the
                        __init__ method
                    further keyword arguments:
                        will update the RC parameter dict yet again. Need be
                        valid matplotlib RC parameters in order to have any
                        effect.

            helpers (dict, optional): helper configuration passed to PlotHelper
                initialization if enabled
            animation (dict, optional): animation configuration
            use_dag (bool, optional): Whether to use the TransformationDAG to
                select and transform data that can be used in the plotting
                function. If not given, will query the plot function attributes
                for whether the DAG should be used.
            **func_kwargs: Passed to the imported function

        Raises:
            ValueError: On superfluous ``helpers`` or ``animation`` arguments
                in cases where these are not supported
        """
        # Get the plotting function
        plot_func = self._resolve_plot_func(
            plot_func=plot_func, module=module, module_file=module_file
        )

        # Generate a style dictionary to be used for context manager creation
        rc_params = self._prepare_style_context(**(style if style else {}))

        # Check if PlotHelper is to be used, defaulting to True for None.
        _use_helper = getattr(plot_func, "use_helper", False)
        if _use_helper is None:
            _use_helper = True

        if _use_helper:
            switch_anim_mode = False

            # Delegate to private helper method that performs the plot or the
            # animation. In case that animation mode is to be entered or
            # exited, adjust the animation-related parameters accordingly.
            try:
                self._plot_with_helper(
                    out_path=out_path,
                    plot_func=plot_func,
                    helpers=helpers,
                    style_context=self._build_style_context(**rc_params),
                    func_kwargs=func_kwargs,
                    use_dag=use_dag,
                    animation=animation,
                )

            except EnterAnimationMode:
                log.note("Entering animation mode ...")
                if not animation:
                    raise ValueError(
                        "Cannot dynamically enter animation mode without any "
                        "`animation` parameters having been specified in the "
                        f"configuration of the {self.classname} "
                        f"'{self.name}' plot!"
                    )

                switch_anim_mode = True
                animation = copy.deepcopy(animation)
                animation["enabled"] = True

            except ExitAnimationMode:
                log.note("Exiting animation mode ...")
                switch_anim_mode = True
                animation = None

            # else: animation was successful.

            # In case of the mode having switched, plot anew.
            if switch_anim_mode:
                log.debug("Plotting anew (with change in animation mode) ...")
                try:
                    self._plot_with_helper(
                        out_path=out_path,
                        plot_func=plot_func,
                        helpers=helpers,
                        style_context=self._build_style_context(**rc_params),
                        func_kwargs=func_kwargs,
                        use_dag=use_dag,
                        animation=animation,
                    )

                except (EnterAnimationMode, ExitAnimationMode):
                    raise RuntimeError(
                        "Cannot repeatedly enter or exit animation mode! Make "
                        f"sure that the plotting function of {self.logstr} "
                        "respects this requirement and that the plot "
                        "configuration you specified does not contradict "
                        "itself."
                    )

        else:
            # Call only the plot function
            # Do not allow helper or animation parameters
            if helpers:
                raise ValueError(
                    "The key 'helpers' was found in the configuration of "
                    f"plot '{self.name}' but usage of the PlotHelper is not "
                    f"supported by plot function '{plot_func.__name__}'!"
                )

            if animation:
                raise ValueError(
                    "The key 'animation' was found in the "
                    f"configuration of plot '{self.name}' but the animation "
                    "feature is only available when using the PlotHelper for "
                    f"plot function '{plot_func.__name__}'!"
                )

            # Prepare the arguments. The DataManager is added to args there
            # and data transformation via DAG occurs there as well.
            args, kwargs = self._prepare_plot_func_args(
                plot_func, use_dag=use_dag, out_path=out_path, **func_kwargs
            )

            # Enter the stlye context
            with self._build_style_context(**rc_params):
                log.debug(
                    "Calling plotting function '%s' ...", plot_func.__name__
                )
                plot_func(*args, **kwargs)
            # Done.

    # .........................................................................
    # Helpers: Main plot routines

    def _plot_with_helper(
        self,
        *,
        out_path: str,
        plot_func: Callable,
        helpers: dict,
        style_context,
        func_kwargs: dict,
        animation: dict,
        use_dag: bool,
    ):
        """A helper method that performs plotting using the
        :py:class:`~dantro.plot.plot_helper.PlotHelper`.

        Args:
            out_path (str): The output path
            plot_func (Callable): The resolved plot function
            helpers (dict): The helper configuration
            style_context: A style context; can also be DoNothingContext, if
                no style adjustments are to take place.
            func_kwargs (dict): Plot function arguments
            animation (dict): Animation parameters
            use_dag (bool): Whether a DAG is used in preprocessing or not
        """
        # Determine if animation is enabled, which is relevant for PlotHelper
        animation = copy.deepcopy(animation) if animation else {}
        animation_enabled = animation.pop("enabled", False)

        # Prepare the arguments. The DataManager is added to args there; if the
        # DAG is used, data transformation and placeholder resolution will
        # happen there as well.
        # In order to apply placeholder resolution to the helper configuration
        # as well, the helpers are passed along here (and popped from the
        # parsed kwargs again a few lines below).
        args, kwargs = self._prepare_plot_func_args(
            plot_func, use_dag=use_dag, helpers=helpers, **func_kwargs
        )

        # Initialize a PlotHelper instance that will take care of figure
        # setup, invoking helper-functions and saving the figure.
        # Then, add the Helper instance to the plot function keyword arguments.
        helpers = kwargs.pop("helpers")
        helper_defaults = getattr(plot_func, "helper_defaults", None)
        hlpr = self.PLOT_HELPER_CLS(
            out_path=out_path,
            helper_defaults=helper_defaults,
            update_helper_cfg=helpers,
            raise_on_error=self.raise_exc,
            animation_enabled=animation_enabled,
        )
        kwargs["hlpr"] = hlpr

        # Check if an animation is to be done; if so, delegate to helper method
        if animation_enabled:
            self._perform_animation(
                hlpr=hlpr,
                style_context=style_context,
                plot_func=plot_func,
                plot_args=args,
                plot_kwargs=kwargs,
                **animation,
            )
            return
        # else: No animation to be done.

        # Enter two context: one for style (could also be DoNothingContext)
        # and one for prevention of figures leaking from the plot function.
        leak_prev = figure_leak_prevention(close_current_fig_on_raise=True)

        with style_context, leak_prev:
            hlpr.setup_figure()

            plot_func_name = plot_func.__name__
            log.info("Now calling plotting function '%s' ...", plot_func_name)
            plot_func(*args, **kwargs)
            log.note("Plotting function '%s' returned.", plot_func_name)

            log.info("Invoking helpers ...")
            hlpr.invoke_enabled(axes="all")

            log.note("Saving figure ...")
            hlpr.save_figure()
            log.remark("Figure saved.")

    # .........................................................................
    # Helpers: specialization of data selection and transformation framework

    def _get_dag_params(
        self, *, _plot_func: Callable, **cfg
    ) -> Tuple[dict, dict]:
        """Extends the parent method by making the plot function callable
        available to the other helper methods and extracting some further
        information from the plot function.
        """
        dag_params, plot_kwargs = super()._get_dag_params(**cfg)

        # Store the plot function, such that it is available as argument in the
        # other subclassed helper methods
        dag_params["init"]["_plot_func"] = _plot_func
        dag_params["compute"]["_plot_func"] = _plot_func

        # Determine whether the DAG object should be passed along to the func
        pass_dag = getattr(_plot_func, "pass_dag_object_along", False)
        dag_params["pass_dag_object_along"] = pass_dag

        # Determine whether the DAG results should be unpacked when passing
        # them to the plot function
        unpack_results = getattr(_plot_func, "unpack_dag_results", False)
        dag_params["unpack_dag_results"] = unpack_results

        return dag_params, plot_kwargs

    def _use_dag(
        self, *, use_dag: bool, plot_kwargs: dict, _plot_func: Callable
    ) -> bool:
        """Whether the DAG should be used or not. This method extends that of
        the base class by additionally checking the plot function attributes
        for any information regarding the DAG
        """
        # If None was given, check the plot function attributes
        if use_dag is None:
            use_dag = getattr(_plot_func, "use_dag", None)

        # Let the parent class do whatever else it does
        use_dag = super()._use_dag(use_dag=use_dag, plot_kwargs=plot_kwargs)

        # Complain, if tags where required, but DAG usage was disabled
        if not use_dag and getattr(_plot_func, "required_dag_tags", None):
            raise ValueError(
                "The plot function {} requires DAG tags to be "
                "computed, but DAG usage was disabled."
                "".format(_plot_func)
            )

        return use_dag

    def _create_dag(
        self, *, _plot_func: Callable, **dag_params
    ) -> TransformationDAG:
        """Extends the parent method by allowing to pass the _plot_func, which
        can be used to adjust DAG behaviour ...
        """
        return super()._create_dag(**dag_params)

    def _compute_dag(
        self,
        dag: TransformationDAG,
        *,
        _plot_func: Callable,
        compute_only: Sequence[str],
        **compute_kwargs,
    ) -> dict:
        """Compute the dag results.

        This extends the parent method by additionally checking whether all
        required tags are defined and (after computation) whether all required
        tags were computed.
        """
        # Extract the required tags from the plot function attributes
        required_tags = getattr(_plot_func, "required_dag_tags", None)

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
                        _plot_func,
                        ", ".join(missing_tags),
                        ", ".join(dag.tags),
                    )
                )

        # If the compute_only argument was not explicitly given, determine
        # whether to compute only the required tags
        if (
            compute_only is None
            and required_tags is not None
            and getattr(_plot_func, "compute_only_required_dag_tags", False)
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
                        _plot_func,
                        ", ".join(missing_tags),
                        ", ".join(required_tags),
                        ", ".join(dag.tags),
                        ", ".join(compute_only),
                    )
                )

        return super()._compute_dag(
            dag, compute_only=compute_only, **compute_kwargs
        )

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

        .. note::

            This behaviour is different than in the parent class, where the
            DAG results are passed on as ``dag_results``.

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
            cfg = dict(data=dag_results, **plot_kwargs)

        # Add the `dag` kwarg, if configured to do so.
        if dag_params["pass_dag_object_along"]:
            cfg["dag"] = dag

        return cfg

    # .........................................................................
    # Helpers: Style and Animation

    def _prepare_style_context(
        self,
        *,
        base_style: Union[str, List[str]] = None,
        rc_file: str = None,
        ignore_defaults: bool = False,
        **update_rc_params,
    ) -> dict:
        """Builds a dictionary with rcparams for use in a matplotlib rc context

        Args:
            base_style (Union[str, List[str]], optional): The matplotlib
                style to use as a basis for the generated rc parameters dict.
            rc_file (str, optional): path to a YAML file containing rc
                parameters. These are used to update those of the base styles.
            ignore_defaults (bool, optional): Whether to ignore the rc
                parameters that were given to the __init__ method
            **update_rc_params: All further parameters update those that are
                already provided by base_style and/or rc_file arguments.

        Returns:
            dict: The rc parameters dictionary, a valid dict to enter a
                matplotlib style context with

        Raises:
            ValueError: On invalid arguments
        """
        # Determine what to base this
        if self._default_rc_params and not ignore_defaults:
            log.debug("Composing RC parameters based on defaults ...")
            rc_dict = self._default_rc_params

        else:
            log.debug("Composing RC parameters ...")
            rc_dict = dict()

        # Make sure base_style is a list of strings
        if not base_style:
            base_style = []

        elif isinstance(base_style, str):
            base_style = [base_style]

        elif not isinstance(base_style, (list, tuple)):
            raise TypeError(
                "Argument `base_style` need be None, a string, "
                f"or a list of strings, was of type {type(base_style)} with "
                f"value '{base_style}'!"
            )

        # Now, base_style definitely is an iterable.
        # Use it to initially populate the RC dict
        if base_style:
            log.debug("Using base styles: %s", ", ".join(base_style))

            # Iterate over it and populate the rc_dict
            for style_name in base_style:
                # If the base_style key is given, load a dictionary with the
                # corresponding rc_params
                if style_name not in plt.style.available:
                    _available = ", ".join(plt.style.available)
                    raise ValueError(
                        f"Style '{style_name}' is not a valid matplotlib "
                        f"style. Available styles: {_available}"
                    )

                rc_dict = recursive_update(
                    rc_dict, plt.style.library[style_name]
                )

        # If a `rc_file` is specifed update the `rc_dict`
        if rc_file:
            path_to_rc = os.path.expanduser(rc_file)

            if not os.path.isabs(path_to_rc):
                raise ValueError(
                    "Argument `rc_file` needs to be an absolute "
                    f"path, was not! Got: {path_to_rc}"
                )

            elif not os.path.exists(path_to_rc):
                raise ValueError(
                    f"No file was found at path {path_to_rc} specified by "
                    "argument `rc_file`!"
                )

            log.debug("Loading RC parameters from file %s ...", path_to_rc)
            rc_dict = recursive_update(rc_dict, load_yml(path_to_rc))

        # If any other rc_params are specified, update the `rc_dict` with them
        if update_rc_params:
            log.debug("Recursively updating RC parameters...")
            rc_dict = recursive_update(rc_dict, update_rc_params)

        return rc_dict

    def _build_style_context(self, **rc_params):
        """Constructs the matplotlib style context manager, if parameters were
        given, otherwise returns the DoNothingContext
        """
        import matplotlib.pyplot as plt  # FIXME Should not be necessary

        if rc_params:
            log.remark(
                "Using custom style context with %d entries ...",
                len(rc_params),
            )
            return plt.rc_context(rc=rc_params)
        return DoNothingContext()

    def _perform_animation(
        self,
        *,
        hlpr: PlotHelper,
        style_context,
        plot_func: Callable,
        plot_args: tuple,
        plot_kwargs: dict,
        writer: str,
        writer_kwargs: dict = None,
        animation_update_kwargs: dict = None,
    ):
        """Prepares the Writer and checks for valid animation config.

        Args:
            hlpr (PlotHelper): The plot helper
            style_context: The style context to enter before starting animation
            plot_func (Callable): plotting function which is to be animated
            plot_args (tuple): positional arguments to ``plot_func``
            plot_kwargs (dict): keyword arguments to ``plot_func``
            writer (str): name of movie writer with which the frames are saved
            writer_kwargs (dict, optional): A dict of writer parameters. These
                are associated with the chosen writer via the top level key
                in ``writer_kwargs``. Each dictionary container has three
                further keys queried, all optional:

                    init:
                        passed to ``Writer.__init__`` method
                    saving:
                        passed to ``Writer.saving`` method
                    grab_frame:
                        passed to ``Writer.grab_frame`` method

            animation_update_kwargs (dict, optional): Passed to the animation
                update generator call.

        Raises:
            ValueError: if the animation is not supported by the ``plot_func``
                or if the writer is not available
        """
        if not getattr(plot_func, "supports_animation", False):
            raise ValueError(
                f"Plotting function '{plot_func.__name__}' was not marked as "
                "supporting an animation! To do so, add the "
                "`supports_animation` flag to the plot function "
                "decorator."
            )

        # Get the kwargs for __init__, saving, and grab_frame of the writer
        writer_name = writer
        writer_cfg = (
            writer_kwargs.get(writer_name, {}) if writer_kwargs else {}
        )

        # Need to extract `dpi`, because matplotlib interface wants it as
        # positional argument. Damn you, matplotlib.
        dpi = writer_cfg.get("saving", {}).pop("dpi", 96)

        # Retrieve the writer; to trigger writer registration with matplotlib,
        # make sure that the movie writers module is actually imported
        from ..utils._file_writer import FileWriter

        if mpl.animation.writers.is_available(writer_name):
            wCls = mpl.animation.writers[writer_name]
            writer = wCls(**writer_cfg.get("init", {}))

        else:
            _available = ", ".join(mpl.animation.writers.list())
            raise ValueError(
                f"The writer '{writer_name}' is not available on your "
                f"system! Available writers: {_available}"
            )

        # Now got the writer.

        # Can enter the style context and perform animation now.
        # In order to not aggregate additional figures during this process,
        # also enter an additional context manager, which prevents figures
        # leaking from the plot function or the animation generator.
        leak_prev = figure_leak_prevention(close_current_fig_on_raise=True)

        log.debug(
            "Performing animation of plot function '%s' using writer %s ...",
            plot_func.__name__,
            writer_name,
        )
        frame_no = -1

        with style_context, leak_prev:
            hlpr.setup_figure()

            # Call the plot function
            plot_func_name = plot_func.__name__
            log.info("Now calling plotting function '%s' ...", plot_func_name)

            plot_func(*plot_args, **plot_kwargs)
            # NOTE This plot is NOT saved as the first frame in order to allow
            #      the animation update generator be a more general method.

            # If helpers are _not_ called later, call them now and be done.
            # While they would not need to be kept enabled, doing so causes no
            # harm and is the more expected behaviour.
            if not hlpr.invoke_before_grab:
                hlpr.invoke_enabled(axes="all", mark_disabled_after_use=False)

            # Enter context manager of movie writer
            with writer.saving(
                hlpr.fig, hlpr.out_path, dpi, **writer_cfg.get("saving", {})
            ):

                # Create the iterator for the animation
                log.info("Invoking animation update generator ...")
                anim_it = hlpr.animation_update(
                    **(
                        animation_update_kwargs
                        if animation_update_kwargs
                        else {}
                    )
                )

                # Create generator and perform the iteration. The return value
                # of the generator currently is ignored.
                log.debug("Iterating animation update generator ...")
                for frame_no, _ in enumerate(anim_it):
                    # Update the figure used in the writer
                    # This is required for cases in which each frame is given
                    # by a new figure.
                    if writer.fig is not hlpr.fig:
                        writer.fig = hlpr.fig

                    # If required, invoke all enabled helpers before grabbing
                    if hlpr.invoke_before_grab:
                        hlpr.invoke_enabled(
                            axes="all", mark_disabled_after_use=False
                        )

                    # The anim_it invocation has already created the new frame.
                    # Grab it; the writer takes care of saving it
                    writer.grab_frame(**writer_cfg.get("grab_frame", {}))
                    log.debug("Grabbed frame %d.", frame_no)

            # Exited 'saving' context
            # Make sure the figure is closed
            hlpr.close_figure()

        # Exited externally given style context and figure_leak_prevention.
        # Done now.
        log.note("Animation finished after %s frames.", frame_no + 1)
