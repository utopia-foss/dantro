"""This module implements the :py:class:`.PyPlotCreator` class, which
specializes on creating :py:mod:`matplotlib.pyplot`-based plots."""

import copy
import logging
import os
from typing import Callable, List, Sequence, Tuple, Union

from ..._import_tools import LazyLoader
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

    DAG_INVOKE_IN_BASE = True
    """Whether DAG invocation should happen in the base class method
    :py:meth:`~dantro.plot.creators.base.BasePlotCreator._prepare_plot_func_args`.
    If False, can/need to invoke the data selection separately in the desired
    place inside the derived class.
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
        style: dict = None,
        helpers: dict = None,
        animation: dict = None,
        use_dag: bool = None,
        **func_kwargs,
    ):
        """Performs the plot operation.

        In addition to the behavior of the base class's
        :py:meth:`~dantro.plot.creators.base.BasePlotCreator.plot`, this method
        integrates the :ref:`plot helper framework <pcr_pyplot_helper>`,
        :ref:`style contexts <pcr_pyplot_style>` and the
        :ref:`animation mode <pcr_pyplot_animations>`.

        Alternatively, the base module can be loaded from a file path.

        Args:
            out_path (str): The output path for the resulting file
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
            use_dag (bool, optional): Whether to use the :ref:`dag_framework`
                to select and transform data that can be used in the plotting
                function. If not given, will query the plot function attributes
                for whether the DAG should be used.
                See :ref:`plot_creator_dag` for more information.
            **func_kwargs: Passed to the imported function

        Raises:
            ValueError: On superfluous ``helpers`` or ``animation`` arguments
                in cases where these are not supported
        """
        # Store the output path, needed by methods
        self._out_path = out_path

        # Generate a style dictionary to be used for context manager creation
        rc_params = self._prepare_style_context(**(style if style else {}))

        # Check if PlotHelper is to be used, defaulting to True for None.
        _use_helper = getattr(self.plot_func, "use_helper", False)
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
                    f"supported by plot function '{self.plot_func_name}'!"
                )

            if animation:
                raise ValueError(
                    "The key 'animation' was found in the "
                    f"configuration of plot '{self.name}' but the animation "
                    "feature is only available when using the PlotHelper for "
                    f"plot function '{self.plot_func_name}'!"
                )

            # Prepare the arguments. The DataManager is added to args there
            # and data transformation via DAG occurs there as well.
            args, kwargs = self._prepare_plot_func_args(
                use_dag=use_dag, out_path=out_path, **func_kwargs
            )

            # Enter the style context and plot
            with self._build_style_context(**rc_params):
                self._invoke_plot_func(*args, **kwargs)

    # .........................................................................
    # Plotting with the PlotHelper

    def _plot_with_helper(
        self,
        *,
        out_path: str,
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
            use_dag=use_dag, helpers=helpers, **func_kwargs
        )

        # Initialize a PlotHelper instance that will take care of figure
        # setup, invoking helper-functions and saving the figure.
        # Then, add the Helper instance to the plot function keyword arguments.
        helpers = kwargs.pop("helpers")
        helper_defaults = getattr(self.plot_func, "helper_defaults", None)
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

            self._invoke_plot_func(*args, **kwargs)

            log.info("Invoking helpers ...")
            hlpr.invoke_enabled(axes="all")

            log.note("Saving figure ...")
            hlpr.save_figure()
            log.remark("Figure saved.")

    # .........................................................................
    # Style

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

    # .........................................................................
    # Animation

    def _perform_animation(
        self,
        *,
        hlpr: PlotHelper,
        style_context,
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
        if not getattr(self.plot_func, "supports_animation", False):
            raise ValueError(
                f"Plotting function '{self.plot_func_name}' was not "
                "marked as supporting an animation! To do so, add the "
                "`supports_animation` flag to the plot function decorator."
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
            self.plot_func_name,
            writer_name,
        )
        frame_no = -1

        with style_context, leak_prev:
            hlpr.setup_figure()

            # Call the plot function
            self._invoke_plot_func(*plot_args, **plot_kwargs)
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
