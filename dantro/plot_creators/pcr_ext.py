"""This module implements the ExternalPlotCreator class"""

import os
import logging
import importlib
import importlib.util
import inspect
from collections import defaultdict
from typing import Callable, Union, List

import matplotlib.pyplot as plt

from ..tools import load_yml, recursive_update, DoNothingContext
from .pcr_base import BasePlotCreator
from .pcr_ext_modules.plot_helper import PlotHelper, PlotHelperWarning


# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class ExternalPlotCreator(BasePlotCreator):
    """This PlotCreator uses external scripts to create plots."""

    # Settings of functionality implemented in parent classes
    EXTENSIONS = 'all'  # no checks performed
    DEFAULT_EXT = None
    DEFAULT_EXT_REQUIRED = False

    # For relative module imports, see the following as the base package
    BASE_PKG = "dantro.plot_creators.ext_funcs"

    # Configuration for the PlotManager's auto-detection feature ..............

    # Whether to ignore function attributes (e.g. `creator_name`) when
    # deciding whether a plot function is to be used with this creator
    _AD_IGNORE_FUNC_ATTRS = False

    # The parameters below are for inspecting the plot function signature. See
    # https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
    # for more information on the specification for argument kinds.

    # Exactly how many POSITIONAL_ONLY arguments to allow; -1 to not check
    _AD_NUM_POSITIONAL_ONLY = -1

    # Exactly how many POSITIONAL_OR_KEYWORD arguments to allow; -1: no check
    _AD_NUM_POSITIONAL_OR_KEYWORD = 1
    
    # Whether to allow *args
    _AD_ALLOW_VAR_POSITIONAL = True

    # The KEYWORD_ONLY arguments that are required to be (explicitly!) accepted
    _AD_KEYWORD_ONLY = ['out_path']

    # Whether to allow **kwargs
    _AD_ALLOW_VAR_KEYWORD = True


    # .........................................................................
    # Main API functions, required by PlotManager

    def __init__(self, name: str, *, base_module_file_dir: str=None,
                 style: dict=None, **parent_kwargs):
        """Initialize an ExternalPlotCreator.
        
        Args:
            name (str): The name of this plot
            base_module_file_dir (str, optional): If given, `module_file`
                arguments to the `_plot` method that are relative paths will
                be seen relative to this directory
            style (dict, optional): The default style context defintion to
                enter before calling the plot function. This can be used to
                specify the aesthetics of a plot. It is evaluated here once,
                stored as attribute, and can be updated when the plot method
                is called.
            **parent_kwargs: Passed to the parent __init__
        
        Raises:
            ValueError: On invalid `base_module_file_dir` argument
        """
        super().__init__(name, **parent_kwargs)

        # If given, check the base module file dir argument is valid
        if base_module_file_dir:
            bmfd = os.path.expanduser(base_module_file_dir)
            
            if not os.path.isabs(bmfd):
                raise ValueError("Argument `base_module_file_dir` needs to be "
                                 "an absolute path, was not! Got: "+str(bmfd))
            
            elif not os.path.exists(bmfd) or not os.path.isdir(bmfd):
                raise ValueError("Argument `base_module_file_dir` does not "
                                 "exists or does not point to a directory!")

        self.base_module_file_dir = base_module_file_dir

        # Parse default RC parameters
        self._default_rc_params = None

        if style is not None:
            self._default_rc_params = self._prepare_style_context(**style)
            

    def plot(self, *, out_path: str, plot_func: Union[str, Callable],
             module: str=None, module_file: str=None, style: dict=None,
             helpers: dict=None, **func_kwargs):
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
                the `plot_func` in. If `base_module_file_dir` is given, this
                can also be a path relative to that directory.
            style (dict, optional): Parameters that determine the aesthetics of
                the created plot; basically matplotlib rcParams. From them, a
                style context is entered before calling the plot function.
                Valid keys:
                    - `base_style` (str, List[str], optional) names of valid
                        matplotlib styles
                    - `rc_file` (str, optional) path to a YAML RC parameter
                        file that is used to update the base style
                    - `ignore_defaults` (bool, optional) Whether to ignore the
                        default style passed to the __init__ method
                    - further parameters will update the RC parameter dict yet
                        again. Need be valid matplotlib RC parameters in order
                        to have any effect.
            helpers (dict, optional): helper configuration passed to PlotHelper
                initialization if enabled
            **func_kwargs: Passed to the imported function
        """
        # Get the plotting function
        plot_func = self._resolve_plot_func(plot_func=plot_func,
                                            module=module,
                                            module_file=module_file)
        
        # Now have the plotting function
        # Generate a style dictionary
        rc_params = self._prepare_style_context(**(style if style else {}))

        # Get style context
        if rc_params:
            log.debug("Using custom style context ...")
            context = plt.rc_context(rc=rc_params)
        else:
            context = DoNothingContext()

        # Check if PlotHelper is to be used
        if getattr(plot_func, "use_helper", False):
            # Initialize a PlotHelper instance that will take care of figure
            # setup, invoking helper-functions and saving the figure
            hlpr = PlotHelper(out_path=out_path,
                              enabled_helpers_defaults=plot_func.enabled_helpers,
                              helper_defaults=plot_func.helper_defaults,
                              **(helpers if helpers else {}))
            
            # Prepare the arguments (the data manager is added to args there)
            args, kwargs = self._prepare_plot_func_args(hlpr=hlpr,
                                                        **func_kwargs)
            # NOTE out_path not needed here; PlotHelper takes care of that

            # Enter the context (either a style context or DoNothingContext)
            with context:
                hlpr.setup_figure()
                log.debug("Calling plotting function '%s' ...",
                          plot_func.__name__)
                plot_func(*args, **kwargs)
                hlpr.invoke_all()
                hlpr.save_figure()

        else:
            # Call only the plot function
            # Warn, if helper arguments were still passed ...
            if helpers:
                raise PlotHelperWarning("The key 'helpers' was found in the "
                                        "configuration of plot '{}' but usage "
                                        "of the PlotHelper is not supported "
                                        "by plot function '{}'! Arguments "
                                        "will be ignored."
                                        "".format(self.name,
                                                  plot_func.__name__))

            # Prepare the arguments (the data manager is added to args there)
            args, kwargs = self._prepare_plot_func_args(out_path=out_path,
                                                        **func_kwargs)

            # Enter the context (either a style context or DoNothingContext)
            with context:
                log.debug("Calling plotting function '%s' ...",
                          plot_func.__name__)
                plot_func(*args, **kwargs)

    def can_plot(self, creator_name: str, **cfg) -> bool:
        """Whether this plot creator is able to make a plot for the given plot
        configuration.
        
        This checks whether the configuration allows resolving a plot function.
        If that is the case, it checks whether it is 
        
        Args:
            creator_name (str): The name for this creator used within the
                PlotManager.
            **cfg: The plot configuration with which to decide this ...
        
        Returns:
            bool: Whether this creator can be used for plotting or not
        """
        log.debug("Checking if %s can plot the given configuration ...",
                  self.logstr)

        # Gather the arguments needed for plot function resolution and remove
        # those that are None
        pf_kwargs = dict(plot_func=cfg.get('plot_func'),
                         module=cfg.get('module'),
                         module_file=cfg.get('module_file'))
        pf_kwargs = {k: v for k, v in pf_kwargs.items() if v is not None}

        # Try to resolve the function
        try:
            pf = self._resolve_plot_func(**pf_kwargs)

        except:
            log.debug("Cannot plot this configuration, because a plotting "
                      "function could not be resolved with the given "
                      "arguments: %s", pf_kwargs)
            return False

        # else: was able to resolve a plotting function

        # The function might have an attribute that specifies the name of the
        # creator to use
        if not self._AD_IGNORE_FUNC_ATTRS:
            if self._declared_plot_func_by_attrs(pf, creator_name):
                return True

        # Check that function's signature and decide accordingly
        return self._valid_plot_func_signature(inspect.signature(pf))

    # .........................................................................
    # Helpers, used internally

    def _resolve_plot_func(self, *,
                           plot_func: Union[str, Callable], module: str=None,
                           module_file: str=None) -> Callable:
        """
        Args:
            plot_func (Union[str, Callable]): The plot function or a name or
                module string under which it can be imported.
            module (str): If plot_func was the name of the plot
                function, this needs to be the name of the module to import
            module_file (str): Path to the file to load and look for
                the `plot_func` in. If `base_module_file_dir` is given, this
                can also be a path relative to that directory.
        
        Returns:
            Callable: The resolved plot function
        
        Raises:
            TypeError: Upon wrong argument types
        """
        if callable(plot_func):
            log.debug("Received plotting function:  %s", str(plot_func))
            return plot_func
        
        elif not isinstance(plot_func, str):
            raise TypeError("Argument `plot_func` needs to be a string or a "
                            "callable, was {} with value '{}'."
                            "".format(type(plot_func), plot_func))
        
        # else: need to resolve the module and find the plot_func in it
        # First resolve the module
        if module_file:
            # Get it from a file
            mod = self._get_module_from_file(module_file)

        elif isinstance(module, str):
            # Import module via importlib, allowing relative imports
            # from the dantro.plot_funcs subpackage
            mod = importlib.import_module(module, package=self.BASE_PKG)
        
        else:
            raise TypeError("Could not import a module, because neither "
                            "argument `module_file` was given nor did "
                            "argument `module` have the correct type "
                            "(needs to be string but was {} with value "
                            "'{}')."
                            "".format(type(module), module))

        # plot_func could be something like "A.B.C.d", so go along modules
        # recursively to get to "C"
        attr_names = plot_func.split(".")
        for attr_name in attr_names[:-1]:
            mod = getattr(mod, attr_name)

        # This is now the last module. Get the actual function
        plot_func = getattr(mod, attr_names[-1])

        log.debug("Resolved plotting function:  %s", str(plot_func))

        return plot_func

    def _get_module_from_file(self, path: str):
        """Returns the module corresponding to the file at the given `path`"""
        # Pre-processing
        path = os.path.expanduser(path)

        # Make it absolute
        if not os.path.isabs(path):
            if not self.base_module_file_dir:
                raise ValueError("Need to specify `base_module_file_dir` "
                                 "during initialization to use relative paths "
                                 "for `module_file` argument!")

            path = os.path.join(self.base_module_file_dir, path)

        # Extract a name from the path to use as module name
        mod_name = "from_file." + os.path.basename(path).split(".")[0]
        # NOTE The name does not really matter

        # Is an absolute path now
        # Create a module specification and, from that, import the module
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Now, it is loaded
        return mod

    def _prepare_plot_func_args(self, *args, **kwargs) -> tuple:
        """Prepares the args and kwargs passed to the plot function.
        
        The passed args and kwargs are carried over, while the positional
        arguments are prepended with passing of the data manager.

        When subclassing this function, the parent method (this one) should
        still be called to maintain base functionality.
        
        Args:
            *args: Additional args
            **kwargs: Additional kwargs
        
        Returns:
            tuple: (args: tuple, kwargs: dict)
        """
        return ((self.dm,) + args, kwargs)

    def _prepare_style_context(self, *, base_style: Union[str, List[str]]=None,
                               rc_file: str=None, ignore_defaults: bool=False,
                               **update_rc_params) -> dict:
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
            raise TypeError("Argument `base_style` need be None, a string, "
                            "or a list of strings, was of type {} with "
                            "value '{}'!".format(type(base_style), base_style))

        # Now, base_style definitely is an iterable.
        # Use it to initially populate the RC dict
        if base_style:
            log.debug("Using base styles: %s", ", ".join(base_style))

            # Iterate over it and populate the rc_dict
            for style_name in base_style:
                # If the base_style key is given, load a dictionary with the
                # corresponding rc_params
                if style_name not in plt.style.available:
                    raise ValueError("Style '{}' is not a valid matplotlib "
                                     "style. Available styles: {}"
                                     "".format(style_name,
                                               ", ".join(plt.style.available)))

                rc_dict = recursive_update(rc_dict,
                                           plt.style.library[style_name])
        
        # If a `rc_file` is specifed update the `rc_dict`
        if rc_file:
            path_to_rc = os.path.expanduser(rc_file)

            if not os.path.isabs(path_to_rc):
                raise ValueError("Argument `rc_file` needs to be an absolute "
                                 "path, was not! Got: {}".format(path_to_rc))

            elif not os.path.exists(path_to_rc):
                raise ValueError("No file was found at path {} specified by "
                                 "argument `rc_file`!".format(path_to_rc))

            log.debug("Loading RC parameters from file %s ...", path_to_rc)
            rc_dict = recursive_update(rc_dict, load_yml(path_to_rc))

        # If any other rc_params are specified, update the `rc_dict` with them
        if update_rc_params:
            log.debug("Recursively updating RC parameters...")
            rc_dict = recursive_update(rc_dict, update_rc_params)

        return rc_dict

    def _declared_plot_func_by_attrs(self, pf: Callable,
                                     creator_name: str) -> bool:
        """Checks whether the given function has attributes set that declare
        it as a plotting function that is to be used with this creator.
        
        Args:
            pf (Callable): The plot function to check attributes of
            creator_name (str): The name under which this creator type is
                registered to the PlotManager.
        
        Returns:
            bool: Whether the plot function attributes declare the given plot
                function as suitable for working with this specific creator.
        """
        
        if hasattr(pf, 'creator_type') and pf.creator_type is not None:
            if isinstance(self, pf.creator_type):
                log.debug("The desired type of the plot function, %s, is the "
                          "same or a parent type of this %s.",
                          pf.creator_type, self.logstr)
                return True

        if hasattr(pf, 'creator_name') and pf.creator_name == creator_name:
            log.debug("The plot function's desired creator name '%s' "
                      "matches the name under which %s is known to the "
                      "PlotManager.", pf.creator_name, self.classname)
            return True

        log.debug("Checked plot function attributes, but neither the type "
                  "nor the creator name were specified or matched this "
                  "creator.")
        return False

    def _valid_plot_func_signature(self, sig: inspect.Signature,
                                   raise_if_invalid: bool=False) -> bool:
        """Determines whether the plot function signature is valid
        
        Args:
            sig (inspect.Signature): The inspected signature of the plot func
            raise_if_invalid (bool, optional): Whether to raise an error when
                the signature is not valid
        
        Returns:
            bool: Whether the signature is a valid plot function signature
        """
        def p2s(*params) -> str:
            """Given some parameters, returns a comma-joined string of their
            names"""
            return ", ".join([p.name for p in params])

        # Shortcut for the inspect.Parameter class, to access the kinds
        Param = inspect.Parameter

        log.debug("Inspecting plot function signature: %s", sig)

        # Aggregate parameters by their kind
        pbk = defaultdict(list)
        for p in sig.parameters.values():
            pbk[p.kind].append(p)

        # List of error strings
        errs = []

        # Check the number of positional arguments is as expected
        if self._AD_NUM_POSITIONAL_ONLY >= 0:
            if len(pbk[Param.POSITIONAL_ONLY]) != self._AD_NUM_POSITIONAL_ONLY:
                errs.append("Expected {} POSITIONAL_ONLY argument(s) but the "
                            "plot function allowed {}: {}. Change the plot "
                            "function signature to only take the expected "
                            "number of positional-only arguments."
                            "".format(self._AD_NUM_POSITIONAL_ONLY,
                                      len(pbk[Param.POSITIONAL_ONLY]),
                                      p2s(*pbk[Param.POSITIONAL_ONLY])))

        # Check the number of "positional or keyword" arguments is as expected
        if self._AD_NUM_POSITIONAL_OR_KEYWORD >= 0:
            if (   len(pbk[Param.POSITIONAL_OR_KEYWORD])
                != self._AD_NUM_POSITIONAL_OR_KEYWORD):
                errs.append("Expected {} POSITIONAL_OR_KEYWORD argument(s) "
                            "but the plot function allowed {}: {}. Make sure "
                            "the `*` is set to specify the beginning of the "
                            "keyword-only arguments section of the signature."
                            "".format(self._AD_NUM_POSITIONAL_OR_KEYWORD,
                                      len(pbk[Param.POSITIONAL_OR_KEYWORD]),
                                      p2s(*pbk[Param.POSITIONAL_OR_KEYWORD])))

        # Check that variable *args and **kwargs are as expected
        if not self._AD_ALLOW_VAR_POSITIONAL:
            if len(pbk[Param.VAR_POSITIONAL]) > 0:
                errs.append("VAR_POSITIONAL arguments are not allowed, but "
                            "the plot function gathers them via argument *{}!"
                            "".format(pbk[Param.VAR_POSITIONAL][0]))
        
        if not self._AD_ALLOW_VAR_KEYWORD:
            if len(pbk[Param.VAR_KEYWORD]) > 0:
                errs.append("VAR_KEYWORD arguments are not allowed, but "
                            "the plot function gathers them via argument **{}!"
                            "".format(pbk[Param.VAR_KEYWORD][0]))

        # Check that the required keyword-only arguments are available
        if not all([p in sig.parameters for p in self._AD_KEYWORD_ONLY]):
            errs.append("Did not find all of the expected KEYWORD_ONLY "
                        "arguments ({}) in the plot function!"
                        "".format(", ".join(self._AD_KEYWORD_ONLY)))

        # Decide how to continue
        is_valid = not errs
        log.debug("Signature is %s", "valid." if is_valid else "NOT valid!")

        if raise_if_invalid and not is_valid:
            # Not valid and configured to raise
            raise ValueError("The given plot function with signature '{}' is "
                             "not valid! The following issues were identified "
                             "by inspecting its signature:\n  - {}\n"
                             "".format(sig, "\n  - ".join(errs)))
        elif not is_valid:
            log.debug("Issues with the plot function's signature:\n  - %s",
                      "\n  - ".join(errs))

        return is_valid

# -----------------------------------------------------------------------------

class is_plot_func:
    """This is a decorator class declaring the decorated function as a
    plotting function to use with ExternalPlotCreator-derived plot creators
    """

    def __init__(self, *, creator_type: type=None, creator_name: str=None,
                 use_helper: bool=True, enabled_helpers: list=None,
                 helper_defaults: Union[dict, str]=None,
                 **additional_attributes):
        """Initialize the decorator. Note that the function to be decorated is
        not passed to this method.
        
        Args:
            creator_type (type, optional): The type of plot creator to use
            creator_name (str, optional): The name of the plot creator to use
            use_helper (bool, optional): Whether to use a PlotHelper
            enabled_helpers (list, optional): Key strings for helpers that
                are enabled by default
            helper_defaults (Union[dict, str], optional): Default
                configurations for helpers in enabled_helpers
            **additional_attributes: Additional attributes to add to the
                plot function
        """
        if isinstance(helper_defaults, str):
            # Interpret as path to yaml file
            log.debug("Loading helper defaults from file %s ...",
                      helper_defaults)
            helper_defaults = load_yml(helper_defaults)

        self.pf_attrs = dict(creator_type=creator_type,
                             creator_name=creator_name,
                             use_helper=use_helper,
                             enabled_helpers=enabled_helpers,
                             helper_defaults=helper_defaults,
                             **additional_attributes)

    def __call__(self, func: Callable):
        """If there are decorator arguments, __call__() is only called
        once, as part of the decoration process and expects as only argument
        the function to be decorated.
        """
        # Do not actually wrap the function, but add attributes to it
        for k, v in self.pf_attrs.items():
            setattr(func, k, v)

        # Return the function, now with attributes set
        return func
