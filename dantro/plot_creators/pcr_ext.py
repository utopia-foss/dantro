"""This module implements the ExternalPlotCreator class"""

import os
import logging
import importlib
import importlib.util
import inspect
from collections import defaultdict
from typing import Callable, Union, List

from .pcr_base import BasePlotCreator


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
                 **parent_kwargs):
        """Initialize an ExternalPlotCreator.
        
        Args:
            name (str): The name of this plot
            base_module_file_dir (str, optional): If given, `module_file`
                arguments to the `_plot` method that are relative paths will
                be seen relative to this directory
            **parent_kwargs: Passed to the parent __init__
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

    def plot(self, *, out_path: str, plot_func: Union[str, Callable],
             module: str=None, module_file: str=None, **func_kwargs):
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
            **func_kwargs: Passed to the imported function
        """
        # Get the plotting function
        plot_func = self._resolve_plot_func(plot_func=plot_func,
                                            module=module,
                                            module_file=module_file)
        
        # Now have the plotting function
        # Prepare the arguments (the data manager is added to args there)
        args, kwargs = self._prepare_plot_func_args(out_path=out_path,
                                                    **func_kwargs)

        # Call it
        log.debug("Calling plotting function '%s'...", plot_func.__name__)

        plot_func(*args, **kwargs)

        log.debug("Plotting function returned.")

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
            if hasattr(pf, "creator_name") and pf.creator_name == creator_name:
                log.debug("The plot function's desired creator name '%s' "
                          "matches the name under which %s is known to the "
                          "PlotManager.", creator_name, self.classname)
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

    def __init__(self, *, creator_name: str=None):
        """Initialize the decorator. Note that the function to be decorated is
        not passed to this method.
        
        Args:
            creator_name (str, optional): The name of the plot creator to use
        """
        self.creator_name = creator_name

    def __call__(self, func: Callable):
        """If there are decorator arguments, __call__() is only called
        once, as part of the decoration process and expects as only argument
        the function to be decorated.
        """
        # Do not actually wrap the function, but add attributes to it
        func.creator_name = self.creator_name

        # Return the function, now with attributes set
        return func
