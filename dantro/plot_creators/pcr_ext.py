"""This module implements the ExternalPlotCreator class"""

import os
import logging
import importlib
import importlib.util
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

    # .........................................................................

    def __init__(self, name: str, *, base_module_file_dir: str=None, **parent_kwargs):
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


    def _plot(self, *, out_path: str, plot_func: Union[str, Callable], module: str=None, module_file: str=None, **func_kwargs):
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
        # Call it
        log.info("Calling plotting function '%s'...", plot_func.__name__)

        plot_func(self.dm, out_path=out_path, **func_kwargs)

        log.info("Plotting function returned.")

    def _resolve_plot_func(self, *, plot_func: Union[str, Callable], module: str=None, module_file: str=None) -> Callable:
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
