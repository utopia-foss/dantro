"""Implements utilities that revolve around the plotting function which is then
invoked by the :ref:`plot creators <plot_creators>`:

- a decorator to declare a function as a plot function
- the tools to resolve a plotting function from a module or file
"""

import importlib
import importlib.util
import logging
import os
import warnings
from typing import Callable, Sequence, Union

from ..._import_tools import (
    import_module_from_file as _import_module_from_file,
)

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class PlotFuncResolver:
    """Takes care of resolving a plot function"""

    BASE_PKG: str = "dantro.plot.funcs"
    """The default module string to use for relative module imports, where this
    module becomes the base package. Evaluated in :py:meth:`.__init__`.
    """

    def __init__(
        self, *, base_module_file_dir: str = None, base_pkg: str = None
    ):
        """Set up the plot function resolver.

        Args:
            base_module_file_dir (str, optional): If given, ``module_file``
                arguments to :py:meth:`.resolve` that are relative paths will
                be seen relative to this directory. Needs to be an absolute
                directory path and supports ``~`` expansion.
            base_pkg (str, optional): If given, use this base package instead
                for relative module imports instead of :py:attr:`.BASE_PKG`.

        Raises:
            ValueError: If ``base_module_file_dir`` was not absolute
            FileNotFoundError: If ``base_module_file_dir`` is missing or not a
                directory.
        """
        if base_module_file_dir:
            bmfd = os.path.expanduser(base_module_file_dir)

            if not os.path.isabs(bmfd):
                raise ValueError(
                    "Argument `base_module_file_dir` needs to be "
                    f"an absolute path, was not! Got: {bmfd}"
                )

            elif not os.path.exists(bmfd) or not os.path.isdir(bmfd):
                raise FileNotFoundError(
                    "Argument `base_module_file_dir` does not "
                    "exists or does not point to a directory!"
                )

        self.base_module_file_dir = base_module_file_dir
        self.base_pkg = base_pkg if base_pkg else self.BASE_PKG

    def resolve(
        self,
        *,
        plot_func: Union[str, Callable],
        module: str = None,
        module_file: str = None,
    ) -> Callable:
        """Resolve and return the plot function callable

        Args:
            plot_func (Union[str, Callable]): The name or module string of the
                plot function as it can be imported from ``module``. If this is
                a callable will directly return that callable.
            module (str): If ``plot_func`` was the name of the plot
                function, this needs to be the name of the module to import
                that name from.
            module_file (str): Path to the file to load and look for
                the ``plot_func`` in. If ``base_module_file_dir`` is given
                during initialization, this can also be a path relative to that
                directory.

        Returns:
            Callable: The resolved plot function

        Raises:
            TypeError: On bad argument types
        """
        if callable(plot_func):
            log.debug("Received plotting function:  %s", str(plot_func))
            return self._attach_attributes(plot_func)

        elif not isinstance(plot_func, str):
            raise TypeError(
                "Argument `plot_func` needs to be a string or a "
                f"callable, was {type(plot_func)} with value '{plot_func}'."
            )

        # else: need to resolve the module and find the callable in it
        # For less confusing variable names, do some renaming
        plot_func_modstr = plot_func
        del plot_func

        # First resolve the module, either from file or via import
        if module_file:
            mod = self._get_module_from_file(
                module_file, base_module_file_dir=self.base_module_file_dir
            )

        elif isinstance(module, str):
            mod = self._get_module_via_import(
                module=module, base_pkg=self.base_pkg
            )

        else:
            raise TypeError(
                "Could not import a module, because neither argument "
                "`module_file` was given nor did argument `module` have the "
                f"correct type (needs to be string but was {type(module)} "
                f"with value '{module}')."
            )

        # plot_func could be something like "A.B.C.d"; go along the segments to
        # allow for more versatile plot function retrieval
        attr_names = plot_func_modstr.split(".")
        for attr_name in attr_names[:-1]:
            mod = getattr(mod, attr_name)

        # This is now the last module. Get the actual function
        plot_func = getattr(mod, attr_names[-1])

        log.debug("Resolved plotting function:  %s", str(plot_func))

        # Decorate with some attributes, then return the result
        return self._attach_attributes(
            plot_func, module=mod, plot_func_modstr=plot_func_modstr
        )

    def _get_module_from_file(self, path: str, *, base_module_file_dir: str):
        """Returns the module corresponding to the file at the given ``path``.

        This uses :py:func:`~dantro._import_tools.import_module_from_file`
        to carry out the import.
        """
        try:
            return _import_module_from_file(
                path, base_dir=base_module_file_dir
            )
        except ValueError as err:
            raise ValueError(
                "Need to specify `base_module_file_dir` during initialization "
                "to use relative paths for `module_file` argument!"
            ) from err

    def _get_module_via_import(self, *, module: str, base_pkg: str):
        """Returns the module via import.

        Imports ``module`` via importlib, allowing relative imports from the
        package defined as base package.
        """
        return importlib.import_module(module, package=base_pkg)

    def _attach_attributes(
        self,
        plot_func: Callable,
        /,
        *,
        module=None,
        plot_func_modstr: str = None,
    ) -> Callable:
        """Attaches some informational attributes to the plot function."""
        plot_func.modstr = plot_func_modstr
        # TODO attach a `name` here, while the information is available
        return plot_func


# -----------------------------------------------------------------------------


class is_plot_func:
    """This is a decorator class declaring the decorated function as a
    plotting function to use with
    :py:class:`~dantro.plot.creators.base.BasePlotCreator` or derived creators.

    .. note::

        This decorator has a set of specializations that make sense only when
        using a specific creator type!
        For example, the ``helper``-related arguments are only used by
        :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` and are ignored
        without warning otherwise.
    """

    def __init__(
        self,
        *,
        creator: str = None,
        creator_type: type = None,
        creator_name: str = None,
        use_dag: bool = None,
        required_dag_tags: Sequence[str] = None,
        compute_only_required_dag_tags: bool = True,
        pass_dag_object_along: bool = False,
        unpack_dag_results: bool = False,
        use_helper: bool = None,
        helper_defaults: Union[dict, str] = None,
        supports_animation=False,
        add_attributes: dict = None,
    ):
        """Initialize the decorator.

        .. note::

            Some arguments are only evaluated when using a certain creator
            type, e.g. :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`.

        Args:
            creator (str, optional): The creator to use; needs to be registered
                with the PlotManager under this name.
            creator_type (type, optional): The type of plot creator to use.
                This argument is DEPRECATED, use ``creator`` instead.
            creator_name (str, optional): The name of the plot creator to use.
                This argument is DEPRECATED, use ``creator`` instead.
            use_dag (bool, optional): Whether to use the data transformation
                framework.
            required_dag_tags (Sequence[str], optional): The DAG tags that are
                required by the plot function.
            compute_only_required_dag_tags (bool, optional): Whether to compute
                only those DAG tags that are specified as required by the plot
                function. This is ignored if no required DAG tags were given
                and can be overwritten by the ``compute_only`` argument.
            pass_dag_object_along (bool, optional): Whether to pass on the DAG
                object to the plot function
            unpack_dag_results (bool, optional): Whether to unpack the results
                of the DAG computation directly into the plot function instead
                of passing it as a dictionary.
            use_helper (bool, optional): Whether to use the
                :py:class:`~dantro.plot.plot_helper.PlotHelper` with this plot.
                Needs :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`.
                If None, will default to True for supported creators and False
                otherwise.
            helper_defaults (Union[dict, str], optional): Default
                configurations for helpers; these are automatically considered
                to be enabled. If not dict-like, will assume this is an
                absolute path (supporting ``~`` expansion) to a YAML file and
                will load the dict-like configuration from there.
                Needs :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`.
            supports_animation (bool, optional): Whether the plot function
                supports animation.
                Needs :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`.
            add_attributes (dict, optional): Additional attributes to add to
                the plot function.

        Raises:
            ValueError: If ``helper_defaults`` was a string but not an absolute
                path.
        """
        from ..._yaml import load_yml

        if helper_defaults and not isinstance(helper_defaults, dict):
            # Interpret as absolute path to yaml file
            fpath = os.path.expanduser(helper_defaults)
            if not os.path.isabs(fpath):
                raise ValueError(
                    "`helper_defaults` string argument was a "
                    f"relative path: {fpath}, but needs to be either a "
                    "dict or an absolute path (~ allowed)."
                )

            log.debug("Loading helper defaults from file %s ...", fpath)
            helper_defaults = load_yml(fpath)

        # Evaluate the creator argument
        if creator:
            if creator_name or creator_type:
                raise ValueError(
                    "Cannot pass both `creator` and `creator_name` or "
                    "`creator_type`!"
                )

        else:
            if creator_name and creator_type:
                raise ValueError(
                    "Cannot pass both of the deprecated decorator arguments "
                    "`creator_name` and `creator_type`. Use `creator` instead."
                )

            _warn_msg = (
                "The `{}` argument is deprecated! Use `creator` instead."
            )
            if creator_name:
                warnings.warn(
                    _warn_msg.format("creator_name"), DeprecationWarning
                )
                creator = creator_name

            elif creator_type:
                warnings.warn(
                    _warn_msg.format("creator_type"), DeprecationWarning
                )
                creator = creator_type

        # Gather those attributes that are to be set as function attributes
        self.pf_attrs = dict(
            is_plot_func=True,
            creator=creator,
            use_dag=use_dag,
            required_dag_tags=required_dag_tags,
            compute_only_required_dag_tags=compute_only_required_dag_tags,
            pass_dag_object_along=pass_dag_object_along,
            use_helper=use_helper,
            helper_defaults=helper_defaults,
            supports_animation=supports_animation,
            **(add_attributes if add_attributes else {}),
        )

    def __call__(self, func: Callable):
        """If there are decorator arguments, __call__() is only called
        once, as part of the decoration process and expects as only argument
        the function to be decorated.
        """
        # Do not actually wrap the function call, but add attributes to it
        for k, v in self.pf_attrs.items():
            setattr(func, k, v)

        # Return the function, now with attributes set
        return func
