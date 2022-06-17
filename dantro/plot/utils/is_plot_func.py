"""Implements a decorator to declare a function as a plot function"""

import logging
import os
from typing import Callable, Sequence, Union

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class is_plot_func:
    """This is a decorator class declaring the decorated function as a
    plotting function to use with
    :py:class:`~dantro.plot.creators.base.BasePlotCreator` or derived creators.

    It has a set of specializations that make sense only when using a specific
    creator type. For example, the ``helper``-related arguments are only used
    by :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`.
    """

    def __init__(
        self,
        *,
        creator_type: type = None,
        creator_name: str = None,
        use_dag: bool = None,
        required_dag_tags: Sequence[str] = None,
        compute_only_required_dag_tags: bool = True,
        pass_dag_object_along: bool = False,
        unpack_dag_results: bool = False,
        use_helper: bool = True,
        helper_defaults: Union[dict, str] = None,
        supports_animation=False,
        add_attributes: dict = None,
    ):
        """Initialize the decorator. Note that the function to be decorated is
        not passed to this method.

        .. note::

            Some arguments are only evaluated when using a certain creator
            type, e.g. :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`.

        Args:
            creator_type (type, optional): The type of plot creator to use
            creator_name (str, optional): The name of the plot creator to use
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

        # Gather those attributes that are to be set as function attributes
        self.pf_attrs = dict(
            is_plot_func=True,
            creator_type=creator_type,
            creator_name=creator_name,
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
