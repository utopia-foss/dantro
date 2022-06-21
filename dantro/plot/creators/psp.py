"""Plot creators working on :py:class:`~dantro.groups.psp.ParamSpaceGroup`.
These are based on the :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator`
and provide additional functionality for data that is stored such a format.

See :ref:`pcr_psp` for more information.
"""

import copy
import logging
import time
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from paramspace import ParamDim, ParamSpace

from ..._import_tools import LazyLoader
from ...abc import PATH_JOIN_CHAR
from ...dag import DAGNode, DAGReference, DAGTag, TransformationDAG
from ...groups import ParamSpaceGroup, ParamSpaceStateGroup
from ...tools import is_iterable, recursive_update
from .base import SkipPlot
from .pyplot import PyPlotCreator

log = logging.getLogger(__name__)

xr = LazyLoader("xarray")


# -----------------------------------------------------------------------------


class MultiversePlotCreator(PyPlotCreator):
    """A MultiversePlotCreator is an PyPlotCreator that allows data to be
    selected before being passed to the plot function.
    """

    PSGRP_PATH: str = None
    """Where the :py:class:`~dantro.groups.psp.ParamSpaceGroup` object is
    expected within the :py:class:`~dantro.data_mngr.DataManager`"""

    # .........................................................................

    def __init__(self, *args, psgrp_path: str = None, **kwargs):
        """Initialize a MultiversePlotCreator

        Args:
            *args: Passed on to parent class.
            psgrp_path (str, optional): The path to the associated
                :py:class:`~dantro.groups.psp.ParamSpaceGroup` that is to
                be used for these multiverse plots.
            **kwargs: Passed on to parent
        """
        super().__init__(*args, **kwargs)

        if psgrp_path:
            self.PSGRP_PATH = psgrp_path

    @property
    def psgrp(self) -> ParamSpaceGroup:
        """Retrieves the parameter space group associated with this plot
        creator by looking up a certain path in the data manager.
        """
        if self.PSGRP_PATH is None:
            raise ValueError(
                "Missing class variable PSGRP_PATH! Either set "
                "it directly or pass the `psgrp_path` argument "
                "to the __init__ function."
            )

        # Retrieve the parameter space group
        return self.dm[self.PSGRP_PATH]

    def _check_skipping(self, *, plot_kwargs: dict):
        """Adds a skip condition for plots with this creator:

        Controlled by the ``expected_multiverse_ndim`` argument, this
        plot will be skipped if the dimensionality of the associated
        :py:class:`~dantro.groups.psp.ParamSpaceGroup` is *not* specified in
        the set of permissible dimensionalities.
        If that argument is not given or None, this check will not be carried
        out.
        """
        super()._check_skipping(plot_kwargs=plot_kwargs)

        # Extract the parameter value; popping intended and required here.
        ndim = plot_kwargs.pop("expected_multiverse_ndim", None)

        # Only need to continue if there are any requirements
        if ndim is None:
            return

        # Get the parameter space group's dimensionality
        mv_ndim = self.psgrp.pspace.num_dims

        # Make sure its a set of integers
        ndims = set(ndim) if is_iterable(ndim) else {ndim}
        if not all([isinstance(nd, int) for nd in ndims]):
            raise TypeError(
                "Expected sequence or set of integers for specifying required "
                f"multiverse dimensionality, but got: {repr(ndim)}"
            )

        if mv_ndim not in ndims:
            raise SkipPlot(
                f"{self.psgrp.logstr} dimensionality {mv_ndim} ∉ {ndims}."
            )

    def _prepare_plot_func_args(
        self,
        *args,
        select: dict = None,
        select_and_combine: dict = None,
        **kwargs,
    ) -> Tuple[tuple, dict]:
        """Prepares the arguments for the plot function.

        This also implements the functionality to select and combine data from
        the Multiverse and provide it to the plot function. It can do so via
        the associated :py:class:`~dantro.groups.psp.ParamSpaceGroup`
        directly or by creating a :py:class:`~dantro.dag.TransformationDAG`
        that leads to the same results.

        .. warning::

            The ``select_and_combine`` argument behaves slightly different to
            the ``select`` argument! In the long term, the ``select`` argument
            will be deprecated.

        Args:
            *args: Positional arguments to the plot function.
            select (dict, optional): If given, selects and combines multiverse
                data using
                :py:meth:`~dantro.groups.psp.ParamSpaceGroup.select`.
                The result is an ``xr.Dataset`` and it is made available to
                the plot function as ``mv_data`` argument.
            select_and_combine (dict, optional): If given, interfaces with the
                DAG to select, transform, and combine data from the multiverse
                via the DAG.
            **kwargs: Keyword arguments for the plot function. If DAG usage is
                enabled, these contain further arguments like ``transform``
                that are filtered out accordingly.

        Returns:
            Tuple[tuple, dict]: The (args, kwargs) tuple for calling the plot
                function. These now include either the DAG results or the
                additional ``mv_data`` key.

        Raises:
            TypeError: If both or neither of the arguments ``select`` and/or
                ``select_and_combine`` were given.
        """
        # Distinguish between the new DAG-based selection interface and the
        # old (and soon-to-be-deprecated) one.
        if select and not select_and_combine:
            # Select multiverse data via the ParamSpaceGroup
            kwargs["mv_data"] = self.psgrp.select(**select)

        elif select_and_combine:
            # Pass both arguments along
            kwargs["select"] = select
            kwargs["select_and_combine"] = select_and_combine

        else:
            raise TypeError(
                "Expected at least one of the arguments `select` "
                "or `select_and_combine`, got neither!"
            )

        # Let the parent method (from PyPlotCreator) do its thing.
        # It will invoke the specialized _get_dag_params and _create_dag helper
        # methods that are implemented by this class.
        return super()._prepare_plot_func_args(*args, **kwargs)

    # .........................................................................
    # DAG specialization

    def _get_dag_params(
        self, *, select_and_combine: dict, **cfg
    ) -> Tuple[dict, dict]:
        """Extends the parent method by extracting the select_and_combine
        argument that handles MultiversePlotCreator behaviour
        """
        dag_params, plot_kwargs = super()._get_dag_params(**cfg)

        # Add the select_and_combine argument; converting None to an empty dict
        select_and_combine = select_and_combine if select_and_combine else {}
        dag_params["init"]["select_and_combine"] = select_and_combine

        return dag_params, plot_kwargs

    def _create_dag(
        self,
        *,
        select_and_combine: dict,
        select: dict = None,
        transform: Sequence[dict] = None,
        select_base: str = None,
        select_path_prefix: str = None,
        **dag_init_params,
    ) -> TransformationDAG:
        """Extends the parent method by translating the ``select_and_combine``
        argument into selection of tags from a universe subspace, subsequent
        transformations, and a ``combine`` operation, that aligns the data in
        the desired fashion.

        This way, the :py:meth:`~dantro.groups.psp.ParamSpaceGroup.select`
        method's behaviour is emulated in the DAG.

        Args:
            select_and_combine (dict): The parameters to define which data from
                the universes to select and combine before applying further
                transformations.
            select (dict, optional): Additional select operations; these are
                *not* applied to *each* universe but only globally, after the
                ``select_and_combine`` nodes are added.
            transform (Sequence[dict], optional): Additional transform
                operations that are added to the DAG after both the
                ``select_and_combine``- and ``select``-related transformations
                were added.
            select_base (str, optional): The select base for the ``select``
                argument. These are *not* relevant for the selection that
                occurs via the ``select_and_combine`` argument and is only set
                after all ``select_and_combine``-related transformations are
                added to the DAG.
            select_path_prefix (str, optional): The selection path prefix for
                the ``select`` argument. Cannot be used here.
            **dag_init_params: Further initialization arguments to the DAG.

        Returns:
            TransformationDAG: The populated DAG object.
        """

        def add_uni_transformations(
            dag: TransformationDAG,
            *,
            uni_name: str,
            coords: dict,
            path: str,
            missing: List[str],
            allow_missing_or_failing: bool,
            transform: Sequence[dict] = None,
            **select_kwargs,
        ) -> DAGReference:
            """Adds the sequence of select and transform operations that is
            to be applied to the data from a *single* universe; this is in
            preparation to the combination of all single-universe DAG strands.
            The last transformation node that is added by this helper is the
            one that is used as input to the combination methods.

            The easiest way to add a sequence of transformations that is based
            on a selection from the DataManager is to use TransformationDAG's
            :py:meth:`~dantro.dag.TransformationDAG.add_nodes` method. To that
            end, this helper function creates the necessary parameters for the
            ``select`` argument to that method.

            .. note::

                To not crowd the tag space, tags are omitted on these transform
                operations, unless manually specified.
            """
            # Keep track of missing parameter space states
            if uni_name not in self.psgrp:
                missing.append(uni_name)

            # Create the full path that is needed to get from the selection
            # base (the ParamSpaceGroup) to the desired path within the
            # current universe and prepare arguments for the select operation
            field_path = PATH_JOIN_CHAR.join([uni_name, path])

            select = dict()
            select[uni_name] = dict(
                path=field_path,
                transform=transform,
                omit_tag=True,
                **select_kwargs,
            )
            # Add the nodes that handle the selection and optional transform
            # operations on the selected data. This is all user-determined.
            # The selection base is the DataManager.
            dag.add_nodes(select=select)

            # Prepare coordinates for expanding dimensions
            _coords = {k: [v] for k, v in coords.items()}

            # Set allow_failure only on the last node of this branch, such
            # that all other nodes may still have their separate fallbacks;
            # the fallback used here is only relevant if everything else failed
            # irrecoverably. By using an empty xr.Dataset, the combination via
            # xr.merge will succeed and propagate the coordinates onward, but
            # have null data.
            extra_kwargs = dict()
            if allow_missing_or_failing:
                extra_kwargs["allow_failure"] = allow_missing_or_failing
                extra_kwargs["fallback"] = xr.Dataset(coords=_coords)

            # With the latest-added transformation as input, add the parameter
            # space coordinates to it such that all single universes can be
            # aligned properly. Best not to cache this result.
            return dag.add_node(
                operation="dantro.expand_dims",
                args=[DAGNode(-1)],
                kwargs=dict(dim={k: [v] for k, v in coords.items()}),
                file_cache=dict(read=False, write=False),
                **extra_kwargs,
            )

        def add_transformations(
            dag: TransformationDAG,
            *,
            path: str,
            tag: str,
            subspace: dict,
            combination_method: str,
            allow_missing_or_failing: bool,
            combination_kwargs: dict = None,
            transform_after_combine: List[dict] = None,
            **select_kwargs,
        ) -> None:
            """Adds the sequence of transformations that is necessary to select
            data from a single universe and transform it, i.e.: all the preps
            necessary to arrive at another input argument to the combiation.
            """

            # Get the parameter space object
            psp = copy.deepcopy(self.psgrp.pspace)

            # Apply the subspace mask, if given
            if subspace:
                psp.activate_subspace(**subspace)

            # Prepare iterators and extract shape information
            psp_it = psp.iterator(
                with_info=("state_no_str", "current_coords"), omit_pt=True
            )

            # For each universe in the subspace, add a sequence of transform
            # operations that lead to the data being selected and (optionally)
            # further transformed. Keep track of the reference to the last
            # node of each branch, which is a node that assigns coordinates to
            # each point of the parameter space.
            # Also, keep track of missing universes and generate a warning
            # message that should help circumventing the problem.
            missing = []
            refs = [
                add_uni_transformations(
                    dag,
                    uni_name=state_no_str,
                    coords=coords,
                    path=path,
                    missing=missing,
                    allow_missing_or_failing=allow_missing_or_failing,
                    **select_kwargs,
                )
                for state_no_str, coords in psp_it
            ]

            # Handle missing universes and behavior upon transformation failure
            if missing and not allow_missing_or_failing:
                log.caution(
                    "The following %d parameter space states are missing from "
                    "%s: %s\nThis will probably lead to an error during "
                    "computation of the data transformation results.\n"
                    "Consider using the `select_and_combine.subspace` "
                    "argument for field '%s'; alternatively, use the "
                    "`allow_missing_or_failing` option.",
                    len(missing),
                    self.psgrp.logstr,
                    ", ".join(missing),
                    tag,
                )

            if allow_missing_or_failing and combination_method != "merge":
                log.caution(
                    "With `allow_missing_or_failing` set for field '%s', "
                    "combination method '%s' is incompatible! "
                    "Using 'merge' instead.",
                    tag,
                    combination_method,
                )
                combination_method = "merge"

            # Depending on the chosen combination method, create corresponding
            # additional transformations for combination via merge or via
            # concatenation.
            if combination_method == "merge":
                dag.add_node(
                    operation="dantro.merge",
                    args=[refs],
                    kwargs=dict(reduce_to_array=True),
                )

            elif combination_method == "concat":
                # For concatenation, it's best to have the data in an ndarray
                # of xr.DataArray's, such that sequential applications of the
                # xr.concat method along the array axes can be used for
                # combining the data.
                dag.add_node(
                    operation="populate_ndarray",
                    args=[refs],
                    kwargs=dict(shape=psp.shape, dtype="object"),
                )

                dag.add_node(
                    operation="dantro.multi_concat",
                    args=[DAGNode(-1)],
                    kwargs=dict(dims=list(psp.dims.keys())),
                )

            elif isinstance(combination_method, dict):
                op = combination_method.pop("operation")
                pass_pspace = combination_method.pop("pass_pspace", False)
                log.remark("Using custom combination operation:  '%s'", op)
                dag.add_node(
                    operation=op,
                    args=[refs],
                    kwargs=dict(
                        **combination_method,
                        **(dict(pspace=psp) if pass_pspace else {}),
                    ),
                )

            else:
                raise ValueError(
                    f"Invalid combination method '{combination_method}'! "
                    "Available methods: 'merge', 'concat', or a custom "
                    "combination method (passing a dict with key `operation`)."
                )

            # Now have the data combined into an xr.DataArray
            # Might want to add more transformations here
            if transform_after_combine:
                dag.add_nodes(transform=transform_after_combine)

            # Finally, attach the tag and pass combination kwargs
            dag.add_node(
                operation="pass",
                args=[DAGNode(-1)],
                tag=tag,
                **(combination_kwargs if combination_kwargs else {}),
            )

        def add_sac_transformations(
            dag: TransformationDAG,
            *,
            fields: dict,
            subspace: dict = None,
            combination_method: str = "concat",
            allow_missing_or_failing: bool = None,
            transform_after_combine: List[dict] = None,
            base_path: str = None,
        ) -> None:
            """Adds transformations to the given DAG that select data from the
            selected multiverse subspace.

            Args:
                dag (TransformationDAG): The DAG to add nodes to that represent
                    the select-and-combine operations.
                fields (dict): Which fields to select from the separate
                    universes.
                subspace (dict, optional): The (default) subspace to select the
                    data from.
                combination_method (str, optional): The (default) combination
                    method of the multidimensional data.
                allow_missing_or_failing (bool, optional): If set, will use an
                    automatic fallback for missing data or failure during
                    transformations.
                    This may be overwritten by each field's separate option.
                transform_after_combine (List[dict], optional):
                    Transformations that are applied to each field's output
                    *after* combination.
                    This may be overwritten by each field's separate option.
                base_path (str, optional): If given, ``path`` specifications
                    of each field can be seen as relative to this path.
            """
            # To make selections shorter, add a transformation to get to the
            # ParamSpaceGroup and set that as the selection base.
            dag.select_base = dag.add_node(
                operation="getitem", args=[DAGTag("dm"), self.PSGRP_PATH]
            )

            # For all tags, update the default values with custom arguments
            # and then add transformations
            for tag, spec in fields.items():
                # For safety, work on a copy
                spec = copy.deepcopy(spec)

                # The field might be given in short (path-only) syntax
                if not isinstance(spec, dict):
                    spec = dict(path=spec)

                # If a base path was given, prepend it. This is the path that
                # is selected *within* each universe.
                if base_path is not None:
                    spec["path"] = PATH_JOIN_CHAR.join(
                        [base_path, spec["path"]]
                    )

                # Parse parameters, i.e.: Use defaults defined on this level
                # if the spec does not provide more specific information.
                spec["subspace"] = spec.get(
                    "subspace", copy.deepcopy(subspace)
                )
                spec["combination_method"] = spec.get(
                    "combination_method", copy.deepcopy(combination_method)
                )
                spec["allow_missing_or_failing"] = spec.get(
                    "allow_missing_or_failing", allow_missing_or_failing
                )
                spec["transform_after_combine"] = spec.get(
                    "transform_after_combine",
                    copy.deepcopy(transform_after_combine),
                )

                # Add the transformations for this specific tag
                add_transformations(dag, tag=tag, **spec)

            # Done. :)
            log.remark(
                "Added select-and-combine transformations for tags:  %s",
                ", ".join(fields.keys()),
            )
            # NOTE Resetting the selection base is not necessary here, because
            #      the user-specified value is set directly after this function
            #      returns.

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # To not create further confusion regarding base paths, raise an error
        # if it is attempted to set the `select_path_prefix` argument.
        if select_path_prefix:
            raise ValueError(
                "The select_path_prefix argument cannot be used "
                f"within {self.logstr}! Use the select_and_combine.base_path "
                "argument instead."
            )

        # Initialize an (empty) DAG, i.e.: without select and transform args
        # and without setting the selection base
        dag = super()._create_dag(**dag_init_params)

        # Add nodes that perform the "select and combine" operations, based on
        # selections from the DataManager
        add_sac_transformations(dag, **select_and_combine)

        # Can now set the selection base to the user-intended value and add
        # the other user-specified transformations
        dag.select_base = select_base
        dag.add_nodes(select=select, transform=transform)

        return dag


# -----------------------------------------------------------------------------


class UniversePlotCreator(PyPlotCreator):
    """A UniversePlotCreator is an PyPlotCreator that allows looping of
    all or a selected subspace of universes.
    """

    PSGRP_PATH: str = None
    """Where the :py:class:`~dantro.groups.psp.ParamSpaceGroup` object is
    expected within the :py:class:`~dantro.data_mngr.DataManager`"""

    # .........................................................................

    def __init__(self, *args, psgrp_path: str = None, **kwargs):
        """Initialize a UniversePlotCreator

        Args:
            *args: Passed on to parent class
            psgrp_path (str, optional): Specifies the location of the
                :py:class:`~dantro.groups.psp.ParamSpaceGroup` within the
                data tree. If given, overwrites the class variable default.
            **kwargs: Passed on to parent class
        """
        super().__init__(*args, **kwargs)

        if psgrp_path:
            self.PSGRP_PATH = psgrp_path

        # Add custom attributes
        self._without_pspace = False

        # Cache attributes
        self._psp = None
        self._psp_active_smap_cache = None

    @property
    def psgrp(self) -> ParamSpaceGroup:
        """Retrieves the parameter space group associated with this plot
        creator by looking up a certain path in the data manager.
        """
        if self.PSGRP_PATH is None:
            raise ValueError(
                "Missing class variable PSGRP_PATH! Either set "
                "it directly or pass the `psgrp_path` argument "
                "to the __init__ function."
            )

        # Retrieve the parameter space group
        return self.dm[self.PSGRP_PATH]

    def prepare_cfg(
        self, *, plot_cfg: dict, pspace: Union[dict, ParamSpace]
    ) -> Tuple[dict, ParamSpace]:
        """Converts a regular plot configuration to one that can be configured
        to iterate over multiple universes via a parameter space.

        This is implemented in the following way:

            1. Extracts the ``universes`` key from the configuration and parses
               it, ensuring it is a valid dict for subspace specification
            2. Creates a new ParamSpace object that additionally contains the
               parameter dimensions corresponding to the universes. These are
               stored in a _coords dict inside the returned plot configuration.
            3. Apply the parsed ``universes`` key to activate a subspace of the
               newly created parameter space.
            4. As a mapping from coordinates to state numbers is needed, the
               corresponding active state mapping is saved as an attribute to
               the plot creator, such that it is available later when
               the state number needs to be retrieved only be the info of the
               current coordinates.
        """
        # If a pspace was given, need to extract the dict from there, because
        # the steps below will lead to additional paramspace dimensions
        if pspace is not None:
            if isinstance(pspace, ParamSpace):
                # FIXME internal API usage
                pspace = copy.deepcopy(pspace._dict)

            plot_cfg = recursive_update(pspace, plot_cfg)

        # Now have the plot config
        # Identify those keys that specify which universes to loop over
        try:
            unis = plot_cfg.pop("universes")

        except KeyError as err:
            raise ValueError(
                "Missing required keyword-argument `universes` "
                "in plot configuration!"
            ) from err

        # Get the parameter space, as it might be needed for certain values of
        # the `universes` argument
        self._psp = copy.deepcopy(self.psgrp.pspace)

        # -- Case 1: No parameter space available in the first place
        # Only default point is available, which should be handled differently
        if self._psp.num_dims == 0 or self.psgrp.only_default_data_present:
            if unis not in ("all", "single", "first", "random", "any"):
                raise ValueError(
                    "Could not select a universe for plotting because the "
                    "associated parameter space has no dimensions available "
                    "or only data for the default point was available in "
                    f"{self.psgrp.logstr}. For these cases, the only valid "
                    "values for the `universes` argument are: "
                    "'all', 'single', 'first', 'random', or 'any'."
                )

            # Set a flag to carry information to _prepare_plot_func_args
            self._without_pspace = True

            # Distinguish cases where plot_cfg was given and those were not
            if pspace is not None:
                # There was a recursive update step; return the plot config
                # as parameter space
                return dict(), ParamSpace(plot_cfg)

            # else: Only need to return the plot configuration
            return plot_cfg, None

        # -- Case 2: Explicitly given universe names
        if isinstance(unis, (list, tuple)):
            if any([not isinstance(n, int) for n in unis]):
                raise TypeError(
                    "Got at least one non-integer value in universe ID list!\n"
                    "When supplying a list or tuple to the `universes` "
                    "argument, each element needs to be an integer value "
                    "denoting the universe IDs to create plots for. Make "
                    f"sure that this is the case for the given list:\n  {unis}"
                )

            plot_cfg["_uni_id"] = ParamDim(
                default=0, values=unis, name="uni_id", order=-np.inf
            )

            # Convert plot config into "multi plot config", including the
            # information of the universe IDs to use for plotting; this info
            # is extracted in `_prepare_plot_func_args` and used for selection.
            return {}, ParamSpace(plot_cfg)
            # NOTE Don't need the state map in this approach, so there's no
            #      point in caching it and/or calling `activate_subspace`, as
            #      needs to be done in the approach below.

        # -- Case 3: Subspace selector
        # Parse it such that it is a valid subspace selector
        if isinstance(unis, str):
            if unis in ("all",):
                # is equivalent to an empty specifier -> empty dict
                unis = dict()

            elif unis in ("single", "first", "random", "any"):
                # Find the state number from the universes available in the
                # parameter space group. Then retrieve the coordinates from
                # the corresponding parameter space state map

                # Create a list of available universe IDs
                uni_ids = [int(_id) for _id in self.psgrp.keys()]

                # Select the first or a random ID
                if unis in ["single", "first"]:
                    uni_id = min(uni_ids)
                else:
                    uni_id = np.random.choice(uni_ids)

                # Now retrieve the point from the (full) state map
                smap = self._psp.state_map
                point = smap.where(smap == uni_id, drop=True)
                # NOTE Universe IDs are unique, so that's ok.

                # And find its coordinates
                unis = {k: c.item() for k, c in point.coords.items()}

            else:
                raise ValueError(
                    "Invalid value for `universes` argument. Got "
                    f"'{unis}', but expected one of: 'all', 'single', "
                    "'first', 'random', or 'any'."
                )

        elif not isinstance(unis, dict):
            raise TypeError(
                "Need parameter `universes` to be either a list of universe "
                "state numbers, a string or a dictionary of subspace "
                f"selectors, but got: {type(unis)} {unis}."
            )

        # else: was a dict, can be used as a subspace selector

        # Ensure that no invalid dimension names were selected
        for pdim_name in unis.keys():
            if pdim_name not in self._psp.dims.keys():
                _dim_names = ", ".join([n for n in self._psp.dims])
                raise ValueError(
                    f"No parameter dimension '{pdim_name}' was available "
                    "in the parameter space associated with "
                    f"{self.psgrp.logstr}! Available parameter "
                    f"dimensions: {_dim_names}"
                )

        # Copy parameter dimension objects for each coordinate
        # As the parameter space with the coordinates has a different
        # hierarchy than psp, the ordering has to be manually adjusted to be
        # the same as in the original psp.
        coords = dict()
        for dim_num, (name, pdim) in enumerate(self._psp.dims.items()):
            # Need to use a copy, as it will need to be changed
            _pdim = copy.deepcopy(pdim)

            # Make sure the name is the same as in the parameter space
            _pdim._name = name
            # FIXME internal API usage

            # Adjust the order to put them in front in the parameter space, but
            # keep the ordering the same as in `psp`.
            # NOTE The actual _order attribute does no longer play a role, as
            #      the psp.dims are already sorted according to those values.
            #      We just need to generate an offset to those parameter dims
            #      that might be defined in the plot configuration ...
            _pdim._order = -100000000 + dim_num
            # FIXME internal API usage

            # Now store it in the dict
            coords[name] = _pdim

        # Add these as a new key to the plot configuration
        if "_coords" in plot_cfg:
            raise ValueError(
                "The given plot configuration may _not_ contain "
                "the key '_coords' on the top-most level!"
            )

        plot_cfg["_coords"] = coords

        # Convert the whole dict to a parameter space, the "multi plot config"
        mpc = ParamSpace(plot_cfg)

        # Activate only a certain subspace of the multi-plot configuration;
        # this determines which values will be iterated over.
        mpc.activate_subspace(**unis)

        # Now, also need the regular parameter space (i.e. without additional
        # plot configuration coordinates) to use in _prepare_plot_func_args.
        # Need to apply the universe selection to that as well
        self._psp.activate_subspace(**unis)
        self._psp_active_smap_cache = self._psp.active_state_map

        # Only return the configuration as a parameter space; all is included
        # in there now.
        return {}, mpc

    def _prepare_plot_func_args(
        self, *args, _coords: dict = None, _uni_id: int = None, **kwargs
    ) -> Tuple[tuple, dict]:
        """Prepares the arguments for the plot function and implements the
        special arguments required for ParamSpaceGroup-like data: selection of
        a single universe from the given coordinates.

        Args:
            *args: Passed along to parent method
            _coords (dict, optional): The current coordinate descriptor which
                is then used to retrieve a certain point in parameter space
                from the state map attribute.
            _uni_id (int, optional): If given, use this ID to select a universe
                from the ParamSpaceGroup (and ignore the ``_coords`` argument)
            **kwargs: Passed along to parent method

        Returns:
            tuple: (args, kwargs) for the plot function
        """
        # Need to distinguish between cases with or without pspace given. The
        # aim is to retrieve a Universe ID to use for selection.
        if self._without_pspace:
            # Only the default universe is available, always having ID 0.
            uni_id = 0

        elif _uni_id is not None:
            # This is a parameter sweep over explicitly given IDs.
            uni_id = _uni_id

        else:
            # This is a parameter sweep over coordinate space.
            # Given the coordinates, retrieve the data for a single universe
            # from the state map. As _coords is created by the _prepare_cfg
            # method, it will unambiguously selects a universe ID.
            uni_id = int(self._psp_active_smap_cache.sel(_coords))

        # Select the corresponding universe from the ParamSpaceGroup
        uni = self.psgrp[uni_id]
        log.note("Using data of:        %s", uni.logstr)

        # Let the parent function, implemented in PyPlotCreator, do its
        # thing. This will return the (args, kwargs) tuple and will also take
        # care of data transformation using the DAG framework, for which some
        # behaviour is specialized for selection from the passed `uni` using
        # additional helper methods; see below.
        return super()._prepare_plot_func_args(*args, uni=uni, **kwargs)

    def _get_dag_params(
        self, *, uni: ParamSpaceStateGroup, **cfg
    ) -> Tuple[dict, dict]:
        """Makes the selected universe available and adjusts DAG parameters
        such that selections can be based on that universe.
        """
        dag_params, plot_kwargs = super()._get_dag_params(**cfg)

        # Extend the DAG parameters such that they perform a base_transform
        # to get to the selected universe and subsequently use that as the
        # selection base.
        uni_path = PATH_JOIN_CHAR.join([self.PSGRP_PATH, uni.name])
        base_transform = [
            dict(
                getitem=[DAGTag("dm"), uni_path],
                tag="uni",
                file_cache=dict(read=False, write=False),
            )
        ]
        dag_params["init"] = dict(
            **dag_params["init"],
            base_transform=base_transform,
            select_base="uni",
        )

        return dag_params, plot_kwargs
