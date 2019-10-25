"""This implements the ParamSpaceGroup plot creators, based on the
ExternalPlotCreator and providing additional functionality for data that is
stored in a ParamSpaceGroup.
"""

import copy
import logging
from typing import Union, Tuple, Callable, Sequence

import numpy as np
import xarray as xr

from paramspace import ParamSpace

from .pcr_ext import ExternalPlotCreator
from ..groups import ParamSpaceGroup, ParamSpaceStateGroup
from ..tools import recursive_update
from ..abc import PATH_JOIN_CHAR
from ..dag import TransformationDAG, DAGReference, DAGTag, DAGNode

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class MultiversePlotCreator(ExternalPlotCreator):
    """A MultiversePlotCreator is an ExternalPlotCreator that allows data to be
    selected before being passed to the plot function.
    """
    # Where the ``ParamSpaceGroup`` object is expected within the data manager
    PSGRP_PATH = None

    # Configure the auto-detection feature implemented in ExternalPlotCreator:
    # The KEYWORD_ONLY arguments that are required to be (explicitly!) accepted
    _AD_KEYWORD_ONLY = ['out_path', 'mv_data']

    # .........................................................................

    def __init__(self, *args, psgrp_path: str=None, **kwargs):
        """Initialize a MultiversePlotCreator"""
        super().__init__(*args, **kwargs)

        if psgrp_path:
            self.PSGRP_PATH = psgrp_path

    @property
    def psgrp(self) -> ParamSpaceGroup:
        """Retrieves the parameter space group associated with this plot
        creator by looking up a certain path in the data manager.
        """
        if self.PSGRP_PATH is None:
            raise ValueError("Missing class variable PSGRP_PATH! Either set "
                             "it directly or pass the `psgrp_path` argument "
                             "to the __init__ function.")

        # Retrieve the parameter space group
        return self.dm[self.PSGRP_PATH]

    def _prepare_plot_func_args(self, *args,
                                select: dict=None,
                                select_and_combine: dict=None,
                                **kwargs) -> Tuple[tuple, dict]:
        """Prepares the arguments for the plot function.
        
        This also implements the functionality to select and combine data from
        the Multiverse and provide it to the plot function. It can do so via
        the associated :py:class:`~dantro.groups.ParamSpaceGroup` directly or
        by creating a :py:class:`~dantro.dag.TransformationDAG` that leads to
        the same results.
        
        .. warning::
        
            The ``select_and_combine`` argument behaves slightly different to
            the ``select`` argument! In the long term, the ``select`` argument
            will be deprecated.
        
        Args:
            *args: Positional arguments to the plot function.
            select (dict, optional): If given, selects and combines multiverse
                data using :py:meth:`~dantro.groups.ParamSpaceGroup.select`.
                The result is an ``xr.Dataset`` and it is made available to
                the plot function as ``mv_data`` argument.
            select_and_combine (dict, optional): Interfaces with the DAG to
                select, transform, and combine data from the multiverse via
                the DAG.
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
        # TODO make this more elegant
        if select and not select_and_combine:
            # Select multiverse data via the ParamSpaceGroup
            kwargs['mv_data'] = self.psgrp.select(**select)

        elif select_and_combine and not select:
            # Pass the select_and_combine argument along
            kwargs['select_and_combine'] = select_and_combine

        else:
            raise TypeError("Expected only one of the arguments `select` and "
                            "`select_and_combine`, got both or neither!")

        # Let the parent method (from ExternalPlotCreator) do its thing.
        # It will invoke the specialized _get_dag_params and _create_dag helper
        # methods that are implemented by this class.
        return super()._prepare_plot_func_args(*args, **kwargs)

    # .........................................................................
    # DAG specialization

    def _get_dag_params(self, *, select_and_combine: dict=None,
                        **cfg) -> Tuple[dict, dict]:
        """Extends the parent method by..."""
        dag_params, plot_kwargs = super()._get_dag_params(**cfg)

        # Additionally, store the `select_and_combine` argument
        dag_params['init']['select_and_combine'] = select_and_combine

        return dag_params, plot_kwargs

    def _create_dag(self, *, _plot_func: Callable,
                    select_and_combine: dict,
                    select: dict=None, transform: Sequence[dict]=None,
                    select_base: str=None, **dag_init_params
                    ) -> TransformationDAG:
        """Extends the parent method by translating the ``select_and_combine``
        argument into selection of tags from a universe subspace, subsequent
        transformations, and a ``combine`` operation, that aligns the data in
        the desired fashion.

        This way, the :py:meth:`~dantro.groups.ParamSpaceGroup.select` method's
        behaviour is emulated in the DAG.
        """
        def add_single_uni_transformations(dag: TransformationDAG, *,
                                           uni: ParamSpaceStateGroup,
                                           coords: dict, path: str,
                                           transform: Sequence[dict]=None,
                                           **select_kwargs
                                           ) -> DAGReference:
            """Adds the sequence of select and transform operations that is
            to be applied to the data from a single universe; this is in
            preparation to the combination of all single-universe DAG strands.
            The last transformation node that is added by this helper is the
            one that combines
            
            The easiest way to add a sequence of transformations that is based
            on a selection from the DataManager is to use TransformationDAG's
            add_nodes method. To that end, this helper function creates the
            necessary parameters for the ``select`` argument to that method.
            """
            # Create the full path that is needed to get from the DataManager
            # to the path within the current universe.
            uni_path = PATH_JOIN_CHAR.join([self.PSGRP_PATH, uni.name, path])
            
            # Prepare arguments for the select operation
            select = dict()
            select[uni.name] = dict(path=uni_path, transform=transform,
                                    omit_tag=True, **select_kwargs)
            
            # Add the nodes that handle the selection and optional transform
            # operations on the selected data. This is all user-determined.
            dag.add_nodes(select=select)

            # With the latest-added transformation as input, add the parameter
            # space coordinates to it such that all single universes can be
            # aligned properly. Don't cache this result.
            return dag.add_node(operation='dantro.expand_dims',
                                args=[DAGNode(-1)], kwargs=dict(dim=coords),
                                file_cache=dict(read=False, write=False))

        def add_transformations(dag: TransformationDAG, *,
                                tag: str, path: str, subspace: dict,
                                transform: Sequence[dict]=None,
                                combination_method: str,
                                combination_kwargs: dict=None,
                                **select_kwargs):
            """Adds the sequence of transformations that is necessary to select
            data from a single universe and transform it, i.e.: all the preps
            necessary to arrive at another input argument to the combiation.
            """
            # Get the parameter space object
            psp = copy.deepcopy(self.psgrp.pspace)

            # Apply the subspace mask, if given
            if subspace:
                psp.activate_subspace(**subspace)

            # Prepare multi-dimensional array for gathering DAGReferences
            refs = np.zeros(psp.shape, dtype='object')
            refs.fill(dict())  # these are ignored in xr.merge

            # Prepare iterators
            psp_it = psp.iterator(with_info=('state_no', 'current_coords'),
                                  omit_pt=True)
            arr_it = np.nditer(refs, flags=('multi_index', 'refs_ok'))

            # For each universe in the subspace, add a sequence of transform
            # operations that lead to the data being selected and (optionally)
            # further transformed. Keep track of those by aggregating them
            # into an ndarray.
            for (state_no, coords), _ in zip(psp_it, arr_it):
                # Get the universe, i.e. a ParamSpaceStateGroup
                uni = self.psgrp[state_no]
                # TODO error handling

                # Add the transformations for this single universe and keep
                # track of the reference to the last node of that branch, which
                # is a node that creates an xr.DataArray with the coordinates
                # within the parameter space assigned to it.
                ref = add_single_uni_transformations(dag, uni=uni,
                                                     coords=coords, path=path,
                                                     transform=transform,
                                                     **select_kwargs)

                # Keep track of the reference objects
                refs[arr_it.multi_index] = ref

            # Add the object to the argument buffer, which returns a reference
            # to the object in the TransformationDAG's object database.
            arg_ref = dag.add_object(refs, name="{}-refs".format(tag),
                                     dependencies=list(refs.flat))
            ref_buf = dag.add_node(operation='dag.get_object',
                                   args=[DAGTag('dag')],
                                   kwargs=dict(obj_hash=arg_ref.ref,
                                               depends_on=list(refs.flat)))

            # Depending on the chosen combination method, create corresponding
            # additional transformations for combination via merge or via
            # concatenation.
            if combination_method in ['merge']:
                dag.add_node(operation='dantro.merge',
                             args=[ref_buf],
                             tag=tag,
                             **(combination_kwargs if combination_kwargs
                                else {}))

            elif combination_method in ['concat']:
                dag.add_node(operation='dantro.concat',
                             args=[ref_buf],
                             kwargs=dict(dims=list(psp.dims.keys())),
                             tag=tag,
                             **(combination_kwargs if combination_kwargs
                                else {}))

            else:
                raise ValueError("Invalid combination method '{}'! Available "
                                 "methods: merge, concat."
                                 "".format(combination_method))

            # Done, yay! :)

        def add_sac_transformations(dag: TransformationDAG, *,
                                    fields: dict, subspace: dict=None,
                                    combination_method: str='concat',
                                    base_path: str=None) -> None:
            """Adds transformations to the given DAG that select data from the
            selected multiverse subspace.
            
            Args:
                dag (TransformationDAG): The DAG to add nodes t o
                fields (dict): Which fields to select from the separate
                    universes.
                subspace (dict, optional): The (default) subspace to select the
                    data from.
                combination_method (str, optional): The (default) combination
                    method of the multidimensional data.
                base_path (str, optional): If given, ``path`` specifications
                    of each field can be seen as relative to this path.
            """
            for tag, spec in fields.items():
                # Parse parameters, i.e.: Use defaults defined on this level
                # if the spec does not provide more specific information.
                spec = copy.deepcopy(spec)

                if base_path is not None:
                    spec['path'] = base_path + PATH_JOIN_CHAR + spec['path']

                spec['subspace'] = spec.get('subspace',copy.deepcopy(subspace))
                spec['combination_method'] = spec.get('combination_method',
                                                      combination_method)

                # Add the transformations for this specific tag
                add_transformations(dag, tag=tag, **spec)
        
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Initialize an (empty) DAG, i.e.: without select and transform args
        # and without setting the selection base
        dag = super()._create_dag(_plot_func=_plot_func, **dag_init_params)

        # Add nodes that perform the "select and combine" operations, based on
        # selections from the DataManager
        add_sac_transformations(dag, **select_and_combine)

        # Can now set the selection base to the intended value
        if select_base is not None:
            dag.select_base = select_base

        # Now add the additional transformations, which may use existing nodes.
        dag.add_nodes(select=select, transform=transform)
        
        return dag


# -----------------------------------------------------------------------------

class UniversePlotCreator(ExternalPlotCreator):
    """A UniversePlotCreator is an ExternalPlotCreator that allows looping of
    all or a selected subspace of universes.
    """
    # Where the `ParamSpaceGroup` object is expected within the data manager
    PSGRP_PATH = None

    # Configure the auto-detection feature implemented in ExternalPlotCreator:
    # The KEYWORD_ONLY arguments that are required to be (explicitly!) accepted
    _AD_KEYWORD_ONLY = ['out_path', 'uni']

    # .........................................................................

    def __init__(self, *args, psgrp_path: str=None, **kwargs):
        """Initialize a UniversePlotCreator"""
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
            raise ValueError("Missing class variable PSGRP_PATH! Either set "
                             "it directly or pass the `psgrp_path` argument "
                             "to the __init__ function.")

        # Retrieve the parameter space group
        return self.dm[self.PSGRP_PATH]

    def prepare_cfg(self, *, plot_cfg: dict, pspace: Union[dict, ParamSpace]
                    ) -> Tuple[dict, ParamSpace]:
        """Converts a regular plot configuration to one that can be configured
        to iterate over multiple universes via a parameter space.

        This is implemented in the following way:
            1. Extracts the `universes` key from the configuration and parses
               it, ensuring it is a valid dict for subspace specification
            2. Creates a new ParamSpace object that additionally contains the
               parameter dimensions corresponding to the universes. These are
               stored in a _coords dict inside the returned plot configuration.
            3. Apply the parsed `universes` key to activate a subspace of the
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
            plot_cfg = recursive_update(copy.deepcopy(pspace._dict), plot_cfg)
            # FIXME internal API usage

        # Now have the plot config
        # Identify those keys that specify which universes to loop over
        try:
            unis = plot_cfg.pop('universes')
        
        except KeyError as err:
            raise ValueError("Missing required keyword-argument `universes` "
                             "in plot configuration!") from err

        # Get the parameter space, as it might be needed for certain values of
        # the `universes` argument
        self._psp = copy.deepcopy(self.psgrp.pspace)

        # If there was no parameter space available in the first place, only
        # the default point is available, which should be handled differently
        if self._psp.num_dims == 0 or self.psgrp.only_default_data_present:
            if unis not in ['all', 'single', 'first', 'random', 'any']:
                raise ValueError("Could not select a universe for plotting "
                                 "because the associated parameter space has "
                                 "no dimensions available or only data for "
                                 "the default point was available in {}. "
                                 "For these cases, the only valid values for "
                                 "the `universes` argument are: 'all', "
                                 "'single', 'first', 'random', or 'any'."
                                 "".format(self.psgrp.logstr))

            # Set a flag to carry information to _prepare_plot_func_args
            self._without_pspace = True

            # Distinguish cases where plot_cfg was given and those were not
            if pspace is not None:
                # There was a recursive update step; return the plot config
                # as parameter space
                return dict(), ParamSpace(plot_cfg)

            # else: Only need to return the plot configuration
            return plot_cfg, None

        # Parse it such that it is a valid subspace selector
        if isinstance(unis, str):
            if unis in ['all']:
                # is equivalent to an empty specifier -> empty dict
                unis = dict()

            elif unis in ['single', 'first', 'random', 'any']:
                # Find the state number from the universes available in the
                # parameter space group. Then retrieve the coordinates from
                # the corresponding parameter space state map

                # Create a list of available universe IDs
                uni_ids = [int(_id) for _id in self.psgrp.keys()]

                # Select the first or a random ID
                if unis in ['single', 'first']:
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
                raise ValueError("Invalid value for `universes` argument. Got "
                                 "'{}', but expected one of: 'all', 'single', "
                                 "'first', 'random', or 'any'.".format(unis))

        elif not isinstance(unis, dict):
            raise TypeError("Need parameter `universes` to be either a "
                            "string or a dictionary of subspace selectors, "
                            "but got: {} {}.".format(type(unis), unis))

        # else: was a dict, can be used as a subspace selector

        # Ensure that no invalid dimension names were selected
        for pdim_name in unis.keys():
            if pdim_name not in self._psp.dims.keys():
                raise ValueError("No parameter dimension '{}' was available "
                                 "in the parameter space associated with {}! "
                                 "Available parameter dimensions: {}"
                                 "".format(pdim_name, self.psgrp.logstr,
                                           ", ".join([n for n
                                                      in self._psp.dims])))

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
        if '_coords' in plot_cfg:
            raise ValueError("The given plot configuration may _not_ contain "
                             "the key '_coords' on the top-most level!")

        plot_cfg['_coords'] = coords

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

    def _prepare_plot_func_args(self, *args, _coords: dict=None,
                                **kwargs) -> Tuple[tuple, dict]:
        """Prepares the arguments for the plot function and implements the
        special arguments required for ParamSpaceGroup-like data: selection of
        a single universe from the given coordinates.
        
        Args:
            *args: Passed along to parent method
            _coords (dict): The current coordinate descriptor which is then
                used to retrieve a certain point in parameter space from the
                state map attribute.
            **kwargs: Passed along to parent method
        
        Returns:
            tuple: (args, kwargs) for the plot function
        """
        # Need to distinguish between cases with or without pspace given. The
        # aim is to retrieve a Universe ID to use for selection.
        if self._without_pspace:
            # Only the default universe is available, always having ID 0.
            uni_id = 0

        else:
            # This is a parameter sweep.
            # Given the coordinates, retrieve the data for a single universe
            # from the state map. As _coords is created by the _prepare_cfg
            # method, it will unambiguously selects a universe ID.
            uni_id = int(self._psp_active_smap_cache.sel(**_coords))

        # Select the corresponding universe from the ParamSpaceGroup
        uni = self.psgrp[uni_id]
        log.note("Using data of:        %s", uni.logstr)
        uni_name = "uni{:d}".format(uni_id)
        uni_path = PATH_JOIN_CHAR.join([self.PSGRP_PATH, uni.name])
        # TODO Should be possible to do via uni.path, but is not (yet)

        # Create the parameters for the DAG transformation interface which uses
        # selections based in the universe group
        base_transform = [dict(getitem=[DAGTag('dm'), uni_path],
                               tag=uni_name,
                               file_cache=dict(read=False, write=False))]

        # Compile the DAG options dict, based on potentially existing options
        kwargs['dag_options'] = dict(**kwargs.get('dag_options', {}),
                                     base_transform=base_transform,
                                     select_base=uni_name)
        # NOTE If DAG usage is not enabled, these parameters have no effect

        # Let the parent function, implemented in ExternalPlotCreator, do its
        # thing. This will return the (args, kwargs) tuple
        return super()._prepare_plot_func_args(*args, uni=uni, **kwargs)
