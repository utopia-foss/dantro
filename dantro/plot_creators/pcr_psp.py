"""This implements the ParamSpaceGroup plot creators, based on the
ExternalPlotCreator and providing additional functionality for data that is
stored in a ParamSpaceGroup.
"""

import copy
import logging
from typing import Union

import numpy as np
import xarray as xr

from paramspace import ParamSpace

from .pcr_ext import ExternalPlotCreator
from ..groups import ParamSpaceGroup
from ..tools import recursive_update

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class MultiversePlotCreator(ExternalPlotCreator):
    """A MultiversePlotCreator is an ExternalPlotCreator that allows data to be
    selected before being passed to the plot function.
    """
    # Where the `ParamSpaceGroup` object is expected within the data manager
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

    def _prepare_plot_func_args(self, *args, select: dict, **kwargs) -> tuple:
        """Prepares the arguments for the plot function and implements the
        select functionality for multiverse data.
        
        Args:
            *args: Passed along to parent method
            select (dict): Which data to select from the multiverse
            **kwargs: Passed along to parent method
        
        Returns:
            tuple: The (args, kwargs) tuple for calling the plot function
        """
        # Select the multiverse data
        mv_data = self.psgrp.select(**select)

        # Let the parent function, implemented in ExternalPlotCreator, do its
        # thing. This will return the (args, kwargs) tuple
        return super()._prepare_plot_func_args(*args, **kwargs,
                                               mv_data=mv_data)


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
        self._state_map = None
        self._without_pspace = False

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

    @property
    def state_map(self) -> xr.DataArray:
        """Returns the temporarily stored state mapping"""
        if self._state_map is None:
            raise RuntimeError("No state mapping was stored yet; this should "
                               "not have happened!")
        return self._state_map

    def prepare_cfg(self, *,
                    plot_cfg: dict, pspace: Union[dict, ParamSpace]) -> tuple:
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
        psp = self.psgrp.pspace

        # If there was no parameter space available in the first place, only
        # the default point is available, which should be handled differently
        if psp.num_dims == 0 or self.psgrp.only_default_data_present:
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

            else:
                # Only need to return the plot configuration
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

                # Create a list of universe IDs
                uni_ids = [int(_id) for _id in self.psgrp.keys()]

                # Select the first or a random ID
                if unis in ['single', 'first']:
                    uni_id = min(uni_ids)
                else:
                    uni_id = np.random.choice(uni_ids)

                # Now retrieve the point from the state map
                smap = psp.state_map
                point = smap.where(smap == uni_id, drop=True)

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
            if pdim_name not in psp.dims.keys():
                raise ValueError("No parameter dimension '{}' was available "
                                 "in the parameter space associated with {}! "
                                 "Available parameter dimensions: {}"
                                 "".format(pdim_name, self.psgrp.logstr,
                                           ", ".join([n for n in psp.dims])))

        # Copy parameter dimension objects for each coordinate
        # As the parameter space with the coordinates has a different
        # hierarchy than psp, the ordering has to be manually adjusted to be
        # the same as in the original psp.
        coords = dict()
        for dim_num, (name, pdim) in enumerate(psp.dims.items()):
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

        # Activate only a certain subspace
        mpc.activate_subspace(**unis)

        # Now, retrieve the mapping for the active subspace and store it as an
        # attribute of the plot creator
        self._state_map = mpc.active_state_map

        # Only return the configuration as a parameter space; all is included
        # in there now.
        return {}, mpc

    def _prepare_plot_func_args(self, *args,
                                _coords: dict=None, **kwargs) -> tuple:
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
        # Need to distinguish between cases with or without pspace given
        if self._without_pspace:
            # Only the default universe is available; pass that one
            uni = self.psgrp[0]
            return super()._prepare_plot_func_args(*args, **kwargs, uni=uni)

        # else: this is a parameter sweep
        # Given the coordinates, retrieve the data for a single universe from
        # the state map. As _coords is created by the _prepare_cfg method, it
        # can be assumed that it unambiguously selects a universe ID
        uni_id = self.state_map.sel(**_coords).item()

        # Select the corresponding universe from the ParamSpaceGroup
        uni = self.psgrp[uni_id]

        # Let the parent function, implemented in ExternalPlotCreator, do its
        # thing. This will return the (args, kwargs) tuple
        return super()._prepare_plot_func_args(*args, **kwargs, uni=uni)
