"""This module implements the PlotHelper class"""

import os
import copy
import logging
import matplotlib.pyplot as plt

from dantro.tools import recursive_update

log = logging.getLogger(__name__)

class PlotHelperWarning(UserWarning):
    pass

# -----------------------------------------------------------------------------

class PlotHelper:
    """The PlotHelper takes care of the figure setup and saving and allows
    accessing matplotlib utilities through the plot configuration.
    """

    def __init__(self, *, out_path: str, enabled_helpers_defaults: list=None,
                 helper_defaults: dict=None, **helper_cfg):
        """Initialize a Plot Helper.
        
        Args:
            out_path (str): path to store the created figure
            enabled_helpers_defaults (list, optional): Names of enabled helpers
            helper_defaults (dict, optional): Helper configuration
            **helper_cfg: External helper configuration with which the
                defaults are updated
        
        Raises:
            ValueError: No matching helper function defined
                (unknown key in enabled_helpers)
        """
        # Determine available and enabled helpers
        self._AVAILABLE_HELPERS = [attr_name[6:] for attr_name in dir(self)
                                   if attr_name.startswith('_hlpr_')]
        self._enabled = enabled_helpers_defaults if enabled_helpers_defaults else []
        self._cfg = helper_defaults if helper_defaults else {}

        # Update defaults with helper_cfg
        if helper_cfg:
            self._enabled += [helper_name for helper_name in helper_cfg.keys()
                              if helper_name not in self._enabled
                              and helper_name not in ['setup_figure', 'save_figure']]
            # NOTE Need to distinguish setup_figure and save_figure here,
            #      because they are "special" helper functions, but their
            #      configuration is saved alongside the other helpers' config

            self._cfg = recursive_update(self._cfg, helper_cfg)

        # Check that the enabled helpers are actually available
        for helper_name in self._enabled:
            if helper_name not in self._AVAILABLE_HELPERS:
                raise ValueError("No helper with name '{}' available! "
                                 "Available helpers: {}"
                                 "".format(helper_name,
                                           ", ".join(self._AVAILABLE_HELPERS)))

        # Initialize remaining attributes
        self._out_path = out_path
        self._fig = None

    # .........................................................................
    # Properties

    @property
    def cfg(self) -> dict:
        """Returns deepcopy of dict with the helper configuration"""
        return copy.deepcopy(self._cfg)

    @property
    def fig(self):
        """Returns the current figure"""
        if not self._fig:
            raise ValueError("No figure initialized! Use 'setup_figure' to "
                             "create a figure instance first.")
        return self._fig

    @property
    def ax(self):
        """Returns the current axis of the associated figure"""
        return self.fig.gca()
    
    @property
    def enabled_helpers(self) -> list:
        """Returns copy of list with enabled helpers"""
        return copy.copy(self._enabled)

    @property
    def out_path(self) -> str:
        """Returns the output path of the plot"""
        return self._out_path
    

    # .........................................................................
    
    def _get_helper_params(self, helper_name: str) -> dict:
        """Gets the parameters for a certain helper
        
        Args:
            helper_name (str): Name of the helper
        
        Returns:
            dict: Contains the helper configuration
        """
        return self.cfg.get(helper_name, {})

    def setup_figure(self):
        """Sets up a matplotlib figure instance with the given configuration

        Raises:
            ValueError: Calling this function twice within the same PlotHelper
        """
        if self._fig is not None:
            raise ValueError("Figure is already initialized! The setup_figure "
                             "method may only be called once.")

        figure_cfg = self._get_helper_params('setup_figure')
        self._fig = plt.figure(**figure_cfg)

    def save_figure(self):
        """Saves and closes the current figure"""
        save_kwargs = self._get_helper_params('save_figure')
        self.fig.savefig(self.out_path, **save_kwargs)
        plt.close(self.fig)

    def invoke_helper(self, helper_name: str, *,
                      mark_disabled_after_use: bool=True,
                      **update_helper_params):
        """Invokes a single helper with the given configuration, then disables
        the helper.
        
        Args:
            helper_name (str): helper which is invoked
            mark_disabled_after_use (bool, optional): If True, the helper is
                marked as disabled after invoking it
            **update_helper_params: dict with which the helper config
                is updated
        
        Raises:
            ValueError: No matching helper function defined
            Warning: already enabled helper is invoked without avoiding
                repitition
        """
        # Get the helper function
        try:
            helper = getattr(self, "_hlpr_" + helper_name)

        except AttributeError as err:
            raise ValueError("No helper with name '{}' available! "
                             "Available helpers: {}"
                             "".format(helper_name,
                                       ", ".join(self._AVAILABLE_HELPERS))
                             ) from err

        # prepare helper parameters
        helper_params = self._get_helper_params(helper_name)
        if update_helper_params:
            helper_params = recursive_update(helper_params, update_helper_params)

        # Invoke helper
        log.debug("Invoking helper function '%s' ...", helper_name)
        helper(**helper_params)

        if not mark_disabled_after_use and helper_name in self._enabled:
            raise PlotHelperWarning("The already enabled helper '{}' was "
                                    "invoked. Its second invocation might "
                                    "lead to losing the effect of this "
                                    "invocation; set the "
                                    "`mark_disabled_after_use` parameter to "
                                    "avoid this."
                                    "".format(helper_name))

        elif mark_disabled_after_use:
            self.mark_disabled(helper_name)

    def invoke_helpers(self, *helper_names, mark_disabled_after_use: bool=True,
                       **update_kwargs):
        """Invoke the according helper for each name in helper_names
        
        Args:
            helper_names (list): Helpers to be invoked
            **update_kwargs: dict containing update parameters for helper
                configuration containing helper names as keys.
        """
        for helper_name in helper_names:
            self.invoke_helper(helper_name,
                               mark_disabled_after_use=mark_disabled_after_use,
                               **update_kwargs.get(helper_name, {}))

    def invoke_all(self, **update_kwargs):
        """Invokes all enabled helpers with their current configuration
        
        Args:
            **update_kwargs: dict containing update parameters for helper
                configuration containing helper names as keys.
        """
        for helper_name in self.enabled_helpers:
            self.invoke_helper(helper_name,
                               **update_kwargs.get(helper_name, {}))

    def provide_cfg(self, helper_name: str, **update_kwargs):
        """Update or add a single entry to a helper's configuration.
        
        Args:
            helper_name (str): Change config for this Helper
            **update_kwargs: dict containing the helper parameters with
                which the config is updated recursively
        
        Raises:
            ValueError: No matching helper function defined
        """
        if (    helper_name not in self._AVAILABLE_HELPERS
            and helper_name not in ['setup_figure', 'save_figure']):

            raise ValueError("No helper with name '{}' available! "
                             "Available helpers: {}"
                             "".format(helper_name,
                                       ", ".join(self._AVAILABLE_HELPERS)))

        # All good. Update or create the configuration entry
        if helper_name in self.cfg:
            recursive_update(self._cfg[helper_name], update_kwargs)
        else:
            self._cfg[helper_name] = update_kwargs

    def mark_enabled(self, *helper_names):
        """Marks the specified helpers as enabled
        
        Args:
            helper_names (list): Helpers to be enabled
        
        Raises:
            ValueError: No helper with such a name available
        """
        for helper_name in helper_names:
            if helper_name in self._AVAILABLE_HELPERS:
                if helper_name not in self._enabled:
                    self._enabled.append(helper_name)
            else:
                raise ValueError("No helper with name '{}' available! "
                                 "Available helpers: {}"
                                 "".format(helper_name,
                                           ", ".join(self._AVAILABLE_HELPERS)))

    def mark_disabled(self, *helper_names):
        """Marks the according helpers as disabled
        
        Args:
            helper_names (list): Helpers to be disabled
        
        Raises:
            ValueError: No helper with such a name available
        """
        for helper_name in helper_names:
            if helper_name in self._AVAILABLE_HELPERS:
                if helper_name in self._enabled:
                    self._enabled.remove(helper_name)
            else:
                raise ValueError("No helper with name '{}' available! "
                                 "Available helpers: {}"
                                 "".format(helper_name,
                                           ", ".join(self._AVAILABLE_HELPERS)))

    # .........................................................................
    # Helper Methods

    def _hlpr_set_title(self, *, title: str, **title_kwargs):
        """Set the plot title.
        
        Args:
            title (str): Title to be set
            **title_kwargs: Passed to matplotlib.pyplot.axis.set_title
        """
        self.ax.set_title(title, **title_kwargs)
