"""This module implements the DataManager class, the root of the data tree."""

import os
import copy
import logging
import datetime
from typing import Union

from dantro.base import PATH_JOIN_CHAR
from dantro.group import OrderedDataGroup
from dantro.tools import load_yml, recursive_update

# Local constants
log = logging.getLogger(__name__)


# Exception classes ...........................................................

class DataManagerError(Exception):
    """All DataManager exceptions derive from this one"""
    pass

class RequiredDataMissingError(DataManagerError):
    """Raised if required data was missing."""
    pass

class MissingDataError(DataManagerError):
    """Raised if data was missing, but is not required."""
    pass

class MissingLoaderError(DataManagerError):
    """Raised if a data loader was not available"""
    pass

# -----------------------------------------------------------------------------

class DataManager(OrderedDataGroup):
    """The DataManager is the root of a data tree, coupled to a specific data directory.
    
    It handles the loading of data and can be used for interactive work with
    the data.
    """

    # Define as class variables what should be the default groups / containers
    _DefaultDataGroupClass = OrderedDataGroup

    # .........................................................................
    # Initialisation

    def __init__(self, data_dir: str, *, name: str=None, load_cfg: Union[dict, str]=None, out_dir: Union[str, bool]="_output/{date:}"):
        """Initialises a DataManager object.
        
        Args:
            data_dir (str): the directory the data can be found in. If this is
                a relative path, it is considered relative to the current
                working directory.
            name (str, optional): which name to give to the DataManager. If no
                name is given, the data directories basename will be used
            load_cfg (Union[dict, str], optional): The base configuration used
                for loading data. If a string is given, assumes a yaml file and
                loads that. If none is given, it can still be supplied to the
                load_data method.
            out_dir (Union[str, bool], optional): where output is written to.
                If this is given as a relative path, it is considered relative
                to the _data directory_. A formatting operation with the keys 
                `date` and `name` is performed on this, where the latter is
                the name of the data manager. If set to False, the output
                directory is not created.
        """

        # Find a name if none was given
        if not name:
            basename = os.path.basename(os.path.abspath(data_dir))
            name = "{}_Manager".format(basename.replace(" ", "_"))
        
        log.info("Initialising %s '%s'...", self.classname, name)

        # Initialise as a data group via parent class
        super().__init__(name=name)

        # Initialise directories
        self.dirs = self._init_dirs(data_dir=data_dir, out_dir=out_dir)

        # If specific values for the load configuration was given, use these to set the class constants of this instance.
        self.load_cfg = {}

        if load_cfg and isinstance(load_cfg, str):
            # Assume this is the path to a configuration file and load it
            log.debug("Loading the default load config from a path:\n  %s",
                      load_cfg)
            self.load_cfg = load_yml(load_cfg)

        elif load_cfg:
            # Assume this is already a mapping and store it as the default
            log.debug("Using the given %s as default load configuration.",
                      type(load_cfg))
            self.load_cfg = load_cfg

        # Done
        log.debug("%s initialised.", self.logstr)

    def _init_dirs(self, *, data_dir: str, out_dir: Union[str, bool]) -> dict:
        """Initialises the directories managed by this DataManager and returns
        a dictionary that stores the absolute paths to these directories.
        
        If they do not exist, they will be created.
        
        Args:
            data_dir (str): the directory the data can be found in. If this is
                a relative path, it is considered relative to the current
                working directory.
            out_dir (Union[str, bool]): where output is written to.
                If this is given as a relative path, it is considered relative
                to the _data directory_. A formatting operation with the keys 
                `date` and `name` is performed on this, where the latter is
                the name of the data manager. If set to False, the output
                directory is not created.
        
        Returns:
            dict: The directories
        """

        # Make the data directory absolute
        log.debug("Received `data_dir` argument:\n  %s", data_dir)
        data_dir = os.path.abspath(data_dir)

        # Save dictionary that will hold info on directories
        dirs = dict(data=data_dir)

        # See if an output directory should be created
        if out_dir:
            log.debug("Received `out_dir` argument:\n  %s", out_dir)

            # Make current date and time available for formatting operations
            datestr = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

            # Perform a format operation on the output directory
            out_dir = out_dir.format(name=self.name.lower(), date=datestr)

            # If it is relative, assume it to be relative to the data directory
            if not os.path.isabs(out_dir):
                # By joining them together, out_dir is now relative
                out_dir = os.path.join(data_dir, out_dir)
    
            # Make path absolute and store in dict
            dirs['out'] = os.path.abspath(out_dir)

            # Create the directory
            os.makedirs(dirs['out'])
        else:
            dirs['out'] = False

        # Inform about the managed directories, then return
        log.debug("Managed directories:\n%s",
                  "\n".join(["  {:<8s} : {}".format(k, v)
                             for k, v in dirs.items()]))

        return dirs

    # .........................................................................
    # Loading data

    def load_data(self, *, load_cfg: dict=None, update_load_cfg: dict=None, overwrite_existing: bool=False, print_tree: bool=False) -> None:
        """Load the data using the specified load configuration.
        
        Args:
            load_cfg (dict, optional): The load configuration to use. If not
                given, the one specified during initialisation is used.
            update_load_cfg (dict, optional): If given, it is used to update
                the load configuration recursively
            overwrite_existing (bool, optional): If True, existing data will
                be overwritten. If False, they will be skipped (without
                raising an error)
            print_tree (bool, optional): If True, a tree representation of the
                DataManager is printed after the data was loaded
        """

        if not load_cfg:
            log.debug("Using default load_cfg.")
            load_cfg = self.load_cfg

        # Make sure to work on a copy, be it on the defaults or on the passed
        load_cfg = copy.deepcopy(load_cfg)

        if update_load_cfg:
            # Recursively update with the given keywords
            load_cfg = recursive_update(load_cfg, update_load_cfg)
            log.debug("Updated the default load configuration for this call of `load_data`.")

        log.info("Loading %d data entries ...", len(load_cfg))

        # Loop over the data entries that were configured to be loaded.
        for entry_name, params in load_cfg.items():
            log.note("Loading data entry '%s' ...", entry_name)

            # Warn if the entry_name is already present
            if entry_name in self:
                if overwrite_existing:
                    log.warning("The data entry '%s' was already loaded and "
                                "will be overwritten.", entry_name)
                else:
                    log.debug("The data entry '%s' was already loaded; not "
                              "loading it again ...", entry_name)
                    continue

            # Extract the group path, which is not needed by _load_entry
            group_path = params.pop('group_path', None)

            # Try loading the data and handle specific DataManagerErrors
            try:
                _entry = self._load_entry(entry_name=entry_name, **params)

            except RequiredDataMissingError:
                log.error("Required entry '%s' could not be loaded!", entry_name)
                raise

            except MissingDataError:
                log.warning("No files were found to import.")
                # Does not raise
                _entry = None

            except MissingLoaderError:
                # Loader was not available.
                raise

            else:
                # Everything as desired, _entry is now the imported data
                log.debug("Data successfully imported.")

            # Save the entry
            # See if a base_group was specified in the parameters. If yes, do not load the data under self[entry_name] but into the base group with the specified name. If that group is not present, create it.
            if not group_path:
                # Save it under the name of this entry
                self[entry_name] = _entry
                log.progress("Imported and saved data to entry_name '%s'.",
                             entry_name)

            else:
                # Find the group the entry is to be added to, create if needed
                if group_path in self:
                    group = self[group_path]
                
                else:
                    if len(group_path.split("PATH_JOIN_CHAR")) > 1:
                        raise NotImplementedError("Cannot create intermediate "
                                                  "groups yet for path '{}'!"
                                                  "".format(group_path))
                    # TODO implement creation of empty groups on the way

                    log.note("Creating group '%s' ...", group_path)
                    group = self._DefaultDataGroupClass(name=group_path)

                    # Let it be managed by this DataManager
                    self[group_path] = group

                # Recursively save to that group
                group.recursive_update(_entry)
                log.progress("Imported and saved data into group '%s'.",
                             group_path)

            # Done with this config entry, continue with next
        else:
            # All done
            log.info("Successfully loaded %d data entries.", len(self.data))
            log.info("Available data entries:\n  %s",
                     ",  ".join(self.data.keys()))

        if print_tree:
            print("{dm:name} tree:\n{dm:tree}".format(dm=self))

    def _load_entry(self, entry_name, **load_params):
        pass
