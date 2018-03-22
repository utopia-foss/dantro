"""This module implements the DataManager class, the root of the data tree."""

import os
import copy
import datetime
import uuid
import re
import glob
import logging
import warnings
from typing import Union, Callable

from dantro.base import PATH_JOIN_CHAR, BaseDataContainer, BaseDataGroup
from dantro.group import OrderedDataGroup
import dantro.tools as tools

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

class LoaderError(DataManagerError):
    """Raised if a data loader was not available"""
    pass

class MissingDataWarning(UserWarning):
    """Used as warning instead of MissingDataError"""
    pass

class ExistingDataWarning(UserWarning):
    """If there was data already existing ..."""
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
            self.load_cfg = tools.load_yml(load_cfg)

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

        def get_target_group(target_group_path: str) -> BaseDataGroup:
            """A helper function to resolve the target group"""
            # Determine to which group to save the entry that will be loaded
            if not target_group_path:
                # Save it in the root level
                return self

            # else: Find or create the group that the entry is to be added to
            if target_group_path not in self:
                if len(target_group_path.split(PATH_JOIN_CHAR)) > 1:
                    raise NotImplementedError("Cannot create intermediate "
                                              "groups yet for "
                                              "target_group path '{}'!"
                                              "".format(target_group_path))
                # TODO implement creation of empty groups on the way

                log.debug("Creating group '%s' ...", target_group_path)
                _grp = self._DefaultDataGroupClass(name=target_group_path)
                # FIXME this assumes target_group_path to *not* be a path

                # Add to the root level
                self.add(_grp)

            # Resolve the entry and return
            return self[target_group_path]


        if not load_cfg:
            log.debug("Using load configuration given at initialisation.")
            load_cfg = self.load_cfg
        else:
            log.debug("Using the given load configuration.")

        # Make sure to work on a copy, be it on the defaults or on the passed
        load_cfg = copy.deepcopy(load_cfg)

        if update_load_cfg:
            # Recursively update with the given keywords
            load_cfg = tools.recursive_update(load_cfg, update_load_cfg)
            log.debug("Updated the load config.")

        log.info("Loading %d data entries ...", len(load_cfg))

        
        # Loop over the data entries that were configured to be loaded . . . .
        for entry_name, params in load_cfg.items():
            # Some initial checks
            if not isinstance(params, dict):
                raise TypeError("Got invalid load specifications for entry "
                                "'{}'! Expected dict, got {} with value '{}'. "
                                "Check the correctness of the given load "
                                "configuration!".format(entry_name,
                                                        type(params), params))

            log.info("Loading data entry '%s' ...", entry_name)

            # Extract the target group path parameter and resolve it
            target_group = get_target_group(params.pop('target_group', None))

            # Extract the name of the target container or group . . . . . . . .
            target_basename = params.pop('target_basename', entry_name)

            # Check if the target already exists
            if target_basename in target_group:
                _warnmsg = ("The data entry with target basename '{}' "
                            "already exists in target group '{}'."
                            "".format(target_basename, target_group))
                if not overwrite_existing:
                    warnings.warn(_warnmsg + " It will not be loaded again.",
                                  ExistingDataWarning)
                    # go to the next entry
                    continue
                # else: load it, overwriting data
                warnings.warn(_warnmsg + " It will be overwritten!",
                              ExistingDataWarning)

            # Try loading the data and handle specific DataManagerErrors . . .
            try:
                _entry = self._entry_loader(target_group=target_group,
                                            name=target_basename, **params)

            except RequiredDataMissingError:
                log.error("Required entry '%s' could not be loaded!",
                          entry_name)
                raise

            except MissingDataError as err:
                warnings.warn("No files were found to import!\n"+str(err),
                              MissingDataWarning)
                continue  # Does not raise, but does not save anything either

            except LoaderError:
                raise

            else:
                # Everything as desired, _entry is now the imported data
                log.debug("Data successfully imported.")

            # Loaded now. Recursively save there . . . . . . . . . . . . . . . 
            target_group.recursive_update(_entry)
            log.debug("Saved data to target group %s.", target_group.path)

            # Done with this config entry, continue with next
        
        # All done
        log.info("Successfully loaded %d data entries.", len(self.data))
        log.info("Available data entries:\n  %s", ",  ".join(self.data.keys()))

        if print_tree:
            print("{dm:name} tree:\n{dm:tree}".format(dm=self))

    def _entry_loader(self, *, target_group: BaseDataGroup, name: str, loader: str, glob_str: str, ignore: list=None, always_create_group: bool=False, required: bool=False, name_regex: str=None, progress_indicator: bool=True, parallel: bool=False, **loader_kwargs) -> Union[BaseDataContainer, BaseDataGroup]:
        """Helper function that loads a data entry.
        
        Args:
            target_group (BaseDataGroup): The group the entry is loaded to;
                this is used to check whether the entry exists or not
            name (str): The name of the entry
            loader (str): The loader to use
            glob_str (str): The glob string to search files in the data dir
            ignore (list, optional): The exact file names in this list will
                be ignored during loading. Paths are seen as elative to the data directory.
            always_create_group (bool, optional): If False (default), no group
                will be created if only a single file was loaded. If True,
                will create a group even if only one file is loaded.
            required (bool, optional): If True, will raise an error if no files
                were found.
            name_regex (str, optional): The regex applied to the relative path
                of the files that were found. It is used to generate the name
                of the target container. If not given, the basename is used.
            progress_indicator (bool, optional): Description
            parallel (bool, optional): If True, data is loaded in parallel -
                not implemented yet!
            **loader_kwargs: passed on to the loader function
        
        Returns:
            Union[BaseDataContainer, BaseDataGroup]: The loaded entry
        
        Raises:
            MissingDataError: If data was missing, but was not required
            LoaderError: If the loader could not be found
            NotImplementedError: For `parallel == True`
            RequiredDataMissingError: If required data was missing
        """

        def prepare_target_cont_class(*, target_group, load_func, filepath: str, re_pattern):
            """Fetches the class that the load function specifies and prepares it to be used for initialisation by the load function."""
            # Find a suitable name
            if filepath:
                if re_pattern:
                    # Use the specified regex pattern to extract a name
                    try:
                        # Only take into account the first matching group
                        tname = re.findall(re_pattern, filepath)[0]
                    except IndexError:
                        warnings.warn("Could not extract a name using the "
                                      "pattern '{}' on the file path:\n{}\n "
                                      "Using the hash instead."
                                      "".format(re_pattern, filepath),
                                      UserWarning)
                        tname = hash(filepath)[:12]

                else:
                    # use the file's basename, without extension
                    tname = os.path.splitext(os.path.basename(filepath))[0].lower()

            # Ensure that there is nothing under that name in the target group
            if tname in target_group:
                warnings.warn("Possible duplicate entry '{}' in {}! "
                              "Adding a random string to the end ..."
                              "".format(tname, target_group.path),
                              UserWarning)
                tname += "_" + str(uuid.uuid4().hex[:12])

            # Try to resolve the class that is to 
            try:
                TargetCls = getattr(load_func, 'TargetCls')
            except AttributeError as err:
                raise LoaderError("Load function {} misses required attribute "
                                  "'TargetCls'. Check your mixin!"
                                  "".format(load_func)) from err

            # Create a new init function where the name is already resolved
            return lambda **kws: TargetCls(name=tname, **kws)

        # Get the load function
        load_func_name = '_load_' + loader.lower()
        try:
            load_func = getattr(self, load_func_name)
        except AttributeError as err:
            raise LoaderError("Loader '{}' was not available to {}! Make sure "
                              "to use a mixin class that supplies the '{}' "
                              "loader method.".format(loader, self.logstr,
                                                      load_func_name)) from err
        else:
            log.debug("Resolved '%s' loader function.", loader)

        # Generate an absolute glob string and a list of files
        glob_str = os.path.join(self.dirs['data'], glob_str)
        log.debug("Created absolute glob string:\n  %s", glob_str)
        files = glob.glob(glob_str, recursive=True)

        # See if some files should be ignored
        if ignore:
            log.debug("Got list of files to ignore:\n  %s", ignore)
            
            # Make absolute and generate list of files to exclude
            ignore = [os.path.join(self.dirs['data'], path) for path in ignore]

            log.debug("Removing them one by one now ...")

            # Remove the elements one by one
            while ignore:
                rmf = ignore.pop()
                try:
                    files.remove(rmf)
                except ValueError:
                    log.debug("%s was not found in files list.", rmf)
                else:
                    log.debugv("%s removed from files list.", rmf)

        # Now the file list is final
        log.debug("Found %d files for loader '%s':\n  %s",
                  len(files), loader, "\n  ".join(files))

        if not files:
            # No files found; can exit here, one way or another
            if not required:
                raise MissingDataError("No files matching glob_str '{}' "
                                       "(and ignoring {}).".format(glob_str,
                                                                   ignore))
            raise RequiredDataMissingError("No files matching '{}' (and "
                                           "ignoring {}) were found, but this "
                                           "entry was marked as required."
                                           "".format(glob_str, ignore))

        # else: there was at least one file to load.

        # If a regex pattern was specified, compile it
        re_pattern = re.compile(name_regex) if name_regex else None

        # Distinguish between cases where a group should be created and one where the DataContainer-object can be directly returned
        if len(files) == 1 and not always_create_group:
            # Use the single file available
            file = files[0]

            # Prepare the target class, which will be filled by the load func
            # The helper function takes care of the naming of the target cont
            TargetCls = prepare_target_cont_class(load_func=load_func,
                                                  target_group=target_group,
                                                  filepath=file,
                                                  re_pattern=re_pattern)

            # Load the data using the loader function
            data = load_func(file, TargetCls=TargetCls, **loader_kwargs)

            log.debug("Finished loading a single file for entry %s.", name)

            # And return it
            return data
    
        # else: more than one file -> need to work with groups

        if parallel:
            # TODO could be implemented by parallelising the below for loop
            raise NotImplementedError("Cannot load in parallel yet.")

        # Create a group wherein all entries are gathered
        # It has the same name as the entry
        group = self._DefaultDataGroupClass(name=name)

        # Go over the files and load them
        for n, file in enumerate(files):
            if progress_indicator:
                line = "  Loading ... {}/{}".format(n+1, len(files))
                print(tools.fill_tty_line(line), end="\r")

            # Prepare the target class, which will be filled by the load func
            # The helper function takes care of the naming of the target cont
            TargetCls = prepare_target_cont_class(load_func=load_func,
                                                  target_group=group,
                                                  filepath=file,
                                                  re_pattern=re_pattern)

            # Get the data and add it to the group
            _data = load_func(file, TargetCls=TargetCls, **loader_kwargs)
            group.add(_data)

        log.debug("Finished loading %d files for entry %s.", len(files), name)

        return group
