"""This module implements the DataManager class, the root of the data tree."""

import os
import copy
import datetime
import re
import glob
import logging
import warnings
from typing import Union, Callable, List, Tuple

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

class ExistingDataError(DataManagerError):
    """Raised if data already existed."""
    pass

class ExistingGroupError(DataManagerError):
    """Raised if a group already existed."""
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

class NoMatchWarning(UserWarning):
    """If there was no regex match"""
    pass

# -----------------------------------------------------------------------------

class DataManager(OrderedDataGroup):
    """The DataManager is the root of a data tree, coupled to a specific data directory.
    
    It handles the loading of data and can be used for interactive work with
    the data.
    """

    # Define as class variables what should be the default groups / containers
    _DATA_GROUP_DEFAULT_CLS = OrderedDataGroup

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

        # If a specific value for the load configuration was given, store it
        # and use it as the default for the `load_from_cfg` method
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
                  "\n".join(["  {:>8s} : {}".format(k, v)
                             for k, v in dirs.items()]))

        return dirs

    # .........................................................................

    def new_group(self, name: str, *, Cls: type=None, **kwargs):
        """Creates a new group with the given name.
        
        Args:
            name (str): The name of the group
            Cls (type, optional): If given, use this type to create the
                group. If not given, uses the type of this instance.
            **kwargs: Passed on to Cls.__init__
        
        Returns:
            Cls: the created group
        """
        if Cls is None:
            Cls = self._DATA_GROUP_DEFAULT_CLS

        return super().new_group(name, Cls=Cls, **kwargs)

    # .........................................................................
    # Loading data

    def load_from_cfg(self, *, load_cfg: dict=None, update_load_cfg: dict=None, exists_action: str='raise', print_tree: bool=False) -> None:
        """Load multiple data entries using the specified load configuration.
        
        Args:
            load_cfg (dict, optional): The load configuration to use. If not
                given, the one specified during initialisation is used.
            update_load_cfg (dict, optional): If given, it is used to update
                the load configuration recursively
            exists_action (str, optional): The behaviour upon existing data.
                Can be: raise (default), skip, skip_nowarn, overwrite,
                overwrite_nowarn.  With *_nowarn values, no warning is given
                if an entry already existed.
            print_tree (bool, optional): If True, a tree representation of the
                DataManager is printed after the data was loaded
        
        Raises:
            TypeError: Raised if a given configuration entry was of invalid 
                type, i.e. not a dict
        """
        # Determine which load configuration to use
        if not load_cfg:
            log.debug("No new load configuration given; will use load "
                      "configuration given at initialisation.")
            load_cfg = self.load_cfg

        # Make sure to work on a copy, be it on the defaults or on the passed
        load_cfg = copy.deepcopy(load_cfg)

        if update_load_cfg:
            # Recursively update with the given keywords
            load_cfg = tools.recursive_update(load_cfg, update_load_cfg)
            log.debug("Updated the load configuration.")

        log.info("Loading %d data entries ...", len(load_cfg))

        # Loop over the data entries that were configured to be loaded
        for entry_name, params in load_cfg.items():
            # Check if this is of valid type
            if not isinstance(params, dict):
                raise TypeError("Got invalid load specifications for entry "
                                "'{}'! Expected dict, got {} with value '{}'. "
                                "Check the correctness of the given load "
                                "configuration!".format(entry_name,
                                                        type(params), params))

            # Use the public method to load this single entry
            self.load(entry_name, exists_action=exists_action,
                      print_tree=False,  # to not have prints during loading
                      **params)
        
        # All done
        log.info("Successfully loaded %d data entries.", len(load_cfg))
        log.info("Available top-level entries:\n  %s\n", 
                 ",  ".join(self.data.keys()))

        # Finally, print the tree
        if print_tree:
            print("{:tree}".format(self))


    def load(self, entry_name: str, *, loader: str, glob_str: Union[str, List[str]], target_group: str=None, target_path: str=None, print_tree: bool=False, **load_params) -> None:
        """Performs a single load operation.
        
        Args:
            entry_name (str): Name of this entry; will also be the name of the
                created group or container, unless `target_basename` is given
            loader (str): The name of the loader to use
            glob_str (Union[str, List[str]]): A glob string or a list of glob
                strings by which to identify the files within `data_dir` that
                are to be loaded using the given loader function
            target_group (str, optional): If given, the files to be loaded will
                be stored in this group. This may only be given if the argument
                target_path is _not_ given.
            target_path (str, optional): The path to write the data to. This
                can be a format string. It is evaluated for each file that has
                been matched. If it is not given, the content is loaded to a
                group with the name of this entry at the root level.
                Available keys are: basename, match (if `path_regex` is given)
            print_tree (bool, optional): Whether to print the tree at the end
                of the loading operation.
            **load_params: Further loading parameters, all optional!
                ignore (list): The exact file names in this list will be
                    ignored during loading. Paths are seen as elative to the
                    data directory of the data manager.
                required (bool): If True, will raise an error if no files were
                    found. Default: False.
                path_regex (str): This pattern can be used to match the path of
                    the file that is being loaded. The match result is
                    available to the format string under the `match` key.
                exists_action (str): The behaviour upon existing data.
                    Can be: raise (default), skip, skip_nowarn, overwrite,
                    overwrite_nowarn.
                    With *_nowarn values, no warning is given if an entry
                    already existed.
                progress_indicator (bool): Whether to print a progress
                    indicator or not. Default: True
                parallel (bool): If True, data is loaded in parallel.
                    This feature is not implemented yet!
                any further kwargs: passed on to the loader function
        
        Returns:
            None
        """

        def glob_match_single(glob_str: Union[str, List[str]]) -> bool:
            """Returns True if the given glob str matches at most one file."""
            return bool(isinstance(glob_str, str) and glob_str.find('*') < 0)

        def check_target_path(target_path: str):
            """Check that the target path evaluates correctly."""
            log.debug("Checking target path '%s' ...", target_path)
            try:
                _target_path = target_path.format(basename="basename",
                                                  match="match")

            except (IndexError, KeyError) as err:
                raise ValueError("Invalid argument `target_path`. Will not be "
                                 "able to properly evalaute '{}' later due to "
                                 "a {}: {}".format(target_path,
                                                   type(err), err)) from err
            else:
                log.debug("Target path will be:  %s", _target_path)    

        # Some preparations
        log.info("Loading data entry '%s' ...", entry_name)

        # Parse the arguments that result in the target path
        if target_group:
            if target_path:
                raise ValueError("Received both arguments `target_group` and "
                                 "`target_path`; make sure to only pass one "
                                 "or none of them.")

            if glob_match_single(glob_str):
                target_path = target_group + "/" + entry_name
            else:
                target_path = target_group + "/{basename:}"

        elif not target_path:
            if glob_match_single(glob_str):
                target_path = entry_name
            else:
                target_path = entry_name + "/{basename:}"

        # ...and check that it is working.
        check_target_path(target_path)

        # else: target_path was given

        log.info("Loading entry '%s' ...", )

        # Try loading the data and handle specific DataManagerErrors
        try:
            self._load(target_path=target_path, loader=loader,
                       glob_str=glob_str, **load_params)

        except RequiredDataMissingError:
            raise

        except MissingDataError as err:
            warnings.warn("No files were found to import!\n"+str(err),
                          MissingDataWarning)
            return  # Does not raise, but does not save anything either

        except LoaderError:
            raise

        else:
            # Everything as desired
            log.debug("Data successfully loaded.")

        # Done with this entry
        log.debug("Entry '%s' successfully loaded.", entry_name)

        if print_tree:
            print("{:tree}".format(self))
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # Helpers for loading and storing data

    def _load(self, *, target_path: str, loader: str, glob_str: Union[str, List[str]], ignore: List[str]=None, required: bool=False, path_regex: str=None, exists_action: str='raise', progress_indicator: bool=True, parallel: bool=False, **loader_kwargs) -> None:
        """Helper function that loads a data entry to the specified path.
        
        Args:
            target_path (str): The path to load the result of the loader to.
                This can be a format string; it is evaluated for each file.
                Available keys are: basename, match (if `path_regex` is given)
            loader (str): The loader to use
            glob_str (Union[str, List[str]]): A glob string or a list of glob
                strings to match files in the data directory
            ignore (List[str], optional): The exact file names in this list
                will be ignored during loading. Paths are seen as elative to the data directory.
            required (bool, optional): If True, will raise an error if no files
                were found.
            path_regex (str, optional): The regex applied to the relative path
                of the files that were found. It is used to generate the name
                of the target container. If not given, the basename is used.
            exists_action (str, optional): The behaviour upon existing data.
                Can be: raise (default), skip, skip_nowarn, overwrite,
                overwrite_nowarn.
                With *_nowarn values, no warning is given if an entry already
                existed.
            progress_indicator (bool, optional): Whether to print a progress
                indicator or not
            parallel (bool, optional): If True, data is loaded in parallel -
                not implemented yet!
            **loader_kwargs: passed on to the loader function
        
        Raises:
            NotImplementedError: For `parallel == True`
        """

        def resolve_loader(loader: str) -> Tuple[Callable, str, Callable]:
            """Resolves the loader function"""
            load_func_name = '_load_' + loader.lower()
            try:
                load_func = getattr(self, load_func_name)

            except AttributeError as err:
                raise LoaderError("Loader '{}' was not available to {}! Make "
                                  "sure to use a mixin class that supplies "
                                  "the '{}' loader method."
                                  "".format(loader, self.logstr,
                                            load_func_name)) from err
            else:
                log.debug("Resolved '%s' loader function.", loader)

            try:
                TargetCls = getattr(load_func, 'TargetCls')

            except AttributeError as err:
                raise LoaderError("Load function {} misses required attribute "
                                  "'TargetCls'. Check your mixin!"
                                  "".format(load_func)) from err

            return load_func, load_func_name, TargetCls

        def create_files_list(*, glob_str: Union[str, List[str]], ignore: List[str], required: bool=False, sort: bool=False) -> list:
            """Create the list of file paths to load from.

            Internally, this uses a set, thus ensuring that the paths are
            unique. The set is converted to a list before returning.
            
            Args:
                glob_str (Union[str, List[str]]): The glob pattern or a list of
                    glob patterns
                ignore (List[str]): The list of files to ignore
                required (bool, optional): Will lead to an error being raised
                    if no files could be matched
                sort (bool, optional): If true, sorts the list before returning
            
            Returns:
                list: the file paths to load
            
            Raises:
                MissingDataError: If no files could be matched
                RequiredDataMissingError: If no files could be matched but were
                    required.
            """
            # Create a set to assure that all files are unique
            files = set()

            # Assure it is a list of strings
            if isinstance(glob_str, str):
                # Is a single glob string                
                # Put it into a list to handle the same as the given arg
                glob_str = [glob_str]

            # Assuming glob_str to be lists of strings now
            log.debug("Got %d glob string(s) to create set of matching file "
                      "paths from.", len(glob_str))

            # Go over the given glob strings and add to the files set
            for gs in glob_str:
                # Make the glob string absolute
                gs = os.path.join(self.dirs['data'], gs)
                log.debug("Adding files that match glob string:\n  %s", gs)

                # Add to the set of files; this assures uniqueness of found paths
                files.update(list(glob.glob(gs, recursive=True)))

            # See if some files should be ignored
            if ignore:
                log.debug("Got list of files to ignore:\n  %s", ignore)
                
                # Make absolute and generate list of files to exclude
                ignore = [os.path.join(self.dirs['data'], path)
                          for path in ignore]

                log.debug("Removing them one by one now ...")

                # Remove the elements one by one
                while ignore:
                    rmf = ignore.pop()
                    try:
                        files.remove(rmf)
                    except KeyError:
                        log.debug("%s was not found in set of files.", rmf)
                    else:
                        log.debug("%s removed from set of files.", rmf)

            # Now the file list is final
            log.debug("Found %d files to load:\n  %s",
                      len(files), "\n  ".join(files))

            if not files:
                # No files found; exit here, one way or another
                if not required:
                    raise MissingDataError("No files found matching "
                                           "`glob_str` {} (and ignoring {})."
                                           "".format(glob_str, ignore))
                raise RequiredDataMissingError("No files found matching "
                                               "`glob_str` {} (and ignoring "
                                               "{}) were found, but were "
                                               "marked as required!"
                                               "".format(glob_str, ignore))

            # Convert to list
            files = list(files)

            # Sort, if asked to do so
            if sort:
                files.sort()

            return files

        def prepare_target_path(target_path: str, *, filepath: str, path_sre=None) -> List[str]:
            """Prepare the target path"""
            # The dict to be filled with formatting parameters
            fps = dict()

            # Extract the file basename (without extension)
            fps['basename'] = os.path.splitext(os.path.basename(filepath))[0]
            fps['basename'] = fps['basename'].lower()

            # Use the specified regex pattern to extract a match
            if path_sre:
                try:
                    _match = path_sre.findall(filepath)[0]
                
                except IndexError:
                    # nothing could be found
                    warnings.warn("Could not extract a name using the "
                                  "regex pattern '{}' on the file path:\n"
                                  "{}\nUsing the path's basename instead."
                                  "".format(path_sre, filepath),
                                  NoMatchWarning)
                    _match = fps['basename']
                
                else:
                    log.debug("Matched '%s' in file path '%s'.",
                              _match, filepath)
                
                fps['match'] = _match

            # Parse the format string to generate the file path
            log.debug("Parsing format string '%s' to generate target path ...",      target_path)
            log.debug("  kwargs: %s", fps)
            target_path = target_path.format(**fps)

            log.debug("Generated target path:  %s", target_path)
            return target_path.split(PATH_JOIN_CHAR)

        def skip_path(path: str, *, exists_action: str) -> bool:
            """Check whether a given path exists and — depending on the
            `exists_action` – decides whether to skip this path or now.
            
            Args:
                path (str): The path to check for existence.
                exists_action (str): The behaviour upon existing data. Can be:
                    raise, skip, skip_nowarn, overwrite, overwrite_nowarn.
                    The *_nowarn arguments suppress the warning
            
            Returns:
                bool: Whether to skip this path
            
            Raises:
                ExistingDataError: Raised when `exists_action == 'raise'`
                ValueError: Raised for invalid `exists_action` value
            """
            if path not in self:
                # Does not exist yet -> no need to skip
                return False
            # else: path exists already
            # NOTE that it is not known whether the path points to a group
            # or to a container

            _msg = ("Path '{}' already exists."
                    "".format(PATH_JOIN_CHAR.join(path)))

            # Distinguish different actions
            if exists_action == 'raise':
                raise ExistingDataError(_msg + " Adjust argument "
                                        "`exists_action` to allow skipping "
                                        "or overwriting of existing entries.")

            if exists_action in ['skip', 'skip_nowarn']:
                if exists_action == 'skip':
                    warnings.warn(_msg
                                  + " Loading of this entry will be skipped.",
                                  ExistingDataWarning)
                return True  # will lead to the data not being loaded

            elif exists_action in ['overwrite', 'overwrite_nowarn']:
                if exists_action == 'overwrite':
                    warnings.warn(_msg + " It will be overwritten!",
                                  ExistingDataWarning)
                return False  # will lead to the data being loaded

            else:
                raise ValueError("Invalid value for `exists_action` "
                                 "argument '{}'! Can be: raise, skip, "
                                 "skip_nowarn, overwrite, overwrite_nowarn."
                                 "".format(exists_action))

        def store(obj: Union[BaseDataGroup, BaseDataContainer], *, target_path: List[str]) -> None:
            """Store the given `obj` at the supplied `path`.

            Note that this will automatically overwrite, assuming that all
            checks have been made prior to the call to this function.
            
            Args:
                obj (Union[BaseDataGroup, BaseDataContainer]): Object to store
                target_path (List[str]): The path to store the object at
            
            Returns:
                None
            
            Raises:
                ExistingDataError: If non-group-like data already existed at
                    that path
            """

            # Extract a target group path and a base name
            group_path = target_path[:-1]
            basename = target_path[-1]

            # Resolve the target group object; create it if necessary
            # Need to check whether it is given at all. If not, write into the
            # data manager directly
            if not group_path:
                # Write directly into data manager root
                group = self

            else:
                # Need to retrieve or create the group
                # The difficulty is that the path can also point to a container
                # Need to assure here, that the group path points to a group
                if group_path not in self:
                    # Needs to be created
                    self._create_groups(group_path)

                elif not isinstance(self[group_path], BaseDataGroup):
                    # Already exists, but is no group. Cannot continue
                    group_path = PATH_JOIN_CHAR.join(group_path)
                    target_path = PATH_JOIN_CHAR.join(target_path)
                    raise ExistingDataError("The object at '{}' in {} is not "
                                            "a group but a {}. Cannot store "
                                            "{} there because the target path "
                                            "'{}' requires it to be a group."
                                            "".format(group_path, self.logstr,
                                                      type(self[group_path]),
                                                      obj.logstr, target_path))

                # Now the group path will point to a group
                group = self[group_path]
            
            # Store data, if possible
            if basename in group:
                # Already exists. Delete the old one, then store the new one
                del group[basename]

            # Can add now
            group.add(obj)
            
            # Done

        # End of helper functions . . . . . . . . . . . . . . . . . . . . . . .
        # Get the loader function
        load_func, load_func_name, TargetCls = resolve_loader(loader)

        # Create the list of file paths to load
        files = create_files_list(glob_str=glob_str, ignore=ignore,
                                  required=required, sort=True)
        
        # If a regex pattern was specified, compile it
        path_sre = re.compile(path_regex) if path_regex else None

        # Check if the `match` key is being used in the target_path
        if path_sre is not None and target_path.find("{match:") < 0:
            raise ValueError("Received the `path_regex` argument to match the "
                             "file path, but the `target_path` argument did "
                             "not contain the corresponding `{{match:}}` "
                             "placeholder. `target_path` value: '{}'."
                             "".format(target_path))

        if parallel:
            # TODO could be implemented by parallelising the below for loop
            raise NotImplementedError("Cannot load in parallel yet.")
        
        # Ready for loading files now . . . . . . . . . . . . . . . . . . . . .
        # Go over the files and load them
        for n, file in enumerate(files):
            if progress_indicator:
                line = "  Loading  {}/{}  ...".format(n+1, len(files))
                print(tools.fill_line(line), end="\r")

            # Prepare the target path (a list of strings)
            _target_path = prepare_target_path(target_path, filepath=file,
                                               path_sre=path_sre)

            # Check if it is to be skipped
            if skip_path(_target_path, exists_action=exists_action):
                log.debug("Skipping file '%s' ...", file)
                continue

            # Prepare the target class, which will be filled by the load func
            _TargetCls = lambda **kws: TargetCls(name=_target_path[-1], **kws)
            # This assures that the name is already correct

            # Get the data
            _data = load_func(file, TargetCls=_TargetCls, **loader_kwargs)
            
            # If this succeeded, store the data
            store(_data, target_path=_target_path)

            # Done with this file
            log.debug("Successfully loaded '%s' and stored at '%s' as %s.",
                      file, PATH_JOIN_CHAR.join(_target_path), _data.logstr)

        # Clear the line to get rid of the load indicator, if there was one
        if progress_indicator:
            tools.clear_line()

        # Done
        log.debug("Finished loading %d files.", len(files))

    def _contains_group(self, path: Union[str, List[str]], *, base_group: BaseDataGroup=None) -> bool:
        """Recursively checks if the given path is available _and_ a group.
        
        Args:
            path (Union[str, List[str]]): The path to check.
            base_group (BaseDataGroup): The group to start from. If not
                given, will use self.
        
        Returns:
            bool: Whether the path points to a group
        
        """
        def check(path: str, base_group: BaseDataGroup) -> bool:
            """Returns True if the object at path within base_group is
            a group. False otherwise.
            """
            return (path in base_group
                    and isinstance(base_group[path], BaseDataGroup))

        if not isinstance(path, list):
            path = path.split(PATH_JOIN_CHAR)

        if not base_group:
            base_group = self

        if len(path) > 1:
            # Need to continue recursively
            if check(path[0], base_group):
                return self._contains_group(path[1:],
                                            base_group=base_group[path[0]])
            return False

        # End of recursion
        return check(path[0], base_group)

    def _create_groups(self, path: Union[str, List[str]], *, base_group: BaseDataGroup=None, GroupCls=None, exist_ok: bool=True):
        """Recursively create groups for the given path.
        
        Args:
            path (Union[str, List[str]]): The path to create groups along
            base_group (BaseDataGroup, optional): The group to start from. If
                not given, uses self.
            GroupCls (None, optional): The class to use for creating the groups
                or None if the _DATA_GROUP_DEFAULT_CLS is to be used
            exist_ok (bool, optional): Whether it is ok that groups along the
                path already exist. These might also be of different type.
                Default: True
        
        Raises:
            ExistingDataError: If not `exist_ok`
            ExistingGroupError: If not `exist_ok` and a group already exists
        """
        # Parse arguments
        if isinstance(path, str):
            path = path.split(PATH_JOIN_CHAR)

        if base_group is None:
            base_group = self

        if GroupCls is None:
            GroupCls = self._DATA_GROUP_DEFAULT_CLS

        # Catch the disallowed case as early as possible
        if path[0] in base_group:
            # Check if it is a group that exists there
            if isinstance(base_group[path[0]], BaseDataGroup):
                if not exist_ok:
                    raise ExistingGroupError(path[0])

            else:
                # There is data (that is not a group) existing at the path.
                # Cannot continue
                raise ExistingDataError("Tried to create a group '{}' in {}, "
                                        "but a container was already stored "
                                        "at that path."
                                        "".format(path[0], base_group.logstr))

        # Create the group, if it does not yet exist
        if path[0] not in base_group:
            log.debug("Creating group '%s' in %s ...",
                      path[0], base_group.logstr)
            grp = GroupCls(name=path[0])
            base_group.add(grp)

        # path[0] is now created
        # Check whether to continue recursion
        if len(path) > 1:
            # Continue recursion
            self._create_groups(path[1:], base_group=base_group[path[0]],
                                GroupCls=GroupCls)
