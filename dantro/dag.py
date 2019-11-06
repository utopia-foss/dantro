"""This is an implementation of a DAG for transformations on dantro objects"""

import os
import sys
import glob
import copy
import time
import logging
import pickle as pkl
from itertools import chain

from typing import TypeVar, Dict, Tuple, Sequence, Any, Union, List, Set

import numpy as np
import xarray as xr

from paramspace.tools import recursive_replace, recursive_collect

from .abc import AbstractDataContainer, PATH_JOIN_CHAR
from .base import BaseDataGroup
from .utils import KeyOrderedDict, apply_operation, register_operation
from .tools import recursive_update
from .data_loaders import LOADER_BY_FILE_EXT
from .containers import ObjectContainer, NumpyDataContainer, XrDataContainer
from ._dag_utils import (THash, DAGObjects,
                         DAGReference, DAGTag, DAGNode,
                         parse_dag_syntax as _parse_dag_syntax,
                         parse_dag_minimal_syntax as _parse_dag_minimal_syntax)
from ._hash import _hash, SHORT_HASH_LENGTH, FULL_HASH_LENGTH

# Local constants .............................................................
log = logging.getLogger(__name__)

# The path within the DAG's associated DataManager to which caches are loaded
DAG_CACHE_DM_PATH = 'cache/dag'

# Functions that can store the DAG computation result objects, distinguishing
# by their type.
DAG_CACHE_RESULT_SAVE_FUNCS = {
    # Saving functions of specific dantro objects
    (NumpyDataContainer,):
        lambda obj, p, **kws: obj.save(p+".npy", **kws),
    (XrDataContainer,):
        lambda obj, p, **kws: obj.save(p+".xrdc", **kws),

    # Saving functions of external packages
    (np.ndarray,):
        lambda obj, p, **kws: np.save(p+".npy", obj, **kws),
    (xr.DataArray,):
        lambda obj, p, **kws: obj.to_netcdf(p+".nc_da", **kws),
    (xr.Dataset,):
        lambda obj, p, **kws: obj.to_netcdf(p+".nc_ds", **kws),
}

# Type definitions (extending those from _dag_utils module)
TRefOrAny = TypeVar('TRefOrAny', DAGReference, Any)


# -----------------------------------------------------------------------------

class Transformation:
    """A transformation is the collection of an N-ary operation and its inputs.

    Transformation objects store the name of the operation that is to be
    carried out and the arguments that are to be fed to that operation. After
    a Transformation is defined, the only interaction with them is via the
    ``compute`` method.

    For computation, the arguments are recursively inspected for whether there
    are any DAGReference-derived objects; these need to be resolved first,
    meaning they are looked up in the DAG's object database and -- if they are
    another Transformation object -- their result is computed. This can lead
    to a traversal along the DAG.

    .. warning::

        Objects of this class should under *no* circumstances be changed after
        they were created!
        For performance reasons, the ``hashstr`` property is cached; thus,
        changing attributes that are included into the hash computation will
        not lead to a new hash, thus silently creating wrong behaviour.

        All relevant attributes (operation, args, kwargs, salt) are thus set
        read-only. This should be respected!
    """

    def __init__(self, *, operation: str,
                 args: Sequence[TRefOrAny],
                 kwargs: Dict[str, TRefOrAny],
                 dag: 'TransformationDAG'=None,
                 salt: int=None,
                 file_cache: dict=None):
        """Initialize a Transformation object.
        
        Args:
            operation (str): The operation that is to be carried out.
            args (Sequence[TRefOrAny]): Positional arguments for the
                operation.
            kwargs (Dict[str, TRefOrAny]): Keyword arguments for the
                operation. These are internally stored as a KeyOrderedDict.
            dag (TransformationDAG, optional): An associated DAG that is needed
                for object lookup. Without an associated DAG, args or kwargs
                may NOT contain any object references.
            salt (int, optional): A hashing salt that can be used to let this
                specific Transformation object have a different hash than other
                objects, thus leading to cache misses.
            file_cache (dict, optional): File cache options. Expected keys are
                ``write`` (boolean or dict) and ``read`` (boolean or dict).
                Note that the options given here are NOT reflected in the hash
                of the object!

                The following arguments are possible under the ``read`` key.

                enabled (bool, optional): Whether it should be attempted to
                    read from the file cache.
                load_options (dict, optional): Passed on to the load function,
                    i.e.: :py:meth:`dantro.data_mngr.DataManager.load`.

                The following arguments are possible under the ``write`` key.
                They are evaluated in the order that they are listed here.
                See :py:meth:`dantro.dag.Transformation._cache_result` for more
                information.

                enabled (bool, optional): Whether writing is enabled at all.
                always (bool, optional): If given, will always write.
                allow_overwrite (bool, optional): If False, will not write a
                    cache file if one already exists. If True, a cache file
                    _might_ be written, although one already exists. This is
                    still conditional on the evaluation of the other arguments.
                min_size (int, optional): The minimum size of the result object
                    that allows writing the cache.
                max_size (int, optional): The maximum size of the result object
                    that allows writing the cache.
                min_compute_time (float, optional): The minimal individual
                    computation time of this node that is needed in order for
                    the file cache to be written. Note that this value can be
                    lower if the node result is not computed but looked up from
                    the cache.
                min_cumulative_compute_time (float, optional): The minimal
                    cumulative computation time of this node and all its
                    dependencies that is needed in order for the file cache to
                    be written. Note that this value can be lower if the node
                    result is not computed but looked up from the cache.
                storage_options (dict, optional): Passed on to the cache
                    storage function. The following arguments are available:

                    ignore_groups (bool, optional): Whether to store groups.
                        Disabled by default.
                    attempt_pickling (bool, optional): Whether it should be
                        attempted to store results that could not be stored
                        via a dedicated storage function by pickling them.
                        Enabled by default.
                    raise_on_error (bool, optional): Whether to raise on error
                        to store a result. Disabled by default; useful to
                        enable this when debugging.
                    pkl_kwargs (dict, optional): Arguments passed on to the
                        pickle.dump function.
                    **save_kwargs: Arguments passed on to the dedicated storage
                        function.
            
        """
        # Storage attributes
        self._operation = operation
        self._args = args
        self._kwargs = KeyOrderedDict(**kwargs)
        self._dag = dag
        self._salt = salt
        self._hashstr = None
        self._profile = dict(compute=0., cumulative_compute=0.,
                             hashstr=0., cache_lookup=0., cache_writing=0.)

        # Parse file cache options, making sure it's a dict with default values
        self._fc_opts = file_cache if file_cache is not None else {}
        
        if isinstance(self._fc_opts.get('write', {}), bool):
            self._fc_opts['write'] = dict(enabled=self._fc_opts['write'])
        elif 'write' not in self._fc_opts:
            self._fc_opts['write'] = dict(enabled=False)
        
        if isinstance(self._fc_opts.get('read', {}), bool):
            self._fc_opts['read'] = dict(enabled=self._fc_opts['read'])
        elif 'read' not in self._fc_opts:
            self._fc_opts['read'] = dict(enabled=False)

        # Cache dict, containing the result and whether the cache is in memory
        self._cache = dict(result=None, filled=False)

    # .........................................................................
    # String representation and hashing

    def __str__(self) -> str:
        """A human-readable string characterizing this Transformation"""
        return ("<{t:}, operation: {op:}, {Na:d} args, {Nkw:d} kwargs>\n"
                "  args:   {args:}\n"
                "  kwargs: {kwargs:}\n"
                "".format(t=type(self).__name__, op=self._operation,
                          Na=len(self._args), Nkw=len(self._kwargs),
                          args=self._args, kwargs=self._kwargs))

    def __repr__(self) -> str:
        """A deterministic string representation of this transformation.

        .. note::

            This is also used for hash creation, thus it does not include the
            attributes that are set via the initialization arguments ``dag``
            and ``file_cache``.

        .. warning::

            Changing this method will lead to cache invalidations!
        """
        return ("<{mod:}.{t:}, operation={op:}, args={args:}, "
                "kwargs={kwargs:}, salt={salt:}>"
                "".format(mod=type(self).__module__, t=type(self).__name__,
                          op=repr(self._operation),
                          args=repr(self._args),
                          kwargs=repr(dict(self._kwargs)),# TODO Check sorting!
                          salt=repr(self._salt)))

    @property
    def hashstr(self) -> THash:
        """Computes the hash of this Transformation by creating a deterministic
        representation of this Transformation using ``__repr__`` and then
        applying a checksum hash function to it.

        Note that this does NOT rely on the built-in hash function but on the
        custom dantro ``_hash`` function which produces a platform-independent
        and deterministic hash. As this is a *string*-based (rather than an
        integer-based) hash, it is not implemented as the ``__hash__`` magic
        method but as this separate property.
        """
        if self._hashstr is None:
            t0 = time.time()
            self._hashstr = _hash(repr(self))
            self._update_profile(hashstr=time.time() - t0)

        return self._hashstr

    def __hash__(self) -> int:
        """Computes the python-compatible integer hash of this object from the
        string-based hash of this Transformation.
        """
        return hash(self.hashstr)

    # .........................................................................
    # Properties

    @property
    def dag(self) -> 'TransformationDAG':
        """The associated TransformationDAG; used for object lookup"""
        return self._dag

    @property
    def dependencies(self) -> Set[DAGReference]:
        """Recursively collects the references that are found in the positional
        and keyword arguments of this Transformation.
        """
        return set(recursive_collect(chain(self._args, self._kwargs.values()),
                   select_func=(lambda o: isinstance(o, DAGReference))))

    @property
    def resolved_dependencies(self) -> Set['Transformation']:
        """Transformation objects that this Transformation depends on"""
        return set([ref.resolve_object(dag=self.dag)
                    for ref in self.dependencies])

    @property
    def profile(self) -> Dict[str, float]:
        """The profiling data for this transformation"""
        return self._profile

    # YAML representation .....................................................
    yaml_tag = u'!dag_trf'

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(**constructor.construct_mapping(node, deep=True))

    @classmethod
    def to_yaml(cls, representer, node):
        """A YAML representation of this Transformation, including all its
        arguments (which must again be YAML-representable). In essence, this
        returns a YAML mapping that has the ``!dag_trf`` YAML tag prefixed,
        such that *reading* it in will lead to the ``from_yaml`` method being
        invoked.

        .. note::

            The YAML representation does *not* include the ``file_cache``
            parameters.

        .. warning::

            The YAML representation is used in computing the hashstr that
            identifies this transformation.
            Changing the argument order here or adding further keys to the
            dict will lead to hash changes and thus to cache misses.

        """
        # Collect the attributes that are relevant for the transformation.
        d = dict(operation=node._operation,
                 args=node._args,
                 kwargs=dict(node._kwargs))
        
        # If a specific salt was given, add that to the dict as well
        if node._salt is not None:
            d['salt'] = node._salt

        # Let YAML represent this as a mapping with an additional tag
        return representer.represent_mapping(cls.yaml_tag, d)

    # .........................................................................
    # Compute interface

    def compute(self) -> Any:
        """Computes the result of this transformation by recursively resolving
        objects and carrying out operations.
        
        This method can also be called if the result is already computed; this
        will lead only to a cache-lookup, not a re-computation.
        
        Returns:
            Any: The result of the operation
        """
        def is_DAGReference(obj: Any) -> bool:
            return isinstance(obj, DAGReference)

        def resolve_and_compute(ref: DAGReference):
            """Resolve references to their objects, if necessary computing the
            results of referenced Transformation objects recursively.

            Makes use of arguments from outer scope.
            """
            if self.dag is None:
                raise ValueError("Cannot resolve Transformation arguments "
                                 "that contain DAG references, because no DAG "
                                 "was associated with this Transformation!")

            # Let the reference resolve the corresponding object from the DAG
            obj = ref.resolve_object(dag=self.dag)
            
            # Check if this refers to an object that is NOT a transformation.
            # This could be the DataManager or the DAG itself, but also other
            # objects in the DAG's object database. Performing computation on
            # those is either not possible or would lead to infinite loops.
            if not isinstance(obj, Transformation):
                return obj

            # else: It is another Transformation object. Compute it, which
            # leads to a traversal up the DAG tree.
            return obj.compute()

        # Try to look up an already computed result from memory or file cache
        success, res = self._lookup_result()
        
        if not success:
            # Did not find a result in memory or file cache -> Compute it.
            # First, compute the result of the references in the arguments.
            args =   recursive_replace(copy.deepcopy(self._args),
                                       select_func=is_DAGReference,
                                       replace_func=resolve_and_compute)
            kwargs = recursive_replace(copy.deepcopy(self._kwargs),
                                       select_func=is_DAGReference,
                                       replace_func=resolve_and_compute)
            # NOTE Important to deepcopy here, because otherwise the recursive
            #      replacement and the mutability of both args and kwargs will
            #      lead to DAGReference objects being replaced with the actual
            #      objects, which would break the .hashstr property of this
            #      object. The deepcopy is always possible, because even if
            #      the args and kwargs are nested, they contain only trivial
            #      objects.

            # Carry out the operation
            res = self._perform_operation(args=args, kwargs=kwargs)

        # Allow caching the result, even if it comes from the cache
        self._cache_result(res)

        return res

    def _perform_operation(self, *, args, kwargs) -> Any:
        """Perform the operation, updating the profiling info on the side"""
        # Initialize a dict for profiling info
        prof = dict()

        # Set up profiling
        t0 = time.time()

        # Actually perform the operation
        res = apply_operation(self._operation, *args, **kwargs)
        # TODO Add error handling with node information

        # Prase profiling info and return the result
        self._update_profile(cumulative_compute=(time.time() - t0))

        return res

    def _update_profile(self, *, cumulative_compute: float=None,
                        **times) -> None:
        """Given some new profiling times, updates the profiling information.
        
        Args:
            cumulative_compute (float, optional): The cumulative computation
                time; if given, additionally computes the computation time for
                this individual node.
            **times: Valid profiling data.
        """
        # If cumulative computation time was given, calculate individual time
        if cumulative_compute is not None:
            self._profile['cumulative_compute'] = cumulative_compute

            # Aggregate the dependencies' cumulative computation times
            deps_cctime = sum([dep.profile['cumulative_compute']
                               for dep in self.resolved_dependencies
                               if isinstance(dep, Transformation)])
            # NOTE The dependencies might not have this value set because there
            #      might have been a cache lookup
            self._profile['compute'] = max(0., cumulative_compute-deps_cctime)

        # Store the remaining entries
        self._profile.update(times)

    # Cache handling . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _lookup_result(self) -> Tuple[bool, Any]:
        """Look up the transformation result to spare re-computation"""
        success, res = False, None

        # Retrieve cache parameters
        read_opts = self._fc_opts.get('read', {})
        load_opts = read_opts.get('load_options', {})

        # Check if the cache is already filled. If not, see if the file cache
        # can be read and is configured to be read.
        if self._cache['filled']:
            success = True
            res = self._cache['result']

        elif self.dag is not None and read_opts.get('enabled', False):
            # Setup profiling
            t0 = time.time()

            # Let the DAG check if there is a file cache, i.e. if a file with
            # this Transformation's hash exists in the DAG's cache directory.
            success, res = self.dag._retrieve_from_cache_file(self.hashstr,
                                                              **load_opts)

            # Store the result
            if success:
                self._cache['result'] = res
                self._cache['filled'] = True

            self._update_profile(cache_lookup=(time.time() - t0))

        return success, res
    
    def _cache_result(self, result: Any) -> None:
        """Stores a computed result in the cache"""
        def should_write(*, enabled: bool, always: bool=False,
                         allow_overwrite: bool=False,
                         min_size: int=None, max_size: int=None,
                         min_compute_time: float=None,
                         min_cumulative_compute_time: float=None,
                         storage_options: dict=None
                         ) -> bool:
            """A helper function to evaluate _whether_ the file cache is to be
            written or not.
            
            Args:
                enabled (bool): Whether writing is enabled at all
                always (bool, optional): If given, will always write.
                allow_overwrite (bool, optional): If False, will not write a
                    cache file if one already exists. If True, a cache file
                    _might_ be written, although one already exists. This is
                    still conditional on the evaluation of the other arguments.
                min_size (int, optional): The minimum size of the result object
                    that allows writing the cache.
                max_size (int, optional): The maximum size of the result object
                    that allows writing the cache.
                min_compute_time (float, optional): The minimal individual
                    computation time of this node that is needed in order for
                    the file cache to be written. Note that this value can be
                    lower if the node result is not computed but looked up from
                    the cache.
                min_cumulative_compute_time (float, optional): The minimal
                    cumulative computation time of this node and all its
                    dependencies that is needed in order for the file cache to
                    be written. Note that this value can be lower if the node
                    result is not computed but looked up from the cache.
                storage_options (dict, optional): (ignored here)
            
            Returns:
                bool: Whether to write the file cache or not.
            """
            if not enabled:
                # ... nothing else to check
                return False

            # With always: always write, don't look at other arguments.
            if always:
                return True
            # All checks below are formulated such that they return False.

            # If overwriting is _disabled_ and a cache file already exists, it
            # is already clear that a new one should _not_ be written
            if not allow_overwrite and self.hashstr in self.dag.cache_files:
                return False

            # Evaluate profiling information
            if min_compute_time is not None:
                if self.profile['compute'] < min_compute_time:
                    return False

            if min_cumulative_compute_time is not None:
                if (  self.profile['cumulative_compute']
                    < min_cumulative_compute_time):
                    return False

            # Evaluate object size
            if min_size is not None or max_size is not None:
                size_itvl = [min_size if min_size is not None else 0,
                             max_size if max_size is not None else np.inf]
                obj_size = sys.getsizeof(result)  # from outer scope

                if not (size_itvl[0] < obj_size < size_itvl[1]):
                    return False

            # If this point is reached, the cache file should be written.
            return True

        # Store a reference to the result and mark the cache as being in use
        self._cache['result'] = result
        self._cache['filled'] = True
        # NOTE If instead of a proper computation, the passed result object was
        #      previously looked up from the cache, this will not have an
        #      effect.
        
        # Get file cache writing parameters; don't write if not 
        write_opts = self._fc_opts['write']

        # Determine whether to write to a file
        if self.dag is not None and should_write(**write_opts):
            # Setup profiling
            t0 = time.time()
            
            # Write the result to a file inside the DAG's cache directory. This
            # is handled by the DAG itself, because the Transformation does not
            # know (and should not care) aboute the cache directory ...
            storage_opts = write_opts.get('storage_options', {})
            self.dag._write_to_cache_file(self.hashstr, result=result,
                                          **storage_opts)

            self._update_profile(cache_writing=(time.time() - t0))



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class TransformationDAG:
    """This class collects transformation operations that are (already by
    their own structure) connected into a directed acyclic graph. The aim of
    this class is to maintain base objects, manage references, and allow
    operations on the DAG, the most central of which is computing the result
    of a node.

    Furthermore, this class also implements caching of transformations, such
    that operations that take very long can be stored (in memory or on disk) to
    speed up future operations.

    Objects of this class are initialized with dict-like arguments which
    specify the transformation operations. There are some shorthands that allow
    a simple definition syntax, for example the ``select`` syntax, which takes
    care of selecting a basic set of data from the associated DataManager.
    """

    def __init__(self, *, dm: 'DataManager',
                 select: dict=None, transform: Sequence[dict]=None,
                 cache_dir: str='.cache', file_cache_defaults: dict=None,
                 base_transform: Sequence[Transformation]=None,
                 select_base: Union[DAGReference, str]=None,
                 select_path_prefix: str=None):
        """Initialize a DAG which is associated with a DataManager and load the
        specified transformations configuration into it.
        
        Args:
            dm (DataManager): The associated data manager
            select (dict, optional): Selection specifications, which are
                translated into regular transformations based on ``getitem``
                operations. The ``base_transform`` and ``select_base``
                arguments can be used to define from which object to select.
                By default, selection happens from the associated DataManager.
            transform (Sequence[dict], optional): Transform specifications.
            cache_dir (str, optional): The name of the cache directory to
                create if file caching is enabled. If this is a relative path,
                it is interpreted relative to the associated data manager's
                data directory. If it is absolute, the absolute path is used.
                The directory is only created if it is needed.
            file_cache_defaults (dict, optional): Default arguments for file
                caching behaviour. This is recursively updated with the
                arguments given in each individual select or transform
                specification.
            base_transform (Sequence[Transformation], optional): A sequence of
                transform specifications that are added to the DAG prior to
                those added via ``select`` and ``transform``. These can be used
                to create some other object from the data manager which should
                be used as the basis of ``select`` operations.
            select_base (Union[DAGReference, str], optional): Which tag to
                base the ``select`` operations on. If None, will use the
                (always-registered) tag for the data manager, ``dm``. This
                attribute can also be set via the ``select_base`` property.
            select_path_prefix (str, optional): If given, this path is prefixed
                to all ``path`` specifications made within the ``select``
                argument. Note that unlike setting the ``select_base`` this
                merely joins the given prefix to the given paths, thus leading
                to repeated path resolution. For that reason, using the
                ``select_base`` argument is generally preferred and the
                ``select_path_prefix`` should only be used if ``select_base``
                is already in use.
                If this path ends with a ``/``, it is directly prepended. If
                not, the ``/`` is added before adjoining it to the other path.
        """
        self._dm = dm
        self._objects = DAGObjects()
        self._tags = dict()
        self._nodes = list()
        self._fc_opts = file_cache_defaults if file_cache_defaults else {}
        self._select_base = None
        self._profile = dict(add_node=0., compute=0.)
        self._select_path_prefix = select_path_prefix
        
        # Determine cache directory path; relative path interpreted as relative
        # to the DataManager's data directory
        if os.path.isabs(cache_dir):
            self._cache_dir = cache_dir
        else:
            self._cache_dir = os.path.join(self.dm.dirs['data'], cache_dir)

        # Add the DAG itself and the DataManager as objects with default tags
        self.tags['dag'] = self.objects.add_object(self)
        self.tags['dm'] = self.objects.add_object(self.dm)
        # NOTE The data manager is NOT a node of the DAG, but more like an
        #      external data source, thus being accessible only as a tag

        # Add base transformations that do not rely on select operations
        self.add_nodes(transform=base_transform)

        # Set the selection base tag; the property setter checks availability
        self.select_base = select_base

        # Now add nodes via the main arguments; these can now make use of the
        # select interface, because a select base tag is set and base transform
        # operations were already added.
        self.add_nodes(select=select, transform=transform)

    # .........................................................................
    
    def __str__(self) -> str:
        """A human-readable string characterizing this TransformationDAG"""
        return ("<TransformationDAG, "
                "{:d} node(s), {:d} tag(s), {:d} object(s)>"
                "".format(len(self.nodes), len(self.tags), len(self.objects)))

    # .........................................................................

    @property
    def dm(self) -> 'DataManager':
        """The associated DataManager"""
        return self._dm

    @property
    def hashstr(self) -> THash:
        """Returns the hash of this DAG, which depends solely on the hash of
        the associated DataManager.
        """
        return _hash("<TransformationDAG, coupled to DataManager with ref {}>"
                     "".format(self.dm.hashstr))

    @property
    def objects(self) -> DAGObjects:
        """The object database"""
        return self._objects

    @property
    def tags(self) -> Dict[str, THash]:
        """A mapping from tags to objects' hashes; the hashes can be looked
        up in the object database to get to the objects.
        """
        return self._tags

    @property
    def nodes(self) -> List[THash]:
        """The nodes of the DAG"""
        return self._nodes

    @property
    def cache_dir(self) -> str:
        """The path to the cache directory that is associated with the
        DataManager that is coupled to this DAG. Note that the directory might
        not exist yet!
        """
        return self._cache_dir

    @property
    def cache_files(self) -> Dict[THash, Tuple[str, str]]:
        """Scans the cache directory for cache files and returns a dict that
        has as keys the hashes and as values a tuple of full path and file
        extension.
        """
        info = dict()

        # Go over all files in the cache dir that have an extension
        for path in glob.glob(os.path.join(self.cache_dir, '*.*')):
            if not os.path.isfile(path):
                continue

            # Get filename and extension, then check if it is a hash
            fname, ext = os.path.splitext(os.path.basename(path))
            if len(fname) != FULL_HASH_LENGTH:
                continue
            # else: filename is assumed to be the hash.

            if fname in info:
                raise ValueError("Encountered a duplicate cache file for the "
                                 "transformation with hash {}! Delete all but "
                                 "one of those files from the cache directory "
                                 "{}.".format(fname, self.cache_dir))

            # All good, store info.
            info[fname] = dict(full_path=path, ext=ext)

        return info

    @property
    def select_base(self) -> DAGReference:
        """The reference to the object that is used for select operations"""
        return self._select_base

    @select_base.setter
    def select_base(self, new_base: Union[DAGReference, str]):
        """Set the reference that is to be used as the base of select
        operations. It can either be a reference object or a string, which is
        then interpreted as a tag.
        """
        # Distinguish by type. If it's not a DAGReference, assume it's a tag.
        if new_base is None:
            new_base = DAGTag('dm').convert_to_ref(dag=self)

        elif isinstance(new_base, DAGReference):
            # Make sure it is a proper DAGReference object (hash-based) and not
            # an object of a derived class.
            new_base = new_base.convert_to_ref(dag=self)

        elif new_base not in self.tags:
            raise KeyError("The tag '{}' cannot be the basis of future select "
                           "operations because it is not available! Make sure "
                           "that a node with that tag is added prior to the "
                           "attempt of setting it. Available tags: {}. "
                           "Alternatively, pass a DAGReference object."
                           "".format(new_base, ", ".join(self.tags)))

        else:
            # Tag is available. Create a DAGReference via DAGTag conversion
            log.debug("Setting select_base to tag '%s' ...", new_base)
            new_base = DAGTag(new_base).convert_to_ref(dag=self)

        # Have a DAGReference now. Store it.
        self._select_base = new_base

    @property
    def profile(self) -> Dict[str, float]:
        """Returns the profiling information for the DAG."""
        return self._profile
    
    @property
    def profile_extended(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Builds an extended profile that includes the profiles from all
        transformations and some aggregated information.

        This is calculated anew upon each invocation; the result is not cached.
        """
        prof = copy.deepcopy(self.profile)
        
        # Add tag-specific information
        prof['tags'] = dict()
        for tag, obj_hash in self.tags.items():
            obj = self.objects[obj_hash]
            if not isinstance(obj, Transformation):
                continue

            tprof = copy.deepcopy(obj.profile)
            prof['tags'][tag] = tprof

        # Aggregate the profiled times from all transformations (by item)
        to_aggregate = ('compute', 'hashstr',
                        'cache_lookup', 'cache_writing')
        stat_funcs = dict(mean=lambda d: np.mean(d),
                          std=lambda d: np.std(d),
                          min=lambda d: np.min(d),
                          max=lambda d: np.max(d),
                          q25=lambda d: np.quantile(d, .25),
                          q50=lambda d: np.quantile(d, .50),
                          q75=lambda d: np.quantile(d, .75))
        tprofs = {item: list() for item in to_aggregate}

        for obj_hash, obj in self.objects.items():
            if not isinstance(obj, Transformation):
                continue

            tprof = copy.deepcopy(obj.profile)
            for item in to_aggregate:
                tprofs[item].append(tprof[item])

        # Compute some statistics for the aggregated elements
        prof['aggregated'] = dict()
        for item in to_aggregate:
            prof['aggregated'][item] = {k: (f(tprofs[item])
                                            if tprofs[item] else np.nan)
                                        for k, f in stat_funcs.items()}
        return prof

    # .........................................................................

    def add_node(self, *, operation: str, args: list=None, kwargs: dict=None,
                 tag: str=None, file_cache: dict=None,
                 **trf_kwargs) -> DAGReference:
        """Add a new node by creating a new Transformation object and adding it
        to the node list.
        
        Args:
            operation (str): The name of the operation
            args (list, optional): Positional arguments to the operation
            kwargs (dict, optional): Keyword arguments to the operation
            tag (str, optional): The tag the transformation should be made
                available as.
            file_cache (dict, optional): File cache options for
                this node. If defaults were given during initialization, those
                defaults will be updated with the given dict.
            **trf_kwargs: Passed on to Transformation.__init__
        
        Raises:
            ValueError: If the tag already exists
        
        Returns:
            DAGReference: The reference to the created node
        """
        t0 = time.time()

        # Some helper methods for the recursive replacement
        def not_proper_ref(obj: Any) -> bool:
            return (    isinstance(obj, DAGReference)
                    and type(obj) is not DAGReference)

        def convert_to_ref(obj: DAGReference) -> DAGReference:
            return obj.convert_to_ref(dag=self)

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Handle default values of arguments
        args = copy.deepcopy(args) if args else []
        kwargs = copy.deepcopy(kwargs) if kwargs else {}
        # NOTE Deep copy is important here, because the mutability of nested
        #      args or kwargs may lead to side effects. The deep copy should
        #      always be possible, because args and kwargs should only contain
        #      trivial objects.

        # Recursively replace any derived references to proper DAGReferences,
        # which work hash-based. This is to not have multiple options of how
        # another TransformationDAG object is referenced.
        args =   recursive_replace(args, select_func=not_proper_ref,
                                   replace_func=convert_to_ref)
        kwargs = recursive_replace(kwargs, select_func=not_proper_ref,
                                   replace_func=convert_to_ref)
        
        # Parse file cache parameters
        fc_opts = copy.deepcopy(self._fc_opts)  # Always a dict
        
        if file_cache is not None:
            fc_opts = recursive_update(fc_opts, file_cache)

        # From these arguments, create the Transformation object and add it to
        # the objects database.
        trf = Transformation(operation=operation, args=args, kwargs=kwargs,
                             dag=self, file_cache=fc_opts, **trf_kwargs)
        trf_hash = self.objects.add_object(trf)
        # NOTE From this point on, the object itself has no relevance.
        #      Transformation objects should only be handled via their hash in
        #      order to reduce duplicate calculations and make efficient
        #      caching possible.

        # Store the hash in the node list
        self.nodes.append(trf_hash)

        # If a tag was specified, create a tag
        if tag:
            if tag in self.tags.keys():
                raise ValueError("Tag '{}' already exists! Choose a different "
                                 "one. Already in use: {}"
                                 "".format(tag, ", ".join(self.tags.keys())))
            self.tags[tag] = trf_hash

        # Update the profile
        self._update_profile(add_node=time.time() - t0)

        # Return a reference to the newly created node
        return DAGReference(trf_hash)
    
    def add_nodes(self, *, select: dict=None, transform: Sequence[dict]=None):
        """Adds multiple nodes by parsing the specification given via the
        ``select`` and ``transform`` arguments.
        
        Args:
            select (dict, optional): Selection specifications, which are
                translated into regular transformations based on ``getitem``
                operations. The ``base_transform`` and ``select_base``
                arguments can be used to define from which object to select.
                By default, selection happens from the associated DataManager.
            transform (Sequence[dict], optional): Transform specifications.
        """
        # Parse the arguments and add multiple nodes from those specs
        specs = self._parse_trfs(select=select, transform=transform)
        if not specs:
            return
        
        for spec in specs:
            self.add_node(**spec)
    
    def compute(self, *, compute_only: Sequence[str]=None) -> Dict[str, Any]:
        """Computes all specified tags and returns a result dict.
        
        Args:
            compute_only (Sequence[str], optional): The tags to compute. If not
                given, will compute all associated tags.
            cache_options (dict, optional): Cache options. These will update
                the default cache options given at initialization of the DAG.
        
        Returns:
            Dict[str, Any]: A mapping from tags to fully computed results.
        """
        def postprocess_result(res, *, tag: str):
            """Performs some postprocessing operations on the results of
            individual tag computations.
            """
            # If the object is a detached dantro tree object, use the short
            # transformation hash for its name
            if isinstance(res, (AbstractDataContainer)) and res.parent is None:
                res.name = trf.hashstr[:SHORT_HASH_LENGTH]

            # Unwrap ObjectContainer; these are only meant for usage within the
            # data tree and it makes little sense to keep them in that form.
            if isinstance(res, ObjectContainer):
                res = res.data

            return res
        
        # Initiate start time for profiling
        t0 = time.time()

        # Determine which tags to compute
        compute_only = compute_only if compute_only is not None else 'all'
        if compute_only == 'all':
            compute_only = [t for t in self.tags.keys()
                            if t not in ['dm', 'dag']]

        log.info("Computation invoked on DAG with %d nodes.", len(self.nodes))

        # The results dict
        results = dict()

        if not compute_only:
            log.remark("No tags were selected to be computed. "
                       "Available tags:\n  %s", ", ".join(self.tags))
            return results

        # Compute and collect the results
        for tag in compute_only:
            log.remark("Computing tag '%s' ...", tag)
            # Resolve the transformation, then compute the result
            trf = self.objects[self.tags[tag]]
            res = trf.compute()

            # Postprocess it and store the result under its tag
            results[tag] = postprocess_result(res, tag=tag)

        # Update profiling information
        t1 = time.time()
        self._update_profile(compute=t1-t0)

        # Provide some information to the user
        log.note("Computed %d tag%s in %.2gs: %s",
                 len(compute_only), "s" if len(compute_only) != 1 else "",
                 t1-t0, ", ".join(results.keys()))

        prof_extd = self.profile_extended
        fstr = ("{name:>25s}   {p[mean]:<7s}  ±  {p[std]:<7s}   "
                "({p[min]:<7s} | {p[max]:<7s})")
        log.remark("Profiling results per node:  "
                   "mean ± std (min|max) [s]\n%s",
                   "\n".join([fstr.format(name=k, p={_k: "{:.2g}".format(_v)
                                                     for _k, _v in v.items()})
                              for k, v in prof_extd['aggregated'].items()
                              if k not in ('hashstr',)]))
        # TODO In the future, make this prettier; with proper time formatting!

        return results

    # .........................................................................
    # Helpers: Parsing transformation specifications

    def _parse_trfs(self, *, select: dict,
                    transform: Sequence[dict]) -> Sequence[dict]:
        """Parse the given arguments to bring them into a uniform format: a
        sequence of parameters for transformation operations.
        
        Args:
            select (dict): The shorthand to select certain objects from the
                DataManager. These may also include transformations.
            transform (Sequence[dict]): Actual transformation operations,
                carried out afterwards.
        
        Returns:
            Sequence[dict]: A sequence of transformation parameters that was
                brought into a uniform structure.
        """
        # The to-be-populated list of transformations
        trfs = list()

        # Prepare arguments: make sure they are dicts and deep copies.
        select = copy.deepcopy(select) if select else {}
        transform = copy.deepcopy(transform) if transform else []

        # First, parse the ``select`` argument. This contains a basic operation
        # to select data from the selection base (e.g. the DataManager) and
        # also allows to perform some operations on it.
        for tag, params in sorted(select.items()):
            if isinstance(params, str):
                path = params
                with_previous_result = False
                more_trfs = None
                salt = None
                omit_tag = False

            elif isinstance(params, dict):
                path = params['path']
                with_previous_result = params.get('with_previous_result',False)
                more_trfs = params.get('transform')
                salt = params.get('salt')
                omit_tag = params.get('omit_tag', False)

                if 'file_cache' in params:
                    raise ValueError("For selection from the selection base, "
                                     "the file cache is always disabled! "
                                     "The `file_cache` argument is thus not "
                                     "allowed; remove it from the selection "
                                     "for tag '{}'.".format(tag))

            else:
                raise TypeError("Invalid type for '{}' entry within `select` "
                                "argument! Got {} but expected string or dict."
                                "".format(tag, type(params)))

            # If given, process the path by prepending the prefix
            if self._select_path_prefix:
                if self._select_path_prefix[-1] == PATH_JOIN_CHAR:
                    path = self._select_path_prefix + path
                else:
                    path = PATH_JOIN_CHAR.join([self._select_path_prefix,path])

            # Construct parameters to select from the selection base.
            # Only assign a tag if there are no further transformations;
            # otherwise, the last additional transformation should set the tag.
            sel_trf = dict(operation='getitem',
                           tag=None if (more_trfs or omit_tag) else tag,
                           args=[self.select_base, path],
                           kwargs=dict(),
                           file_cache=dict(read=False, write=False))
            
            # Carry additional parameters only if given
            if salt is not None:
                sel_trf['salt'] = salt

            # Now finished with the formulation of the select operation.
            trfs.append(sel_trf)
            
            if not more_trfs:
                # Done with this select operation.
                continue

            # else: there are additional transformations to be parsed and added
            for i, trf_params in enumerate(more_trfs):
                trf_params = _parse_dag_minimal_syntax(trf_params)

                # Might have to use the previous result ...
                if 'with_previous_result' not in trf_params:
                    trf_params['with_previous_result'] = with_previous_result
                
                # Can now parse the regular syntax
                trf_params = _parse_dag_syntax(**trf_params)

                # If the tag is not to be omitted, the last transformation for
                # the selected tag needs to set the tag.
                if i+1 == len(more_trfs) and not omit_tag:
                    if trf_params.get('tag'):
                        raise ValueError("The tag of the last transform "
                                         "operation within a select routine "
                                         "cannot be set manually. Check the "
                                         "parameters for selection of tag "
                                         "'{}'.".format(tag))
                        # TODO Could actually allow multiple tags here ...
                    
                    # Add the tag to the parameters
                    trf_params['tag'] = tag

                # Can append it now
                trfs.append(trf_params)

        # Now, parse the normal `transform` argument. The operations defined
        # here are added after the instructions from the `select` section.
        for trf_params in transform:
            trf_params = _parse_dag_minimal_syntax(trf_params)
            trfs.append(_parse_dag_syntax(**trf_params))

        # Done parsing, yay.
        return trfs

    # .........................................................................
    # Helpers: Profiling

    def _update_profile(self, **times):
        """Updates profiling information by adding the given time to the
        matching key.
        """
        for key, t in times.items():
            self._profile[key] = self._profile.get(key, 0.) + t

    # .........................................................................
    # Cache writing and reading
    # NOTE This is done here rather than in Transformation because this is the
    #      more central entity and it is a bit easier ...

    def _retrieve_from_cache_file(self, trf_hash: str,
                                  **load_kwargs) -> Tuple[bool, Any]:
        """Retrieves a transformation's result from a cache file."""
        success, res = False, None
        cache_files = self.cache_files

        if trf_hash not in cache_files.keys():
            # Bad luck, no cache file
            return success, res

        # Parse load options
        if 'exists_action' not in load_kwargs:
            load_kwargs['exists_action'] = 'skip_nowarn'
            
        # else: There was a file. Let the DataManager load it.
        file_ext = cache_files[trf_hash]['ext']
        self.dm.load('dag_cache',
                     loader=LOADER_BY_FILE_EXT[file_ext[1:]],
                     base_path=self.cache_dir,
                     glob_str=trf_hash + file_ext,
                     target_path=DAG_CACHE_DM_PATH + "/{basename:}",
                     required=True,
                     **load_kwargs)
        # NOTE If a file was already loaded from the cache, it will not be
        #      loaded again. Thus, the DataManager acts as a persistent
        #      storage for loaded cache files. Consequently, these are shared
        #      among all TransformationDAG objects.
        
        # Retrieve from the DataManager
        res = self.dm[DAG_CACHE_DM_PATH][trf_hash]

        # Done.
        success = True
        return success, res

    def _write_to_cache_file(self, trf_hash: str, *, result: Any,
                             ignore_groups: bool=True,
                             attempt_pickling: bool=True,
                             raise_on_error: bool=False,
                             pkl_kwargs: dict=None,
                             **save_kwargs) -> bool:
        """Writes the given result object to a hash file, overwriting existing
        ones.
        
        Args:
            trf_hash (str): The hash; will be used for the file name
            result (Any): The result object to write as a cache file
        """        
        # Cannot store groups
        if isinstance(result, BaseDataGroup):
            if not ignore_groups:
                raise NotImplementedError("Cannot currently write dantro "
                    "groups to a cache file. Sorry. Adjust the ignore_groups "
                    "argument in the file cache write options for the "
                    "transformation resulting in {}.".format(result))
            return False

        # Make sure the directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Prepare the file path (still lacks an extension)
        fpath = os.path.join(self.cache_dir, trf_hash)

        # Go over the saving functions and see if the type agrees. If so, use
        # that function to write the data.
        for types, sfunc in DAG_CACHE_RESULT_SAVE_FUNCS.items():
            if not isinstance(result, types):
                continue

            # else: type matched, invoke saving function
            try:
                sfunc(result, fpath, **save_kwargs)

            except Exception as exc:
                msg = ("Failed saving transformation cache file for result of "
                       "type {} using storage function for type(s) {}. Value "
                       "of result: {}. Additional keyword arguments: {}"
                       "".format(type(result), types, result, save_kwargs))
                if raise_on_error:
                    raise RuntimeError(msg) from exc

                log.warning("%s. %s: %s", msg, exc.__class__.__name__, exc)
                # ... will attempt pickling below
            
            else:
                # Success
                return True

        # Reached the end of the loop without returning -> not saved yet
        if not attempt_pickling:
            return False
        
        # Try to pickle it
        try:
            with open(fpath + ".pkl", mode='wb') as pkl_file:
                pkl.dump(result, pkl_file,
                         **(pkl_kwargs if pkl_kwargs else {}))
        
        except Exception as exc:
            msg = ("Failed saving transformation cache file. Cannot pickle "
                   "result object of type {} and with value {}."
                   "".format(type(result), result))
            if raise_on_error:
                raise RuntimeError(msg) from exc
                
            log.warning("%s. %s: %s", msg, exc.__class__.__name__, exc)
            return False
        
        # else: Success
        return True

