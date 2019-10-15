"""This is an implementation of a DAG for transformations on dantro objects"""

import os
import sys
import glob
import copy
import time
import logging
import pickle as pkl
from itertools import chain

from typing import TypeVar, Dict, Tuple, Sequence, Any, Union, List

import numpy as np
import xarray as xr

from .abc import AbstractDataContainer
from .base import BaseDataGroup
from .utils import KeyOrderedDict, OPERATIONS, apply_operation
from .tools import recursive_update
from .data_loaders import LOADER_BY_FILE_EXT
from .containers import ObjectContainer, NumpyDataContainer, XrDataContainer
from ._dag_utils import THash, DAGObjects, DAGReference, DAGTag, DAGNode
from ._hash import _hash, SHORT_HASH_LENGTH, FULL_HASH_LENGTH
from ._yaml import yaml_dumps as _serialize

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

    For computation, the arguments are inspected for whether there are any
    DAGReference-derived objects; these need to be resolved first, meaning they
    are looked up in the DAG's object database and -- if they are another
    Transformation object -- their result is computed. This can lead to a
    recursion.
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
        """
        # Storage attributes
        self._operation = operation
        self._args = args
        self._kwargs = KeyOrderedDict(**kwargs)
        self._dag = dag
        self._salt = salt

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

        # Profiling info from the last computation
        self._profile = dict()

        # Cache dict, containing the result and whether the cache is in memory
        self._cache = dict(result=None, filled=False)

    # .........................................................................
    # Properties

    @property
    def dag(self) -> 'TransformationDAG':
        """The associated TransformationDAG; used for object lookup"""
        return self._dag

    @property
    def hashstr(self) -> str:
        """Computes the hash of this Transformation by serializing itself into
        a YAML string which is then hashed.

        Note that this does NOT rely on the built-in hash function but on the
        custom ``_hash`` function which produces a platform-independent and
        deterministic hash.

        As this is a string-based hash, it is not implemented as the __hash__
        magic method but as a separate property.
        """
        dag_classes = (DAGNode, DAGReference, DAGTag, Transformation,)
        serialization_params = dict(canonical=True)
        # WARNING Changing the above leads to cache invalidations!

        return _hash(_serialize(self, register_classes=dag_classes,
                                **serialization_params))

    @property
    def dependencies(self) -> List[THash]:
        """Hashes of those objects that this transformation depends on, i.e.
        hashes of other referenced DAG nodes.
        """
        return [r.ref for r in chain(self._args, self._kwargs.values())
                if isinstance(r, DAGReference)]

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
        arguments (which must again be YAML-representable) ...

        WARNING Changing the argument order here or adding further keys to the
                dict will lead to hash changes and thus to cache misses.
        """
        # Collect the 
        d = dict(operation=node._operation,
                 args=node._args,
                 kwargs=dict(node._kwargs))
        
        # If a specific salt was given, add that to the dict
        if node._salt is not None:
            d['salt'] = node._salt

        # Can now represent it ...
        return representer.represent_mapping(cls.yaml_tag, d)

    # .........................................................................
    # Compute interface

    def compute(self) -> Any:
        """Computes the result of this transformation by recursively resolving
        objects and carrying out operations.
        
        This method can also be called if the result is already computed; this
        will lead only to a cache-lookup, not a re-computation.
        
        Args:
            cache_options (dict, optional): Can be used to configure the
                behaviour of the cache.
        
        Returns:
            Any: The result of the operation
        """
        def resolve(element):
            """Resolve references to their objects, if necessary computing the
            results of referenced transformations recursively.

            Makes use of arguments from outer scope.
            """
            if not isinstance(element, DAGReference):
                # Is an argument. Nothing to do, return it as it is.
                return element

            elif self.dag is None:
                raise ValueError("Cannot resolve Transformation arguments "
                                 "that contain DAG references, because no DAG "
                                 "was associated with this Transformation!")

            # Is a reference; let it resolve the corresponding object from DAG
            obj = element.resolve_object(dag=self.dag)
            
            # Check if this refers to the DataManager, which cannot perform any
            # further computation ...
            if obj is self.dag.dm:
                return self.dag.dm

            # else: wasn't the DataManager. It should thus be another
            # Transformation object. Compute, passing on the cache options.
            # This is a traversal up the DAG tree.
            return obj.compute()

        # Return the already computed and cached result, if possible
        success, res = self._lookup_result()
        if success:
            return res

        # else: could not read the result from a cache -> Need to compute it.
        # First, resolve the references in the arguments
        args = [resolve(e) for e in self._args]
        kwargs = {k:resolve(e) for k, e in self._kwargs.items()}

        # Carry out the operation
        res = self._perform_operation(args=args, kwargs=kwargs)

        # Allow caching the result
        self._cache_result(res)

        return res

    def _perform_operation(self, *, args, kwargs) -> Any:
        """Perform the operation, updating the profiling info on the side"""
        # Initialize a dict for profiling info
        prof = dict()

        # Set up profiling
        t0 = time.time()

        # Actually perform the operation
        res = apply_operation(self._operation, *args, **kwargs,
                              _maintain_container_type=True)

        # Prase profiling info and return the result
        self._update_profile(cumulative_compute=(time.time() - t0))

        return res

    def _update_profile(self, **times) -> None:
        """Given some time, updates the profiling information
        
        Args:
            **times: Valid profiling data.
        """
        self._profile.update(times)
        # TODO do some more processing here, e.g. calculating the time it took
        #      for this specific transformation (removing the time taken by
        #      the depending transformations)
        # TODO calculate the `compute` key

    # Cache handling . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _lookup_result(self) -> Tuple[bool, Any]:
        """Look up the transformation result to spare re-computation"""
        success, res = False, None

        # Retrieve cache parameters
        read_opts = self._fc_opts.get('read', {})

        # Setup profiling
        t0 = time.time()
        
        # Check if the cache is already filled. If not, see if the file cache
        # can be read and is configured to be read.
        if self._cache['filled']:
            success = True
            res = self._cache['result']

        elif self.dag is not None and read_opts.get('enabled', False):
            # Let the DAG check if there is a file cache, i.e. if a file with
            # this Transformation's hash exists in the DAG's cache directory.
            success, res = self.dag._retrieve_from_cache_file(self.hashstr)

            # Store the result
            if success:
                self._cache['result'] = res
                self._cache['filled'] = True

        self._update_profile(cache_lookup=(time.time() - t0))

        return success, res
    
    def _cache_result(self, result: Any) -> None:
        """Stores a computed result in the cache"""
        def should_write(*, enabled: bool, always: bool=False,
                         overwrite: bool=False,
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
                overwrite (bool, optional): If False, will not write a cache
                    file if one already exists. If True, a cache file might be
                    written, although one already exists.
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

            # If overwriting is disabled, check if a cache file already exists
            if not overwrite and self.hashstr in self.dag.cache_files:
                return False

            # Evaluate profiling information
            if min_compute_time is not None:
                if self.profile['compute'] <= min_compute_time:
                    return False

            if min_cumulative_compute_time is not None:
                if (   self.profile['cumulative_compute']
                    <= min_cumulative_compute_time):
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

        # Start profiling
        t0 = time.time()

        # Determine whether to write to a file
        if self.dag is not None and should_write(**write_opts):
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
                 cache_dir: str='.cache', file_cache_defaults: dict=None):
        """Initialize a DAG which is associated with a DataManager and load the
        specified transformations configuration into it.
        """
        self._dm = dm

        self._objects = DAGObjects()
        self._tags = dict()
        self._nodes = list()

        self._trfs = self._parse_trfs(select=select, transform=transform)

        # Add the DataManager as an object and a default tag
        self.tags['dm'] = self.objects.add_object(self.dm)
        # NOTE The data manager is NOT a node of the DAG, but more like an
        #      external data source, thus being accessible only as a tag

        # Default file cache options
        self._fc_opts = file_cache_defaults if file_cache_defaults else {}
        
        # Determine cache directory path; relative path interpreted as relative
        # to the DataManager's data directory
        if os.path.isabs(cache_dir):
            self._cache_dir = cache_dir
        else:
            self._cache_dir = os.path.join(self.dm.dirs['data'], cache_dir)

        # Build the DAG by subsequently adding nodes from the parsed parameters
        for trf_params in self._trfs:
            self._add_node(**trf_params)

    # .........................................................................

    @property
    def dm(self) -> 'DataManager':
        """The associated DataManager"""
        return self._dm

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
        for path in glob.glob(os.path.join(self._cache_dir, '*.*')):
            if not os.path.isfile(path):
                continue

            # Get filename and extension, then check if it is a hash
            fname, ext = os.path.splitext(os.path.basename(path))
            if len(fname) != FULL_HASH_LENGTH:
                continue

            # else: filename is assumed to be the hash. Store info.
            info[fname] = dict(full_path=path, ext=ext)

        return info

    # .........................................................................

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
        def parse_minimal_syntax(params: Union[str, dict]) -> dict:
            """Parses the minimal syntax"""
            if isinstance(params, dict):
                return params

            elif isinstance(params, str):
                return dict(operation=params, carry_result=True)

            # else:
            raise TypeError("Expected either dict or string for minimal "
                            "syntax, got {} with value: {}"
                            "".format(type(params), params))


        def parse_params(*, operation: str=None,
                         args: list=None, kwargs: dict=None,
                         tag: str=None, carry_result: bool=False,
                         salt: int=None, file_cache: dict=None,
                         **ops) -> dict:
            """Given the parameters of a transform operation, possibly in a
            shorthand notation, returns a dict with normalized content by
            expanding the shorthand notation.
            
            Keys that will remain in the resulting dict:
                ``operation``, ``args``, ``kwargs``, ``tag``.
            
            Args:
                operation (str, optional): Which operation to carry out;
                    can only be specified if there is no ``ops`` argument.
                args (list, optional): Positional arguments for the operation;
                    can only be specified if there is no ``ops`` argument.
                kwargs (dict, optional): Keyword arguments for the operation;
                    can only be specified if there is no ``ops`` argument.
                tag (str, optional): The tag to attach to this transformation
                carry_result (bool, optional): Whether the result is to be
                    carried through from the previous operation; if true, the
                    first positional argument is set to a reference to the
                    previous node's result.
                salt (int, optional): A salt to the Transformation object,
                    thereby changing its hash.
                file_cache (dict, optional): File cache parameters
                **ops: The operation that is to be carried out. May contain
                    one and only one operation.
            
            Returns:
                dict: The normalized dict of transform parameters.
            
            Raises:
                ValueError: For len(ops) != 1
            """
            # Distinguish between explicit and shorthand mode
            if operation and not ops:
                # Explicit parametrization
                args = args if args else []
                kwargs = kwargs if kwargs else {}

            elif ops and not operation:
                # Shorthand parametrization
                # Make sure there are no stray argument
                if args is not None or kwargs is not None:
                    raise ValueError("When using shorthand notation, the args "
                                     "and kwargs need to be specified under "
                                     "the key that specifies the operation!")

                elif len(ops) > 1:
                    raise ValueError("For shorthand notation, there can only "
                                     "be a single operation specified, but "
                                     "got: {}.".format(ops))

                # Extract operation name and parameters
                operation, op_params = list(ops.items())[0]

                # Depending on type, regard parameters as args or kwargs. If
                # the argument is not a container, assume it's a single
                # positional argument.
                if isinstance(op_params, dict):
                    args, kwargs = [], op_params
                
                elif isinstance(op_params, (list, tuple)):
                    args, kwargs = list(op_params), {}
                
                elif op_params is not None:
                    args, kwargs = [op_params], {}
                
                else:
                    args, kwargs = [], {}

            elif not operation and not ops:
                raise ValueError("Missing operation specification. Either use "
                                 "the `operation` key to specify one or use "
                                 "shorthand notation by using the name of the "
                                 "operation as a key and adding the arguments "
                                 "to it as values.")

            else:
                raise ValueError("Got two specifications of operations, one "
                                 "via the `operation` argument ({}), another "
                                 "via the shorthand notation ({}). Remove "
                                 "one of them.".format(operation, ops))

            # Have variables operation, args, and kwargs set now.

            # If the result is to be carried on, the first _positional_
            # argument is set to be a reference to the previous node
            if carry_result:
                args.insert(0, DAGNode(-1))

            # Done. Construct the dict.
            # Mandatory parameters
            d = dict(operation=operation, args=args, kwargs=kwargs, tag=tag)
            
            # Add optional parameters only if they were specified
            if salt is not None:
                d['salt'] = salt

            if file_cache is not None:
                d['file_cache'] = file_cache

            return d

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # The to-be-populated list of transformations
        trfs = list()

        # Prepare arguments: make sure they are dicts and deep copies.
        select = copy.deepcopy(select) if select else {}
        transform = copy.deepcopy(transform) if transform else []

        # First, parse the ``select`` argument. This contains a basic operation
        # to select data from the DataManager and also allows to perform some
        # operations on it.
        for field_name, params in sorted(select.items()):
            if isinstance(params, str):
                path = params
                carry_result = False
                more_trfs = None
                salt = None

            elif isinstance(params, dict):
                path = params['path']
                carry_result = params.get('carry_result', False)
                more_trfs = params.get('transform')
                salt = params.get('salt')

                if 'file_cache' in params:
                    raise ValueError("For selection from DataManager, the "
                                     "file cache is always disabled. Please "
                                     "remove the file_cache argument.")

            else:
                raise TypeError("Invalid type for '{}' entry within `select` "
                                "argument! Got {} but expected string or dict."
                                "".format(field_name, type(params)))

            # Construct parameters to select from the DataManager. Only assign
            # a tag if there are no further transformations; otherwise, the
            # last additional transformation should set the tag.
            sel_trf = dict(operation='getitem',
                           tag=field_name if not more_trfs else None,
                           args=[DAGTag('dm'), path],
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
                # Parse it
                trf_params = parse_minimal_syntax(trf_params)

                if 'carry_result' not in trf_params:
                    trf_params['carry_result'] = carry_result
                
                trf_params = parse_params(**trf_params)

                # On the last transformation for the selected tag, need to set
                # the tag to the specified field name. This is to avoid
                # confusion about the result of the select operation.
                if i+1 == len(more_trfs):
                    if trf_params.get('tag'):
                        raise ValueError("The tag of the last transform "
                                         "operation within a select routine "
                                         "cannot be set manually. Check the "
                                         "parameters for selection of tag "
                                         "'{}'.".format(field_name))
                    
                    # Set it to the field name
                    trf_params['tag'] = field_name

                # Done, can append it now
                trfs.append(trf_params)

        # Now, parse the normal ``transform`` argument. The operations defined
        # here are added after all the instructions from the ``select`` section
        # above. Furthermore, the ``carry_result`` feature is not available
        # here but everything has to be specified explicitly.
        for trf_params in transform:
            trf_params = parse_minimal_syntax(trf_params)
            trfs.append(parse_params(**trf_params))

        # Done parsing, yay.
        return trfs
    
    def _add_node(self, *, operation: str, args: list=None, kwargs: dict=None,
                  tag: str=None, file_cache: dict=None,
                  **trf_kwargs) -> THash:
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
        """
        def is_ref_subclass_instance(obj: DAGReference) -> bool:
            """True if obj is an instance of a true subclass of DAGReference"""
            return (    isinstance(obj, DAGReference)
                    and type(obj) is not DAGReference)

        # Handle default values of arguments if they are not given
        args = args if args else []
        kwargs = kwargs if kwargs else {}

        # Resolve any derived references to proper hash references
        args = [(v if not is_ref_subclass_instance(v)
                 else v.convert_to_ref(dag=self))
                for v in args]
        kwargs = {k: (v if not is_ref_subclass_instance(v)
                      else v.convert_to_ref(dag=self))
                  for k,v in kwargs.items()}

        # Parse file cache parameters
        fc_opts = copy.deepcopy(self._fc_opts)  # Always a dict
        
        if file_cache is not None:
            fc_opts = recursive_update(fc_opts, file_cache)

        # From these arguments, create the Transformation object and add it to
        # the objects database.
        trf_hash = self.objects.add_object(Transformation(operation=operation,
                                                          args=args,
                                                          kwargs=kwargs,
                                                          dag=self,
                                                          file_cache=fc_opts,
                                                          **trf_kwargs))
        # NOTE From this point on, the object itself has no relevance; that is
        #      why it is not stored here. Transformation objects should only
        #      be handled via their hash in order to reduce duplicate
        #      calculations and make efficient caching possible.

        # Store the hash in the node list
        self.nodes.append(trf_hash)

        # If a tag was specified, create a tag
        if tag:
            if tag in self.tags.keys():
                raise ValueError("Tag '{}' already exists! Choose a different "
                                 "one. Already in use: {}"
                                 "".format(tag, ", ".join(self.tags.keys())))
            self.tags[tag] = trf_hash
    
    # .........................................................................
    
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
        # Determine which tags to compute
        compute_only = compute_only if compute_only is not None else 'all'
        if compute_only == 'all':
            compute_only = [t for t in self.tags.keys() if t != 'dm']

        log.info("Tags to be computed: {}".format(", ".join(compute_only)))

        # Compute and collect the results
        results = dict()
        for tag in compute_only:
            # Resolve the transformation, then compute the result
            trf = self.objects[self.tags[tag]]
            res = trf.compute()

            # If the object is a detached dantro tree object, use the short
            # transformation hash for its name
            if isinstance(res, (AbstractDataContainer)) and res.parent is None:
                res.name = trf.hashstr[:SHORT_HASH_LENGTH]

            # Unwrap ObjectContainer; these are only meant for usage within the
            # data tree and it makes little sense to keep them in that form.
            if isinstance(res, ObjectContainer):
                res = res.data

            # Store the result under its tag
            results[tag] = res

        return results

    # .........................................................................
    # Cache writing and reading
    # NOTE This is done here rather than in Transformation because this is the
    #      more central entity and it is a bit easier ...

    def _retrieve_from_cache_file(self, trf_hash: str) -> Tuple[bool, Any]:
        """Retrieves a transformation's result from a cache file."""
        success, res = False, None
        cache_files = self.cache_files

        if trf_hash not in cache_files.keys():
            # Bad luck, no cache file
            return success, res
            
        # else: There was a file. Let the DataManager load it.
        file_ext = cache_files[trf_hash]['ext']
        self.dm.load('dag_cache',
                     loader=LOADER_BY_FILE_EXT[file_ext[1:]],
                     base_path=self.cache_dir,
                     glob_str=trf_hash + file_ext,
                     target_path=DAG_CACHE_DM_PATH + "/{basename:}",
                     required=True)
        
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
