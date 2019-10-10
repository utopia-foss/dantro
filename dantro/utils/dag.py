"""This is an implementation of a DAG for transformations on dantro objects"""

import copy
import collections
import hashlib
import logging

from typing import NewType, TypeVar
from typing import Dict, Tuple, Sequence, Any, Hashable, Callable, Union, List

from .ordereddict import KeyOrderedDict
from .data_ops import OPERATIONS, apply_operation
from .link import Link
from .._dag_utils import THash, DAGReference, DAGTag, DAGNode
from ..base import BaseDataGroup, BaseDataContainer
from ..containers import LinkContainer, ObjectContainer
from ..tools import is_hashable

# Local constants
log = logging.getLogger(__name__)

# Type definitions (extending those from _dag_utils module)
TObjectMap = NewType('TObjectMap', Dict[THash, Hashable])
TRefOrAny = TypeVar('TRefOrAny', DAGReference, Any)

# -----------------------------------------------------------------------------

def _serialize(obj) -> str:
    """Serializes the given object using YAML"""
    try:
        return repr(obj) # FIXME Implement YAML serialization
    
    except Exception as err:
        raise ValueError("Could not serialize the given {} object!"
                         "".format(type(obj))) from err

def _hash(s: Union[str, Any]) -> str:
    """Returns a deterministic hash of the given string or object. If the
    given object is not already a string, it is tried to be serialized.

    This uses the hashlib.md5 algorithm which returns a 32 character string.
    Note that the hash is used for a checksum, not for security purposes.
    """
    if not isinstance(s, str):
        s = _serialize(s)
    return hashlib.md5(s.encode('utf-8')).hexdigest()

# -----------------------------------------------------------------------------

class DAGObjects:
    """An objects database for the DAG framework.

    It wraps a flat, key-ordered dict, containing (hash, object ref) pairs.
    """

    def __init__(self):
        """Initialize an empty objects database"""
        self._d = KeyOrderedDict()

    def add_object(self, obj: Hashable) -> THash:
        """Add an object to the object database, storing it under its hash."""
        key = _hash(obj)
        self._d[key] = obj
        return key

    def __getitem__(self, key: THash) -> Hashable:
        """Return the object associated with the given hash"""
        return self._d[key]

    def get(self, key: THash, default=None) -> Union[Hashable, None]:
        """Return the object associated with the given hash, if it is in the
        object database, else return default.
        """
        return self._d.get(key, default=default)

    def __len__(self) -> int:
        """Returns the number of objects in the objects database"""
        return len(self._d)

# -----------------------------------------------------------------------------

class Transformation(collections.abc.Hashable):
    """A transformation is the collection of an N-ary operation and its inputs.

    Transformation objects do not actually store any objects, but only their
    hashes, which then have to be looked up in the associated TransformationDAG
    object.
    """

    def __init__(self, *, operation: str,
                 args: Sequence[TRefOrAny], kwargs: Dict[str, TRefOrAny]):
        """
        Args:
            operation (str): The operation that is to be carried out.
            args (Sequence[TRefOrAny]): Positional arguments for the
                operation.
            kwargs (Dict[str, TRefOrAny]): Keyword arguments for the
                operation. These are internally stored as a KeyOrderedDict.
        """
        # Storage attributes
        self._operation = operation
        self._args = args
        self._kwargs = KeyOrderedDict(**kwargs)

        # A cache attribute for the result of the computation
        self._result = None

    def __repr__(self) -> str:
        """Returns a string representation of this object by combining the 
        representations of the stored objects."""
        return ("Transformation(operation={}, args={}, kwargs={})"
                "".format(repr(self._operation),
                          repr(self._args), repr(self._kwargs)))

    def __hash__(self) -> str:
        """Uses the hashes of the inputs and the operation name to generate
        a new hash. Note that this does NOT rely on the built-in hash function
        but on a custom hash method which produces a deterministic and platform
        independent hash to be used as a checksum.
        """
        return _hash(repr(self))

    @property
    def result(self) -> Any:
        """Return the result of this transformation. Will be None if it was
        not yet computed.
        """
        return self._result

    def compute(self, *, dag: 'TransformationDAG') -> Any:
        """Computes the result of this transformation by recursively resolving
        objects and carrying out operations.
        
        Args:
            dag (TransformationDAG): The associated DAG
        
        Returns:
            Any: The result of the operation
        """
        def resolve(element, *, dag):
            """Resolve references to their objects"""
            if not isinstance(element, DAGReference):
                # Is an argument. Nothing to do, return it as it is.
                return element

            # Is a reference; resolve the corresponding object
            obj = element.resolve_object(dag=dag)
            
            # Check if this refers to the DataManager, which cannot perform any
            # further computation ...
            if obj is dag.dm:
                return dag.dm

            # Wasn't the DataManager; should now be a Transformation object
            if not isinstance(obj, Transformation):
                raise TypeError("Unexpected object of type {}!"
                                "".format(type(obj)))

            # Compute the result of the referenced transformation
            return obj.compute(dag=dag)

        # Return the already computed result, if available
        if self.result is not None:
            return self.result

        # Not available, compute it.
        # First, need to resolve the arguments. If any of the objects is not
        # fully resolved, do so. This enters the recursion ...
        args = [resolve(e, dag=dag) for e in self._args]
        kwargs = {k:resolve(e, dag=dag) for k, e in self._kwargs.items()}

        # Carry out the operation and store the result for next time
        self._result = apply_operation(self._operation, *args, **kwargs,
                                       _maintain_container_type=True)
        return self.result

# -----------------------------------------------------------------------------

class TransformationDAG:
    """This class collects transformation operations that are (already by
    their own structure) connected into a directed acyclic graph. The aim of
    this class is to maintain base objects, manage references, and allow
    operations on the DAG, the most central of which is computing the result
    of a node.

    Furthermore, this class can also implement caching of transformations.

    Objects of this class are initialized with dict-like arguments which
    specify the transformation operations. There are some shorthands that allow
    a simple 
    """

    def __init__(self, *, dm: 'DataManager',
                 select: dict=None, transform: Sequence[dict]=None,
                 compute_tags: Union[str, Sequence[str]]='all'):
        """Initialize a DAG which is associated with a DataManager and load the
        specified transformations configuration into it.
        """
        self._dm = dm

        self._objects = DAGObjects()
        self._tags = dict()
        self._nodes = list()

        self._compute_tags = compute_tags
        self._trfs = self._parse_trfs(select=select, transform=transform)

        # Add the DataManager as an object and a default field
        self.tags['dm'] = self.objects.add_object(self.dm)
        # NOTE The data manager is NOT a node of the DAG, but more like an
        #      external data source, thus being accessible only as a field

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
        """A mapping from field names to object hashes"""
        return self._tags

    @property
    def nodes(self) -> List[THash]:
        """The nodes of the DAG"""
        return self._nodes

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

            elif not isinstance(params, str):
                raise TypeError("Expected either dict or string for minimal "
                                "syntax, got {} with value: {}"
                                "".format(type(params), params))

            return dict(operation=params, carry_result=True)

        def parse_params(*, operation: str=None,
                         args: list=None, kwargs: dict=None,
                         tag: str=None,  carry_result: bool=False,
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
                    carried through to the next operation; if true, the first
                    positional argument is set to a reference to the previous
                    node.
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
                if 'args' in ops or 'kwargs' in ops:
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

            elif bool(operation) == bool(ops):
                # Both or none were given
                raise ValueError("Bad parameter format! Need either explicit "
                                 "or shorthand arguments, but got neither or "
                                 "both!")

            # Have variables operation, args, and kwargs set now.

            # If the result is to be carried on, the first _positional_
            # argument is set to be a reference to the previous node
            if carry_result:
                args.insert(0, DAGNode(-1))

            # Done. Construct the dict.
            return dict(operation=operation, args=args, kwargs=kwargs, tag=tag)

        # The to-be-populated list of transformations
        trfs = list()

        # Parse the arguments to assert that they are not None and deep copies
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

            elif isinstance(params, dict):
                path = params['path']
                carry_result = params.get('carry_result', False)
                more_trfs = params.get('transform')

            else:
                raise TypeError("Invalid type for '{}' entry within `select` "
                                "argument! Got {} but expected string or dict."
                                "".format(field_name, type(params)))

            # Construct parameters to select from the DataManager
            trfs.append(dict(operation='getitem',
                             tag=field_name if not more_trfs else None,
                             args=[DAGTag('dm'), path],
                             kwargs=dict()))
            # NOTE If more transformations are to occur on this element, the
            #      tag of this first transformation need be None.
            
            if not more_trfs:
                # Done with this select operation.
                continue

            # else: there are additional transformations to be parsed and added
            for i, trf_params in enumerate(more_trfs):
                # Parse it
                trf_params = parse_minimal_syntax(trf_params)
                trf_params = parse_params(**trf_params,
                                          carry_result=carry_result)

                # On the last transformation for the selected tag, need to set
                # the tag to the specified field name. This is to avoid
                # confusion about the result of the select operation.
                if i+1 == len(more_trfs):
                    if trf_params.get('tag'):
                        raise ValueError("The tag of the last transform "
                                         "operation within a select routine "
                                         "cannot be set manually.")
                    
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
    
    def _add_node(self, *, operation: str,
                  args: list=None, kwargs: dict=None, tag: str=None) -> THash:
        """Add a new node by creating a new Transformation object and adding it
        to the node list.
        """
        def is_ref_subclass_instance(obj: DAGReference) -> bool:
            """True if obj is an instance of a true subclass of DAGReference"""
            return (    isinstance(obj, DAGReference)
                    and type(obj) is not DAGReference)

        # Handle default values of arguments if they are not given
        args = args if args else []
        kwargs = kwargs if kwargs else {}

        # Resolve any derived references to proper hash references
        args =   [(v if not is_ref_subclass_instance(v)
                   else v.convert_to_ref(dag=self))
                  for v in args]
        kwargs = {k: (v if not is_ref_subclass_instance(v)
                      else v.convert_to_ref(dag=self))
                  for k,v in kwargs.items()}

        # From these arguments, create the Transformation object
        trf = Transformation(operation=operation, args=args, kwargs=kwargs)

        # Store the object in the object database
        trf_hash = self.objects.add_object(trf)

        # Store the hash in the node list
        self.nodes.append(trf_hash)

        # If a tag was specified, create a tag
        if tag:
            if tag in self.tags.keys():
                raise ValueError("Tag '{}' already exists!".format(tag))
            self.tags[tag] = trf_hash
    
    # .........................................................................
    
    def compute(self, *, compute_only: Sequence[str]=None,
                ResultCls: BaseDataGroup=None) -> BaseDataGroup:
        """Computes all specified tags."""
        # Determine which tags to compute
        if compute_only:
            to_compute = compute_only
        elif self._compute_tags == 'all':
            to_compute = [t for t in self.tags.keys() if t != 'dm']
        else:
            to_compute = self._compute_tags

        log.info("Tags to be computed: {}".format(", ".join(to_compute)))

        # Determine which group class to use for storing results
        if ResultCls is None:
            ResultCls = self.dm._DATA_GROUP_DEFAULT_CLS

        # Compute the results and return them
        results = ResultCls(name='TransformationResults') # TODO name
        for tag in to_compute:
            result = self.objects[self.tags[tag]].compute(dag=self)

            # Handle dantro objects and non-dantro objects in different ways
            if isinstance(result, (BaseDataGroup, BaseDataContainer)):
                # If the resulting object is attached somewhere, add a link;
                # otherwise, rename it and add it directly
                if result.parent is not None:
                    # Compute the relative path from the DataManager to the
                    # result object. Note: DM != root of path system
                    rel_path = result.path[len(self.dm.path) + 1:]

                    # Create and add the link to the results group
                    link = Link(anchor=self.dm, rel_path=rel_path)
                    results.add(LinkContainer(name=tag, data=link))

                else:
                    result._name = tag  # FIXME internal API usage
                    results.add(result)

            else:
                # Create an ObjectContainer
                # TODO use more specific containers, if available
                results.add(ObjectContainer(name=tag, data=result))

        return results
