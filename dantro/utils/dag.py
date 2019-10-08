"""This is an implementation of a DAG for transformations on dantro objects"""

import collections
import hashlib
import logging
from typing import NewType, TypeVar
from typing import Dict, Tuple, Sequence, Any, Hashable, Callable, Union

from .ordereddict import KeyOrderedDict
from .data_ops import OPERATIONS, apply_operation
from ..tools import is_hashable

# Local constants
log = logging.getLogger(__name__)

# Type definitions
THash = NewType('THash', str)
TObjectMap = NewType('TObjectMap', Dict[THash, Hashable])

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

class Transformation(collections.abc.Hashable):
    """A transformation is the collection of an N-ary operation and its inputs.

    Transformation objects do not actually store any objects, but only their
    hashes, which then have to be looked up in the associated TransformationDAG
    object.
    """

    def __init__(self, *,
                 input_args: Sequence[THash],
                 input_kwargs: Dict[str, THash],
                 operation: THash):
        """
        Args:
            input_args (Sequence[THash]): Positional arguments for the
                operation.
            input_kwargs (Dict[str, THash]): Keyword arguments for the
                operation. These are internally stored as a KeyOrderedDict.
            operation (Operation): The operation that is to be carried out.
        """
        self._input_args = input_args
        self._input_kwargs = KeyOrderedDict(**input_kwargs)
        self._operation = operation
        self._result = None

        # TODO Make sure that all are hashable

    def __repr__(self) -> str:
        """Returns a deterministic string representation of this object"""
        return ("Transformation(input_args={}, input_kwargs={}, operation={})"
                "".format(repr(self._input_args),
                          repr(self._input_kwargs),
                          repr(self._operation)))

    def __hash__(self) -> str:
        """Uses the hashes of the inputs and the operation name to generate
        a new hash. Note that this does NOT rely on the built-in hash function
        but on a custom hash method which produces a deterministic and platform
        independent hash to be used as a checksum.
        """
        return _hash(repr(self))

    def _resolve_operation(self, *, dag: 'TransformationDAG') -> Callable:
        return OPERATIONS[dag.objects[self._operation]]

    def compute(self, *, dag: 'TransformationDAG') -> Any:
        """Computes the result of this transformation by recursively resolving
        objects and carrying out operations."""
        # Resolve the arguments and the operation objects
        args = [objects[h] for h in self._input_args]
        kwargs = {k:objects[h] for k, h in self._input_kwargs.items()}
        op = objects[self._operation]

        # If any of the objects is not fully resolved, do so.
        # TODO

        # Carry out the operation ...
        return op(*args, **kwargs)


class DAGObjects:
    """An objects database for the TransformationDAG class.

    It wraps a flat, key-ordered dict, containing (hash, object ref) pairs.
    """

    def __init__(self):
        self._d = KeyOrderedDict()

    def add_object(self, obj: Hashable) -> THash:
        """Add an object to the """
        key = _hash(obj)
        self._d[key] = obj
        return key

    def __getitem__(self, key: THash) -> Hashable:
        return self._d[key]


class TransformationDAG:
    """This class collects transformation operations that are (already by
    their own structure) connected into a directed acyclic graph. The aim of
    this class is to maintain base objects, manage references, and allow
    operations on the DAG, the most central of which is computing the result
    of a node.

    Furthermore, this class can also implement caching of transformations.
    """

    def __init__(self, *, dm: 'DataManager',
                 select: dict=None, transform: dict=None,
                 compute_fields: Union[str, Sequence[str]]='all'):
        """Initialize a DAG which is associated with a DataManager and load the
        specified transformations configuration into it.
        """
        self._dm = dm
        self._fields = dict()
        self._objects = DAGObjects()

        self._compute_fields = compute_fields
        self._trfs = self._parse_trfs(select=select, transform=transform)

        self._build_dag()

    def _parse_trfs(self, *, select: dict, transform: dict) -> Sequence[dict]:
        """Parse the given arguments to bring them into a uniform format: a
        sequence of parameters for transformation operations.
        
        Args:
            select (dict): The shorthand to select certain objects from the
                DataManager. These may also include transformations.
            transform (dict): Actual transformation operations, carried out
                afterwards.
        
        Returns:
            Sequence[dict]: Description
        """
        trfs = 

        # TODO Do further parsing here ...

        return d

    def _build_dag(self) -> None:
        """Builds the actual directed acyclic graph using the information
        from the configuration.
        """
        pass
        # TODO
    
    # .........................................................................

    @property
    def dm(self) -> 'DataManager':
        """The associated DataManager"""
        return self._dm

    @property
    def fields(self) -> Dict[str, THash]:
        """A mapping from field names to object hashes"""
        return self._fields

    @property
    def objects(self) -> DAGObjects:
        """The object database"""
        return self._objects

    # .........................................................................
    
    def compute(self, *, compute_only: Sequence[str]=None) -> Dict[str, Any]:
        """Computes all specified fields."""
        raise NotImplementedError()
