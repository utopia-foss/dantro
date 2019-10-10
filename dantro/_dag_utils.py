"""Private low-level helper classes and functions for the DAG framework

NOTE This is imported by dantro.tools to register classes with YAML.
"""

from typing import NewType, Any

# Type definitions
THash = NewType('THash', str)

# -----------------------------------------------------------------------------

class DAGReference:
    """The DAGReference class is the base class of all DAG reference objects.

    While it does not implement __hash__ by itself, it is yaml-representable
    and thus hashable after a parent object created a YAML representation.
    """
    def __init__(self, ref: THash):
        """Initialize a DAGReference object from a hash."""
        if not isinstance(ref, str):
            raise TypeError("DAGReference requires a string-like argument, "
                            "got {}!".format(type(ref)))

        self._data = ref

    def __eq__(self, other) -> bool:
        """Only objects with exactly the same type and data are regarded as
        equal; specifically, this makes instances of subclasses always unequal
        to instances of the DAGReference base class.
        """
        if type(other) == type(self):
            return self._data == other._data
        return False

    def __repr__(self) -> str:
        return "<{} {}>".format(type(self).__name__, repr(self._data))

    def _resolve_ref(self, *, dag: 'TransformationDAG') -> THash:
        """Return the hash reference; for the base class, the data is already
        the hash reference"""
        return self._data

    def convert_to_ref(self, *, dag: 'TransformationDAG') -> 'DAGReference':
        """Create a new object that is a hash ref to the same object this
        tag refers to."""
        return DAGReference(self._resolve_ref(dag=dag))

    def resolve_object(self, *, dag: 'TransformationDAG') -> Any:
        """Resolve the object by looking up the reference in the DAG's object
        database.
        """
        return dag.objects[self._resolve_ref(dag=dag)]

    # YAML representation . . . . . . . . . . . . . . . . . . . . . . . . . . .
    yaml_tag = u'!dag_ref'

    @classmethod
    def from_yaml(cls, constructor, node):
        """Construct a DAGReference from a scalar YAML node"""
        return cls(constructor.construct_scalar(node))

    @classmethod
    def to_yaml(cls, representer, node):
        """Create a YAML representation of a DAGReference, carrying only the
        _data attribute over...
        
        As YAML expects scalar data to be str-like, a type cast is done. The
        subclasses that rely on certain argument types should take care that
        they can parse arguments that are str-like.
        """
        return representer.represent_scalar(cls.yaml_tag, str(node._data))


# .............................................................................

class DAGTag(DAGReference):
    """A DAGTag object stores a name of a tag, which serves as a named
    reference to some object in the DAG.

    While it does not implement __hash__ by itself, it is yaml-representable
    and thus hashable after a parent object created a YAML representation.
    """
    yaml_tag = u'!dag_tag'

    def __init__(self, name: str):
        """Initialize a DAGTag object, storing the specified field name"""
        self._data = name

    @property
    def name(self) -> str:
        """The name of the tag within the DAG that this object references"""
        return self._data

    def _resolve_ref(self, *, dag: 'TransformationDAG') -> THash:
        """Return the hash reference by looking up the tag in the DAG"""
        return dag.tags[self.name]

# .............................................................................

class DAGNode(DAGReference):
    """A DAGNode is a reference by the index within the DAG's node list.

    While it does not implement __hash__ by itself, it is yaml-representable
    and thus hashable after a parent object created a YAML representation.
    """
    yaml_tag = u'!dag_node'

    def __init__(self, idx: int):
        """Initialize a DAGNode object with a node index.
        
        Args:
            idx (int): The idx value to set this reference to. Can also be a
                negative value, in which case the node list is traversed from
                the back.
        
        Raises:
            TypeError: On invalid type (not int-convertible)
        """
        if not isinstance(idx, int):
            # Try an integer conversion, to be a bit more robust
            try:
                idx = int(idx)
            except:
                raise TypeError("DAGNode requires an int-convertible "
                                "argument, got {}!".format(type(idx)))

        self._data = idx

    @property
    def idx(self) -> int:
        """The idx to the referenced node within the DAG's node list"""
        return self._data

    def _resolve_ref(self, *, dag: 'TransformationDAG') -> THash:
        """Return the hash reference by looking up the node index in the DAG"""
        return dag.nodes[self.idx]
