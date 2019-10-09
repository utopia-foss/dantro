"""Private low-level helper classes and functions for the DAG framework

NOTE This is imported by dantro.tools and should not be imported by the
     dantro.utils.dag module itself!
"""

from typing import NewType, Union, Any

# Type definitions
THash = NewType('THash', str)

# -----------------------------------------------------------------------------

class DAGReference:
    """The DAGReference class is the base class of all DAG reference objects.

    It is yaml-representable and thus hashable after serialization.
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
        return cls(constructor.construct_scalar(node))

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_data(node.data)


# .............................................................................

class DAGTag(DAGReference):
    """A DAGTag object stores a name of a tag, which serves as a named
    reference to some object in the DAG.

    It is yaml-representable and thus hashable after serialization.
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

    It is yaml-representable and thus hashable after serialization.
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
