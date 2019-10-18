"""Private low-level helper classes and functions for the DAG framework

NOTE This is imported by dantro.tools to register classes with YAML.
"""

from typing import NewType, TypeVar, Any

# Type definitions
THash = NewType('THash', str)
TDAGHashable = TypeVar('TDAGHashable', 'DataManager', 'Transformation')

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

    @property
    def ref(self) -> THash:
        """The associated reference of this object"""
        return self._data

    def _resolve_ref(self, *, dag: 'TransformationDAG') -> THash:
        """Return the hash reference; for the base class, the data is already
        the hash reference, so no DAG is needed. Derived classes _might_ need
        the DAG to resolve their reference hash.
        """
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

# -----------------------------------------------------------------------------

class DAGObjects:
    """An objects database for the DAG framework.

    It uses a flat dict containing (hash, object ref) pairs. The interface is
    slightly restricted compared to a regular dict; especially, item deletion
    is not made available.

    Objects are added to the database via the ``add_object`` method. They need
    to have a ``hashstr`` property, which returns a hash string
    deterministically representing the object; note that this is not
    equivalent to the Python builtin hash() function which invokes the __hash__
    magic method.
    """

    def __init__(self):
        """Initialize an empty objects database"""
        self._d = dict()

    def add_object(self, obj: TDAGHashable) -> THash:
        """Add an object to the object database, storing it under its hash.

        Note that the object cannot be just any object that is hashable but it
        needs to return a string-based hash via the ``hashstr`` property. This
        is a dantro DAG framework-internal interface.

        Also note that the object will NOT be added if an object with the same
        hash is already present. The object itself is of no importance, only
        the returned hash is.
        """
        key = obj.hashstr  # DAG-framework internal hash method

        # Only add the new object, if the hash does not exist yet.
        if key not in self:
            self._d[key] = obj
        return key

    def __getitem__(self, key: THash) -> TDAGHashable:
        """Return the object associated with the given hash"""
        return self._d[key]

    def __len__(self) -> int:
        """Returns the number of objects in the objects database"""
        return len(self._d)

    def __contains__(self, key: THash) -> bool:
        """Whether the given hash refers to an object in this database"""
        return key in self._d

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()
