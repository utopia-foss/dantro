"""Private low-level helper classes and functions for the DAG framework

NOTE This is imported by dantro.tools and should not be imported by the
     dantro.utils.dag module itself!
"""

from typing import NewType, Union

# Type definitions
THash = NewType('THash', str)

# -----------------------------------------------------------------------------

class DAGObject:
    """The DAGObject class is the base class of DAG-related reference objects.

    It is yaml-representable and thus hashable after serialization.
    """
    def __init__(self, data):
        """Initialize a DAGObject object and store the given data"""
        self._data = data

    @property
    def data(self):
        return self._data

    def __eq__(self, other) -> bool:
        if isinstance(other, DAGObject):
            return self.data == other.data
        return False

    def __repr__(self) -> str:
        return "<{} {}>".format(type(self).__name__, repr(self.data))

    # YAML representation . . . . . . . . . . . . . . . . . . . . . . . . . . .
    yaml_tag = u'!dag_obj'

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(constructor.construct_scalar(node))

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_data(node.data)


# .............................................................................

class DAGField(DAGObject):
    """A DAGField object stores the name of a field, which serves as a named
    reference to some object in the DAG.

    It is yaml-representable and thus hashable after serialization.
    """
    yaml_tag = u'!field'

    def __init__(self, field_name: str):
        """Initialize a DAGField object, storing the specified field name"""
        super().__init__(field_name)

    @property
    def name(self) -> str:
        """The name of the field within the DAG that this object references"""
        return self._data

# .............................................................................

class DAGNode(DAGObject):
    """A DAGNode is a relative reference that uses an offset within the
    DAG's node list to specify a reference.

    It is yaml-representable and thus hashable after serialization.
    """
    yaml_tag = u'!dag_node'

    def __init__(self, offset: int=-1):
        """Initialize a DAGNode object from an offset value."""
        if not isinstance(offset, int):
            try:
                offset = int(offset)
            except:
                raise TypeError("DAGNode requires an int-like argument, got "
                                "{}!".format(type(offset)))

        super().__init__(offset)

    @property
    def offset(self) -> THash:
        """The offset to the referenced node"""
        return self._data

# .............................................................................

class DAGReference(DAGObject):
    """A DAGReference object stores the hash of a DAG object in order and is a
    reference to any object inside the DAG.

    It is yaml-representable and thus hashable after serialization.
    """
    yaml_tag = u'!dag_ref'

    def __init__(self, ref: THash):
        """Initialize a DAGReference object from a hash."""
        if not isinstance(ref, str):
            raise TypeError("DAGReference requires a string-like argument, "
                            "got {}!".format(type(ref)))

        super().__init__(ref)

    @property
    def ref(self) -> THash:
        """The reference to an object within the DAG."""
        return self._data
