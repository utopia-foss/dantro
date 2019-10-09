"""Private low-level helper classes and functions for the DAG framework

NOTE This is imported by dantro.tools and should not be imported by the
     dantro.utils.dag module itself!
"""

from typing import NewType, Union

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

    @property
    def data(self):
        """The stored data"""
        return self._data

    def __eq__(self, other) -> bool:
        if isinstance(other, DAGReference):
            return self.data == other.data
        return False

    def __repr__(self) -> str:
        return "<{} {}>".format(type(self).__name__, repr(self.data))

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

# .............................................................................

class DAGNode(DAGReference):
    """A DAGNode is a relative reference that uses an offset within the
    DAG's node list to specify a reference.

    It is yaml-representable and thus hashable after serialization.
    """
    yaml_tag = u'!dag_node'

    def __init__(self, offset: int=None):
        """Initialize a DAGNode object from an offset value.
        
        Args:
            offset (int, optional): The offset value to set this reference to.
                If not given, will default to -1, i.e. the previous node.
        
        Raises:
            TypeError: On invalid type (not int-convertible)
            ValueError: On positive offset value (makes no sense)
        """
        if offset is None:
            offset = -1

        elif not isinstance(offset, int):
            # Try an integer conversion, to be a bit more robust
            try:
                offset = int(offset)
            except:
                raise TypeError("DAGNode requires an int-like argument, got "
                                "{}!".format(type(offset)))

        if offset >= 0:
            raise ValueError("DAGNode offset need be negative, was {}!"
                             "".format(offset))

        self._data = offset

    @property
    def offset(self) -> int:
        """The offset to the referenced node"""
        return self._data
