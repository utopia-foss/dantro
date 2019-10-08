"""Private low-level helper classes and functions for the DAG framework

NOTE This is imported by dantro.tools and should not be imported by the
     dantro.utils.dag module itself!
"""

from typing import Union

class DAGField:
    def __init__(self, field_name: str):
        self._name = field_name

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other) -> bool:
        if isinstance(other, DAGField):
            return self.name == other.name
        return False

    def __repr__(self) -> str:
        return "<DAGField {}>".format(repr(self.name))

    # YAML representation
    yaml_tag = u'!field'

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(constructor.construct_scalar(node))

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_data(node.name)
    

class DAGReference:
    def __init__(self, ref: Union[str, int]):
        try:
            ref = int(ref)
        except:
            pass

        self._ref = ref

    @property
    def ref(self) -> str:
        return self._ref

    def __eq__(self, other) -> bool:
        if isinstance(other, DAGReference):
            return self.ref == other.ref
        return False

    def __repr__(self) -> str:
        return "<DAGReference {}>".format(repr(self.ref))

    # YAML representation
    yaml_tag = u'!dag_ref'

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(constructor.construct_scalar(node))

    @classmethod
    def to_yaml(cls, representer, node):
        return representer.represent_data(node.ref)
    
