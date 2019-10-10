"""Takes care of all YAML-related imports and configuration"""

import os
import io
import logging
from typing import Any

import ruamel.yaml

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

# Import yaml from paramspace and configure it
from paramspace import yaml
yaml.default_flow_style = False

# Register further classes
from ._dag_utils import DAGReference, DAGTag, DAGNode

yaml.register_class(DAGReference)
yaml.register_class(DAGTag)
yaml.register_class(DAGNode)

# Special constructors ........................................................
# For the case of a reference to the previous node
yaml.constructor.add_constructor(u'!dag_prev', lambda l, n: DAGNode(-1))

# -----------------------------------------------------------------------------

def load_yml(path: str, *, mode: str='r') -> dict:
    """Loads a yaml file from a path
    
    Args:
        path (str): The path to the yml file
        mode (str, optional): Read mode
    
    Returns:
        dict: The parsed dictionary data
    """
    log.debug("Loading YAML file... mode: %s, path:\n  %s", mode, path)

    with open(path, mode) as yaml_file:
        return yaml.load(yaml_file)

def write_yml(d: dict, *, path: str, mode: str='w'):
    """Write a dict as a yaml file to a path
    
    Args:
        d (dict): The dict to convert to dump
        path (str): The path to write the yml file to
        mode (str, optional): Write mode of the file
    """
    log.debug("Dumping %s to YAML file... mode: %s, target:\n  %s",
              type(d).__name__, mode, path)

    # Make sure the directory is present
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, mode) as yaml_file:
        # Add the yaml '---' prefix
        yaml_file.write("---\n")

        # Now dump the rest
        yaml.dump(d, stream=yaml_file)

        # Ensure new line
        yaml_file.write("\n")


# -----------------------------------------------------------------------------
# More specific YAML-based functions

def yaml_dumps(obj: Any, *, register_classes: tuple=(), **dump_params) -> str:
    """Serializes the given object using a newly created YAML dumper.
    
    The aim of this function is to provide YAML dumping that is not dependent
    on any package configuration; all parameters can be passed here.
    
    In other words, his function does _not_ use the dantro._yaml.yaml object
    for dumping but each time creates a new dumper with fixed settings. This
    reduces the chance of interference from elsewhere. Compared to the time
    needed for serialization in itself, the extra time needed to create the
    new ruamel.yaml.YAML object and register the classes is negligible.
    
    Args:
        obj (Any): The object to dump
        register_classes (tuple, optional): Additional classes to register
        **dump_params: Dumping parameters
    
    Returns:
        str: The output of serialization
    
    Raises:
        ValueError: On failure to serialize the given object
    """
    s = io.StringIO()
    y = ruamel.yaml.YAML()

    # Register classes; then apply dumping parameters via object properties
    for Cls in register_classes:
        y.register_class(Cls)

    for k, v in dump_params.items():
        setattr(y, k, v)

    # Serialize
    try:
        y.dump(obj, stream=s)
    
    except Exception as err:
        raise ValueError("Could not serialize the given {} object!"
                         "".format(type(obj))) from err

    return s.getvalue()
