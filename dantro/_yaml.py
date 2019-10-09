"""Takes care of all YAML-related imports and configuration"""

import os
import logging

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
yaml.constructor.add_constructor(u'!dag_prev',
                                 lambda l, n: DAGNode(l.construct_object(n)))

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
