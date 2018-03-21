"""This module implements tools that are generally useful in dantro"""

import collections
import logging

import yaml

# Local constants
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

def load_yml(path: str) -> dict:
    """Loads a yaml file from a path
    
    Args:
        path (str): The path to the yml file
    
    Returns:
        dict: The parsed dictionary data
    """
    log.debug("Loading YAML file from %s ...", path)

    with open(path, 'r') as yaml_file:
        d = yaml.load(yaml_file)

    return d


def recursive_update(d: dict, u: dict) -> dict:
    """Recursively updates the Mapping-like object `d` with the Mapping-like
    object `u` and returns it. Note that this does not create a copy of `d`!
    
    Based on: http://stackoverflow.com/a/32357112/1827608
    
    Args:
        d (dict): The mapping to update
        u (dict): The mapping whose values are used to update `d`
    
    Returns:
        dict: The updated dict `d`
    """
    for k, v in u.items():
        if isinstance(d, collections.Mapping):
            # Already a Mapping
            if isinstance(v, collections.Mapping):
                # Already a Mapping, continue recursion
                d[k] = recursive_update(d.get(k, {}), v)
                # This already creates a mapping if the key was not available
            else:
                # Not a mapping -> at leaf -> update value
                d[k] = v    # ... which is just u[k]

        else:
            # Not a mapping -> create one
            d = {k: u[k]}
    return d
