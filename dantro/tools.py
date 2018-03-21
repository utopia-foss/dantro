"""This module implements tools that are generally useful in dantro"""

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
