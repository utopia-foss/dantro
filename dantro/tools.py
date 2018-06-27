"""This module implements tools that are generally useful in dantro"""

import sys
import subprocess
import collections
import logging

import yaml

# Local constants
log = logging.getLogger(__name__)

# Terminal, TTY-related
IS_A_TTY = sys.stdout.isatty()
try:
    _, TTY_COLS = subprocess.check_output(['stty', 'size']).split()
except:
    # Probably not run from terminal --> set value manually
    TTY_COLS = 79
else:
    TTY_COLS = int(TTY_COLS)
log.debug("Determined TTY_COLS: %d, IS_A_TTY: %d", TTY_COLS, IS_A_TTY)

# -----------------------------------------------------------------------------
# Loading from / writing to files

def load_yml(path: str) -> dict:
    """Loads a yaml file from a path
    
    Args:
        path (str): The path to the yml file
    
    Returns:
        dict: The parsed dictionary data
    """
    log.debug("Loading YAML file... path:\n  %s", path)

    with open(path, 'r') as yaml_file:
        d = yaml.load(yaml_file)

    return d


def write_yml(d: dict, *, path: str):
    """Write a dict as a yaml file to a path
    
    Args:
        d (dict): The dict to convert to dump
        path (str): The path to write the yml file to
    """
    log.debug("Dumping %s to YAML file... target:\n  %s", type(d), path)

    with open(path, 'w') as yaml_file:
        yaml.dump(d, stream=yaml_file)

# -----------------------------------------------------------------------------
# Dictionary operations

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

# -----------------------------------------------------------------------------
# Terminal messaging

def clear_line(only_in_tty=True, break_if_not_tty=True):
    """Clears the current terminal line and resets the cursor to the first
    position using a POSIX command.
    
    Based on: https://stackoverflow.com/a/25105111/1827608
    
    Args:
        only_in_tty (bool, optional): If True (default) will only clear the
            line if the script is executed in a TTY
        break_if_not_tty (bool, optional): If True (default), will insert a
            line break if the script is not executed in a TTY
    """
    # Differentiate cases
    if (only_in_tty and IS_A_TTY) or not only_in_tty:
        # Print the POSIX character
        print('\x1b[2K\r', end='')

    if break_if_not_tty and not IS_A_TTY:
        # print linebreak (no flush)
        print('\n', end='')

    # flush manually (there might not have been a linebreak)
    sys.stdout.flush()

def fill_tty_line(s: str, fill_char: str=" ", align: str="left") -> str:
    """If the terminal is a tty, returns a string that fills the whole tty 
    line with the specified fill character."""
    if not IS_A_TTY:
        return s

    fill_str = fill_char * (TTY_COLS - len(s))

    if align in ["left", "l", None]:
        return s + fill_str

    elif align in ["right", "r"]:
        return fill_str + s

    elif align in ["center", "centre", "c"]:
        return fill_str[:len(fill_str)//2] + s + fill_str[len(fill_str)//2:]

    else:
        raise ValueError("align argument '{}' not supported".format(align))

def tty_centred_msg(s: str, fill_char: str="·") -> str:
    """Shortcut for a common fill_tty_line use case."""
    return fill_tty_line(s, fill_char=fill_char, align='centre')

# -----------------------------------------------------------------------------
# Environment managers

class tmp_sys_path:
    """Temporarily adjusts the paths in sys.path"""

    def __init__(self, *paths, mode: str='append'):
        """
        Args:
            *paths: The paths that should be temporarily available in sys.path
            mode (str, optional): The mode. Can be: append, prepend, replace
        """
        # Store arguments
        self._paths = paths
        self._mode = mode

        # Store the current system path values
        self._old_syspath = [p for p in sys.path]

    def __enter__(self):
        """Enter the environment, i.e. add the paths."""
        if mode in ['append', 'prepend']:
            # Determine the function to use for inserting values
            add_path = sys.path.append if mode == 'append' else sys.path.insert

            # Go over the paths and append them
            for p in self._paths:
                if p not in self._old_syspath:
                    add_path(p)

        elif mode in ['replace']:
            # Clear the path
            sys.path = []

            # Add the values
            for p in self._paths:
                sys.path.append(p)

        else:
            raise ValueError("Invalid mode: {}".format(mode))

    # def __exit__(self, ...): # TODO continue here
