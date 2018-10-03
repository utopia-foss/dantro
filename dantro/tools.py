"""This module implements tools that are generally useful in dantro"""

import sys
import subprocess
import collections
import logging

# Get a logger instance
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

# Import yaml and configure
from paramspace import yaml
yaml.default_flow_style = False

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

def fill_line(s: str, *, num_cols: int=TTY_COLS, fill_char: str=" ", align: str="left") -> str:
    """Extends the given string such that it fills a whole line of `num_cols` columns.
    
    Args:
        s (str): The string to extend to a whole line
        num_cols (int, optional): The number of colums of the line; defaults to
            the number of TTY columns or – if those are not available – 79
        fill_char (str, optional): The fill character
        align (str, optional): The alignment. Can be: 'left', 'right', 'center'
            or the one-letter equivalents.
    
    Returns:
        str: The string of length `num_cols`
    
    Raises:
        ValueError: For invalid `align` or `fill_char` argument
    """
    if len(fill_char) != 1:
        raise ValueError("Argument `fill_char` needs to be string of length 1 "
                         "but was: "+str(fill_char))

    fill_str = fill_char * (num_cols - len(s))

    if align in ["left", "l", None]:
        return s + fill_str

    elif align in ["right", "r"]:
        return fill_str + s

    elif align in ["center", "centre", "c"]:
        return fill_str[:len(fill_str)//2] + s + fill_str[len(fill_str)//2:]

    raise ValueError("align argument '{}' not supported".format(align))

def center_in_line(s: str, *, num_cols: int=TTY_COLS, fill_char: str="·", spacing: int=1) -> str:
    """Shortcut for a common fill_line use case.
    
    Args:
        s (str): The string to center in the line
        num_cols (int, optional): The number of columns in the line
        fill_char (str, optional): The fill character
        spacing (int, optional): The spacing around the string `s`
    
    Returns:
        str: The string centered in the line
    """
    spacing = " " * spacing
    return fill_line(spacing + s + spacing, num_cols=num_cols,
                     fill_char=fill_char, align='centre')


# Misc ------------------------------------------------------------------------

def recursive_call(obj, *, func, n):
    """Recursively calls func n times on obj.
    
    Args:
        obj: The object to call func on
        func: The function to call
        n (int): How many times to call func
    
    Returns:
        The result of n recursive calls of func on obj.
    """
    if n > 0:
        # Recursive branch
        return recursive_call(func(obj), n=n-1, func=func)

    # End of recursion
    return obj
