"""This module implements tools that are generally useful in dantro"""

import os
import sys
import subprocess
import collections
import logging

import numpy as np

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
        if isinstance(d, collections.abc.Mapping):
            # Already a Mapping
            if isinstance(v, collections.abc.Mapping):
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


# -----------------------------------------------------------------------------
# Numerics

def apply_along_axis(func, axis: int, arr: np.ndarray, *args, **kwargs) -> np.ndarray:
    """This is like numpy's function of the same name, but does not try to
    cast the results of func to an np.ndarray but tries to keep them as dtype
    object. Thus, the return value of this function will always have one fewer
    dimension then the input array.

    This goes along the equivalent formulation of np.apply_along_axis, outlined
    in their documentation of the function.
    """
    # Get the shapes of the outer and inner iteration; both are tuples!
    shape_outer, shape_inner = arr.shape[:axis], arr.shape[axis+1:]
    num_outer = len(shape_outer)

    # These together give the shape of the output array
    out = np.zeros(shape_outer + shape_inner, dtype='object')
    out.fill(None)

    log.debug("apply_along_axis called")
    log.debug("  input array:     %s, %s", arr.shape, arr.dtype)
    log.debug("  axis to reduce:  %d", axis)
    log.debug("  output will be:  %s, %s", out.shape, out.dtype)

    # Now loop over the output array and at each position fill it with the
    # result of the function call.
    it = np.nditer(out, flags=('refs_ok', 'multi_index'))
    for _ in it:
        midx = it.multi_index

        # Build selector, which has the ellipsis at position `axis`, thus one
        # dimension higher than the out array and matching the input `arr`.
        sel = tuple(midx[:num_outer]) + (Ellipsis,) + tuple(midx[num_outer:])
        log.debug("  midx: %s  -->  selector: %s", midx, sel)

        # Apply function to selected parts of array, then write to the current
        # point in the iteration over the output array.
        out[midx] = func(arr[sel], *args, **kwargs)

    log.debug("  finished iteration, returning output array...")
    return out

# -----------------------------------------------------------------------------
# Fun with byte strings

def decode_bytestrings(obj) -> str:
    """Checks whether the given attribute value is or contains byte
    strings and if so, decodes it to a python string.
    
    Args:
        obj: The object to try to decode into holding python strings
    
    Returns:
        str: Either the unchanged object or the decoded one
    """
    # Check for data loaded as array of bytestring
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ['S', 'a']:
            obj = obj.astype('U')

        # If it is of dtype object, decode all bytes objects
        if obj.dtype == np.dtype('object'):
            def decode_if_bytes(val):
                if isinstance(val, bytes):
                    return val.decode('utf8')
                return val

            # Apply element-wise
            obj = np.vectorize(decode_if_bytes)(obj)

    # ... or as bytes
    elif isinstance(obj, bytes):
        # Decode bytestring to unicode
        obj = obj.decode('utf8')

    return obj


# -----------------------------------------------------------------------------
# rc_context

class DoNothingContext:
    """A context manager that ... does nothing.
    """

    def __init__(self):
        pass

    def __enter__(self):
        """Called upon entering the context using the `with` statement"""
        pass

    def __exit__(self, *args):
        """Called upon exiting the context, with *args representing exceptions etc"""
        pass

