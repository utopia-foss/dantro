"""This module implements tools that are generally useful in dantro"""

import collections
import logging
import os
import subprocess
import sys
from typing import List, Mapping, Sequence, Set, Tuple, Union

import numpy as np

# Get a logger instance
log = logging.getLogger(__name__)

# Terminal, TTY-related
IS_A_TTY = sys.stdout.isatty()
try:
    _, TTY_COLS = subprocess.check_output(["stty", "size"]).split()
except:
    # Probably not run from terminal --> set value manually
    TTY_COLS = 79
else:
    TTY_COLS = int(TTY_COLS)
log.debug("Determined TTY_COLS: %d, IS_A_TTY: %d", TTY_COLS, IS_A_TTY)

# -----------------------------------------------------------------------------
# Import private yaml module, where everything is configured

from ._yaml import load_yml, write_yml, yaml

# -----------------------------------------------------------------------------
# Dictionary operations


def recursive_update(d: dict, u: dict) -> dict:
    """Recursively updates the Mapping-like object ``d`` with the Mapping-like
    object ``u`` and returns it. Note that this does *not* create a copy of
    ``d``, but changes it mutably!

    Based on: http://stackoverflow.com/a/32357112/1827608

    Args:
        d (dict): The mapping to update
        u (dict): The mapping whose values are used to update ``d``

    Returns:
        dict: The updated dict ``d``
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
                d[k] = v  # ... which is just u[k]

        else:
            # Not a mapping -> create one
            d = {k: u[k]}
    return d


def recursive_getitem(obj: Union[Mapping, Sequence], keys: Sequence):
    """Go along the sequence of ``keys`` through ``obj`` and return the target
    item.

    Args:
        obj (Union[Mapping, Sequence]): The object to get the item from
        keys (Sequence): The sequence of keys to follow

    Returns:
        The target item from ``obj``, specified by ``keys``

    Raises:
        ValueError: If any index or key in the key sequence was not available
    """

    def handle_error(exc: Exception, *, key, keys, obj):
        raise ValueError(
            f"Invalid {'key' if isinstance(exc, KeyError) else 'index'} "
            f"'{key}' during recursive getitem of key sequence "
            f"{' -> '.join([repr(k) for k in keys])}! "
            f"{exc.__class__.__name__}: {exc} raised on the following "
            f"object:\n{obj}"
        ) from exc

    if len(keys) > 1:
        # Continue recursion
        try:
            return recursive_getitem(obj[keys[0]], keys=keys[1:])
        except (KeyError, IndexError) as err:
            handle_error(err, key=keys[0], keys=keys, obj=obj)

    # else: reached the end of the recursion
    try:
        return obj[keys[0]]
    except (KeyError, IndexError) as err:
        handle_error(err, key=keys[0], keys=keys, obj=obj)


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
        print("\x1b[2K\r", end="")

    if break_if_not_tty and not IS_A_TTY:
        # print linebreak (no flush)
        print("\n", end="")

    # flush manually (there might not have been a linebreak)
    sys.stdout.flush()


def fill_line(
    s: str,
    *,
    num_cols: int = TTY_COLS,
    fill_char: str = " ",
    align: str = "left",
) -> str:
    """Extends the given string such that it fills a whole line of `num_cols`
    columns.

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
        raise ValueError(
            "Argument `fill_char` needs to be string of length 1 but was: "
            + str(fill_char)
        )

    fill_str = fill_char * (num_cols - len(s))

    if align in ["left", "l", None]:
        return s + fill_str

    elif align in ["right", "r"]:
        return fill_str + s

    elif align in ["center", "centre", "c"]:
        return (
            fill_str[: len(fill_str) // 2] + s + fill_str[len(fill_str) // 2 :]
        )

    raise ValueError("align argument '{}' not supported".format(align))


def print_line(s: str, *, end="\r", **kwargs):
    """Wrapper around :py:func:`~dantro.tools.fill_line` that also prints
    a line with carriage return (without new line) as end character. This is
    useful for progress report lines that overwrite the previously printed
    content repetitively.
    """
    print(fill_line(s, **kwargs), end=end)


def center_in_line(
    s: str, *, num_cols: int = TTY_COLS, fill_char: str = "·", spacing: int = 1
) -> str:
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
    return fill_line(
        spacing + s + spacing,
        num_cols=num_cols,
        fill_char=fill_char,
        align="centre",
    )


def make_columns(
    items: List[str],
    *,
    wrap_width: int = TTY_COLS,
    fstr: str = "  {item:<{width:}s}  ",
) -> str:
    """Given a sequence of string items, returns a string with these items
    spread out over several columns. Iteration is first within the row and
    then into the next row.

    The number of columns is determined automatically from the wrap width, the
    length of the longest item in the items list, and the length of the
    evaluated format string.

    Args:
        items (List[str]): The string items to represent in columns.
        wrap_width (int, optional): The maximum width of each full row
        fstr (str, optional): The format string to use. Needs to accept the
            keys ``item`` and ``width``, the latter of which will be used for
            padding. The format string should lead to strings of equal length,
            otherwise the column layout will be messed up!
    """
    if not items:
        return ""

    max_item_width = max(len(item) for item in items)
    item_str_width = len(
        fstr.format(item=" " * max_item_width, width=max_item_width)
    )
    num_cols = wrap_width // item_str_width

    rows = []
    for i, item in enumerate(items):
        item_str = fstr.format(item=item, width=max_item_width)

        # New row or new column?
        if i % num_cols == 0:
            rows.append(item_str)
        else:
            rows[-1] += item_str

    return "\n".join(rows) + "\n"


# -----------------------------------------------------------------------------
# Numerics


def apply_along_axis(
    func, axis: int, arr: np.ndarray, *args, **kwargs
) -> np.ndarray:
    """This is like numpy's function of the same name, but does not try to
    cast the results of func to an np.ndarray but tries to keep them as dtype
    object. Thus, the return value of this function will always have one fewer
    dimension then the input array.

    This goes along the equivalent formulation of np.apply_along_axis, outlined
    in their documentation of the function.
    """
    # Get the shapes of the outer and inner iteration; both are tuples!
    shape_outer, shape_inner = arr.shape[:axis], arr.shape[axis + 1 :]
    num_outer = len(shape_outer)

    # These together give the shape of the output array
    out = np.zeros(shape_outer + shape_inner, dtype="object")
    out.fill(None)

    log.debug("Applying function '%s' along axis ...", func.__name__)
    log.debug("  input array:     %s, %s", arr.shape, arr.dtype)
    log.debug("  axis to reduce:  %d", axis)
    log.debug("  output will be:  %s, %s", out.shape, out.dtype)

    # Now loop over the output array and at each position fill it with the
    # result of the function call.
    it = np.nditer(out, flags=("refs_ok", "multi_index"))
    for _ in it:
        midx = it.multi_index

        # Build selector, which has the ellipsis at position `axis`, thus one
        # dimension higher than the out array and matching the input `arr`.
        sel = tuple(midx[:num_outer]) + (Ellipsis,) + tuple(midx[num_outer:])
        log.debug("  midx: %s  -->  selector: %s", midx, sel)

        # Apply function to selected parts of array, then write to the current
        # point in the iteration over the output array.
        out[midx] = func(arr[sel], *args, **kwargs)

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
        if obj.dtype.kind in ["S", "a"]:
            obj = obj.astype("U")

        # If it is of dtype object, decode all bytes objects
        if obj.dtype == np.dtype("object"):

            def decode_if_bytes(val):
                if isinstance(val, bytes):
                    return val.decode("utf8")
                return val

            # Apply element-wise
            obj = np.vectorize(decode_if_bytes)(obj)

    # ... or as bytes
    elif isinstance(obj, bytes):
        # Decode bytestring to unicode
        obj = obj.decode("utf8")

    return obj


# -----------------------------------------------------------------------------
# Misc


def is_iterable(obj) -> bool:
    """Tries whether the given object is iterable."""
    try:
        (e for e in obj)
    except:
        return False
    return True


def is_hashable(obj) -> bool:
    """Tries whether the given object is hashable."""
    try:
        hash(obj)
    except:
        return False
    return True


class DoNothingContext:
    """A context manager that ... does nothing."""

    def __init__(self):
        pass

    def __enter__(self):
        """Called upon entering the context using the ``with`` statement"""
        pass

    def __exit__(self, *args):
        """Called upon exiting context, with ``*args`` being exceptions etc."""
        pass


class adjusted_log_levels:
    """A context manager that temporarily adjusts log levels"""

    def __init__(self, *new_levels: Sequence[Tuple[str, int]]):
        self.new_levels = {n: l for n, l in new_levels}
        self.old_levels = dict()

    def __enter__(self):
        """When entering the context, sets these levels"""
        for name, new_level in self.new_levels.items():
            logger = logging.getLogger(name)
            self.old_levels[name] = logger.level
            logger.setLevel(new_level)

    def __exit__(self, *_):
        """When leaving the context, resets the levels to their old state"""
        for name, old_level in self.old_levels.items():
            logging.getLogger(name).setLevel(old_level)


def total_bytesize(files: List[str]) -> int:
    """Returns the total size of a list of files"""
    return sum([os.path.getsize(fpath) for fpath in files])


def format_bytesize(num: int, *, precision: int = 1) -> str:
    """Formats a size in bytes to a human readable (binary) format.

    Stripped down from https://stackoverflow.com/a/63839503/1827608 .

    Args:
        num (int): Number of bytes
        precision (int, optional): The decimal precision to use, can be 0..3

    Returns:
        str: The formatted, human-readable byte size
    """
    UNIT_LABELS = ("B", "kiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    PRECISION_OFFSETS = (0.5, 0.05, 0.005, 0.0005)
    FSTRS = (
        "{}{:.0f} {}",
        "{}{:.1f} {}",
        "{}{:.2f} {}",
        "{}{:.3f} {}",
    )

    last_label = UNIT_LABELS[-1]
    unit_step = 1024
    unit_step_thresh = unit_step - PRECISION_OFFSETS[precision]
    is_negative = num < 0

    if is_negative:
        num = abs(num)

    # Special case for Bytes, where there should be no decimal point
    if num < 1024:
        return FSTRS[0].format("-" if is_negative else "", num, "B")

    for unit in UNIT_LABELS:
        if num < unit_step_thresh:
            # Below threshold now, can go to formatting
            break

        # Unless we reached the highest prefix, shrink the number such that
        # it represents the value in the next higher unit
        if unit != last_label:
            num /= unit_step

    return FSTRS[precision].format("-" if is_negative else "", num, unit)


class PoolCallbackHandler:
    """A simple callback handler for multiprocessing pools"""

    def __init__(
        self,
        n_max: int,
        *,
        silent: bool = False,
        fstr: str = "  Loaded  {n}/{n_max} .",
    ):
        """
        Args:
            n_max (int): Number of tasks
            silent (bool, optional): If true, will *not* print a message
            fstr (str, optional): The format string for the status message.
                May contain keys ``n`` and ``n_max``.
        """
        self._n = 0
        self._n_max = n_max
        self.silent = silent
        self._fstr = fstr

    def __call__(self, _):
        self._n += 1
        if not self.silent:
            print_line(self._fstr.format(n=self._n, n_max=self._n_max))


class PoolErrorCallbackHandler:
    """A simple callback handler for errors in multiprocessing pools"""

    def __init__(self):
        self._errors = set()

    def __call__(self, error: Exception):
        self.track_error(error)

    def track_error(self, error: Exception):
        self._errors.update({error})

    @property
    def errors(self) -> Set[Exception]:
        return self._errors
