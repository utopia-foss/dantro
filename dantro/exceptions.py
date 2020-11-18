"""Custom dantro exception classes."""

from difflib import get_close_matches as _get_close_matches

# -----------------------------------------------------------------------------


class DantroError(Exception):
    """Base class for all dantro-related errors"""


class DantroWarning(UserWarning):
    """Base class for all dantro-related warnings"""


# General data tree errors ....................................................


class ItemAccessError(KeyError, IndexError):
    """Raised upon bad access via __getitem__ or similar magic methods.

    This derives from both native exceptions KeyError and IndexError as these
    errors may be equivalent in the context of the dantro data tree, which is
    averse to the underlying storage container.

    See :py:class:`~dantro.base.BaseDataGroup` for example usage.
    """

    def __init__(
        self,
        obj: "AbstractDataContainer",
        *,
        key: str,
        show_hints: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Set up an ItemAccessError object, storing some metadata that is
        used to create a helpful error message.

        Args:
            obj (AbstractDataContainer): The object from which item access was
                attempted but failed
            key (str): The key with which ``__getitem__`` was called
            show_hints (bool, optional): Whether to show hints in the error
                message, e.g. available keys or "Did you mean ...?"
            prefix (str, optional): A prefix string for the error message
            suffix (str, optional): A suffix string for the error message

        Raises:
            TypeError: Upon ``obj`` without attributes ``logstr`` and ``path``;
                or ``key`` not being a string.
        """
        if not (hasattr(obj, "logstr") and hasattr(obj, "path")):
            raise TypeError(
                "ItemAccessError can only be used for objects that supply the "
                "`logstr` and `path` attributes, i.e. objects similar to "
                f"those comprising the dantro data tree! Got {type(obj)}."
            )

        if not isinstance(key, str):
            raise TypeError(
                "ItemAccessError `key` argument needs to be a "
                f"string, but was {type(key)} {repr(key)}!"
            )

        super().__init__(obj, key, show_hints, prefix, suffix)

    def __str__(self) -> str:
        """Parse an error message, using the additional information to give
        hints on where the error occurred and how it can be resolved.
        """
        obj, key, show_hints, prefix, suffix = self.args
        prefix = prefix + " " if prefix else ""
        suffix = suffix if suffix else ""

        # If possible, add a hint.
        hint = ""
        if show_hints and hasattr(obj, "keys") and hasattr(obj, "__len__"):
            if len(obj) == 0:
                hint = "This object is empty (== has zero length). "

            elif len(obj) <= 5:
                _keys = ", ".join(obj.keys())
                hint = f"Available keys:  {_keys} "

            else:
                matches = _get_close_matches(key, obj.keys(), n=5, cutoff=0.0)
                _matches = ", ".join(matches)
                hint = f"Did you mean: {_matches} ? "

        return (
            f"{prefix}"
            f"No item '{key}' available in {obj.logstr} @ {obj.path}! "
            f"{hint}{suffix}"
        )


# DataManager-related .........................................................


class DataManagerError(DantroError):
    """All DataManager exceptions derive from this one"""


class RequiredDataMissingError(DataManagerError):
    """Raised if required data was missing."""


class MissingDataError(DataManagerError):
    """Raised if data was missing, but is not required."""


class ExistingDataError(DataManagerError):
    """Raised if data already existed."""


class ExistingGroupError(DataManagerError):
    """Raised if a group already existed."""


class LoaderError(DataManagerError):
    """Raised if a data loader was not available"""


class MissingDataWarning(DantroWarning):
    """Used as warning instead of MissingDataError"""


class ExistingDataWarning(DantroWarning):
    """If there was data already existing ..."""


class NoMatchWarning(DantroWarning):
    """If there was no regex match"""


# Plotting-related ............................................................


class PlottingError(DantroError):
    """Custom exception class for all plotting errors"""


class PlotConfigError(ValueError, PlottingError):
    """Raised when there were errors in the plot configuration"""


class InvalidCreator(ValueError, PlottingError):
    """Raised when an invalid creator was specified"""


class PlotCreatorError(PlottingError):
    """Raised when an error occured in a plot creator"""
