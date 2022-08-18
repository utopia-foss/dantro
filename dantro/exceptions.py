"""Custom dantro exception classes."""

import inspect
from difflib import get_close_matches as _get_close_matches
from typing import Callable, List, Tuple

# -----------------------------------------------------------------------------


def raise_improved_exception(
    exc: Exception,
    *,
    hints: List[Tuple[Callable, str]] = [],
) -> None:
    """Improves the given exception by appending one or multiple hint messages.

    The ``hints`` argument should be a list of 2-tuples, consisting of a unary
    matching function, expecting the exception as only argument, and a hint
    that is part of the new error message.
    """
    matching_hints = []
    for match_func, hint in hints:
        if match_func(exc):
            matching_hints.append(hint)

    if matching_hints:
        _hints = "\n".join(f"  - {h}" for h in matching_hints)
        raise type(exc)(
            str(exc) + f"\n\nHint(s) how to resolve this:\n{_hints}"
        ) from exc

    # Re-raise the active exception
    raise


# -----------------------------------------------------------------------------


class DantroError(Exception):
    """Base class for all dantro-related errors"""


class DantroWarning(UserWarning):
    """Base class for all dantro-related warnings"""


class DantroMessagingException(DantroError):
    """Base class for exceptions that are used for messaging"""


# General data tree warnings ..................................................


class UnexpectedTypeWarning(DantroWarning):
    """Given when there was an unexpected type passed to a data container."""


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


# For data operations .........................................................


class DataOperationWarning(DantroWarning):
    """Base class for warnings related to data operations"""


class DataOperationError(DantroError):
    """Base class for errors related to data operations"""


class BadOperationName(DataOperationError, ValueError):
    """Raised upon bad data operation name"""


class DataOperationFailed(DataOperationError, RuntimeError):
    """Raised upon failure to apply a data operation"""


class MetaOperationError(DataOperationError):
    """Base class for errors related to meta operations"""


class MetaOperationSignatureError(MetaOperationError):
    """If the meta-operation signature was erroneous"""


class MetaOperationInvocationError(MetaOperationError, ValueError):
    """If the invocation of the meta-operation was erroneous"""


# Data transformations ........................................................


class DAGError(DantroError):
    """For errors in the data transformation framework"""


class MissingDAGReference(DAGError, ValueError):
    """If there was a missing DAG reference"""


class MissingDAGTag(MissingDAGReference, ValueError):
    """Raised upon bad tag names"""


class MissingDAGNode(MissingDAGReference, ValueError):
    """Raised upon bad node index"""


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


class DataLoadingError(DataManagerError):
    """Raised if loading data failed for some reason"""


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


# Messaging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


class SkipPlot(DantroMessagingException):
    """A custom exception class that denotes that a plot is to be skipped.

    This is typically handled by the :py:class:`~dantro.plot_mngr.PlotManager`
    and can thus be raised anywhere below it: in the plot creators, in the
    user-defined plotting functions, ...
    """

    def __init__(self, what: str = ""):
        super().__init__(what)


class EnterAnimationMode(DantroMessagingException):
    """An exception that is used to convey to any
    :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` or derived
    creator that animation mode is to be entered instead of a regular
    single-file plot.

    It can and should be invoked via
    :py:meth:`~dantro.plot.plot_helper.PlotHelper.enable_animation`.

    This exception can be raised from within a plot function to dynamically
    decide whether animation should happen or not. Its counterpart is
    :py:exc:`~dantro.exceptions.ExitAnimationMode`.
    """


class ExitAnimationMode(DantroMessagingException):
    """An exception that is used to convey to any
    :py:class:`~dantro.plot.creators.pyplot.PyPlotCreator` or derived
    creator that animation mode is to be exited and a regular single-file plot
    should be carried out.

    It can and should be invoked via
    :py:meth:`~dantro.plot.plot_helper.PlotHelper.disable_animation`.

    This exception can be raised from within a plot function to dynamically
    decide whether animation should happen or not. Its counterpart is
    :py:exc:`~dantro.exceptions.ExitAnimationMode`.
    """


class PlotHelperError(PlotConfigError):
    """Raised upon failure to invoke a specific plot helper function, this
    custom exception type stores metadata on the helper invocation in order
    to generate a useful error message.
    """

    def __init__(
        self,
        upstream_error: Exception,
        *,
        name: str,
        params: dict,
        ax_coords: Tuple[int, int] = None,
    ):
        """Initializes a PlotHelperError"""
        self.err = upstream_error
        self.name = name
        self.ax_coords = ax_coords
        self.params = params

    def __str__(self):
        """Generates an error message for this particular helper"""
        _params = "\n".join(
            [f"      {k}: {repr(v)}" for k, v in self.params.items()]
        )
        _where = (
            "the figure" if not self.ax_coords else f"axis {self.ax_coords}"
        )
        return (
            f"'{self.name}' failed on {_where}!  "
            f"{self.err.__class__.__name__}: {self.err}\n"
            f"    Invocation parameters were:\n{_params}\n"
        )

    @property
    def docstring(self) -> str:
        """Returns the docstring of this helper function"""
        from .plot import PlotHelper

        func = getattr(PlotHelper, "_hlpr_" + self.name)
        return f"{self.name}\n{'-'*len(self.name)}\n{inspect.getdoc(func)}\n"


class PlotHelperErrors(ValueError):
    """This custom exception type gathers multiple individual instances of
    :py:class:`~dantro.exceptions.PlotHelperError`.
    """

    def __init__(self, *errors, show_docstrings: bool = True):
        """Bundle multiple PlotHelperErrors together

        Args:
            *errors: The individual instances of
                :py:class:`~dantro.exceptions.PlotHelperError`
            show_docstrings (bool, optional): Whether to show docstrings in the
                error message.
        """
        self._errors = list()
        self._axes = set()
        self._num = len(errors)
        self._docstrings = dict()
        self.show_docstrings = show_docstrings

        for error in errors:
            self._errors.append(error)
            self._axes.add(error.ax_coords)
            self._docstrings[error.name] = error.docstring

    @property
    def errors(self):
        return self._errors

    def __str__(self) -> str:
        """Generates a combined error message for *all* registered errors"""
        s = (
            f"Encountered {self._num} error(s) "
            f"for {len(self._docstrings)} different plot helper(s) "
            f"on {len(self._axes)} different axes!\n\n"
        )

        s += "\n".join([f"-- {e}" for e in self._errors]) + "\n"

        if self.show_docstrings:
            s += "Relevant Docstrings\n"
            s += "===================\n\n"
            s += "\n\n".join(self._docstrings.values())

        return s
