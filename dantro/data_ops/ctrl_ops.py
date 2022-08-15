"""Implements operations that control the behaviour of the transformation or
pipeline in general, including functions that can be used for debugging."""

import logging
from typing import Any

from ..base import BaseDataContainer, BaseDataGroup
from ..exceptions import SkipPlot

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def raise_SkipPlot(
    cond: bool = True, *, reason: str = "", passthrough: Any = None
):
    """Raises :py:exc:`~dantro.exceptions.SkipPlot` to trigger that a plot is
    skipped without error, see :ref:`plot_mngr_skipping_plots`.

    If ``cond`` is False, this will do nothing but return the passthrough.

    Args:
        cond (bool, optional): Whether to actually raise the exception
        reason (str, optional): The reason for skipping, optional
        passthrough (Any, optional): A passthrough value which is returned if
            ``cond`` did not evaluate to True.
    """
    if cond:
        raise SkipPlot(reason)
    return passthrough


def print_data(
    data: Any, *, end: str = "\n", fstr: str = None, **fstr_kwargs
) -> Any:
    """Prints and passes on the data using :py:func:`print`.

    The print operation distinguishes between dantro types (in which case some
    more information is shown) and non-dantro types. If a custom format string
    is given, will always use that one.

    .. note::

        This is a passthrough-function: ``data`` is always returned without any
        changes. However, the print operation may lead to resolution of
        :py:mod:`~dantro.proxy` objects.

    Args:
        data (Any): The data to print
        end (str, optional): The ``end`` argument to the ``print`` call
        fstr (str, optional): If given, will use this to format the data for
            printing. The data will be the passed as first *positional*
            argument to the format string, thus addressable by ``{0:}`` or
            ``data`` (e.g. to access attributes via format-string syntax).
            If the format string is not ``None``, will *always* use the format
            string and not use the custom formatting for dantro objects.
        **fstr_kwargs: Keyword arguments passed to the :py:func:`format`
            call.

    Returns:
        Any: the given ``data``
    """
    if fstr is None and isinstance(data, BaseDataContainer):
        print(f"{data}, with data:\n{str(data.data)}\n", end=end)

    elif fstr is None and isinstance(data, BaseDataGroup):
        print(f"{data.tree}\n", end=end)

    else:
        fstr = fstr if fstr is not None else "{0:}"
        print(fstr.format(str(data), data=data, **fstr_kwargs), end=end)

    return data
