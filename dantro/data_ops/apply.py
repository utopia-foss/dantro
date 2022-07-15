"""Implements the application of operations on the given arguments and data"""

import logging
from difflib import get_close_matches as _get_close_matches
from typing import Any

from ..exceptions import *
from ..tools import make_columns as _make_columns
from .db import _OPERATIONS
from .db_tools import get_operation

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def apply_operation(
    op_name: str,
    *op_args,
    _ops: dict = None,
    _log_level: int = 5,
    **op_kwargs,
) -> Any:
    """Apply an operation with the given arguments and then return its return
    value.
    This is used by the :ref:`dag_framework` and allows to invoke operations
    from the data operations database, see :ref:`data_processing`.

    Args:
        op_name (str): The name of the operation to carry out; need to be part
            of the operations database ``_ops``.
        *op_args: The positional arguments to the operation
        _ops (dict, optional): The operations database object to use; if None,
            uses the :ref:`dantro operations database <data_ops_available>`.
        _log_level (int, optional): Log level of the log messages created by
            this function.
        **op_kwargs: The keyword arguments to the operation

    Returns:
        Any: The result of the operation

    Raises:
        BadOperationName: On invalid operation name
        DataOperationError: On failure to *apply* the operation
    """
    if _ops is None:
        _ops = _OPERATIONS

    op = get_operation(op_name, _ops=_ops)

    # Compute and return the results, allowing messaging exceptions through ...
    log.log(_log_level, "Performing operation '%s' ...", op_name)
    try:
        return op(*op_args, **op_kwargs)

    except DantroMessagingException:
        raise

    except Exception as exc:
        # Provide information about arguments, such that it is easier to
        # debug the error. Need to parse them nicely to let the output become
        # garbled up ...
        _op_args = "[]"
        if op_args:
            _op_args = "\n" + "\n\n".join(
                f"    {i:>2d}:  {arg}" for i, arg in enumerate(op_args)
            )

        _op_kwargs = "{}"
        if op_kwargs:
            _l = max(len(str(k)) for k in op_kwargs)
            _op_kwargs = "\n" + "\n\n".join(
                f"    {k:>{_l}s}:  {kwarg}" for k, kwarg in op_kwargs.items()
            )

        raise DataOperationFailed(
            f"Operation '{op_name}' failed with a {exc.__class__.__name__}, "
            "see below!\nIt was called with the following arguments:\n"
            f"  args:    {_op_args}\n\n"
            f"  kwargs:  {_op_kwargs}\n\n"
            f"{exc.__class__.__name__}: {exc}\n"
        ) from exc
