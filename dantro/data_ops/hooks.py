"""Implements operation hooks for the DAG parser implemented in
:py:mod:`~dantro._dag_utils`.
"""

import logging
from typing import Tuple

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

DAG_PARSER_OPERATION_HOOKS = dict()
"""Contains hooks that are invoked when a certain operation is parsed.
The values should be callables that receive ``operation, *args, **kwargs``
and return a 3-tuple of the manipulated ``operation, args, kwargs``.
The return values will be those that the Transformation object is created
from.

See the :ref:`dag_op_hooks` page for more information on integration and
available hooks.

Example of defining a hook and registering it:

.. testcode:: register_dag_parser_hook
    :hide:

    from typing import Tuple

.. testcode:: register_dag_parser_hook

    # Define hook function
    def _op_hook_my_operation(
        operation, *args, **kwargs
    ) -> Tuple[str, list, dict]:
        # ... do stuff here ...
        return operation, args, kwargs

    # Register with hooks registry
    from dantro.data_ops import DAG_PARSER_OPERATION_HOOKS

    DAG_PARSER_OPERATION_HOOKS["my_operation"] = _op_hook_my_operation

.. todo::

    Implement a decorator to automatically register operation hooks
"""


# -- Operation Hooks ----------------------------------------------------------
# NOTE:
#   - Names should follow ``op_hook_<operation-name>``
#   - A documentation entry should be added in doc/data_io/dag_op_hooks.rst


def op_hook_expression(operation, *args, **kwargs) -> Tuple[str, list, dict]:
    """An operation hook for the ``expression`` operation, attempting to
    auto-detect which symbols are specified in the given expression.
    From those, ``DAGTag`` objects are created, making it more convenient to
    specify an expression that is based on other DAG tags.

    The detected symbols are added to the ``kwargs.symbols``, if no symbol of
    the same name is already explicitly defined there.

    This hook accepts as positional arguments both the ``(expr,)`` form and
    the ``(prev_node, expr)`` form, making it more robust when the
    ``with_previous_result`` flag was set.

    If the expression contains the ``prev`` or ``previous_result`` symbols,
    the corresponding :py:class:`~dantro._dag_utils.DAGNode` will be added to
    the symbols additionally.

    For more information on operation hooks, see :ref:`dag_op_hooks`.
    """
    from sympy import Symbol
    from sympy.parsing.sympy_parser import parse_expr

    from .._dag_utils import DAGNode, DAGTag

    # Extract the expression string
    if len(args) == 1:
        expr = args[0]
    elif len(args) == 2:
        _, expr = args
    else:
        raise TypeError(
            f"Got unexpected positional arguments: {args}; expected either "
            "(expr,) or (prev_node, expr)."
        )

    # Try to extract all symbols from the expression
    all_symbols = parse_expr(expr, evaluate=False).atoms(Symbol)

    # Some symbols might already be given; only add those that were not given.
    # Also, convert the ``prev`` and ``previous_result`` symbols the
    # corresponding DAGNode object
    symbols = kwargs.get("symbols", {})
    for symbol in all_symbols:
        symbol = str(symbol)
        if symbol in symbols:
            log.remark(
                "Symbol '%s' was already specified explicitly! It "
                "will not be replaced.",
                symbol,
            )
            continue

        if symbol in ("prev", "previous_result"):
            symbols[symbol] = DAGNode(-1)
        else:
            symbols[symbol] = DAGTag(symbol)

    # For the case of missing ``symbols`` key, need to write it back to kwargs
    kwargs["symbols"] = symbols

    # For args, return _only_ ``expr``, as expected by the operation
    return operation, (expr,), kwargs


DAG_PARSER_OPERATION_HOOKS["expression"] = op_hook_expression
