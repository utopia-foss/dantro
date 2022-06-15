"""Implements data operations that work with expressions, e.g. lambda function
definitions or symbolic math"""

import logging
import math
import re
from typing import Callable, Tuple, Union

import numpy as np

from .._import_tools import LazyLoader

log = logging.getLogger(__name__)

xr = LazyLoader("xarray")
nx = LazyLoader("networkx")
pd = LazyLoader("pandas")
scipy = LazyLoader("scipy")

# -----------------------------------------------------------------------------


def expression(
    expr: str,
    *,
    symbols: dict = None,
    evaluate: bool = True,
    transformations: Tuple[Callable] = None,
    astype: Union[type, str] = float,
):
    """Parses and evaluates a symbolic math expression using SymPy.

    For parsing, uses sympy's :py:func:`sympy.parsing.sympy_parser.parse_expr`.
    The ``symbols`` are provided as ``local_dict``; the ``global_dict`` is not
    explicitly set and subsequently uses the sympy default value, containing
    all basic sympy symbols and notations.

    .. note::

        The expression given here is not Python code, but symbolic math.
        You cannot call arbitrary functions, but only those that are imported
        by ``from sympy import *``.

    .. hint::

        When using this expression as part of the :ref:`dag_framework`, it is
        attached to a so-called :ref:`syntax hook <dag_op_hooks_integration>`
        that makes it easier to specify the ``symbols`` parameter.
        See :ref:`here <dag_op_hook_expression>` for more information.

    .. warning::

        While the expression is symbolic math, be aware that smypy by default
        interprets the ``^`` operator as XOR, not an exponentiation!
        For exponentiation, use the ``**`` operator or adjust the
        ``transformations`` argument as specified in the sympy documentation.

    .. warning::

        The return object of this operation will *only* contain symbolic sympy
        objects if ``astype is None``. Otherwise, the type cast will evaluate
        all symbolic objects to the numerical equivalent specified by the given
        ``astype``.

    Args:
        expr (str): The expression to evaluate
        symbols (dict, optional): The symbols to use
        evaluate (bool, optional): Controls whether sympy evaluates ``expr``.
            This *may* lead to a fully evaluated result, but does not guarantee
            that no sympy objects are contained in the result. For ensuring
            a fully numerical result, see the ``astype`` argument.
        transformations (Tuple[Callable], optional): The ``transformations``
            argument for sympy's
            :py:func:`sympy.parsing.sympy_parser.parse_expr`. By default, the
            sympy standard transformations are performed.
        astype (Union[type, str], optional): If given, performs a cast to this
            data type, fully evaluating all symbolic expressions.
            Default: Python :py:class:`float`.

    Raises:
        TypeError: Upon failing ``astype`` cast, e.g. due to free symbols
            remaining in the evaluated expression.
        ValueError: When parsing of ``expr`` failed.

    Returns:
        The result of the evaluated expression.
    """
    from sympy.parsing.sympy_parser import parse_expr as _parse_expr
    from sympy.parsing.sympy_parser import standard_transformations as _std_trf

    log.remark("Evaluating symbolic expression:  %s", expr)

    symbols = symbols if symbols else {}
    parse_kwargs = dict(
        evaluate=evaluate,
        transformations=(
            transformations if transformations is not None else _std_trf
        ),
    )

    # Now, parse the expression
    try:
        res = _parse_expr(expr, local_dict=symbols, **parse_kwargs)

    except Exception as exc:
        raise ValueError(
            f"Failed parsing expression '{expr}'! Got a "
            f"{exc.__class__.__name__}: {exc}. Check that the expression can "
            "be evaluated with the available symbols "
            f"({', '.join(symbols) if symbols else 'none specified'}) "
            "and inspect the chained exceptions for more information. Parse "
            f"arguments were: {parse_kwargs}"
        ) from exc

    # Finished here if no type cast is desired
    if astype is None:
        return res

    # If full evaluation is desired, do so via a numpy type cast. This works on
    # all sympy objects, but _importantly_ also works on numpy arrays
    # containing sympy objects.
    dtype = np.dtype(astype)
    log.debug("Applying %s ...", dtype)

    try:
        return dtype.type(res)

    except Exception as exc:
        raise TypeError(
            f"Failed casting the result of expression '{expr}' from "
            f"{type(res)} to {dtype}! This can also be due to free symbols "
            "remaining in the evaluated expression. Either specify the free "
            f"symbols (got: {', '.join(symbols) if symbols else 'none'}) or "
            "deactivate casting by specifying None as ``dtype`` argument. "
            f"The expression evaluated to:\n\n    {res}\n"
        ) from exc


def generate_lambda(expr: str) -> Callable:
    """Generates a lambda from a string. This is useful when working with
    callables in other operations.

    The ``expr`` argument needs to be a valid Python
    `lambda expression <https://docs.python.org/3/reference/expressions.html#lambda>`_.

    Inside the lambda body, the following names are available for use:

        * A large part of the ``builtins`` module
        * Every name from the Python ``math`` module, e.g. ``sin``, ``cos``, …
        * These modules (and their long form): ``np``, ``xr``, ``scipy``

    Internally, this uses ``eval`` but imposes the following restrictions:

        * The following strings may *not* appear in ``expr``: ``;``, ``__``.
        * There can be no nested ``lambda``, i.e. the only allowed lambda
          string is that in the beginning of ``expr``.
        * The dangerous parts from the ``builtins`` module are *not* available.

    Args:
        expr (str): The expression string to evaluate into a lambda.

    Returns:
        Callable: The generated Callable.

    Raises:
        SyntaxError: Upon failed evaluation of the given expression, invalid
            expression pattern, or disallowed strings in the lambda body.
    """
    ALLOWED_BUILTINS = (
        "abs",
        "all",
        "any",
        "callable",
        "chr",
        "divmod",
        "format",
        "hash",
        "hex",
        "id",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "print",
        "repr",
        "round",
        "sorted",
        "sum",
        "None",
        "Ellipsis",
        "False",
        "True",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "list",
        "map",
        "range",
        "reversed",
        "set",
        "slice",
        "str",
        "tuple",
        "type",
        "zip",
        "open",
    )
    DISALLOWED_STRINGS = ("lambda", ";", "__")
    LAMBDA_PATTERN = r"^\s*lambda\s[\w\s\,\*]+\:(.+)$"
    #                   arguments definitions  : lambda body (capture group)
    # See also:  https://regex101.com/r/OmI8NY/2/

    # Check if the given expression matches this pattern
    pattern = re.compile(LAMBDA_PATTERN)
    match = pattern.match(expr)
    if match is None:
        raise SyntaxError(
            f"The given expression '{expr}' was not a valid lambda expression!"
        )

    # Check if the lambda body contains disallowed strings
    lambda_body = match[1]
    if any([bad_str in lambda_body for bad_str in DISALLOWED_STRINGS]):
        raise SyntaxError(
            "Encountered one or more disallowed strings in the "
            f"body ('{lambda_body}') of the given lambda "
            f"expression ('{expr}'). Make sure none of the "
            "following strings appears there: "
            f"{DISALLOWED_STRINGS}"
        )

    # Ok, sanitized enough now. Prepare the globals dict, restricting access to
    # a subset of builtins and only allowing commonly used math functionality
    _g = dict(
        __builtins__={
            n: f for n, f in __builtins__.items() if n in ALLOWED_BUILTINS
        },
        math=math,
        scipy=scipy,
        pandas=pd,
        numpy=np,
        xarray=xr,
        networkx=nx,
        np=np,
        xr=xr,
        nx=nx,
        pd=pd,
        **{n: f for n, f in math.__dict__.items() if not n.startswith("_")},
    )

    # Try evaluation now (with empty locals)
    try:
        f = eval(expr, _g, {})

    except Exception as exc:
        raise SyntaxError(
            "Failed generating lambda object from expression "
            f"'{expr}'! Got a {exc.__class__.__name__}: {exc}"
        ) from exc

    return f
