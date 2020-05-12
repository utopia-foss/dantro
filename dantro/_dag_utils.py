"""Private low-level helper classes and functions for the DAG framework

NOTE This is imported by dantro.tools to register classes with YAML.
"""
import logging
from typing import Any, Union, Tuple

import sympy as sym
from sympy.parsing.sympy_parser import (parse_expr as _parse_expr,
                                        standard_transformations as _std_trf,
                                        convert_xor as _convert_xor)

# Local constants
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

# Contains hooks that are invoked when a certain operation is parsed.
#
# The values should be callables that receive ``operation, *args, **kwargs``
# and return a 3-tuple of the manipulated ``operation, args, kwargs``.
# The return values will be those that the Transformation object is created
# from.
#
# Example of defining a hook and registering it:
#
# .. code-block:: python
#
#     def _op_hook_my_operation(operation, *args, **kwargs
#                               ) -> Tuple[str, list, dict]:
#         """An operation hook for my_operation"""
#         # ... do stuff ...
#         return operation, args, kwargs
#
#     DAG_PARSER_OPERATION_HOOKS['my_operation'] = _op_hook_my_operation
#
DAG_PARSER_OPERATION_HOOKS = dict()


# -----------------------------------------------------------------------------

class DAGReference:
    """The DAGReference class is the base class of all DAG reference objects.

    While it does not implement __hash__ by itself, it is yaml-representable
    and thus hashable after a parent object created a YAML representation.
    """
    def __init__(self, ref: str):
        """Initialize a DAGReference object from a hash."""
        if not isinstance(ref, str):
            raise TypeError("DAGReference requires a string-like argument, "
                            "got {}!".format(type(ref)))

        self._data = ref

    def __eq__(self, other) -> bool:
        """Only objects with exactly the same type and data are regarded as
        equal; specifically, this makes instances of subclasses always unequal
        to instances of the DAGReference base class.
        """
        if type(other) == type(self):
            return self._data == other._data
        return False

    def __repr__(self) -> str:
        return "<{} {}>".format(type(self).__name__, repr(self._data))

    def __hash__(self) -> int:
        return hash(repr(self))

    @property
    def ref(self) -> str:
        """The associated reference of this object"""
        return self._data

    def _resolve_ref(self, *, dag: 'TransformationDAG') -> str:
        """Return the hash reference; for the base class, the data is already
        the hash reference, so no DAG is needed. Derived classes _might_ need
        the DAG to resolve their reference hash.
        """
        return self._data

    def convert_to_ref(self, *, dag: 'TransformationDAG') -> 'DAGReference':
        """Create a new object that is a hash ref to the same object this
        tag refers to."""
        return DAGReference(self._resolve_ref(dag=dag))

    def resolve_object(self, *, dag: 'TransformationDAG') -> Any:
        """Resolve the object by looking up the reference in the DAG's object
        database.
        """
        return dag.objects[self._resolve_ref(dag=dag)]

    # YAML representation . . . . . . . . . . . . . . . . . . . . . . . . . . .
    yaml_tag = u'!dag_ref'

    @classmethod
    def from_yaml(cls, constructor, node):
        """Construct a DAGReference from a scalar YAML node"""
        return cls(constructor.construct_scalar(node))

    @classmethod
    def to_yaml(cls, representer, node):
        """Create a YAML representation of a DAGReference, carrying only the
        _data attribute over...

        As YAML expects scalar data to be str-like, a type cast is done. The
        subclasses that rely on certain argument types should take care that
        they can parse arguments that are str-like.
        """
        return representer.represent_scalar(cls.yaml_tag, str(node._data))


# .............................................................................

class DAGTag(DAGReference):
    """A DAGTag object stores a name of a tag, which serves as a named
    reference to some object in the DAG.

    While it does not implement __hash__ by itself, it is yaml-representable
    and thus hashable after a parent object created a YAML representation.
    """
    yaml_tag = u'!dag_tag'

    def __init__(self, name: str):
        """Initialize a DAGTag object, storing the specified field name"""
        self._data = name

    @property
    def name(self) -> str:
        """The name of the tag within the DAG that this object references"""
        return self._data

    def _resolve_ref(self, *, dag: 'TransformationDAG') -> str:
        """Return the hash reference by looking up the tag in the DAG"""
        return dag.tags[self.name]

# .............................................................................

class DAGNode(DAGReference):
    """A DAGNode is a reference by the index within the DAG's node list.

    While it does not implement __hash__ by itself, it is yaml-representable
    and thus hashable after a parent object created a YAML representation.
    """
    yaml_tag = u'!dag_node'

    def __init__(self, idx: int):
        """Initialize a DAGNode object with a node index.

        Args:
            idx (int): The idx value to set this reference to. Can also be a
                negative value, in which case the node list is traversed from
                the back.

        Raises:
            TypeError: On invalid type (not int-convertible)
        """
        if not isinstance(idx, int):
            # Try an integer conversion, to be a bit more robust
            try:
                idx = int(idx)
            except:
                raise TypeError("DAGNode requires an int-convertible "
                                "argument, got {}!".format(type(idx)))

        self._data = idx

    @property
    def idx(self) -> int:
        """The idx to the referenced node within the DAG's node list"""
        return self._data

    def _resolve_ref(self, *, dag: 'TransformationDAG') -> str:
        """Return the hash reference by looking up the node index in the DAG"""
        return dag.nodes[self.idx]


# -----------------------------------------------------------------------------

class DAGObjects:
    """An objects database for the DAG framework.

    It uses a flat dict containing (hash, object ref) pairs. The interface is
    slightly restricted compared to a regular dict; especially, item deletion
    is not made available.

    Objects are added to the database via the ``add_object`` method. They need
    to have a ``hashstr`` property, which returns a hash string
    deterministically representing the object; note that this is not
    equivalent to the Python builtin hash() function which invokes the __hash__
    magic method.
    """

    def __init__(self):
        """Initialize an empty objects database"""
        self._d = dict()

    def __str__(self) -> str:
        """A human-readable string representation of the object database"""
        return ("<DAGObjects database with {:d} entr{}>"
                "".format(len(self), "ies" if len(self) != 1 else "y"))

    def add_object(self, obj, *, custom_hash: str=None) -> str:
        """Add an object to the object database, storing it under its hash.

        Note that the object cannot be just any object that is hashable but it
        needs to return a string-based hash via the ``hashstr`` property. This
        is a dantro DAG framework-internal interface.

        Also note that the object will NOT be added if an object with the same
        hash is already present. The object itself is of no importance, only
        the returned hash is.

        Args:
            obj: Some object that has the ``hashstr`` property, i.e. is
                hashable as required by the DAG interface
            custom_hash (str, optional): A custom hash to use instead of the
                hash extracted from ``obj``. Can only be given when ``obj``
                does *not* have a ``hashstr`` property.

        Returns:
            str: The hash string of the given object. If a custom hash string
                was given, it is also the return value

        Raises:
            TypeError: When attempting to pass ``custom_hash`` while ``obj``
                *has* a ``hashstr`` property
            ValueError: If the given ``custom_hash`` already exists.
        """
        if custom_hash is not None:
            if hasattr(obj, 'hashstr'):
                raise TypeError("Cannot use a custom hash for objects that "
                                "provide their own `hashstr` property! Got "
                                "object of type {} and custom hash '{}'."
                                "".format(type(obj), custom_hash))

            elif custom_hash in self:
                raise ValueError("The provided custom hash '{}' for object of "
                                 "type {} already exists! Refusing to add it. "
                                 "Was the object already added? If not, "
                                 "choose a different custom hash."
                                 "".format(custom_hash, type(obj)))
            key = custom_hash

        else:
            # Use the DAG framework's internal hash method
            key = obj.hashstr

        # Only add the new object, if the hash does not exist yet.
        if key not in self:
            self._d[key] = obj
        return key

    def __getitem__(self, key: str) -> object:
        """Return the object associated with the given hash"""
        return self._d[key]

    def __len__(self) -> int:
        """Returns the number of objects in the objects database"""
        return len(self._d)

    def __contains__(self, key: str) -> bool:
        """Whether the given hash refers to an object in this database"""
        return key in self._d

    def keys(self):
        return self._d.keys()

    def values(self):
        return self._d.values()

    def items(self):
        return self._d.items()

# -----------------------------------------------------------------------------

def parse_dag_minimal_syntax(params: Union[str, dict]) -> dict:
    """Parses the minimal syntax parameters, effectively translating a string-
    like argument to a dict with the string specified as the ``operation`` key.
    """
    if isinstance(params, dict):
        return params

    elif isinstance(params, str):
        return dict(operation=params, with_previous_result=True)

    # else:
    raise TypeError("Expected either dict or string for minimal syntax, got "
                    "{} with value: {}".format(type(params), params))


def parse_dag_syntax(*, operation: str=None, args: list=None,
                     kwargs: dict=None, tag: str=None,
                     with_previous_result: bool=False,
                     salt: int=None, file_cache: dict=None,
                     ignore_hooks: bool=False, **ops) -> dict:
    """Given the parameters of a transform operation, possibly in a shorthand
    notation, returns a dict with normalized content by expanding the
    shorthand notation.

    Keys that will be available in the resulting dict:
        ``operation``, ``args``, ``kwargs``, ``tag``.

    Args:
        operation (str, optional): Which operation to carry out; can only be
            specified if there is no ``ops`` argument.
        args (list, optional): Positional arguments for the operation; can
            only be specified if there is no ``ops`` argument.
        kwargs (dict, optional): Keyword arguments for the operation; can only
            be specified if there is no ``ops`` argument.
        tag (str, optional): The tag to attach to this transformation
        with_previous_result (bool, optional): Whether the result of the
            previous transformation is to be used as first positional argument
            of this transformation.
        salt (int, optional): A salt to the Transformation object, thereby
            changing its hash.
        file_cache (dict, optional): File cache parameters
        ignore_hooks (bool, optional): If True, there will be no lookup in the
            operation hooks.
        **ops: The operation that is to be carried out. May contain one and
            only one operation.

    Returns:
        dict: The normalized dict of transform parameters.

    Raises:
        ValueError: For invalid notation, e.g. unambiguous specification of
            arguments or the operation.
    """
    def _raise_error(mode: type, *, operation: str, op_params):
        if mode is dict:
            kind, arg_name = "keyword", "kwargs"
        else:
            kind, arg_name = "positional", "args"

        raise ValueError(
            f"Got superfluous `{arg_name}` argument!"
            f"When specifying {kind} arguments via the shorthand notation "
            f"('{operation}: {op_params}'), there can be no additional "
            f"`{arg_name}` argument specified! Remove that argument or merge "
            f"its content with the arguments specified via the shorthand."
        )

    # Distinguish between explicit and shorthand mode
    if operation and not ops:
        # Explicit parametrization
        args = args if args else []
        kwargs = kwargs if kwargs else {}

    elif ops and not operation:
        # Shorthand parametrization
        # Make sure there are no stray argument
        if len(ops) > 1:
            raise ValueError("For shorthand notation, there can only be a "
                             "single operation specified, but got multiple "
                             f"operations: {ops}")

        # Extract operation name and parameters
        operation, op_params = list(ops.items())[0]

        # Depending on type, regard parameters as args or kwargs. If the
        # argument is not a container, assume it's a single positional
        # argument. The arguments that are not specified by op_params will be
        # set from the existing
        if isinstance(op_params, dict):
            if kwargs:
                _raise_error(dict, operation=operation, op_params=op_params)
            args = args if args else []
            kwargs = op_params

        elif isinstance(op_params, (list, tuple)):
            if args:
                _raise_error(list, operation=operation, op_params=op_params)
            args = list(op_params)
            kwargs = kwargs if kwargs else {}

        elif op_params is not None:
            if args:
                _raise_error(list, operation=operation, op_params=op_params)
            args = [op_params]
            kwargs = kwargs if kwargs else {}

        else:
            args = args if args else []
            kwargs = kwargs if kwargs else {}

    elif not operation and not ops:
        raise ValueError("Missing operation specification. Either use the "
                         "`operation` key to specify one or use shorthand "
                         "notation by using the name of the operation as a "
                         "key and adding the arguments to it as values.")

    else:
        raise ValueError(f"Got two specifications of operations, one via the "
                         f"`operation` argument ('{operation}'), another via "
                         f"the shorthand notation ({ops}). Remove one!")

    # Have variables operation, args, and kwargs set now.

    # If the result is to be carried on, the first _positional_
    # argument is set to be a reference to the previous node
    if with_previous_result:
        args.insert(0, DAGNode(-1))

    # Invoke operation-specific hooks
    if not ignore_hooks and operation in DAG_PARSER_OPERATION_HOOKS:
        hook = DAG_PARSER_OPERATION_HOOKS[operation]
        log.remark("Invoking parser hook for operation '%s' ...", operation)
        try:
            operation, args, kwargs = hook(operation, *args, **kwargs)
        except Exception as exc:
            log.warning("Failed applying operation-specific hook for '%s'! "
                        "Got %s: %s.\nEither correct the error or disable "
                        "the hook for this operation by setting the "
                        "``ignore_hooks`` flag. Otherwise, this operation "
                        "might fail during computation.",
                        operation, exc.__class__.__name__, exc)

    # Done. Construct the dict.
    # Mandatory parameters
    d = dict(operation=operation, args=args, kwargs=kwargs, tag=tag)

    # Add optional parameters only if they were specified
    if salt is not None:
        d['salt'] = salt

    if file_cache is not None:
        d['file_cache'] = file_cache

    return d


# -----------------------------------------------------------------------------
# Operation Hooks
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
    # Extract the expression string
    if len(args) == 1:
        expr = args[0]
    elif len(args) == 2:
        _, expr = args
    else:
        raise TypeError(
            f"Got unexpected positional arguments: {args}; expected either "
            f"(expr,) or (prev_node, expr)."
        )

    # Try to extract all symbols from the expression
    all_symbols = _parse_expr(expr, evaluate=False).atoms(sym.Symbol)

    # Some symbols might already be given; only add those that were not given.
    # Also, convert the ``prev`` and ``previous_result`` symbols the
    # corresponding DAGNode object
    symbols = kwargs.get('symbols', {})
    for symbol in all_symbols:
        symbol = str(symbol)
        if symbol in symbols:
            log.remark("Symbol '%s' was already specified explicitly! It "
                       "will not be replaced.", symbol)
            continue

        if symbol in ('prev', 'previous_result'):
            symbols[symbol] = DAGNode(-1)
        else:
            symbols[symbol] = DAGTag(symbol)

    # For the case of missing ``symbols`` key, need to write it back to kwargs
    kwargs['symbols'] = symbols

    # For args, return _only_ ``expr``, as expected by the operation
    return operation, (expr,), kwargs
DAG_PARSER_OPERATION_HOOKS['expression'] = op_hook_expression
