"""Private low-level helper classes and functions used in :py:mod:`dantro.dag`.

For more information, see :ref:`data transformation framework <dag_framework>`.
"""

import logging
from typing import Any, Tuple, Union

from paramspace.tools import recursive_collect, recursive_replace

# NOTE This is imported by dantro.tools to register classes with YAML.
from .exceptions import *

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class Placeholder:
    """A generic placeholder class for use in the
    :ref:`data transformation framework <dag_framework>`.

    Objects of this class or derived classes are yaml-representable and thus
    hashable after a parent object created a YAML representation. In addition,
    the :py:meth:`.__hash__` method can be used to generate a "hash" that is
    implemented simply via the string representation of this object.

    There are a number of derived classes that play a role as providing
    references within the :py:class:`~dantro.dag.TransformationDAG`:
    :py:class:`~dantro._dag_utils.DAGReference`,
    :py:class:`~dantro._dag_utils.DAGTag`, and
    :py:class:`~dantro._dag_utils.DAGNode`.

    In the context of :ref:`meta operations <dag_meta_ops>`, there are
    placeholder classes for positional and keyword arguments:
    :py:class:`~dantro._dag_utils.PositionalArgument` and
    :py:class:`~dantro._dag_utils.KeywordArgument`.
    """

    __slots__ = ("_data",)

    PAYLOAD_DESC: str = "payload"
    """How to refer to the payload in the ``__str__`` method"""

    def __init__(self, data: Any):
        """Initialize a Placeholder by storing its payload"""
        self._data = data

    def __eq__(self, other) -> bool:
        """Only objects with exactly the same type and data are regarded as
        equal; specifically, this makes instances of subclasses always unequal
        to instances of this base class.
        """
        if type(other) == type(self):
            return self._data == other._data
        return False

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {repr(self._data)}>"

    def __str__(self) -> str:
        return "<{}, {}: {}>".format(
            type(self).__name__, self.PAYLOAD_DESC, self._format_payload()
        )

    def _format_payload(self) -> str:
        return repr(self._data)

    def __hash__(self) -> int:
        """Creates a hash by invoking ``hash(repr(self))``"""
        return hash(repr(self))

    @property
    def data(self) -> Any:
        """The payload of the placeholder"""
        return self._data

    # YAML representation . . . . . . . . . . . . . . . . . . . . . . . . . . .
    yaml_tag = "!dag_placeholder"

    @classmethod
    def from_yaml(cls, constructor, node):
        """Construct a Placeholder from a scalar YAML node"""
        return cls(constructor.construct_scalar(node))

    @classmethod
    def to_yaml(cls, representer, node):
        """Create a YAML representation of a Placeholder, carrying only the
        ``_data`` attribute over...

        As YAML expects scalar data to be str-like, a type cast is done. The
        subclasses that rely on certain argument types should take care that
        their ``__init__`` method can parse arguments that are str-like.
        """
        return representer.represent_scalar(cls.yaml_tag, str(node._data))


class ResultPlaceholder(Placeholder):
    """A placeholder class for a data transformation result.

    This is :ref:`used in the plotting framework <dag_result_placeholder>` to
    inject data transformation results into plot arguments.
    """

    __slots__ = ()
    PAYLOAD_DESC = "result_tag"
    yaml_tag = "!dag_result"

    @property
    def result_name(self) -> str:
        """The name of the transformation result this is a placeholder for"""
        return self._data


# .............................................................................


def resolve_placeholders(
    d: dict,
    *,
    dag: "TransformationDAG",
    Cls: type = ResultPlaceholder,
    **compute_kwargs,
) -> dict:
    """Recursively replaces placeholder objects throughout the given dict.

    Computes :py:class:`~dantro.dag.TransformationDAG` results and replaces
    the placeholder objects with entries from the results dict, thereby
    making it possible to compute configuration values using results of the
    `data transformation framework <dag_framework>`, for example as done in
    the plotting framework; see :ref:`dag_result_placeholder`.

    .. warning::

        While this function has a return value, it resolves the placeholders
        in-place, such that the given ``d`` will be mutated even if the return
        value is ignored on the calling site.

    Args:
        d (dict): The object to replace placeholders in. Will recursively walk
            through all dict- and list-like objects to find placeholders.
        dag (dantro.dag.TransformationDAG): The data transformation tree to
            resolve the placeholders' results from.
        Cls (type, optional): The expected type of the placeholders.
        **compute_kwargs: Passed on to
            :py:meth:`~dantro.dag.TransformationDAG.compute`.
    """
    # First, collect the placeholders
    is_placeholder = lambda obj: isinstance(obj, Cls)
    phs = recursive_collect(d, select_func=is_placeholder)

    # If there weren't any, don't have anything to do
    if not phs:
        log.remark("No placeholders found to resolve.")
        return d

    # Otherwise, get ready for computing results and resolving placeholders
    to_compute = {ph.result_name for ph in phs}

    log.info(
        "Resolving %d placeholder%s ...",
        len(phs),
        "s" if len(phs) != 1 else "",
    )

    try:
        results = dag.compute(compute_only=list(to_compute), **compute_kwargs)

    except ValueError as exc:
        _ph_names = ", ".join(to_compute)
        raise ValueError(
            "Placeholder resolution failed for one or more of the specified "
            f"placeholder names ({_ph_names})!\n{exc}"
        ) from exc

    except RuntimeError as exc:
        _ph_names = ", ".join(to_compute)
        raise RuntimeError(
            "Placeholder resolution failed due to an error during "
            "computation of the transformation result for any of "
            f"the placeholder tags ({_ph_names})!\n{exc}"
        ) from exc

    d = recursive_replace(
        d,
        select_func=is_placeholder,
        replace_func=lambda p: results[p.result_name],
    )
    log.remark("Finished resolving placeholders.")
    return d


# -----------------------------------------------------------------------------
# Used in meta-operations


class PlaceholderWithFallback(Placeholder):
    """A class expanding :py:class:`~Placeholder` that adds the ability to
    read and store a fallback value.
    """

    __slots__ = ("_fallback", "_has_fallback")

    def __init__(self, data: Any, *args):
        super().__init__(data)

        # Evaluate the fallback
        self._fallback = None
        self._has_fallback = False
        if len(args) == 1:
            self._fallback = args[0]
            self._has_fallback = True

        elif len(args) > 1:
            raise TypeError(
                f"{type(self).__name__} only accepts a single fallback value! "
                f"Got:  {args}"
            )

    def __repr__(self) -> str:
        """Representation that includes the fallback value, if there is one."""
        if not self.has_fallback:
            return super().__repr__()
        return "<{} {}, fallback: {}>".format(
            type(self).__name__,
            repr(self._data),
            repr(self._fallback),
        )

    def __str__(self) -> str:
        if not self.has_fallback:
            return super().__str__()
        return "<{}, {}: {}, fallback: {}>".format(
            type(self).__name__,
            self.PAYLOAD_DESC,
            self._format_payload(),
            repr(self._fallback),
        )

    @property
    def fallback(self) -> Any:
        """Returns the fallback value"""
        if not self.has_fallback:
            raise ValueError(f"{self} has no fallback value defined!")
        return self._fallback

    @property
    def has_fallback(self) -> bool:
        """Whether there was a fallback value provided"""
        return self._has_fallback

    @classmethod
    def from_yaml(cls, constructor, node):
        """Constructs a placeholder object from a YAML node.

        For a sequence node, will interpret it as (data, fallback).
        With a scalar node, will not have a fallback.
        """
        import ruamel.yaml

        if isinstance(node, ruamel.yaml.nodes.SequenceNode):
            return cls(*constructor.construct_sequence(node))
        return super().from_yaml(constructor, node)

    @classmethod
    def to_yaml(cls, representer, node):
        """Create a YAML representation of a Placeholder, creating a sequence
        representation in case a fallback value was defined.
        """
        if node.has_fallback:
            return representer.represent_sequence(
                cls.yaml_tag, (node._data, node._fallback)
            )
        return super().to_yaml(representer, node)


class PositionalArgument(PlaceholderWithFallback):
    """A PositionalArgument is a placeholder that holds as payload a positional
    argument's position. This is used, e.g., for meta-operation specification.
    """

    __slots__ = ()
    PAYLOAD_DESC = "position"
    yaml_tag = "!arg"

    def __init__(self, pos: int, *args):
        """Initialize from an integer, also accepting int-convertibles"""
        if not isinstance(pos, int):
            # Need an integer conversion to accept YAML string dumps
            try:
                pos = int(pos)
            except:
                raise TypeError(
                    "PositionalArgument requires an int-convertible argument, "
                    f"got {type(pos)} with value {repr(pos)}!"
                )

        if pos < 0:
            raise ValueError(
                "PositionalArgument requires a non-negative "
                f"position, got {pos}!"
            )

        super().__init__(pos, *args)

    @property
    def position(self) -> int:
        return self._data


class KeywordArgument(PlaceholderWithFallback):
    """A KeywordArgument is a placeholder that holds as payload the name of a
    keyword argument. This is used, e.g., for meta-operation specification.
    """

    __slots__ = ()
    PAYLOAD_DESC = "name"
    yaml_tag = "!kwarg"

    def __init__(self, name: str, *args):
        """Initialize by storing the keyword argument name"""
        if not isinstance(name, str):
            raise TypeError(
                "KeywordArgument requires a string "
                f"as argument name, got {type(name)}!"
            )

        super().__init__(name, *args)

    @property
    def name(self) -> int:
        return self._data


# -----------------------------------------------------------------------------


class DAGReference(Placeholder):
    """The DAGReference class is the base class of all DAG reference objects.
    It extends the generic Placeholder class with the ability to resolve
    references within a :py:class:`~dantro.dag.TransformationDAG`.
    """

    __slots__ = ()
    PAYLOAD_DESC = "hash"
    yaml_tag = "!dag_ref"

    def __init__(self, ref: str):
        """Initialize a DAGReference object from a hash."""
        if not isinstance(ref, str):
            raise TypeError(
                "DAGReference requires a string-like argument, "
                f"got {type(ref)}!"
            )

        self._data = ref

    @property
    def ref(self) -> str:
        """The associated reference of this object"""
        return self._data

    def _format_payload(self) -> str:
        # Make sure to not apply this to derived classes with a different kind
        # of payload, e.g. DAGTag ...
        if self.PAYLOAD_DESC != "hash":
            return str(self._data)
        return f"{self._data[:12]}…"

    def _resolve_ref(self, *, dag: "TransformationDAG") -> str:
        """Return the hash reference; for the base class, the data is already
        the hash reference, so no DAG is needed. Derived classes _might_ need
        the DAG to resolve their reference hash.
        """
        return self._data

    def convert_to_ref(self, *, dag: "TransformationDAG") -> "DAGReference":
        """Create a new object that is a hash ref to the same object this
        tag refers to."""
        return DAGReference(self._resolve_ref(dag=dag))

    def resolve_object(self, *, dag: "TransformationDAG") -> Any:
        """Resolve the object by looking up the reference in the DAG's object
        database.
        """
        return dag.objects[self._resolve_ref(dag=dag)]


# .............................................................................


class DAGTag(DAGReference):
    """A DAGTag object stores a name of a tag, which serves as a named
    reference to some object in the DAG.
    """

    __slots__ = ()
    PAYLOAD_DESC = "tag"
    yaml_tag = "!dag_tag"

    def __init__(self, name: str):
        """Initialize a DAGTag object, storing the specified field name"""
        # Prohibit certain names that would collide with DAGMetaOperationTag
        if DAGMetaOperationTag.SPLIT_STR in name:
            raise ValueError(
                "DAGTag names cannot include the "
                f"'{DAGMetaOperationTag.SPLIT_STR}' substring! "
                f"Adjust the name of tag '{name}' accordingly."
            )

        self._data = name

    @property
    def name(self) -> str:
        """The name of the tag within the DAG that this object references"""
        return self._data

    def _resolve_ref(self, *, dag: "TransformationDAG") -> str:
        """Return the hash reference by looking up the tag in the DAG"""
        try:
            return dag.tags[self.name]

        except KeyError as err:
            _avail = ", ".join(dag.tags)
            raise MissingDAGTag(
                f"Could not find tag '{self.name}'! Available tags: {_avail}\n"
                "Check if you are using the tag before it was defined or if "
                "there is a typo in the tag name."
            ) from err


class DAGMetaOperationTag(DAGTag):
    """A DAGMetaOperationTag stores a name of a tag, just as DAGTag, but can
    only be used inside a meta-operation. When resolving this tag's reference,
    the target is looked up from the stack of the TransformationDAG.
    """

    __slots__ = ()
    PAYLOAD_DESC = "tag"
    yaml_tag = "!mop_tag"

    SPLIT_STR: str = "::"
    """The string by which to split off the meta-operation name from the
    fully qualified tag name."""

    def __init__(self, name: str):
        """Initialize the DAGMetaOperationTag object.

        The ``name`` needs to be of the ``<meta-operation name>::<tag name>``
        pattern and thereby include information on the name of the
        meta-operation this tag is used in.
        """
        try:
            mop, tag = name.split(self.SPLIT_STR)

        except Exception as exc:
            raise ValueError(
                f"Invalid name '{name}' for DAGMetaOperationTag! "
                f"The '{self.SPLIT_STR}' substring "
                "is missing or is used more than once!"
            ) from exc

        self._data = name

    def _resolve_ref(self, *, dag: "TransformationDAG") -> str:
        """Return the hash reference by looking it up in the reference stacks
        of the specified TransformationDAG. The last entry always refers to the
        currently active meta-operation.
        """
        try:
            return dag.ref_stacks[self.name][-1]

        except IndexError as err:
            _avail = ", ".join(dag.ref_stacks)
            raise MissingDAGReference(
                "Failed resolving reference from meta-operation tag "
                f"'{self.name}' via reference stack because the stack was "
                f"empty! Available reference stacks: {_avail}\n"
                "Check if you are using the tag in the meta-operation before "
                "it was defined or if there is a typo in the tag name."
            ) from err

    @classmethod
    def make_name(cls, meta_operation: str, *, tag: str) -> str:
        """Given a meta-operation name and a tag name, generates the name of
        this meta-operation tag.
        """
        return f"{meta_operation}{cls.SPLIT_STR}{tag}"

    @classmethod
    def from_names(
        cls, meta_operation: str, *, tag: str
    ) -> "DAGMetaOperationTag":
        """Generates a DAGMetaOperationTag using the names of a meta-operation
        and the name of a tag.
        """
        return cls(cls.make_name(meta_operation, tag=tag))


# .............................................................................


class DAGNode(DAGReference):
    """A DAGNode is a reference by the index within the DAG's node list."""

    __slots__ = ()
    PAYLOAD_DESC = "node ID"
    yaml_tag = "!dag_node"

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
            # Need an integer conversion to accept YAML string dumps
            try:
                idx = int(idx)
            except:
                raise TypeError(
                    "DAGNode requires an int-convertible argument, got "
                    f"{type(idx)} with value {repr(idx)}!"
                )

        self._data = idx

    @property
    def idx(self) -> int:
        """The idx to the referenced node within the DAG's node list"""
        return self._data

    def _resolve_ref(self, *, dag: "TransformationDAG") -> str:
        """Return the hash reference by looking up the node index in the DAG"""
        try:
            return dag.nodes[self.idx]

        except IndexError as err:
            raise MissingDAGNode(
                f"Failed resolving DAG node via node list: {err}\n"
                f"There are currently {len(dag.nodes)} nodes in the node list "
                f"and the lookup was attempted with dag.nodes[{self.idx}]."
            ) from err


# -----------------------------------------------------------------------------


class DAGObjects:
    """An objects database for the DAG framework.

    It uses a flat dict containing (hash, object ref) pairs. The interface is
    slightly restricted compared to a regular dict; especially, item deletion
    is not made available.

    Objects are added to the database via the ``add_object`` method. They need
    to have a ``hashstr`` property, which returns a hash string
    deterministically representing the object; note that this is not
    equivalent to the Python builtin :py:func:`hash` function which invokes the
    magic ``__hash__`` method of an object.
    """

    def __init__(self):
        """Initialize an empty objects database"""
        self._d = dict()

    def __str__(self) -> str:
        """A human-readable string representation of the object database"""
        return "<DAGObjects database with {:d} entr{}>".format(
            len(self), "ies" if len(self) != 1 else "y"
        )

    def add_object(self, obj, *, custom_hash: str = None) -> str:
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
            if hasattr(obj, "hashstr"):
                raise TypeError(
                    "Cannot use a custom hash for objects that provide their "
                    f"own `hashstr` property! Got object of type {type(obj)} "
                    f"and custom hash '{custom_hash}'."
                )

            elif custom_hash in self:
                raise ValueError(
                    f"The provided custom hash '{custom_hash}' for object of "
                    f"type {type(obj)} already exists! Refusing to add it. "
                    "Was the object already added? If not, choose a different "
                    "custom hash."
                )

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


def parse_dag_minimal_syntax(
    params: Union[str, dict], *, with_previous_result: bool = True
) -> dict:
    """Parses the minimal syntax parameters, effectively translating a string-
    like argument to a dict with the string specified as the ``operation`` key.
    """
    if isinstance(params, dict):
        # Not actually minimal syntax, but a passthrough
        return params

    elif isinstance(params, str):
        return dict(
            operation=params,
            with_previous_result=with_previous_result,
            context=dict(from_minimal_syntax=True, spec=params),
        )

    # else:
    raise TypeError(
        "Expected either dict or string for minimal syntax, got "
        f"{type(params)} with value: {params}"
    )


def parse_dag_syntax(
    *,
    operation: str = None,
    args: list = None,
    kwargs: dict = None,
    tag: str = None,
    force_compute: bool = None,
    with_previous_result: bool = False,
    salt: int = None,
    file_cache: dict = None,
    ignore_hooks: bool = False,
    allow_failure: Union[bool, str] = None,
    fallback: Any = None,
    context: dict = None,
    **ops,
) -> dict:
    """Given the parameters of a transform operation, possibly in a shorthand
    notation, returns a dict with normalized content by expanding the
    shorthand notation. The return value is then suited to initialize a
    :py:class:`~dantro.dag.Transformation` object.

    Keys that will always be available in the resulting dict:
        ``operation``, ``args``, ``kwargs``, ``tag``.

    Optionally available keys:
        ``salt``, ``file_cache``, ``allow_failure``, ``fallback``, ``context``.

    Args:
        operation (str, optional): Which operation to carry out; can only be
            specified if there is no ``ops`` argument.
        args (list, optional): Positional arguments for the operation; can
            only be specified if there is no ``ops`` argument.
        kwargs (dict, optional): Keyword arguments for the operation; can only
            be specified if there is no ``ops`` argument.
        tag (str, optional): The tag to attach to this transformation
        force_compute (bool, optional): Whether to force computation for this
            node.
        with_previous_result (bool, optional): Whether the result of the
            previous transformation is to be used as first positional argument
            of this transformation.
        salt (int, optional): A salt to the Transformation object, thereby
            changing its hash.
        file_cache (dict, optional): File cache parameters
        ignore_hooks (bool, optional): If True, there will be no lookup in the
            operation hooks. See :ref:`dag_op_hooks` for more info.
        allow_failure (Union[bool, str], optional): Whether this Transformation
            allows failure during computation. See :ref:`dag_error_handling`.
        fallback (Any, optional): The fallback value to use in case of failure.
        context (dict, optional): Context information, which may be a dict
            containing any form of data and which is carried through to the
            :py:attr:`~dantro.dag.Transformation.context` attribute.
        **ops: The operation that is to be carried out. May contain one and
            only one operation where the key refers to the name of the
            operation and the value refers to positional or keyword arguments,
            depending on type.

    Returns:
        dict: The normalized dict of transform parameters, suitable for
            initializing a :py:class:`~dantro.dag.Transformation` object.

    Raises:
        ValueError: For invalid notation, e.g. unambiguous specification of
            arguments or the operation.
    """
    from .data_ops.hooks import DAG_PARSER_OPERATION_HOOKS

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
            "its content with the arguments specified via the shorthand."
        )

    # Distinguish between explicit and shorthand mode
    if operation and not ops:
        # Explicit parametrization
        args = args if args else []
        kwargs = kwargs if kwargs else {}

    elif ops and not operation:
        # Shorthand parametrization
        # Make sure there are no stray arguments
        if len(ops) > 1:
            raise ValueError(
                "For shorthand notation, there can only be a "
                "single operation specified, but got multiple "
                f"operations: {ops}"
            )

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
        raise ValueError(
            "Missing operation specification. Either use the "
            "`operation` key to specify one or use shorthand "
            "notation by using the name of the operation as a "
            "key and adding the arguments to it as values."
        )

    else:
        raise ValueError(
            "Got two specifications of operations, one via the "
            f"`operation` argument ('{operation}'), another via "
            f"the shorthand notation ({ops}). Remove one!"
        )

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
            log.warning(
                "Failed applying operation-specific hook for '%s'! "
                "Got %s: %s.\nEither correct the error or disable "
                "the hook for this operation by setting the "
                "``ignore_hooks`` flag. Otherwise, this operation "
                "might fail during computation.",
                operation,
                exc.__class__.__name__,
                exc,
            )

    # Done. Construct the dict.
    # Mandatory parameters
    d = dict(operation=operation, args=args, kwargs=kwargs, tag=tag)

    # Add optional parameters only if they were specified
    if salt is not None:
        d["salt"] = salt

    if file_cache is not None:
        d["file_cache"] = file_cache

    if allow_failure is not None:
        d["allow_failure"] = allow_failure

    if fallback is not None:
        d["fallback"] = fallback

    if context is not None:
        d["context"] = context

    if force_compute is not None:
        d["force_compute"] = force_compute

    return d
