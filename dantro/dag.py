"""This is an implementation of a DAG for transformations on dantro objects.
It revolves around two main classes:

- :py:class:`~dantro.dag.Transformation` that represents a data transformation.
- :py:class:`~dantro.dag.TransformationDAG` that aggregates those
  transformations into a directed acyclic graph.

For more information, see :ref:`data transformation framework <dag_framework>`."""

import copy
import glob
import logging
import os
import pickle as _pickle
import sys
import time
import warnings
from collections import defaultdict as _defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, Union

import numpy as np
from paramspace.tools import recursive_collect, recursive_replace

from ._copy import _deepcopy, _shallowcopy
from ._dag_utils import DAGMetaOperationTag as _MOpTag
from ._dag_utils import DAGNode, DAGObjects, DAGReference, DAGTag
from ._dag_utils import KeywordArgument as _Kwarg
from ._dag_utils import Placeholder as _Placeholder
from ._dag_utils import PositionalArgument as _Arg
from ._dag_utils import ResultPlaceholder as _ResultPlaceholder
from ._dag_utils import parse_dag_minimal_syntax as _parse_dag_minimal_syntax
from ._dag_utils import parse_dag_syntax as _parse_dag_syntax
from ._hash import FULL_HASH_LENGTH, SHORT_HASH_LENGTH, _hash
from ._import_tools import LazyLoader, resolve_types
from .abc import PATH_JOIN_CHAR, AbstractDataContainer
from .base import BaseDataGroup
from .containers import NumpyDataContainer, ObjectContainer, XrDataContainer
from .data_loaders import LOADER_BY_FILE_EXT
from .data_ops import (
    apply_operation,
    available_operations,
    get_operation,
    register_operation,
)
from .exceptions import *
from .tools import adjusted_log_levels as _adjusted_log_levels
from .tools import format_time as _format_time
from .tools import make_columns as _make_columns
from .tools import recursive_update as _recursive_update
from .utils import KeyOrderedDict
from .utils.nx import ATTR_MAPPER_OP_PREFIX, ATTR_MAPPER_OP_PREFIX_DAG
from .utils.nx import manipulate_attributes as _manipulate_attributes

# Local constants .............................................................

log = logging.getLogger(__name__)

# A locally used time formatting function
_fmt_time = lambda seconds: _format_time(seconds, ms_precision=2)

# Lazy module imports
xr = LazyLoader("xarray")
pkl = LazyLoader("dill")

DAG_CACHE_DM_PATH = "cache/dag"
"""The path within the :py:class:`~dantro.dag.TransformationDAG` associated
:py:class:`~dantro.data_mngr.DataManager` to which caches are loaded
"""

DAG_CACHE_CONTAINER_TYPES_TO_UNPACK = (
    ObjectContainer,
    XrDataContainer,
)
"""Types of containers that should be unpacked after loading from cache because
having them wrapped into a dantro object is not desirable after loading them
from cache (e.g. because the name attribute is shadowed by tree objects ...)
"""

# fmt: off
DAG_CACHE_RESULT_SAVE_FUNCS = {
    # Specific dantro types
    (NumpyDataContainer,): (
        lambda obj, p, **kws: obj.save(p + ".npy", **kws)
    ),
    (XrDataContainer,): (
        lambda obj, p, **kws: obj.save(p + ".xrdc", **kws)
    ),

    # External package types; use module strings for delayed import
    (np.ndarray,): (
        lambda obj, p, **kws: np.save(p + ".npy", obj, **kws)
    ),
    ("xarray.DataArray",): (
        lambda obj, p, **kws: obj.to_netcdf(p + ".nc_da", **kws)
    ),
    ("xarray.Dataset",): (
        lambda obj, p, **kws: obj.to_netcdf(p + ".nc_ds", **kws)
    ),
}
"""Functions that can store the DAG computation result objects, distinguishing
by their type.
"""
# NOTE It is important that these methods all _overwrite_ an already existing
#      file at the given location _by default_!
# fmt: on


# -----------------------------------------------------------------------------


class Transformation:
    """A transformation is the collection of an N-ary operation and its inputs.

    Transformation objects store the name of the operation that is to be
    carried out and the arguments that are to be fed to that operation. After
    a Transformation is defined, the only interaction with them is via the
    :py:meth:`.compute` method.

    For computation, the arguments are recursively inspected for whether there
    are any DAGReference-derived objects; these need to be resolved first,
    meaning they are looked up in the DAG's object database and -- if they are
    another Transformation object -- their result is computed. This can lead
    to a traversal along the DAG.

    .. warning::

        Objects of this class should under *no* circumstances be changed after
        they were created! For performance reasons, the
        :py:attr:`~dantro.dag.Transformation.hashstr` property is cached; thus,
        changing attributes that are included into the hash computation will
        not lead to a new hash, hence silently creating wrong behaviour.

        All relevant attributes (``operation``, ``args``, ``kwargs``, ``salt``)
        are thus set read-only. This should be respected!
    """

    __slots__ = (
        "_operation",
        "_args",
        "_kwargs",
        "_dag",
        "_salt",
        "_allow_failure",
        "_fallback",
        "_hashstr",
        "_status",
        "_layer",
        "_context",
        "_profile",
        "_fc_opts",
        "_cache",
    )

    def __init__(
        self,
        *,
        operation: str,
        args: Sequence[Union[DAGReference, Any]],
        kwargs: Dict[str, Union[DAGReference, Any]],
        dag: "TransformationDAG" = None,
        salt: int = None,
        allow_failure: Union[bool, str] = None,
        fallback: Any = None,
        file_cache: dict = None,
        context: dict = None,
    ):
        """Initialize a Transformation object.

        Args:
            operation (str): The operation that is to be carried out.
            args (Sequence[Union[DAGReference, Any]]): Positional arguments
                for the operation.
            kwargs (Dict[str, Union[DAGReference, Any]]): Keyword arguments
                for the operation. These are internally stored as a
                :py:class:`~dantro.utils.ordereddict.KeyOrderedDict`.
            dag (TransformationDAG, optional): An associated DAG that is needed
                for object lookup. Without an associated DAG, args or kwargs
                may NOT contain any object references.
            salt (int, optional): A hashing salt that can be used to let this
                specific Transformation object have a different hash than other
                objects, thus leading to cache misses.
            allow_failure (Union[bool, str], optional): Whether the
                computation of this operation or its arguments may fail.
                In case of failure, the ``fallback`` value is used.
                If ``True`` or ``'log'``, will emit a log message upon failure.
                If ``'warn'``, will issue a warning. If ``'silent'``, will use
                the fallback without *any* notification of failure.
                Note that the failure may occur not only during computation of
                this transformation's operation, but also during the recursive
                computation of the referenced arguments. In other words, if the
                computation of an upstream dependency failed, the fallback will
                be used as well.
            fallback (Any, optional): If ``allow_failure`` was set, specifies
                the alternative value to use for this operation. This may in
                turn be a reference to another DAG node.
            file_cache (dict, optional): File cache options. Expected keys are
                ``write`` (boolean or dict) and ``read`` (boolean or dict).
                Note that the options given here are NOT reflected in the hash
                of the object!

                The following arguments are possible under the ``read`` key:

                    enabled (bool, optional):
                        Whether it should be attempted to read from the file
                        cache.
                    load_options (dict, optional):
                        Passed on to the method that loads the cache,
                        :py:meth:`~dantro.data_mngr.DataManager.load`.

                Under the ``write`` key, the following arguments are possible.
                They are evaluated in the order that they are listed here.
                See :py:meth:`~dantro.dag.Transformation._cache_result` for
                more information.

                    enabled (bool, optional):
                        Whether writing is enabled at all
                    always (bool, optional):
                        If given, will always write.
                    allow_overwrite (bool, optional):
                        If False, will not write a cache file if one already
                        exists. If True, a cache file *might* be written,
                        although one already exists. This is still conditional
                        on the evaluation of the other arguments.
                    min_size (int, optional):
                        The *minimum* size of the result object that allows
                        writing the cache.
                    max_size (int, optional):
                        The *maximum* size of the result object that allows
                        writing the cache.
                    min_compute_time (float, optional):
                        The minimal individual computation time of this node
                        that is needed in order for the file cache to be
                        written.
                        *Note* that this value can be lower if the node result
                        is not computed but looked up from the cache.
                    min_cumulative_compute_time (float, optional):
                        The minimal cumulative computation time of this node
                        and all its dependencies that is needed in order for
                        the file cache to be written.
                        *Note* that this value can be lower if the node result
                        is not computed but looked up from the cache.
                    storage_options (dict, optional):
                        Passed on to the cache storage method,
                        :py:meth:`._write_to_cache_file`.
                        The following arguments are available:

                        ignore_groups (bool, optional):
                            Whether to store groups. Disabled by default.
                        attempt_pickling (bool, optional):
                            Whether it should be attempted to store results
                            that could not be stored via a dedicated storage
                            function by pickling them. Enabled by default.
                        raise_on_error (bool, optional):
                            Whether to raise on error to store a result.
                            Disabled by default; it is useful to enable this
                            when debugging.
                        pkl_kwargs (dict, optional):
                            Arguments passed on to the pickle.dump function.
                        further keyword arguments:
                            Passed on to the chosen storage method.
            context (dict, optional): Some meta-data stored alongside the
                Transformation, e.g. containing information about the context
                it was created in. This is not taken into account for the hash.
        """
        self._operation = operation
        self._args = args
        self._kwargs = KeyOrderedDict(**kwargs)
        self._dag = dag
        self._salt = salt
        self._allow_failure = allow_failure
        self._fallback = fallback
        self._hashstr = None
        self._status = None
        self._layer = None
        self._context = context if context else {}
        self._profile = dict(
            compute=np.nan,
            cumulative_compute=np.nan,
            cache_lookup=np.nan,
            cache_writing=np.nan,
            effective=np.nan,
        )

        # Fallback may only be given if ``allow_failure`` evaluated to true.
        if not allow_failure and fallback is not None:
            raise ValueError(
                "The `fallback` argument for a Transformation may only be "
                f"passed with `allow_failure` set! Got: {repr(fallback)}"
            )

        # allow_failure may only take certain values
        _allowed = ("log", "warn", "silent")
        if isinstance(allow_failure, str) and allow_failure not in _allowed:
            _allowed = ", ".join(_allowed)
            raise ValueError(
                f"Invalid `allow_failure` argument '{allow_failure}'. Choose "
                f"from: True, False, {_allowed}."
            )

        # Parse file cache options, making sure it's a dict with default values
        self._fc_opts = file_cache if file_cache is not None else {}

        if isinstance(self._fc_opts.get("write", {}), bool):
            self._fc_opts["write"] = dict(enabled=self._fc_opts["write"])
        elif "write" not in self._fc_opts:
            self._fc_opts["write"] = dict(enabled=False)

        if isinstance(self._fc_opts.get("read", {}), bool):
            self._fc_opts["read"] = dict(enabled=self._fc_opts["read"])
        elif "read" not in self._fc_opts:
            self._fc_opts["read"] = dict(enabled=False)

        # Cache dict, containing the result and whether the cache is in memory
        self._cache = dict(result=None, filled=False)

        # Set the status
        self.status = "initialized"

    # .........................................................................
    # String representation and hashing

    def __str__(self) -> str:
        """A human-readable string characterizing this Transformation"""
        suffix = ""
        if self._allow_failure:
            suffix = ", allows failure"

        return (
            "<{t:}, operation: {op:}, {Na:d} args, {Nkw:d} kwargs{suffix:}>\n"
            "  args:      {args:}\n"
            "  kwargs:    {kwargs:}\n"
            "  fallback:  {fallback:}\n"
            "".format(
                t=type(self).__name__,
                op=self._operation,
                Na=len(self._args),
                Nkw=len(self._kwargs),
                args=self._args,
                kwargs=self._kwargs,
                fallback=self._fallback,
                suffix=suffix,
            )
        )

    def __repr__(self) -> str:
        """A deterministic string representation of this transformation.

        .. note::

            This is also used for hash creation, thus it does not include the
            attributes that are set via the initialization arguments ``dag``
            and ``file_cache``.

        .. warning::

            Changing this method will lead to cache invalidations!
        """
        suffix = ""
        if self._allow_failure:
            suffix = f", fallback={repr(self._fallback)}"

        return (
            "<{mod:}.{t:}, operation={op:}, args={args:}, "
            "kwargs={kwargs:}, salt={salt:}{suffix:}>"
            "".format(
                mod=type(self).__module__,
                t=type(self).__name__,
                op=repr(self._operation),
                args=repr(self._args),
                kwargs=repr(dict(self._kwargs)),
                salt=repr(self._salt),
                suffix=suffix,
            )
        )

    @property
    def hashstr(self) -> str:
        """Computes the hash of this Transformation by creating a deterministic
        representation of this Transformation using ``__repr__`` and then
        applying a checksum hash function to it.

        Note that this does NOT rely on the built-in hash function but on the
        custom dantro ``_hash`` function which produces a platform-independent
        and deterministic hash. As this is a *string*-based (rather than an
        integer-based) hash, it is not implemented as the ``__hash__`` magic
        method but as this separate property.

        Returns:
            str: The hash string for this transformation
        """
        if self._hashstr is None:
            self._hashstr = _hash(repr(self))

        return self._hashstr

    def __hash__(self) -> int:
        """Computes the python-compatible integer hash of this object from the
        string-based hash of this Transformation.
        """
        return hash(self.hashstr)

    # .........................................................................
    # Properties

    @property
    def operation(self) -> str:
        """The operation this transformation performs"""
        return self._operation

    @property
    def dag(self) -> "TransformationDAG":
        """The associated TransformationDAG; used for object lookup"""
        return self._dag

    @property
    def dependencies(self) -> Set[DAGReference]:
        """Recursively collects the references that are found in the positional
        and keyword arguments of this Transformation as well as in the fallback
        value.
        """
        return set(
            recursive_collect(
                chain(self._args, self._kwargs.values(), (self._fallback,)),
                select_func=(lambda o: isinstance(o, DAGReference)),
            )
        )

    @property
    def resolved_dependencies(self) -> Set["Transformation"]:
        """Transformation objects that this Transformation depends on"""
        return {ref.resolve_object(dag=self.dag) for ref in self.dependencies}

    @property
    def profile(self) -> Dict[str, float]:
        """The profiling data for this transformation"""
        return self._profile

    @property
    def has_result(self) -> bool:
        """Whether there is a memory-cached result available for this
        transformation."""
        return self._cache["filled"]

    @property
    def status(self) -> str:
        """Return this Transformation's status which is one of:

        - ``initialized``: set after initialization
        - ``queued``: queued for computation
        - ``computed``: successfully computed
        - ``used_fallback``: if a fallback value was used instead
        - ``looked_up``: after *file* cache lookup
        - ``failed_here``: if computation failed *in this node*
        - ``failed_in_dependency``: if computation failed *in a dependency*
        """
        return self._status

    @status.setter
    def status(self, new_status: str):
        """Sets the status of this Transformation"""
        ALLOWED_STATUS = (
            "initialized",
            "queued",
            "computed",
            "used_fallback",
            "looked_up",
            "failed_here",
            "failed_in_dependency",
        )
        if new_status not in ALLOWED_STATUS:
            _avail = ", ".join(ALLOWED_STATUS)
            raise ValueError(
                f"Invalid status '{new_status}'! Choose from: {_avail}"
            )

        self._status = new_status

    @property
    def layer(self) -> int:
        """Returns the layer this node can be placed at within the DAG by
        recursively going over dependencies and setting the layer to the
        maximum layer of the dependencies plus one.

        Computation occurs upon first invocation, afterwards the cached value
        is returned.

        .. note::

            Transformations without dependencies have a level of zero.
        """
        if self._layer is None:
            deps = self.resolved_dependencies
            if not deps:
                self._layer = 0
            else:
                get_layer = lambda obj: getattr(obj, "layer", 0)
                self._layer = max(get_layer(dep) for dep in deps) + 1

        return self._layer

    @property
    def context(self) -> dict:
        """Returns a dict that holds information about the context this
        transformation was created in."""
        return self._context

    # YAML representation .....................................................
    yaml_tag = "!dag_trf"

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(**constructor.construct_mapping(node, deep=True))

    @classmethod
    def to_yaml(cls, representer, node):
        """A YAML representation of this Transformation, including all its
        arguments (which must again be YAML-representable). In essence, this
        returns a YAML mapping that has the ``!dag_trf`` YAML tag prefixed,
        such that *reading* it in will lead to the ``from_yaml`` method being
        invoked.

        .. note::

            The YAML representation does *not* include the ``file_cache``
            parameters.

        """
        # Collect the attributes that are relevant for the transformation.
        d = dict(
            operation=node._operation,
            args=node._args,
            kwargs=dict(node._kwargs),
        )

        # Add other arguments only if they differ from the defaults
        if node._salt is not None:
            d["salt"] = node._salt

        if node._allow_failure:
            d["allow_failure"] = node._allow_failure
            d["fallback"] = node._fallback

        if node._context:
            d["context"] = node._context

        # Let YAML represent this as a mapping with an additional tag
        return representer.represent_mapping(cls.yaml_tag, d)

    # .........................................................................
    # Compute interface

    def compute(self) -> Any:
        """Computes the result of this transformation by recursively resolving
        objects and carrying out operations.

        This method can also be called if the result is already computed; this
        will lead only to a cache-lookup, not a re-computation.

        Returns:
            Any: The result of the operation
        """

        # Try to look up an already computed result from memory or file cache
        success, res = self._lookup_result()

        if not success:
            # Did not find a result in memory or file cache -> Compute it.

            # Set the status; at this point, this node is not yet computed
            # (first the dependencies need to be resolved) but queued for it.
            self.status = "queued"

            # First, compute the result of the references in the arguments.
            try:
                args = self._resolve_refs(self._args)
                kwargs = self._resolve_refs(self._kwargs)

            except DataOperationFailed as err:
                # Upstream error; skip further computation, potentially use
                # fallback as result (if allowed)
                self.status = "failed_in_dependency"
                res = self._handle_error_and_fallback(
                    err,
                    context="resolution of positional and keyword arguments",
                )

            else:
                # No error; carry out the operation with the resolved arguments
                res = self._perform_operation(args=args, kwargs=kwargs)

        # Allow caching the result, even if it comes from the cache
        self._cache_result(res)

        return res

    def _perform_operation(self, *, args: list, kwargs: dict) -> Any:
        """Perform the operation, updating the profiling info on the side

        Args:
            args (list): The positional arguments to the operation
            kwargs (dict): The keyword arguments to the operation

        Returns:
            Any: The result of the operation

        Raises:
            BadOperationName: Upon bad operation or meta-operation name
            DataOperationFailed: Upon failure to perform the operation
        """
        t0 = time.time()

        # Actually perform the operation, separately handling invalid operation
        # names or the case where failure was actually allowed.
        try:
            res = apply_operation(self._operation, *args, **kwargs)

        except BadOperationName as err:
            self.status = "failed_here"
            _meta_ops = self.dag.meta_operations
            if _meta_ops:
                _meta_ops = "\n" + _make_columns(self.dag.meta_operations)
            else:
                _meta_ops = " (none)\n"

            raise BadOperationName(
                "Could not find an operation or meta-operation named "
                f"'{self._operation}'!\n\n"
                f"{err}\n\n"
                f"Available meta-operations:{_meta_ops}"
                "To register a new meta-operation, specify it during "
                "initialization of the TransformationDAG."
            )

        except DataOperationFailed as err:
            # Operation itself failed. Handle error, potentially re-raising.
            self.status = "failed_here"
            res = self._handle_error_and_fallback(err, context="computation")

        else:
            self.status = "computed"

        # Parse profiling info and return the result
        self._update_profile(cumulative_compute=(time.time() - t0))

        return res

    def _resolve_refs(self, cont: Sequence) -> Sequence:
        """Resolves DAG references within a deepcopy of the given container by
        iterating over it and computing the referenced nodes.

        Args:
            cont (Sequence): The container containing the references to resolve
        """

        def is_DAGReference(obj: Any) -> bool:
            return isinstance(obj, DAGReference)

        def resolve_and_compute(ref: DAGReference):
            """Resolve references to their objects, if necessary computing the
            results of referenced Transformation objects recursively.
            """
            if self.dag is None:
                raise ValueError(
                    "Cannot resolve Transformation arguments "
                    "that contain DAG references, because no DAG "
                    "was associated with this Transformation!"
                )

            # Let the reference resolve the corresponding object from the DAG
            obj = ref.resolve_object(dag=self.dag)

            # Check if this refers to an object that is NOT a transformation.
            # This could be the DataManager or the DAG itself, but also other
            # objects in the DAG's object database. Performing computation on
            # those is either not possible or would lead to infinite loops.
            if not isinstance(obj, Transformation):
                return obj

            # else: It is another Transformation object. Compute it, which
            # leads to a traversal up the DAG tree.
            return obj.compute()

        # Work on a deep copy; otherwise objects are resolved in-place, which
        # is not desirable as it would change the hashstr by populating it with
        # non-trivial objects. Deep copy is always possible because the given
        # containers are expected to contain only trivial items or references.
        cont = _deepcopy(cont)

        return recursive_replace(
            cont,
            select_func=is_DAGReference,
            replace_func=resolve_and_compute,
        )

    def _handle_error_and_fallback(
        self, err: Exception, *, context: str
    ) -> Any:
        """Handles an error that occured during application of the operation
        or during resolving of arguments (and the recursively invoked
        computations on dependent nodes).

        Without error handling enabled, this will directly re-raise the active
        exception. Otherwise, it will generate a log message and will resolve
        the fallback value.
        """
        if not self._allow_failure:
            raise

        # Generate and communicate the message
        msg = (
            f"Operation '{self.operation}' failed during {context}, but was "
            "allowed to fail; using fallback instead. To suppress this "
            f"message, set `allow_failure: silent`.\n\nThe error was:  {err}"
        )
        if self._allow_failure in (True, "log"):
            log.caution(msg)
        elif self._allow_failure in ("warn",):
            warnings.warn(msg, DataOperationWarning)
        else:
            log.trace(msg)

        # Use the fallback; wrapping it in a 1-list to allow scalars
        log.debug("Using fallback for operation '%s' ...", self.operation)
        res = self._resolve_refs([self._fallback])[0]
        self.status = "used_fallback"
        return res

    def _update_profile(
        self, *, cumulative_compute: float = None, **times
    ) -> None:
        """Given some new profiling times, updates the profiling information.

        Args:
            cumulative_compute (float, optional): The cumulative computation
                time; if given, additionally computes the computation time for
                this individual node.
            **times: Valid profiling data.
        """
        # If cumulative computation time was given, calculate individual time
        if cumulative_compute is not None:
            self._profile["cumulative_compute"] = cumulative_compute

            # Aggregate the dependencies' cumulative computation times
            deps_cctime = sum(
                dep.profile["cumulative_compute"]
                for dep in self.resolved_dependencies
                if isinstance(dep, Transformation)
            )
            # NOTE The dependencies might not have this value set because there
            #      might have been a cache lookup
            self._profile["compute"] = max(
                0.0, cumulative_compute - deps_cctime
            )

        # Store the remaining entries
        self._profile.update(times)

        # Update effective time
        self._profile["effective"] = sum(
            self._profile[k]
            for k in ("compute", "cache_lookup", "cache_writing")
            if not np.isnan(self._profile[k])
        )

    # Cache handling . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    def _lookup_result(self) -> Tuple[bool, Any]:
        """Look up the transformation result to spare re-computation"""
        success, res = False, None

        # Retrieve cache parameters
        read_opts = self._fc_opts.get("read", {})
        load_opts = read_opts.get("load_options", {})

        # Check if the cache is already filled. If not, see if the file cache
        # can be read and is configured to be read.
        if self._cache["filled"]:
            success = True
            res = self._cache["result"]
            log.trace("Re-using memory-cached result for %s.", self.hashstr)
            # Not setting status here because it was set previously

        elif self.dag is not None and read_opts.get("enabled", False):
            t0 = time.time()

            # Let the DAG check if there is a file cache, i.e. if a file with
            # this Transformation's hash exists in the DAG's cache directory.
            success, res = self.dag._retrieve_from_cache_file(
                self.hashstr, **load_opts
            )

            # Store the result
            if success:
                self._cache["result"] = res
                self._cache["filled"] = True
                self.status = "looked_up"

            self._update_profile(cache_lookup=(time.time() - t0))

        return success, res

    def _cache_result(self, result: Any) -> None:
        """Stores a computed result in the cache"""

        def should_write(
            *,
            enabled: bool,
            always: bool = False,
            allow_overwrite: bool = False,
            min_size: int = None,
            max_size: int = None,
            min_compute_time: float = None,
            min_cumulative_compute_time: float = None,
            storage_options: dict = None,
        ) -> bool:
            """A helper function to evaluate _whether_ the file cache is to be
            written or not.

            Args:
                enabled (bool): Whether writing is enabled at all
                always (bool, optional): If given, will always write.
                allow_overwrite (bool, optional): If False, will not write a
                    cache file if one already exists. If True, a cache file
                    _might_ be written, although one already exists. This is
                    still conditional on the evaluation of the other arguments.
                min_size (int, optional): The minimum size of the result object
                    that allows writing the cache.
                max_size (int, optional): The maximum size of the result object
                    that allows writing the cache.
                min_compute_time (float, optional): The minimal individual
                    computation time of this node that is needed in order for
                    the file cache to be written. Note that this value can be
                    lower if the node result is not computed but looked up from
                    the cache.
                min_cumulative_compute_time (float, optional): The minimal
                    cumulative computation time of this node and all its
                    dependencies that is needed in order for the file cache to
                    be written. Note that this value can be lower if the node
                    result is not computed but looked up from the cache.
                storage_options (dict, optional): (ignored here)

            Returns:
                bool: Whether to write the file cache or not.
            """
            if not enabled:
                # ... nothing else to check
                return False

            # With always: always write, don't look at other arguments.
            if always:
                return True
            # All checks below are formulated such that they return False.

            # If overwriting is _disabled_ and a cache file already exists, it
            # is already clear that a new one should _not_ be written
            if not allow_overwrite and self.hashstr in self.dag.cache_files:
                return False

            # Evaluate profiling information
            if min_compute_time is not None:
                if self.profile["compute"] < min_compute_time:
                    return False

            if min_cumulative_compute_time is not None:
                if (
                    self.profile["cumulative_compute"]
                    < min_cumulative_compute_time
                ):
                    return False

            # Evaluate object size
            if min_size is not None or max_size is not None:
                size_itvl = [
                    min_size if min_size is not None else 0,
                    max_size if max_size is not None else np.inf,
                ]
                obj_size = sys.getsizeof(result)  # from outer scope

                if not (size_itvl[0] < obj_size < size_itvl[1]):
                    return False

            # If this point is reached, the cache file should be written.
            return True

        # Store a reference to the result and mark the cache as being in use
        self._cache["result"] = result
        self._cache["filled"] = True
        # NOTE If instead of a proper computation, the passed result object was
        #      previously looked up from the cache, this will not have an
        #      effect.

        # Get file cache writing parameters; don't write if not
        write_opts = self._fc_opts["write"]

        # Determine whether to write to a file
        if self.dag is not None and should_write(**write_opts):
            # Setup profiling
            t0 = time.time()

            # Write the result to a file inside the DAG's cache directory. This
            # is handled by the DAG itself, because the Transformation does not
            # know (and should not care) aboute the cache directory ...
            storage_opts = write_opts.get("storage_options", {})
            self.dag._write_to_cache_file(
                self.hashstr, result=result, **storage_opts
            )

            self._update_profile(cache_writing=(time.time() - t0))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class TransformationDAG:
    """This class collects :py:class:`.Transformation` objects that are
    (already by their own structure) connected into a directed acyclic graph.
    The aim of this class is to maintain base objects, manage references, and
    allow operations on the DAG, the most central of which is computing the
    result of a node.

    Furthermore, this class also implements *caching* of transformations, such
    that operations that take very long can be stored (in memory or on disk) to
    speed up future operations.

    Objects of this class are initialized with dict-like arguments which
    specify the transformation operations. There are some shorthands that allow
    a simple definition syntax, for example the ``select`` syntax, which takes
    care of selecting a basic set of data from the associated
    :py:class:`~dantro.data_mngr.DataManager`.

    See :ref:`dag_framework` for more information and examples.
    """

    SPECIAL_TAGS: Sequence[str] = ("dag", "dm", "select_base")
    """Tags with special meaning"""

    NODE_ATTR_DEFAULT_MAPPERS: Dict[str, str] = {
        "layer": f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_layer",
        "operation": f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_operation",
        "description": f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_description",
        "status": f"{ATTR_MAPPER_OP_PREFIX_DAG}.get_status",
    }
    """The default node attribute mappers when
    :py:meth:`generating a graph object from the DAG <.generate_nx_graph>`.
    These are passed to the ``map_node_attrs`` argument of
    :py:func:`~dantro.utils.nx.manipulate_attributes`.
    """

    # .........................................................................

    def __init__(
        self,
        *,
        dm: "dantro.data_mngr.DataManager",
        define: Dict[str, Union[List[dict], Any]] = None,
        select: dict = None,
        transform: Sequence[dict] = None,
        cache_dir: str = ".cache",
        file_cache_defaults: dict = None,
        base_transform: Sequence[Transformation] = None,
        select_base: Union[DAGReference, str] = None,
        select_path_prefix: str = None,
        meta_operations: Dict[str, Union[list, dict]] = None,
        exclude_from_all: List[str] = None,
        verbosity: int = 1,
    ):
        """Initialize a TransformationDAG by loading the specified
        transformations configuration into it, creating a directed acyclic
        graph of :py:class:`.Transformation` objects.

        See :ref:`dag_framework` for more information and examples.

        Args:
            dm (dantro.data_mngr.DataManager): The associated data manager
                which is made available as a special node in the DAG.
            define (Dict[str, Union[List[dict], Any]], optional): Definitions
                of tags. This can happen in two ways: If the given entries
                contain a list or tuple, they are interpreted as sequences of
                transformations which are subsequently added to the DAG, the
                tag being attached to the last transformation of each sequence.
                If the entries contain objects of *any* other type, including
                ``dict`` (!), they will be added to the DAG via a single node
                that uses the ``define`` operation.
                This argument can be helpful to define inputs or variables
                which may then be used in the transformations added via
                the ``select`` or ``transform`` arguments.
                See :ref:`dag_define` for more information and examples.
            select (dict, optional): Selection specifications, which are
                translated into regular transformations based on ``getitem``
                operations. The ``base_transform`` and ``select_base``
                arguments can be used to define from which object to select.
                By default, selection happens from the associated DataManager.
            transform (Sequence[dict], optional): Transform specifications.
            cache_dir (str, optional): The name of the cache directory to
                create if file caching is enabled. If this is a relative path,
                it is interpreted relative to the associated data manager's
                data directory. If it is absolute, the absolute path is used.
                The directory is only created if it is needed.
            file_cache_defaults (dict, optional): Default arguments for file
                caching behaviour. This is recursively updated with the
                arguments given in each individual select or transform
                specification.
            base_transform (Sequence[Transformation], optional): A sequence of
                transform specifications that are added to the DAG prior to
                those added via ``define``, ``select`` and ``transform``.
                These can be used to create some other object from the data
                manager which should be used as the basis of ``select``
                operations. These transformations should be kept as simple as
                possible and ideally be only used to traverse through the data
                tree.
            select_base (Union[DAGReference, str], optional): Which tag to
                base the ``select`` operations on. If None, will use the
                (always-registered) tag for the data manager, ``dm``. This
                attribute can also be set via the ``select_base`` property.
            select_path_prefix (str, optional): If given, this path is prefixed
                to all ``path`` specifications made within the ``select``
                argument. Note that unlike setting the ``select_base`` this
                merely joins the given prefix to the given paths, thus leading
                to repeated path resolution. For that reason, using the
                ``select_base`` argument is generally preferred and the
                ``select_path_prefix`` should only be used if ``select_base``
                is already in use.
                If this path ends with a ``/``, it is directly prepended. If
                not, the ``/`` is added before adjoining it to the other path.
            meta_operations (dict, optional): Meta-operations are basically
                *function definitions* using the language of the transformation
                framework; for information on how to define and use them, see
                :ref:`dag_meta_ops`.
            exclude_from_all (List[str], optional): Tag names that should not
                be defined as :py:meth:`.compute`
                targets if ``compute_only: all`` is set there.
                Note that, alternatively, tags can be named starting with
                ``.`` or ``_`` to exclude them from that list.
            verbosity (str, optional): Logging verbosity during computation.
                This mostly pertains to the extent of statistics being emitted
                through the logger.

                    - ``0``: No statistics
                    - ``1``: Per-node statistics (mean, std, min, max)
                    - ``2``: Total effective time for the 5 slowest operations
                    - ``3``: Same as ``2`` but for all operations
        """
        self._dm = dm
        self._objects = DAGObjects()
        self._tags = dict()
        self._nodes = list()
        self._meta_ops = dict()
        self._ref_stacks = _defaultdict(list)
        self._force_compute_refs = []
        self._fc_opts = file_cache_defaults if file_cache_defaults else {}
        self._select_base = None
        self._profile = dict(add_node=0.0, compute=0.0)
        self._select_path_prefix = select_path_prefix
        self.exclude_from_all = exclude_from_all if exclude_from_all else []
        self.verbosity = verbosity

        # Determine cache directory path; relative path interpreted as relative
        # to the DataManager's data directory
        if os.path.isabs(cache_dir):
            self._cache_dir = cache_dir
        else:
            self._cache_dir = os.path.join(self.dm.dirs["data"], cache_dir)

        # Add the special tags: the DAG itself, the DataManager, and the
        # (changing) selection base
        self.tags["dag"] = self.objects.add_object(self)
        self.tags["dm"] = self.objects.add_object(self.dm)
        self.tags["select_base"] = self.tags["dm"]  # here: default value only
        # NOTE The data manager is NOT a node of the DAG, but more like an
        #      external data source, thus being accessible only as a tag

        # Populate the registry of meta-operations
        if meta_operations:
            for name, spec in meta_operations.items():
                if isinstance(spec, list):
                    self.register_meta_operation(name, transform=spec)
                else:
                    self.register_meta_operation(name, **spec)
            log.debug("Registered %d meta-operations.", len(self._meta_ops))
        # NOTE While meta-operations may also carry out `select` operations,
        #      they don't need to know the selection base *at this point*.

        # Add base transformations that do not rely on select operations
        self.add_nodes(transform=base_transform)

        # Set the selection base tag; the property setter checks availability
        self.select_base = select_base

        # Now add nodes via the main arguments; these can now make use of the
        # select interface, because a select base tag is set and base transform
        # operations were already added.
        self.add_nodes(define=define, select=select, transform=transform)

    # .........................................................................

    def __str__(self) -> str:
        """A human-readable string characterizing this TransformationDAG"""
        return (
            "<TransformationDAG, {:d} node(s), {:d} tag(s), {:d} object(s)>"
            "".format(len(self.nodes), len(self.tags), len(self.objects))
        )

    # .........................................................................

    @property
    def dm(self) -> "DataManager":
        """The associated DataManager"""
        return self._dm

    @property
    def hashstr(self) -> str:
        """Returns the hash of this DAG, which depends solely on the hash of
        the associated DataManager.
        """
        return _hash(
            "<TransformationDAG, coupled to DataManager "
            f"with ref {self.dm.hashstr}>"
        )

    @property
    def objects(self) -> DAGObjects:
        """The object database"""
        return self._objects

    @property
    def tags(self) -> Dict[str, str]:
        """A mapping from tags to objects' hashes; the hashes can be looked
        up in the object database to get to the objects.
        """
        return self._tags

    @property
    def nodes(self) -> List[str]:
        """The nodes of the DAG"""
        return self._nodes

    @property
    def ref_stacks(self) -> Dict[str, List[str]]:
        """Named reference stacks, e.g. for resolving tags that were defined ´
        inside meta-operations.
        """
        return self._ref_stacks

    @property
    def meta_operations(self) -> List[str]:
        """The names of all registered meta-operations.

        To register new meta-operations, use the dedicated registration method,
        :py:meth:`.register_meta_operation`.
        """
        return list(self._meta_ops)

    @property
    def cache_dir(self) -> str:
        """The path to the cache directory that is associated with the
        DataManager that is coupled to this DAG. Note that the directory might
        not exist yet!
        """
        return self._cache_dir

    @property
    def cache_files(self) -> Dict[str, Tuple[str, str]]:
        """Scans the cache directory for cache files and returns a dict that
        has as keys the hash strings and as values a tuple of full path and
        file extension.
        """
        info = dict()

        # Go over all files in the cache dir that have an extension
        for path in glob.glob(os.path.join(self.cache_dir, "*.*")):
            if not os.path.isfile(path):
                continue

            # Get filename and extension, then check if it is a hash
            fname, ext = os.path.splitext(os.path.basename(path))
            if len(fname) != FULL_HASH_LENGTH:
                continue
            # else: filename is assumed to be the hash.

            if fname in info:
                raise ValueError(
                    "Encountered a duplicate cache file for the "
                    f"transformation with hash {fname}! Delete all but one of "
                    f"those files from the cache directory {self.cache_dir}."
                )

            # All good, store info.
            info[fname] = dict(full_path=path, ext=ext)

        return info

    @property
    def select_base(self) -> DAGReference:
        """The reference to the object that is used for select operations"""
        return self._select_base

    @select_base.setter
    def select_base(self, new_base: Union[DAGReference, str]):
        """Set the reference that is to be used as the base of select
        operations. It can either be a reference object or a string, which is
        then interpreted as a tag.
        """
        # Distinguish by type. If it's not a DAGReference, assume it's a tag.
        if new_base is None:
            new_base = DAGTag("dm").convert_to_ref(dag=self)

        elif isinstance(new_base, DAGReference):
            # Make sure it is a proper DAGReference object (hash-based) and not
            # an object of a derived class.
            new_base = new_base.convert_to_ref(dag=self)

        elif new_base not in self.tags:
            _available = ", ".join(self.tags)
            raise KeyError(
                f"The tag '{new_base}' cannot be used to set `select_base` "
                "because it is not available! Make sure that a node with that "
                "tag is added _prior_ to the attempt of setting it. "
                f"Available tags: {_available}. Alternatively, pass a "
                "DAGReference object."
            )

        else:
            # Tag is available. Create a DAGReference via DAGTag conversion
            log.debug("Setting select_base to tag '%s' ...", new_base)
            new_base = DAGTag(new_base).convert_to_ref(dag=self)

        # Have a DAGReference now. Store it and update the special tag.
        self._select_base = new_base
        self.tags["select_base"] = new_base.ref

    @property
    def profile(self) -> Dict[str, float]:
        """Returns the profiling information for the DAG."""
        return self._profile

    @property
    def profile_extended(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """Builds an extended profile that includes the profiles from all
        transformations and some aggregated information.

        This is calculated anew upon each invocation; the result is not cached.

        The extended profile contains the following information:

            - ``tags``: profiles for each tag, stored under the tag
            - ``aggregated``: aggregated statistics of all nodes with profile
              information on compute time, cache lookup, cache writing
            - ``sorted``: individual profiling times, with NaN values set to 0
        """
        prof = _shallowcopy(self.profile)

        # Add tag-specific information
        prof["tags"] = dict()
        for tag, obj_hash in self.tags.items():
            obj = self.objects[obj_hash]
            if not isinstance(obj, Transformation):
                continue

            tprof = _shallowcopy(obj.profile)
            prof["tags"][tag] = tprof

        # Aggregate the profiled times from all transformations (by item)
        to_aggregate = (
            "compute",
            "cache_lookup",
            "cache_writing",
            "effective",
        )
        stat_funcs = dict(
            mean=lambda d: np.nanmean(d),
            std=lambda d: np.nanstd(d),
            min=lambda d: np.nanmin(d),
            max=lambda d: np.nanmax(d),
            # q25=lambda d: np.nanquantile(d, 0.25),
            q50=lambda d: np.nanquantile(d, 0.50),
            # q75=lambda d: np.nanquantile(d, 0.75),
            sum=lambda d: np.nansum(d),
            count=lambda d: np.count_nonzero(~np.isnan(d)),
        )
        tprofs = {item: list() for item in to_aggregate}

        for obj_hash, obj in self.objects.items():
            if not isinstance(obj, Transformation):
                continue

            tprof = _shallowcopy(obj.profile)
            for item in to_aggregate:
                tprofs[item].append(tprof[item])

        # Compute some statistics for the aggregated elements; need to ignore
        # warnings because values can be NaN, e.g. without cache lookup
        prof["aggregated"] = dict()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for item in to_aggregate:
                prof["aggregated"][item] = {
                    k: (f(tprofs[item]) if tprofs[item] else np.nan)
                    for k, f in stat_funcs.items()
                }

        # Also sort the node profiling results, setting NaNs to zeros
        prof["sorted"] = dict()

        to_sort_by = to_aggregate + ("cumulative_compute",)
        nodes = [self.objects[obj_hash] for obj_hash in self.nodes]
        for sort_by in to_sort_by:
            nct = [
                (n.hashstr, n.profile[sort_by])
                if not np.isnan(n.profile[sort_by])
                else (n.hashstr, 0.0)
                for n in nodes
            ]
            prof["sorted"][sort_by] = sorted(
                nct, key=lambda tup: tup[1], reverse=True
            )

        # Additionally, aggregate effective times by operation
        eff_op_times = _defaultdict(list)
        for node in nodes:
            eff_op_times[node.operation].append(node.profile["effective"])
        eff_op_times = dict(eff_op_times)

        prof["operations"] = dict()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            for op, times in eff_op_times.items():
                prof["operations"][op] = {
                    k: (f(times) if times else np.nan)
                    for k, f in stat_funcs.items()
                }

        prof["slow_operations"] = sorted(
            ((op, prof["operations"][op]["sum"]) for op in prof["operations"]),
            key=lambda tup: tup[1],
            reverse=True,
        )

        return prof

    # .........................................................................

    def register_meta_operation(
        self,
        name: str,
        *,
        select: dict = None,
        transform: Sequence[dict] = None,
    ) -> None:
        """Registers a new meta-operation, i.e. a transformation sequence with
        placeholders for the required positional and keyword arguments.
        After registration, these operations are available in the same way as
        other operations; unlike non-meta-operations, they will lead to
        multiple nodes being added to the DAG.

        See :ref:`dag_meta_ops` for more information.

        Args:
            name (str): The name of the meta-operation; can only be used once.
            select (dict, optional): Select specifications
            transform (Sequence[dict], optional): Transform specifications
        """
        if name in self._meta_ops or name in available_operations():
            raise BadOperationName(
                "An operation or meta-operation with the name "
                f"'{name}' already exists!"
            )

        # First, evaluate the select and transform arguments, returning a list
        # of transformation specifications.
        specs = self._parse_trfs(select=select, transform=transform)

        if not specs:
            raise MetaOperationError(
                "Meta-operations need to contain at least one "
                "transformation, but there was none specified "
                f"for meta-operation '{name}'!"
            )

        # Define some helper lambdas to identify placeholders and tags
        is_placeholder = lambda obj: isinstance(obj, _Placeholder)
        is_normal_tag = lambda obj: (
            isinstance(obj, DAGTag) and obj.name not in self.SPECIAL_TAGS
        )
        is_special_tag = lambda obj: (
            isinstance(obj, DAGTag) and obj.name in self.SPECIAL_TAGS
        )
        is_arg = lambda obj: isinstance(obj, _Arg)
        is_kwarg = lambda obj: isinstance(obj, _Kwarg)

        # Traverse the parsed transformation specifications and extract
        # information on the positional and keyword arguments.
        args = list()
        kwargs = list()

        for i, spec in enumerate(specs):
            args += recursive_collect(spec, select_func=is_arg)
            kwargs += recursive_collect(spec, select_func=is_kwarg)

        # Positional arguments, potentially with fallback values
        _arg_pos = [arg.position for arg in args]
        _num_args = 0 if not _arg_pos else (max(_arg_pos) + 1)
        required_args = {arg.position for arg in args if not arg.has_fallback}
        optional_args = {arg.position for arg in args if arg.has_fallback}

        # Keyword arguments, potentially with fallback values
        kwarg_names = {kw.name for kw in kwargs}
        required_kwargs = {kw.name for kw in kwargs if not kw.has_fallback}
        optional_kwargs = {kw.name for kw in kwargs if kw.has_fallback}

        # .. Need a number of checks now ......................................
        # Make sure they are contiguous
        if list(set(_arg_pos)) != list(range(_num_args)):
            raise MetaOperationSignatureError(
                "The positional arguments specified for the meta-operation "
                f"'{name}' were not contiguous! With the highest argument "
                f"index {_num_args - 1}, there need to be positional "
                f"arguments for all integers from 0 to {_num_args - 1}. "
                f"Got: {set(_arg_pos)}."
            )

        # Make sure the lists of required & optional args/kwargs are disjoint,
        # meaning that their intersection is size zero:
        _args_intersection = required_args.intersection(optional_args)
        _kwargs_intersection = required_kwargs.intersection(optional_kwargs)

        if _args_intersection or _kwargs_intersection:
            _bad_args = ", ".join(f"{a}" for a in _args_intersection)
            _bad_kwargs = ", ".join(f"{k}" for k in _kwargs_intersection)
            raise MetaOperationSignatureError(
                "Got (positional or keyword) arguments that were not clearly "
                "identifiable as optional:\n"
                f"  args:    {_bad_args}\n"
                f"  kwargs:  {_bad_kwargs}\n"
                "Make sure that all these args or kwargs with the same "
                "position or name, respectively, also have the same fallback "
                "specification."
            )

        # Make sure there is no overlap between required and optional
        # _positional_ arguments, as association would become ambiguous
        if max(list(required_args) + [-1]) > min(
            list(optional_args) + [float("inf")]
        ):
            _req_args = ", ".join(f"{p}" for p in required_args)
            _opt_args = ", ".join(f"{p}" for p in optional_args)
            raise MetaOperationSignatureError(
                "Optional positional arguments need to come strictly after "
                "required positional arguments!\n"
                f"  Required positions:   {_req_args}\n"
                f"  Optional positions:   {_opt_args}\n"
                "Remove or add fallback values such that the argument "
                "position ranges do not overlap."
            )

        # Locate all newly defined tags within the meta-operation; these are
        # used to refer to other transformations _within_ a meta-operation.
        # Need to check two things: they should be unique and they should all
        # be used (further down).
        defined_tags = {
            i: spec["tag"] for i, spec in enumerate(specs) if spec.get("tag")
        }
        unused_tags = set(defined_tags.values())

        if len(set(defined_tags.values())) != len(defined_tags):
            _tags = ", ".join(defined_tags.values())
            raise MetaOperationError(
                "Encountered duplicate internal tag definitions in the "
                f"meta-operation '{name}'! Check the defined tags ({_tags}) "
                "and make sure every internal tag's name is unique within the "
                "meta-operation."
            )

        # Need to make sure that fallback values don't contain tags or any
        # other weird stuff
        for arg in chain(args, kwargs):
            if arg.has_fallback and recursive_collect(
                [arg.fallback], select_func=is_placeholder
            ):
                raise MetaOperationError(
                    "Default values of positional or keyword arguments "
                    "to meta-operations may not contain tags or any other "
                    "kind of placeholders!\n"
                    "Got the following default value, which did contain "
                    f"a placeholder:  {arg.fallback}"
                )

        # .....................................................................

        def to_meta_operation_tag(tag: DAGTag) -> _MOpTag:
            """Replacement function for meta-operation-internal references.
            Additionally, this function checks if the tag name is internal and
            updates the set of unused tags.

            .. note::

                This local function relies on the local variables
                ``defined_tags``, ``name``, ``mop_tags``, and ``unused_tags``
                and the mutability of the latter two.
            """
            if tag.name not in defined_tags.values():
                _internal = ", ".join(defined_tags.values())
                _special = ", ".join(self.SPECIAL_TAGS)
                raise MetaOperationError(
                    "Encountered a tag that was not defined as part of the "
                    f"meta-operation '{name}': '{tag.name}'! Within a meta-"
                    f"operation, only internally-defined tags ({_internal}) "
                    f"or special tags ({_special}) can be used. To include "
                    "further input into the meta-operation, use a positional "
                    "or keyword argument instead of referring to a tag."
                )

            # Remove from unused tags; then create a new meta-operation tag
            # and return it (such that it replaces the normal tag)
            unused_tags.discard(tag.name)
            return _MOpTag.from_names(name, tag=tag.name)

        # However, these tags cannot stay as regular DAGTags and need to be
        # converted. Otherwise we would have to deal with duplicate tag names
        # when applying a meta-operation multiple times.
        # Furthermore, relative references (e.g. DAGNode) cannot be used as the
        # relative reference breaks if meta-operation are nested.
        #
        # Thus, we have to use a more elaborated approach: Replace the regular
        # DAGTag objects with DAGMetaOperationTag objects, which contain
        # information on the meta-operation they were used in. With this
        # knowledge, we can later resolve the targets of these tags via the
        # reference stacks, thus allowing arbitrary nesting.
        #
        # NOTE The DAGReference objects that are inserted when parsing the
        #      `select` specification above are not touched by this procedure
        #      and need not be adapted either: they are absolute references and
        #      are not used to denote a reference _within_ a meta-operation.
        for i, spec in enumerate(specs):
            # Replace any (non-special) DAGTags within the specification, as
            # these need to be associated via the reference stack.
            spec = recursive_replace(
                spec,
                select_func=is_normal_tag,
                replace_func=to_meta_operation_tag,
            )

        if unused_tags:
            _unused = ", ".join(unused_tags)
            raise MetaOperationError(
                f"Meta-operation '{name}' defines internal tags that are not "
                f"used within the meta-operation: {_unused}! Either remove "
                "these tags or make sure that they are all used by the "
                "specified transformations. Note that exporting additional "
                "tags from a meta-operation is currently not supported."
            )

        # Finally, store all extracted info in the meta operations registry.
        # The reference stack entry keeps track of the nodes that define an
        # internal tag, such that it can be resolved when adding the node.
        self._meta_ops[name] = dict(
            specs=specs,
            defined_tags=defined_tags,
            num_args=len(required_args) + len(optional_args),
            required_args=required_args,
            optional_args=optional_args,
            kwarg_names=kwarg_names,
            required_kwargs=required_kwargs,
            optional_kwargs=optional_kwargs,
        )

    def add_node(
        self,
        *,
        operation: str,
        args: list = None,
        kwargs: dict = None,
        tag: str = None,
        force_compute: bool = None,
        file_cache: dict = None,
        fallback: Any = None,
        **trf_kwargs,
    ) -> DAGReference:
        """Add a new node by creating a new
        :py:class:`~dantro.dag.Transformation` object and adding it to the
        node list.

        In case of ``operation`` being a meta-operation, this method will add
        multiple Transformation objects to the node list. The ``tag`` and the
        ``file_cache`` argument then refer to the *result* node of the meta-
        operation, while the ``**trf_kwargs`` are passed to *all* these nodes.
        For more information, see :ref:`dag_meta_ops`.

        Args:
            operation (str): The name of the operation or meta-operation.
            args (list, optional): Positional arguments to the operation
            kwargs (dict, optional): Keyword arguments to the operation
            tag (str, optional): The tag the transformation should be made
                available as.
            force_compute (bool, optional): If True, the result of this node
                will always be computed as part of :py:meth:`.compute`.
            file_cache (dict, optional): File cache options for this node. If
                defaults were given during initialization, those defaults will
                be updated with the given dict.
            fallback: (Any, optional): The fallback value in case that the
                computation of this node fails.
            **trf_kwargs: Passed on to
                :py:meth:`~dantro.dag.Transformation.__init__`

        Raises:
            ValueError: If the tag already exists

        Returns:
            DAGReference: The reference to the created node. In case of the
                operation being a meta operation, the return value is a
                reference to the *result* node of the meta-operation.
        """
        # May have to delegate node addition ...
        if operation in self._meta_ops:
            return self._add_meta_operation_nodes(
                operation,
                args=args,
                kwargs=kwargs,
                tag=tag,
                force_compute=force_compute,
                file_cache=file_cache,
                **trf_kwargs,
            )

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Some helper methods for the recursive replacement
        def not_proper_ref(obj: Any) -> bool:
            return (
                isinstance(obj, DAGReference) and type(obj) is not DAGReference
            )

        def convert_to_ref(obj: DAGReference) -> DAGReference:
            return obj.convert_to_ref(dag=self)

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Keep track of the time
        t0 = time.time()

        # Handle default values of arguments
        args = _deepcopy(args) if args else []
        kwargs = _deepcopy(kwargs) if kwargs else {}
        fallback = _deepcopy(fallback) if fallback is not None else None
        # NOTE Deep copy is important here, because the mutability of nested
        #      objects may lead to side effects. The deep copy should always
        #      be possible, because these should only contain trivial objects.

        # Recursively replace any derived references to proper DAGReferences,
        # which work hash-based. This is to not have multiple options of how
        # another TransformationDAG object is referenced.
        # The recursiveness is needed because args and kwargs can be deeply
        # nested structures and we want to replace all references regardless
        # of their position within args and kwargs.
        args = recursive_replace(
            args, select_func=not_proper_ref, replace_func=convert_to_ref
        )
        kwargs = recursive_replace(
            kwargs, select_func=not_proper_ref, replace_func=convert_to_ref
        )

        # Also need to do this for the fallback, which may include references.
        # To allow scalars, wrap it in a 1-sized list temporarily.
        fallback = recursive_replace(
            [fallback], select_func=not_proper_ref, replace_func=convert_to_ref
        )[0]

        # Parse file cache parameters
        fc_opts = self._fc_opts  # Always a dict

        if file_cache is not None:
            if isinstance(file_cache, bool):
                file_cache = dict(read=file_cache, write=file_cache)
            fc_opts = _recursive_update(_deepcopy(fc_opts), file_cache)

        # From these arguments, create the Transformation object and add it to
        # the objects database.
        trf = Transformation(
            operation=operation,
            args=args,
            kwargs=kwargs,
            dag=self,
            file_cache=fc_opts,
            fallback=fallback,
            **trf_kwargs,
        )
        trf_hash = self.objects.add_object(trf)
        # NOTE From this point on, the object itself has no relevance.
        #      Transformation objects should only be handled via their hash in
        #      order to reduce duplicate calculations and make efficient
        #      caching possible.

        # Store the hash in the node list
        self.nodes.append(trf_hash)

        # If a tag was specified, create a tag
        if tag:
            if tag in self.tags.keys():
                _used_tags = ", ".join(self.tags.keys())
                raise ValueError(
                    f"Tag '{tag}' already exists! Choose a different one. "
                    f"Already in use: {_used_tags}"
                )
            self.tags[tag] = trf_hash

        # If it should always be computed, denote it as such
        if force_compute:
            self._force_compute_refs.append(trf_hash)

        # Finish up ...
        self._update_profile(add_node=time.time() - t0)

        return DAGReference(trf_hash)

    def add_nodes(
        self,
        *,
        define: Dict[str, Union[List[dict], Any]] = None,
        select: dict = None,
        transform: Sequence[dict] = None,
    ):
        """Adds multiple nodes by parsing the specification given via the
        ``define``, ``select``, and ``transform`` arguments (in that order).

        .. note::

            The current :py:attr:`.select_base`
            property value is used as basis for all ``getitem`` operations.

        Args:
            define (Dict[str, Union[List[dict], Any]], optional): Definitions
                of tags. This can happen in two ways: If the given entries
                contain a list or tuple, they are interpreted as sequences of
                transformations which are subsequently added to the DAG, the
                tag being attached to the last transformation of each sequence.
                If the entries contain objects of *any* other type, including
                ``dict`` (!), they will be added to the DAG via a single node
                that uses the ``define`` operation.
                This argument can be helpful to define inputs or variables
                which may then be used in the transformations added via
                the ``select`` or ``transform`` arguments.
                See :ref:`dag_define` for more information and examples.
            select (dict, optional): Selection specifications, which are
                translated into regular transformations based on ``getitem``
                operations. The ``base_transform`` and ``select_base``
                arguments can be used to define from which object to select.
                By default, selection happens from the associated DataManager.
            transform (Sequence[dict], optional): Transform specifications.
        """
        specs = self._parse_trfs(
            define=define, select=select, transform=transform
        )
        if not specs:
            return

        for spec in specs:
            self.add_node(**spec)

    def compute(
        self, *, compute_only: Sequence[str] = None, verbosity: int = None
    ) -> Dict[str, Any]:
        """Computes all specified tags and returns a result dict.

        Depending on the ``verbosity`` attribute, a varying level of profiling
        statistics will be emitted via the logger.

        Args:
            compute_only (Sequence[str], optional): The tags to compute.
                If ``None``, will compute *all* non-private tags: all tags
                *not* starting with ``.`` or ``_`` that are *not* included
                in the ``TransformationDAG.exclude_from_all`` list.

        Returns:
            Dict[str, Any]: A mapping from tags to fully computed results.
        """
        if verbosity is None:
            verbosity = self.verbosity

        def postprocess_result(res, *, tag: str = None):
            """Performs some postprocessing operations on the results of
            individual tag computations.
            """
            # If the object is a detached dantro tree object, use the short
            # transformation hash for its name
            if isinstance(res, (AbstractDataContainer)) and res.parent is None:
                res.name = trf.hashstr[:SHORT_HASH_LENGTH]

            # Unwrap ObjectContainer; these are only meant for usage within the
            # data tree and it makes little sense to keep them in that form.
            if isinstance(res, ObjectContainer):
                res = res.data

            return res

        def show_compute_profile_info():
            """Shows info on computation profiles, depending on verbosity"""
            if verbosity < 1:
                return

            prof = self.profile_extended
            to_exclude = (
                ("hashstr", "effective") if verbosity == 1 else ("hashstr",)
            )

            # Show aggregated statistics
            _fstr = (
                "{name:>25s}   {p[mean]:<7s}  ±  {p[std]:<7s}   "
                "({p[min]:<7s} | {p[max]:<7s})"
            )
            _stats = [
                _fstr.format(
                    name=k, p={_k: f"{_v:.2g}" for _k, _v in v.items()}
                )
                for k, v in prof["aggregated"].items()
                if k not in to_exclude
            ]
            log.remark(
                "Profiling results per node:  mean ± std (min|max) [s]\n%s",
                "\n".join(_stats),
            )

            if verbosity < 2:
                return

            # Show operations with highest sum of effective time
            num_ops = 5 if verbosity < 3 else len(prof["slow_operations"])
            _ops = prof["operations"]
            _fstr2 = (
                "{name:>25s}   {p[sum]:<10s} "
                "{cnt:>8d}x  ({p[mean]:<7s} ± {p[std]:<7s})"
            )
            _stats2 = [
                _fstr2.format(
                    name=op,
                    p={_k: f"{_v:.2g}" for _k, _v in _ops[op].items()},
                    cnt=_ops[op]["count"],
                )
                for op, _ in prof["slow_operations"][:num_ops]
            ]
            log.remark(
                "Total effective compute times and #calls (mean ± std) [s]\n"
                "%s",
                "\n".join(_stats2),
            )

        # The to-be-populated results dict
        results = dict()

        # Determine which tags to compute
        compute_only = self._parse_compute_only(compute_only)

        if not compute_only and not self._force_compute_refs:
            log.remark("No tags were selected to be computed.")
            return results

        t0 = time.time()

        # May have to force-compute some tags (not added to the results dict)
        if self._force_compute_refs:
            n_fcr = len(self._force_compute_refs)
            log.note(
                "Computing result of %d nodes with `force_compute` set ...",
                len(self._force_compute_refs),
            )
            for i, ref in enumerate(self._force_compute_refs):
                log.remark("  %2d/%d:  '%s'  ...", i + 1, n_fcr, ref)
                _tt = time.time()

                trf = self.objects[ref]
                trf.compute()

                _dtt = time.time() - _tt
                if _dtt > 1.0:
                    log.remark("Finished in %s.", _fmt_time(_dtt))

        # Compute results of tagged nodes and collect their results
        log.note(
            "Computing result of %d tag%s on DAG with %d nodes ...",
            len(compute_only),
            "s" if len(compute_only) != 1 else "",
            len(self.nodes),
        )
        for i, tag in enumerate(compute_only):
            log.remark("  %2d/%d:  '%s'  ...", i + 1, len(compute_only), tag)
            _tt = time.time()

            # Resolve the transformation, then compute the result, postprocess
            # it and store it under its tag in the results dict
            trf = self.objects[self.tags[tag]]
            res = trf.compute()
            results[tag] = postprocess_result(res, tag=tag)

            _dtt = time.time() - _tt
            if _dtt > 1.0:
                log.remark("Finished in %s.", _fmt_time(_dtt))

        # Update profiling information
        t1 = time.time()
        self._update_profile(compute=t1 - t0)

        # Provide some information to the user
        log.note(
            "Computed %d tag%s in %s.",
            len(compute_only),
            "s" if len(compute_only) != 1 else "",
            _fmt_time(t1 - t0),
        )

        show_compute_profile_info()
        return results

    # .........................................................................
    # DAG representation as nx.DiGraph and visualization/export

    def generate_nx_graph(
        self,
        *,
        tags_to_include: Union[str, Sequence[str]] = "all",
        manipulate_attrs: dict = {},
        include_results: bool = False,
        lookup_tags: bool = True,
        edges_as_flow: bool = True,
    ) -> "networkx.DiGraph":
        """Generates a representation of the DAG as a
        :py:class:`networkx.DiGraph` object, which can be useful for debugging.

        **Nodes** represent
        :py:class:`Transformations <dantro.dag.Transformation>` and are
        identified by their :py:meth:`~dantro.dag.Transformation.hashstr`.
        The :py:class:`~dantro.dag.Transformation` objects are added as node
        property ``obj`` and potentially existing tags are added as ``tag``.

        **Edges** represent dependencies between nodes.
        They can be visualized in two ways:

            - With ``edges_as_flow: true``, edges point in the direction of
              results being computed, representing a flow of results.
            - With ``edges_as_flow: false``, edges point towards the
              dependency of a node that needs to be computed before the node
              itself can be computed.

        See :ref:`dag_graph_vis` for more information.

        .. note::

            The returned graph data structure is *not* used internally but is
            a representation that is *generated* from the internally used
            data structures.
            Subsequently, changes to the graph *structure* will not have an
            effect on this :py:class:`~dantro.dag.TransformationDAG`.

        .. hint::

            Use :py:meth:`.visualize` to generate a visual output.
            For processing the DAG representation elsewhere, you can use the
            :py:func:`~dantro.utils.nx.export_graph` function.

        .. warning::

            Do not modify the associated :py:class:`~dantro.dag.Transformation`
            objects!

            These objects are *not* deep-copied into the graph's node
            properties. Thus, changes to these objects *will* reflect on the
            state of the :py:class:`~dantro.dag.TransformationDAG` which may
            have unexpected effects, e.g. because the hash will not be updated.

        Args:
            tags_to_include (Union[str, Sequence[str]], optional): Which tags
                to include into the directed graph. Can be ``all`` to include
                all tags.
            manipulate_attrs (Dict[str, Union[str, dict]], optional): Allows to
                manipulate node and edge attributes.
                See :py:func:`~dantro.utils.nx.manipulate_attributes` for more
                information.

                By default, this includes a number of default node attribute
                mappers, defined in :py:attr:`.NODE_ATTR_DEFAULT_MAPPERS`.
                These can be overwritten or extended via the ``map_node_attrs``
                key within this argument.

                .. note::

                    This method registers specialized data operations with the
                    :ref:`operations database <data_ops_available>` that are
                    meant for handling the case where node attributes
                    are associated with :py:class:`~dantro.dag.Transformation`
                    objects.

                    Available operations (with prefix ``attr_mapper``):

                        - ``{prefix}.get_operation`` returns the operation
                          associated with a node.
                        - ``{prefix}.get_operation`` generates a string from
                          the positional and keyword arguments to a node.
                        - ``{prefix}.get_layer`` returns the layer, i.e. the
                          distance from the farthest dependency; nodes without
                          dependencies have layer 0.
                          See :py:attr:`dantro.dag.Transformation.layer`.
                        - ``{prefix}.get_description`` creates a description
                          string that is useful for visualization (e.g. as
                          node label).

                    To implement your own operation, take care to follow the
                    syntax of :py:func:`~dantro.utils.nx.map_attributes`.

                .. note::

                    By default, there are *no* attributes associated with the
                    edges of the DAG.

            include_results (bool, optional): Whether to include results into
                the node attributes.

                .. note::

                    These will all be ``None`` unless :py:meth:`.compute` was
                    invoked before generating the graph.

            lookup_tags (bool, optional): Whether to lookup tags for each node,
                storing it in the ``tag`` node attribute. The tags in
                ``tags_to_include`` are always included, but the reverse lookup
                of tags can be costly, in which case this should be disabled.
            edges_as_flow (bool, optional): If true, edges point from a node
                towards the nodes that *require* the computed result; if false,
                they point towards the *dependency* of a node.
        """
        import networkx as nx

        # .....................................................................

        def get_node_attrs(
            obj: Union[Transformation, Any],
            *,
            hashstr: str,
            g: "networkx.DiGraph",
            tag: str = None,
        ) -> dict:
            """Retrieves a dict of node attributes for the given node object.

            Args:
                obj: The object that represents the node
                hashstr (str): The hash string for this node
                g (networkx.classes.digraph.DiGraph): The graph object the node
                    will be part of.
                tag (str): If known, the tag to associate with the node.
            """

            def get_tag(hashstr: str) -> str:
                """To avoid repetitive tag search, see if a tag is already
                associated. Also uses the potentially existing tag information.
                """
                if tag:
                    # Tag explicitly specified (in outer scope!)
                    return tag

                if hashstr in g.nodes():
                    # Node already in graph, avoid search
                    return g.nodes()[hashstr].get("tag")

                # Node not yet in graph, need to search for it, if enabled
                if lookup_tags:
                    return self._find_tag(hashstr)

                # No lookup!
                return None

            def get_status(obj) -> str:
                """Retrieves the node status"""
                if isinstance(obj, Transformation):
                    return obj.status
                return "none"

            # Aggregate the node attributes
            node_attrs = dict(
                obj=obj,
                tag=get_tag(hashstr),
                status=get_status(obj),
            )

            if include_results:
                if isinstance(obj, Transformation):
                    _has_result, _result = obj._lookup_result()
                    node_attrs["has_result"] = _has_result
                    node_attrs["result"] = _result
                else:
                    node_attrs["has_result"] = False
                    node_attrs["result"] = None

            return node_attrs

        def add_nodes_and_edges(
            g: "networkx.DiGraph",
            obj: Union[Transformation, Any],
            *,
            tag: str = None,
        ) -> str:
            """Recursively adds nodes and edges into graph ``g``: ``obj``will
            be a new node, and recursion continues on its dependencies.
            Returns the node index of the added node.
            """
            node_attrs = get_node_attrs(
                obj,
                hashstr=obj.hashstr,
                g=g,
                tag=tag,
            )

            # Now add the node
            g.add_node(obj.hashstr, **node_attrs)

            # Can continue recursion only on Transformation objects
            if not isinstance(obj, Transformation):
                return

            for dep in obj.resolved_dependencies:
                add_nodes_and_edges(g, dep)

                # Now have both nodes in the graph, can the edge between them
                if edges_as_flow:
                    g.add_edge(dep.hashstr, obj.hashstr)
                else:
                    g.add_edge(obj.hashstr, dep.hashstr)

            return obj.hashstr

        # .....................................................................

        tags_to_include = self._parse_compute_only(tags_to_include)
        g = nx.DiGraph()

        log.note(
            "Generating DAG representation for %d tag%s ...",
            len(tags_to_include),
            "s" if len(tags_to_include) != 1 else "",
        )

        if not len(self.nodes):
            log.remark("The TransformationDAG is empty.")
            return g

        elif not tags_to_include:
            log.remark("No tags were selected to be included into the graph.")
            return g

        # Build the graph by adding all tags and their dependencies.
        # With DiGraph not allowing multi-edges and nodes with the same
        # identifier, can simply add them directly and let networkx figure out
        # if the node or edge is already present or not.
        for tag in tags_to_include:
            obj = self.objects[self.tags[tag]]
            node = add_nodes_and_edges(g, obj, tag=tag)
            g.nodes()[node]["tag"] = tag

        # Label DataManager explicitly
        if self.dm.hashstr in g.nodes():
            g.nodes()[self.dm.hashstr]["tag"] = "dm"

        # Now apply mappings
        manipulate_attrs = manipulate_attrs if manipulate_attrs else {}
        manipulate_attrs["map_node_attrs"] = _recursive_update(
            copy.copy(self.NODE_ATTR_DEFAULT_MAPPERS),
            manipulate_attrs.get("map_node_attrs", {}),
        )
        _manipulate_attributes(g, **manipulate_attrs)

        # Done.
        log.remark(
            "Generated DAG representation with %d nodes and %d edges.",
            g.number_of_nodes(),
            g.number_of_edges(),
        )
        return g

    def visualize(
        self,
        *,
        out_path: str,
        g: "networkx.DiGraph" = None,
        generation: dict = {},
        drawing: dict = {},
        use_defaults=True,
        scale_figsize: Union[bool, Tuple[float, float]] = (0.25, 0.2),
        show_node_status: bool = True,
        node_status_color: dict = None,
        layout: dict = {},
        figure_kwargs: dict = {},
        annotate_kwargs: dict = {},
        save_kwargs: dict = {},
    ) -> "networkx.DiGraph":
        """Uses :py:meth:`.generate_nx_graph` to generate a DAG representation
        as a :py:class:`networkx.DiGraph` and then creates a visualization.

        .. warning::

            The plotted graph may contain overlapping edges or nodes, depending
            on the size and structure of your DAG. This is less pronounced if
            `pygraphviz <https://pygraphviz.github.io>`_ is installed, which
            provides vastly more capable layouting algorithms.

            To alleviate this, the default layouting and drawing arguments will
            generate a graph with partly transparent nodes and edges and wiggle
            node positions around, thus making edges more discernible.

        Args:
            out_path (str): Where to store the output
            g (networkx.DiGraph, optional): If given, will use this graph
                instead of generating a new one.
            generation (dict, optional): Arguments for graph generation, passed
                on to :py:meth:`.generate_nx_graph`. Not allowed if ``g`` was
                given.
            drawing (dict, optional): Drawing arguments, containing the
                ``nodes``, ``edges`` and ``labels`` keys. The ``labels`` key
                can contain the ``from_attr`` key which will read the attribute
                specified there and use it for the label.
            use_defaults (dict, optional): Whether to use default drawing
                arguments which are optimized for a simple representation.
                These are recursively updated by the ones given in ``drawing``.
                Set to false to use the networkx defaults instead.
            scale_figsize (Union[bool, Tuple[float, float]], optional): If True
                or a tuple, will set the figure size according to:
                ``(width_0 * max_occup. * s_w,  height_0 * max_level * s_h)``
                where ``s_w`` and ``s_h`` are the scaling factors. The maximum
                occupation refers to the highest number of nodes on a single
                layer. This figure size scaling avoids nodes overlapping for
                larger graphs.

                .. note::

                    The default values here are a heuristic and depend very
                    much on the size of the node labels and the font size.

            show_node_status (bool, optional): If true, will color-code the
                node status (computed, not computed, failed), setting the
                ``nodes.node_color`` key correspondingly.

                .. note::

                    Node color is plotted *behind* labels, thus requiring some
                    transparency for the labels.

            node_status_color (dict, optional): If ``show_node_status`` is set,
                will use this map to determine the node colours. It should
                contain keys for all possible values of
                :py:attr:`dantro.dag.Transformation.status`. In addition, there
                needs to be a ``fallback`` key that is used for nodes where no
                status can be determined.
            layout (dict, optional): Passed to (currently hard-coded) layouting
                functions.
            figure_kwargs (dict, optional): Passed to
                :py:func:`matplotlib.pyplot.figure` for setting up the figure
            annotate_kwargs (dict, optional): Used for annotating the graph
                with a title and a legend (for ``show_node_status``).
                Supported keys: ``title``, ``title_kwargs``, ``add_legend``,
                ``legend_kwargs``, ``handle_kwargs``.
            save_kwargs (dict, optional): Passed to
                :py:func:`matplotlib.pyplot.savefig` for saving the figure

        Returns:
            networkx.DiGraph: The passed or generated graph object.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        from .plot.funcs.graph import _draw_graph

        def setup_figure(**figure_kwargs) -> "matplotlib.figure.Figure":
            """Creates a new figure"""
            return plt.figure(**figure_kwargs)

        def annotate_plot(
            *,
            ax,
            show_node_status: bool,
            node_status_color: dict,
            title: str = None,
            title_kwargs: dict = None,
            add_legend: bool = True,
            handle_kwargs: dict = None,
            legend_kwargs: bool = None,
        ) -> list:
            """Add a few annotations to the plot: title & legend for the
            various node status colours.

            Uses figure-level legend and title to reduce overlapping within
            the axes.
            """
            fig = ax.get_figure()
            artists = []

            def create_patch(c, label, **kws):
                from matplotlib.lines import Line2D

                label = label.replace("_", " ")
                return Line2D(
                    [0], [0], marker="o", label=label, markerfacecolor=c, **kws
                )

            if title is not None:
                t = fig.suptitle(
                    title, **(title_kwargs if title_kwargs else {})
                )
                artists.append(t)

            if show_node_status and add_legend:
                handle_kwargs = handle_kwargs if handle_kwargs else {}
                legend_kwargs = legend_kwargs if legend_kwargs else {}
                handles = [
                    create_patch(c, l, **handle_kwargs)
                    for l, c in node_status_color.items()
                ]

                lgd = fig.legend(handles=handles, **legend_kwargs)
                artists.append(lgd)

            return artists

        def save_plot(*, out_path: str, bbox_inches="tight", **save_kwargs):
            """Saves the matplotlib plot to the given output path"""
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(
                out_path,
                bbox_inches=bbox_inches,
                **(save_kwargs if save_kwargs else {}),
            )

        # .....................................................................

        drawing = drawing if drawing else dict(nodes={}, edges={}, labels={})

        # Generate the graph object
        if g is None:
            g = self.generate_nx_graph(**generation)

        elif generation:
            raise ValueError(
                "With a graph given for visualization, the `generation` "
                "argument is not allowed!"
            )

        # Specify defaults
        # TODO Consider defining these elsewhere
        if use_defaults:
            figure_defaults = dict()
            layout_defaults = dict()
            drawing_defaults = dict()

            figure_defaults["figsize"] = (9, 7)

            layout_defaults["model"] = "graphviz_dot"
            layout_defaults["model_kwargs"] = dict(
                graphviz_dot=dict(),
                multipartite=dict(
                    align="horizontal",
                    subset_key="layer",
                    scale=-1,
                    wiggle=dict(x=0.005, seed=123),
                ),
            )
            layout_defaults["fallback"] = "multipartite"
            layout_defaults["silent_fallback"] = True

            drawing_defaults["nodes"] = dict(
                alpha=0.7,
                node_color="w",
                node_size=600,
                linewidths=0,
            )
            drawing_defaults["edges"] = dict(
                arrows=True,
                alpha=0.7,
                arrowsize=12,
                min_target_margin=20,
                min_source_margin=20,
                node_size=drawing_defaults["nodes"].get("node_size"),
            )
            drawing_defaults["labels"] = dict(
                from_attr="description",
                font_size=6,
                bbox=dict(
                    fc="#fffa", ec="#666", linewidth=0.5, boxstyle="round"
                ),
            )

            annotate_defaults = dict(
                # FIXME Positioning is not ideal ...
                # title_kwargs=dict(y=1.02),
                legend_kwargs=dict(
                    loc="lower center",  # upper center is better
                    # bbox_to_anchor=(0.5, 0.98),
                    fontsize=5,
                    ncol=4,
                    framealpha=0,
                ),
                handle_kwargs=dict(
                    color="k",
                    linewidth=0,
                    markersize=6,
                    markeredgewidth=0.2,
                    alpha=0.7,
                ),
            )

            # Use them as basis ...
            figure_kwargs = _recursive_update(figure_defaults, figure_kwargs)
            annotate_kwargs = _recursive_update(
                annotate_defaults, annotate_kwargs
            )
            layout = _recursive_update(layout_defaults, layout)
            drawing = _recursive_update(drawing_defaults, drawing)

        # Figure size scaling
        if scale_figsize:
            if isinstance(scale_figsize, bool):
                scale_figsize = (0.25, 0.22)

            sw, sh = scale_figsize

            figsize = figure_kwargs.pop(
                "figsize", plt.rcParams["figure.figsize"]
            )

            # Compute the maximum layer (assuming starting from 0) and the
            # maximum layer occupation (count of the mode of the layers list)
            layers = [
                lyr if lyr else 0
                for _, lyr in nx.get_node_attributes(g, "layer").items()
            ]
            layers.append(1)  # to ensure that there is at least one element
            max_layer = max(layers)
            max_occupation = layers.count(max(set(layers), key=layers.count))

            figure_kwargs["figsize"] = (
                figsize[0] * sw * max_occupation,
                figsize[1] * sh * max_layer,
            )

        # Show status
        if show_node_status:
            if node_status_color is None:
                # TODO Is there a better location for this?!
                node_status_color = dict(
                    initialized="lightskyblue",
                    queued="cornflowerblue",
                    computed="limegreen",
                    looked_up="forestgreen",
                    failed_here="red",
                    failed_in_dependency="firebrick",
                    used_fallback="gold",
                    no_status="silver",
                )
            drawing["nodes"]["node_color"] = [
                node_status_color.get(s, node_status_color["no_status"])
                for _, s in nx.get_node_attributes(g, "status").items()
            ]

        annotate_kwargs["show_node_status"] = show_node_status
        annotate_kwargs["node_status_color"] = node_status_color

        # ... and draw
        artists = []
        try:
            fig = setup_figure(**figure_kwargs)
            ax = plt.gca()
            artists += _draw_graph(g, ax=ax, layout=layout, drawing=drawing)
            artists += annotate_plot(ax=ax, **annotate_kwargs)
            save_plot(
                out_path=out_path, bbox_extra_artists=artists, **save_kwargs
            )

        finally:
            plt.close(fig)

        return g

    # .........................................................................
    # Helpers

    def _parse_trfs(
        self, *, select: dict, transform: Sequence[dict], define: dict = None
    ) -> Sequence[dict]:
        """Parse the given arguments to bring them into a uniform format: a
        sequence of parameters for transformation operations.
        The arguments are parsed starting with the ``define`` tags, followed by
        the ``select`` and the ``transform`` argument.

        Args:
            select (dict): The shorthand to select certain objects from the
                DataManager. These may also include transformations.
            transform (Sequence[dict]): Actual transformation operations,
                carried out afterwards.
            define (dict, optional): Each entry corresponds either to a
                transformation sequence (if type is list or tuple) where the
                key is used as the tag and attached to the last transformation
                of each sequence.
                For *any* other type, will add a single transformation directly
                with the content of each entry.

        Returns:
            Sequence[dict]: A sequence of transformation parameters that was
                brought into a uniform structure.

        Raises:
            TypeError: On invalid type within entry of ``select``
            ValueError: When ``file_cache`` is given for selection from base
        """
        # The to-be-populated list of transformations
        trfs = list()

        # Prepare arguments: make sure they are dicts and deep copies.
        define = _deepcopy(define) if define else {}
        select = _deepcopy(select) if select else {}
        transform = _deepcopy(transform) if transform else []

        # First, parse the entries in the ``define`` argument
        for tag, define_spec in sorted(define.items()):
            # To allow a wider range of syntax, convert non-sequence like
            # arguments into a single transformation node that is basically
            # a defintion of the given object. Don't need to do anything else
            # in that case.
            if not isinstance(define_spec, (list, tuple)):
                trfs.append(
                    dict(operation="define", args=[define_spec], tag=tag)
                )
                continue

            # Now parse and add them
            for i, trf_params in enumerate(define_spec):
                # Parse minimal and regular syntax
                trf_params = _parse_dag_minimal_syntax(trf_params)
                trf_params = _parse_dag_syntax(**trf_params)
                trfs.append(trf_params)

            # Use an additional passthrough transformation to set the tag
            trfs.append(dict(operation="pass", args=[DAGNode(-1)], tag=tag))

        # Second, parse the ``select`` argument.
        # This contains a basic operation to select data from the selection
        # base (e.g. the DataManager) and also allows to perform some
        # operations on it.
        for tag, params in sorted(select.items()):
            if isinstance(params, str):
                path = params
                with_previous_result = False
                more_trfs = None
                salt = None
                omit_tag = False
                allow_failure = None
                fallback = None

            elif isinstance(params, dict):
                path = params["path"]
                with_previous_result = params.get(
                    "with_previous_result", False
                )
                more_trfs = params.get("transform")
                salt = params.get("salt")
                omit_tag = params.get("omit_tag", False)
                allow_failure = params.get("allow_failure", None)
                fallback = params.get("fallback", None)

                if "file_cache" in params:
                    raise ValueError(
                        "For selection from the selection base, the file "
                        "cache is always disabled! The `file_cache` argument "
                        "is thus not allowed; remove it from the selection "
                        f"for tag '{tag}'."
                    )

            else:
                raise TypeError(
                    f"Invalid type for '{tag}' entry within `select` argument!"
                    f" Got {type(params)} but expected string or dict."
                )

            # If given, process the path by prepending the prefix
            if self._select_path_prefix:
                if self._select_path_prefix[-1] == PATH_JOIN_CHAR:
                    path = self._select_path_prefix + path
                else:
                    path = PATH_JOIN_CHAR.join(
                        [self._select_path_prefix, path]
                    )

            # Construct parameters to select from the selection base.
            # Only assign a tag if there are no further transformations;
            # otherwise, the last additional transformation should set the tag.
            sel_trf = dict(
                operation="getitem",
                tag=None if (more_trfs or omit_tag) else tag,
                args=[DAGTag("select_base"), path],
                kwargs=dict(),
                file_cache=dict(read=False, write=False),
            )

            # Carry additional parameters on only if they were given
            if salt is not None:
                sel_trf["salt"] = salt

            if allow_failure is not None:
                sel_trf["allow_failure"] = allow_failure

            if fallback is not None:
                sel_trf["fallback"] = fallback

            # Now finished with the formulation of the select operation.
            trfs.append(sel_trf)

            if not more_trfs:
                # Done with this select operation.
                continue

            # else: there are additional transformations to be parsed and added
            for i, trf_params in enumerate(more_trfs):
                trf_params = _parse_dag_minimal_syntax(trf_params)

                # Might have to use the previous result ...
                if "with_previous_result" not in trf_params:
                    trf_params["with_previous_result"] = with_previous_result

                # Can now parse the regular syntax
                trf_params = _parse_dag_syntax(**trf_params)

                # If the tag is not to be omitted, the last transformation for
                # the selected tag needs to set the tag.
                if i + 1 == len(more_trfs) and not omit_tag:
                    if trf_params.get("tag"):
                        raise ValueError(
                            "The tag of the last transform operation within a "
                            "select routine cannot be set manually. Check the "
                            f"parameters for selection of tag '{tag}'."
                        )

                    # Add the tag to the parameters
                    trf_params["tag"] = tag

                # Can append it now
                trfs.append(trf_params)

        # Now, parse the normal `transform` argument. The operations defined
        # here are added after the instructions from the `select` section.
        for trf_params in transform:
            trf_params = _parse_dag_minimal_syntax(trf_params)
            trfs.append(_parse_dag_syntax(**trf_params))

        # Done parsing, yay.
        return trfs

    def _add_meta_operation_nodes(
        self,
        operation: str,
        *,
        args: list = None,
        kwargs: dict = None,
        tag: str = None,
        force_compute: bool = None,
        file_cache: dict = None,
        allow_failure: Union[bool, str] = None,
        fallback: Any = None,
        **trf_kwargs,
    ) -> DAGReference:
        """Adds Transformation nodes for meta-operations

        This method resolves the placeholder references in the specified meta-
        operation such that they point to the ``args`` and ``kwargs``.
        It then calls :py:meth:`.add_node`
        repeatedly to add the actual nodes.

        .. note::

            The last node added by this method is considered the "result" of
            the selected meta-operation. Subsequently, the arguments ``tag``,
            ``file_cache``, ``allow_failure`` and ``fallback`` are *only*
            applied to this last node.

            The ``trf_kwargs`` (which include the ``salt``) on the other hand
            are passed to *all* transformations of the meta-operation.

        Args:
            operation (str): The meta-operation to add nodes for
            args (list, optional): Positional arguments to the meta-operation
            kwargs (dict, optional): Keyword arguments to the meta-operation
            tag (str, optional): The tag that is to be attached to the *result*
                of this meta-operation.
            file_cache (dict, optional): File caching options for the *result*.
            allow_failure (Union[bool, str], optional): Specifies the error
                handling for the *result* node of this meta-operation.
            fallback (Any, optional): Specifies the fallback for the *result*
                node of this meta-operation.
            **trf_kwargs: Transformation keyword arguments, passed on to *all*
                transformations that are to be added.
        """
        # Retrieve the meta-operation information. The specs need to be a deep
        # copy, as they are recursively replaced later on.
        meta_op = self._meta_ops[operation]
        specs = _deepcopy(meta_op["specs"])

        # Check that all the required args and kwargs are present
        args = args if args else []
        kwargs = kwargs if kwargs else {}

        if len(args) < len(meta_op["required_args"]):
            raise MetaOperationInvocationError(
                f"Meta-operation '{operation}' requires at least "
                f"{len(meta_op['required_args'])} positional argument(s) but "
                f"got only {len(args)}!"
            )
        elif len(args) > meta_op["num_args"]:
            raise MetaOperationInvocationError(
                f"Meta-operation '{operation}' expects "
                f"{len(meta_op['required_args'])} required and "
                f"{len(meta_op['optional_args'])} optional positional "
                f"argument(s) but got {len(args)}!"
            )

        if any(_k not in kwargs for _k in meta_op["required_kwargs"]):
            _rq_kws = meta_op["required_kwargs"]
            _missing = ", ".join(_k for _k in _rq_kws if _k not in kwargs)
            raise MetaOperationInvocationError(
                f"Meta-operation '{operation}' misses the following required "
                f"keyword argument(s):  {_missing}"
            )

        elif any(_k not in meta_op["kwarg_names"] for _k in kwargs):
            _allowed = meta_op["kwarg_names"]
            _bad_kwargs = ", ".join(_k for _k in kwargs if _k not in _allowed)
            raise MetaOperationInvocationError(
                f"Meta-operation '{operation}' got superfluous keyword "
                f"arguments ({_bad_kwargs})! Remove them."
            )

        # Replace potentially existing relative references in the outside
        # scope -- like a DAGNode(-1) -- into absolute references.
        # This needs to happen in this outer scope, before any nodes are added,
        # because otherwise the reference would point to a node that was added
        # as part of the meta operation.
        # As nested meta-operations also end up at this point, this applies to
        # all scopes.
        is_rel_ref = lambda o: isinstance(o, DAGNode)
        convert_to_abs_ref = lambda ref: ref.convert_to_ref(dag=self)

        args = recursive_replace(
            args, select_func=is_rel_ref, replace_func=convert_to_abs_ref
        )
        kwargs = recursive_replace(
            kwargs, select_func=is_rel_ref, replace_func=convert_to_abs_ref
        )

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Define the helper methods and a placeholder resolution function for
        # the positional and keyword arguments ...
        is_arg = lambda obj: isinstance(obj, _Arg)
        is_kwarg = lambda obj: isinstance(obj, _Kwarg)

        def perform_lookup(
            ph: Union[_Arg, _Kwarg], lookup: Union[list, dict]
        ) -> Any:
            try:
                return lookup[ph.data]
            except (IndexError, KeyError) as err:
                if ph.has_fallback:
                    return ph.fallback
                # Will never get here, because these cases where checked
                # at the beginning of the invocation. If we get here, something
                # is really wrong.
                raise

        # The replacement functions then have full access to args and kwargs,
        # regardless of where in the meta-operation they appear.
        replace_arg = lambda ph: _deepcopy(perform_lookup(ph, args))
        replace_kwarg = lambda ph: _deepcopy(perform_lookup(ph, kwargs))

        # For the internally-defined meta-operation tags:
        is_mop_tag = lambda obj: isinstance(obj, _MOpTag)
        replace_mop_tag = lambda tag: tag.convert_to_ref(dag=self)

        # Keep track of the tags that are being added, such that the reference
        # stacks can be cleaned-up after all nodes were added.
        added_tags = list()

        def fill_placeholders_and_add_node(
            spec: dict, **node_kwargs
        ) -> DAGReference:
            """Fills all placeholder objects (positional and keyword arguments
            as well as internally-defined meta-operation tags) and then adds a
            node to the DAG via the regular method.

            This also adjusts the reference stacks for this meta-operation.
            """
            spec = recursive_replace(
                spec, select_func=is_arg, replace_func=replace_arg
            )
            spec = recursive_replace(
                spec, select_func=is_kwarg, replace_func=replace_kwarg
            )
            spec = recursive_replace(
                spec, select_func=is_mop_tag, replace_func=replace_mop_tag
            )

            # If a tag were to be defined, remove it here; have to handle it
            # via the reference stacks, as there would otherwise be duplicate
            # tags registered via add_node.
            tag = spec.pop("tag", None)

            # Update the context
            spec["context"] = _recursive_update(
                spec.get("context", {}), dict(meta_operation=operation)
            )
            spec["context"] = _recursive_update(
                spec["context"], node_kwargs.pop("context", {})
            )
            spec["context"]["meta_op_internal_tag"] = tag

            # Can now add the node
            ref = self.add_node(**spec, **node_kwargs)

            # If this node tried to define a tag, need to put the reference of
            # this particular instantiation of this tag on the stack, such that
            # the replacement above is unique.
            if tag:
                _mop_tag_name = _MOpTag.make_name(operation, tag=tag)
                added_tags.append(_mop_tag_name)
                self.ref_stacks[_mop_tag_name].append(ref.ref)  # ... as str!

            return ref

        # Now, add nodes for all transform specifications.
        for spec in specs:
            fill_placeholders_and_add_node(spec, **trf_kwargs)

        # Add the result node separately, passing the singular associations
        # for tag, file_cache etc. here and thereby allowing the meta-operation
        # to do internally whatever it wants.
        ref = self.add_node(
            operation="pass",
            args=[DAGNode(-1)],
            tag=tag,
            force_compute=force_compute,
            file_cache=file_cache,
            allow_failure=allow_failure,
            fallback=fallback,
            context=dict(meta_operation=operation),
        )

        # With all nodes added now, can pop off all the references for the tags
        # that were defined by this specific use of the meta-operation. This is
        # to ensure that tags used in meta-operations that are nested further
        # outside point to the correct reference.
        for _mop_tag in added_tags:
            self.ref_stacks[_mop_tag].pop()

        return ref

    def _update_profile(self, **times):
        """Updates profiling information by adding the given time to the
        matching key.
        """
        for key, t in times.items():
            self._profile[key] = self._profile.get(key, 0.0) + t

    def _parse_compute_only(
        self, compute_only: Union[str, List[str]]
    ) -> List[str]:
        """Prepares the ``compute_only`` argument for use in
        :py:meth:`.compute`.
        """
        compute_only = compute_only if compute_only is not None else "all"

        if compute_only == "all":
            compute_only = [
                t
                for t in self.tags.keys()
                if (
                    t not in self.exclude_from_all
                    and t not in self.SPECIAL_TAGS
                    and not (t.startswith(".") or t.startswith("_"))
                )
            ]

        else:
            # Check that all given tags actually are available
            invalid_tags = [t for t in compute_only if t not in self.tags]
            if invalid_tags:
                _invalid_tags = ", ".join(invalid_tags)
                _available_tags = ", ".join(self.tags)
                raise ValueError(
                    "Some of the tags specified in `compute_only` were not "
                    "available in the TransformationDAG!\n"
                    f"  Invalid tags:   {_invalid_tags}\n"
                    f"  Available tags: {_available_tags}"
                )

        return compute_only

    def _find_tag(self, trf: Union[Transformation, str]) -> Union[str, None]:
        """Looks up a tag given a transformation or its hashstr.

        If no tag is associated returns None. If multiple tags are associated,
        returns only the first.

        Args:
            trf (Union[Transformation, str]): The transformation, either as
                the object or as its hashstr.
        """
        if not isinstance(trf, str):
            trf = trf.hashstr

        candidate_tags = [n for n, h in self.tags.items() if h == trf]
        # TODO consider warning
        # if len(candidate_tags) > 1:
        #     pass

        return candidate_tags[0] if candidate_tags else None

    # .........................................................................
    # Cache writing and reading
    # NOTE This is done here rather than in Transformation because this is the
    #      more central entity and it is a bit easier ...

    def _retrieve_from_cache_file(
        self, trf_hash: str, **load_kwargs
    ) -> Tuple[bool, Any]:
        """Retrieves a transformation's result from a cache file and stores it
        in the data manager's cache group.

        .. note::

            If a file was already loaded from the cache, it will not be loaded
            again. Thus, the DataManager acts as a persistent storage for
            loaded cache files. Consequently, these are shared among all
            TransformationDAG objects.
        """
        success, res = False, None

        # Check if the file was already loaded; only go through the trouble of
        # checking all the hash files and invoking the load method if the
        # desired cache file was really not loaded
        try:
            res = self.dm[DAG_CACHE_DM_PATH][trf_hash]
        except ItemAccessError:
            pass
        else:
            success = True

        if not success:
            cache_files = self.cache_files

            if trf_hash not in cache_files.keys():
                # Bad luck, no cache file
                log.trace("No cache file found for %s.", trf_hash)
                return success, res

            # Parse load options
            if "exists_action" not in load_kwargs:
                load_kwargs["exists_action"] = "skip_nowarn"

            # else: There was a file. Let the DataManager load it.
            file_ext = cache_files[trf_hash]["ext"]
            log.trace("Loading result %s from cache file ...", trf_hash)

            with _adjusted_log_levels(("dantro.data_mngr", logging.WARNING)):
                self.dm.load(
                    "dag_cache",
                    loader=LOADER_BY_FILE_EXT[file_ext[1:]],
                    base_path=self.cache_dir,
                    glob_str=trf_hash + file_ext,
                    target_path=DAG_CACHE_DM_PATH + "/{basename:}",
                    required=True,
                    **load_kwargs,
                )

            # Can now retrieve the loaded data
            res = self.dm[DAG_CACHE_DM_PATH][trf_hash]
            success = True

        # Have to unpack some container types
        if isinstance(res, DAG_CACHE_CONTAINER_TYPES_TO_UNPACK):
            res = res.data

        # Done.
        return success, res

    def _write_to_cache_file(
        self,
        trf_hash: str,
        *,
        result: Any,
        ignore_groups: bool = True,
        attempt_pickling: bool = True,
        raise_on_error: bool = False,
        pkl_kwargs: dict = None,
        **save_kwargs,
    ) -> bool:
        """Writes the given result object to a hash file, overwriting existing
        ones.

        Args:
            trf_hash (str): The hash; will be used for the file name
            result (Any): The result object to write as a cache file
            ignore_groups (bool, optional): Whether to store groups. Disabled
                by default.
            attempt_pickling (bool, optional): Whether it should be attempted
                to store results that could not be stored via a dedicated
                storage function by pickling them. Enabled by default.
            raise_on_error (bool, optional): Whether to raise on error to
                store a result. Disabled by default; it is useful to enable
                this when debugging.
            pkl_kwargs (dict, optional): Arguments passed on to the
                pickle.dump function.
            **save_kwargs: Passed on to the chosen storage method.

        Returns:
            bool: Whether a cache file was saved

        Raises:
            NotImplementedError: When attempting to store instances of
                :py:class:`~dantro.base.BaseDataGroup` or a derived class
            RuntimeError: When ``raise_on_error`` was given and there was an
                error during saving.
        """
        # Cannot store groups
        if isinstance(result, BaseDataGroup):
            if not ignore_groups:
                raise NotImplementedError(
                    "Cannot currently write dantro "
                    "groups to a cache file. Sorry. Adjust the ignore_groups "
                    "argument in the file cache write options for the "
                    f"transformation resulting in {result}."
                )
            return False

        # Make sure the directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Prepare the file path (still lacks an extension)
        fpath = os.path.join(self.cache_dir, trf_hash)

        # Delete all potentially existing cache files for this hash
        for to_delete in glob.glob(fpath + ".*"):
            log.debug("Removing already existing cache file for this hash ...")
            os.remove(to_delete)

        # Go over the saving functions and see if the type agrees. If so, use
        # that function to write the data.
        for types, sfunc in DAG_CACHE_RESULT_SAVE_FUNCS.items():
            if not isinstance(result, resolve_types(types)):
                continue

            # else: type matched, invoke saving function
            try:
                sfunc(result, fpath, **save_kwargs)

            except Exception as exc:
                # Make sure that sfunc had no side effects, e.g. a corrupt file
                # that would lead to duplicate cache files.
                # As we ensure that there are no duplicate cache files and
                # because storage functions do not fail upon an existing file,
                # we can use the hash for deletion; and have to, because we do
                # not know the file extension ...
                for bad_fpath in glob.glob(fpath + ".*"):
                    os.remove(bad_fpath)
                    log.trace(
                        "Removed cache file after storage failure: %s",
                        bad_fpath,
                    )

                # Generate an error or warning message
                msg = (
                    "Failed saving transformation cache file for result of "
                    f"type {type(result)} using storage function for type(s) "
                    f"{types}. Value of result:\n{result}.\n\n"
                    f"Additional keyword arguments: {save_kwargs}\n"
                    f"Upstream error was a {exc.__class__.__name__}: {exc}"
                )
                if raise_on_error:
                    raise RuntimeError(msg) from exc
                log.warning("%s.\n%s: %s", msg, exc.__class__.__name__, exc)

                # Don't return here; might attempt pickling below or have
                # another specialized saving method available ...

            else:
                # Success
                log.trace("Successfully stored cache file for %s.", trf_hash)
                return True

        # Reached the end of the loop without returning -> not saved yet
        if not attempt_pickling:
            return False

        # Try to pickle it
        try:
            with open(fpath + ".pkl", mode="wb") as pkl_file:
                pkl.dump(
                    result, pkl_file, **(pkl_kwargs if pkl_kwargs else {})
                )

        except Exception as exc:
            msg = (
                "Failed saving transformation cache file. Cannot pickle "
                f"result object of type {type(result)} and with value "
                f"{result}. Consider deactivating file caching or pickling "
                "for this transformation.\n"
                f"Upstream error was a {exc.__class__.__name__}: {exc}"
            )
            if raise_on_error:
                raise RuntimeError(msg) from exc
            log.warning("%s. %s: %s", msg, exc.__class__.__name__, exc)

            return False

        # else: Success
        log.trace("Successfully stored cache file for %s.", trf_hash)
        return True
