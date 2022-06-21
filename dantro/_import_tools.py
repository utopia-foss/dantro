"""Tools for module importing, e.g. lazy imports."""

import copy
import importlib
import importlib.util
import logging
import os
import sys
from types import ModuleType
from typing import Any, Callable, Sequence, Tuple, Union

log = logging.getLogger(__name__)

# -- Context managers ---------------------------------------------------------


class added_sys_path:
    """A :py:data:`sys.path` context manager temporarily adding a path and
    removing it again upon exiting.
    If the given path already exists in :py:data`sys.path`, it is neither added
    nor removed and :py:data`sys.path` remains unchanged.

    .. todo::

        Expand to allow multiple paths being added
    """

    def __init__(self, path: str):
        """Initialize the context manager.

        Args:
            path (str): The path to add to :py:data:`sys.path`.
        """
        self.path = path
        self.path_already_exists = self.path in sys.path

    def __enter__(self):
        if not self.path_already_exists:
            log.debug("Temporarily adding '%s' to sys.path ...")
            sys.path.insert(0, self.path)

    def __exit__(self, *_):
        if not self.path_already_exists:
            log.debug("Removing temporarily added path from sys.path ...")
            sys.path.remove(self.path)


class temporary_sys_modules:
    """A context manager for the :py:data:`sys.modules` cache, ensuring that it
    is in the same state after exiting as it was before entering the context.

    .. note::

        This works solely on module *names*, not on the module objects! If a
        module object itself is overwritten, this context manager is not
        able to discern that as long as the key does not change.
    """

    def __init__(self, *, reset_only_on_fail: bool = False):
        """Set up the context manager for a temporary :py:data:`sys.modules`
        cache.

        Args:
            reset_only_on_fail (bool, optional): If True, will reset the cache
                only in case the context is exited with an exception.
        """
        self._modules = {ms: mod for ms, mod in sys.modules.items()}
        self.reset_only_on_fail = reset_only_on_fail

    def __enter__(self):
        pass

    def __exit__(self, exc_type: type, *_):
        if self.reset_only_on_fail and exc_type is None:
            return

        elif sys.modules == self._modules:
            return

        # else: Reset module cache
        # NOTE Cannot simply assign here, but need to insert and delete,
        #      otherwise module objects may become disassociated
        to_remove = [ms for ms in sys.modules if ms not in self._modules]
        to_add_back = [ms for ms in self._modules if ms not in sys.modules]

        for ms in to_remove:
            del sys.modules[ms]
        for ms in to_add_back:
            sys.modules[ms] = self._modules[ms]

        log.debug(
            "Reset sys.modules cache to state before entering context manager."
        )


# -- Various import functions -------------------------------------------------


def get_from_module(mod: ModuleType, *, name: str):
    """Retrieves an attribute from a module, if necessary traversing along the
    module string.

    Args:
        mod (ModuleType): Module to start looking at
        name (str): The ``.``-separated module string leading to the desired
            object.
    """
    obj = mod

    for attr_name in name.split("."):
        try:
            obj = getattr(obj, attr_name)

        except AttributeError as err:
            raise AttributeError(
                f"Failed to retrieve attribute or attribute sequence '{name}' "
                f"from module '{mod.__name__}'! Intermediate "
                f"{type(obj).__name__} {obj} has no attribute '{attr_name}'!"
            ) from err

    return obj


def import_module_or_object(
    module: str = None, name: str = None, *, package: str = "dantro"
) -> Any:
    """Imports a module or an object using the specified module string and the
    object name. Uses :py:func:`importlib.import_module` to retrieve the module
    and then uses :py:func:`~dantro._import_tools.get_from_module` for getting
    the ``name`` from that module (if given).

    Args:
        module (str, optional): A module string, e.g. ``numpy.random``. If
            this is not given, it will import from the :py:mod`builtins`
            module. If this is a relative module string, will resolve starting
            from ``package``.
        name (str, optional): The name of the object to retrieve from the
            chosen module and return. This may also be a dot-separated sequence
            of attribute names which can be used to traverse along attributes,
            which uses :py:func:`~dantro._import_tools.get_from_module`.
        package (str, optional): Where to import from if ``module`` was a
            relative module string, e.g. ``.data_mngr``, which would lead to
            resolving the module from ``<package><module>``.

    Returns:
        Any: The chosen module or object, i.e. the object found at
            ``<module>.<name>``

    Raises:
        AttributeError: In cases where part of the ``name`` argument could not
            be resolved due to a bad attribute name.
    """
    module = module if module else "builtins"
    mod = (
        importlib.import_module(module, package=package)
        if module
        else __builtin__
    )

    if not name:
        return mod
    return get_from_module(mod, name=name)


def import_name(modstr: str):
    """Given a module string, import a name, treating the last segment of the
    module string as the name.

    .. note::

        If the last segment of ``modstr`` is *not* the name, use
        :py:func:`~dantro._import_tools.import_module_or_object` instead of
        this function.

    Args:
        modstr (str): A module string, e.g. ``numpy.random.randint``, where
            ``randint`` will be the name to import.
    """
    ms = modstr.split(".")
    return import_module_or_object(module=".".join(ms[:-1]), name=ms[-1])


def import_module_from_path(
    *, mod_path: str, mod_str: str, debug: bool = True
) -> Union[None, ModuleType]:
    """Helper function to import a module that is importable only when adding
    the module's parent directory to :py:data:`sys.path`.

    .. note::

        The ``mod_path`` directory needs to contain an ``__init__.py`` file.
        If that is not the case, you cannot use this function, because the
        directory does not represent a valid Python module.

        Alternatively, a single file can be imported as a module using
        :py:func:`~dantro._import_tools.import_module_from_file`.

    Args:
        mod_path (str): Path to the module's root *directory*, ``~`` expanded
        mod_str (str): Name under which the module can be imported with
            ``mod_path`` being in :py:data:`sys.path`. This is also used to
            add the module to the :py:data:`sys.modules` cache.
        debug (bool, optional): Whether to raise exceptions if import failed

    Returns:
        Union[None, ModuleType]: The imported module or None, if importing
            failed and ``debug`` evaluated to False.

    Raises:
        ImportError: If ``debug`` is set and import failed for whatever reason
        FileNotFoundError: If ``mod_path`` did not point to an existing
            *directory*
    """
    mod_path = os.path.expanduser(mod_path)

    if not os.path.isdir(mod_path):
        raise FileNotFoundError(
            "The `mod_path` argument to import a module from a path should be "
            f"the path to an existing directory! Given path:  {mod_path}"
        )

    # Need the parent directory in the path, because the import is only
    # possible from there. This, in turn, depends on the depth of the module
    # string, so the parent directory should be chosen accordingly.
    mod_parent_dir = mod_path
    for _ in mod_str.split("."):
        mod_parent_dir = os.path.dirname(mod_parent_dir)

    try:
        # Use the temporary environments to prevent that a *failed* import ends
        # up generating an erroneous sys.modules cache entry.
        _add_sys_path = added_sys_path(mod_parent_dir)
        _tmp_sys_modules = temporary_sys_modules(reset_only_on_fail=True)
        with _add_sys_path, _tmp_sys_modules:
            mod = importlib.import_module(mod_str)

    except Exception as exc:
        if debug:
            raise ImportError(
                f"Failed importing module '{mod_str}'!\n"
                f"Make sure that {mod_path}/__init__.py can be loaded without "
                "errors (with its parent directory being part of sys.path) "
                "and that the `mod_str` argument is correct. "
                "To debug, inspect the chained traceback."
            ) from exc

        log.debug(
            "Importing module '%s' from %s failed: %s", mod_str, mod_path, exc
        )
        return

    else:
        log.debug("Successfully imported module '%s'.", mod_str)

    return mod


def import_module_from_file(
    mod_file: str,
    *,
    base_dir: str = None,
    mod_name_fstr: str = "from_file.{filename:}",
) -> ModuleType:
    """Returns the module corresponding to the file at the given ``mod_file``.

    This uses :py:func:`importlib.util.spec_from_file_location` and
    :py:func:`importlib.util.module_from_spec` to construct a module from the
    given file, regardless of whether there is a ``__init__.py`` file beside
    the file or not.

    Args:
        mod_file (str): The path to a python module *file* to load as a module
        base_dir (str, optional): If given, uses this to resolve relative
            ``mod_file`` paths.
        mod_name_fstr (str): How to name the module. Should be a format string
            that is supplied with the ``filename`` argument.

    Returns:
        ModuleType: The imported module

    Raises:
        ValueError: If ``mod_file`` was a relative path but no ``base_dir`` was
            given.
    """
    mod_file = os.path.expanduser(mod_file)

    # Evaluate relative paths
    if not os.path.isabs(mod_file):
        if not base_dir:
            raise ValueError(
                f"Cannot import from a relative `mod_file` path ({mod_file}) "
                "if no `base_dir` argument was given!"
            )

        mod_file = os.path.join(base_dir, mod_file)

    # Extract a name from the path to use as module name, needed for building
    # a module specification.
    fname, _ = os.path.splitext(os.path.basename(mod_file))
    mod_name = mod_name_fstr.format(filename=fname)

    # Create a module specification and, from that, import the module
    spec = importlib.util.spec_from_file_location(mod_name, mod_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


# -- LazyLoader ---------------------------------------------------------------


class LazyLoader:
    """Delays import until the module's attributes are accessed.

    This is inspired by an implementation by Dboy Liao, see
    `here <https://levelup.gitconnected.com/python-trick-lazy-module-loading-df9b9dc111af>`_.

    It extends on it by allowing a ``depth`` until which loading will be lazy.
    """

    def __init__(self, mod_name: str, *, _depth: int = 0):
        """Initialize a placeholder for a module.

        .. warning::

            Values of ``_depth > 0`` may lead to unexpected behaviour of the
            root module, i.e. this object, because attribute calls do not
            yield an actual object. Only use this in scenarios where you are
            in full control over the attribute calls.

            We furthermore suggest to not make the LazyLoader instance publicly
            available in such cases.

        Args:
            mod_name (str): The module name to lazy-load upon attribute call.
            _depth (int, optional): With a depth larger than zero, attribute
                calls are not leading to an import *yet*, but to the creation
                of another LazyLoader instance (with depth reduced by one).
                Note the warning above regarding usage.
        """
        self.mod_name = mod_name
        self._mod = None
        self._depth = _depth
        log.debug("LazyLoader set up for '%s'.", self.mod_name)

    def __getattr__(self, name: str):
        # If depth is zero, do the import
        if self._depth <= 0:
            if self._mod is None:
                self._mod = self.resolve()
            return getattr(self._mod, name)

        # else: create a new lazy loader for the combined module string
        return LazyLoader(f"{self.mod_name}.{name}", _depth=self._depth - 1)

    def resolve(self):
        log.debug("Now lazy-loading '%s' ...", self.mod_name)

        modstr_parts = self.mod_name.split(".")
        return import_module_or_object(
            module=modstr_parts[0], name=".".join(modstr_parts[1:])
        )


def resolve_lazy_imports(d: dict, *, recursive: bool = True) -> dict:
    """In-place resolves lazy imports in the given dict, recursively.

    .. warning::

        Only recurses on dicts, not on other mutable objects!

    Args:
        d (dict): The dict to resolve lazy imports in
        recursive (bool, optional): Whether to recurse through the dict

    Returns:
        dict: ``d`` but with in-place resolved lazy imports
    """
    for k, v in d.items():
        if isinstance(v, LazyLoader):
            d[k] = v.resolve()

        elif recursive and isinstance(v, dict):
            d[k] = resolve_lazy_imports(d[k], recursive=recursive)

    return d


# -- Misc. --------------------------------------------------------------------


def remove_from_sys_modules(cond: Callable):
    """Removes cached module imports from :py:data:`sys.modules` if their
    fully qualified module name fulfills a certain condition.

    Args:
        cond (Callable): A unary function expecting a single ``str`` argument,
            the module name, e.g. ``numpy.random``. If the function returns
            True, will remove that module.
    """
    for modstr in [m for m in sys.modules if cond(m)]:
        del sys.modules[modstr]


def resolve_types(types: Sequence[Union[type, str]]) -> Sequence[type]:
    """Resolves multiple types, that may be given as module strings, into a
    tuple of types such that it can be used in :py:func:`isinstance` or
    similar functions.

    Args:
        types (Sequence[Union[type, str]]): The types to potentially resolve

    Returns:
        Sequence[type]: The resolved types sequence as a :py:class:`tuple`
    """
    return tuple(t if isinstance(t, type) else import_name(t) for t in types)
