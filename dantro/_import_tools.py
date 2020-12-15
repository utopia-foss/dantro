"""Tools for module importing, e.g. lazy imports."""

import logging
from importlib import import_module as _import_module

log = logging.getLogger(__name__)


def get_from_module(mod, *, name: str):
    """Retrieves an attribute from a module, if necessary traversing along the
    module string.

    Args:
        mod: Module to start looking at
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


def import_module_or_object(module: str = None, name: str = None):
    """Imports a module or an object using the specified module string and the
    object name.

    Args:
        module (str, optional): A module string, e.g. numpy.random. If this is
            not given, it will import from the :py:mod`builtins` module. Also,
            relative module strings are resolved from :py:mod:`dantro`.
        name (str, optional): The name of the object to retrieve from the
            chosen module and return. This may also be a dot-separated sequence
            of attribute names which can be used to traverse along attributes.

    Returns:
        The chosen module or object, i.e. the object found at <module>.<name>

    Raises:
        AttributeError: In cases where part of the ``name`` argument could not
            be resolved due to a bad attribute name.
    """
    module = module if module else "builtins"
    mod = _import_module(module, package="dantro") if module else __builtin__

    if not name:
        return mod
    return get_from_module(mod, name=name)


class LazyLoader:
    """Delays import until the module's attributes are accessed.

    This is inspired by the implementation by Dboy Liao:
        https://levelup.gitconnected.com/python-trick-lazy-module-loading-df9b9dc111af

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
