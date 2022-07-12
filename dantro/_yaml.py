"""Takes care of all YAML-related imports and configuration

The ``ruamel.yaml.YAML`` object used here is imported from :py:mod:`paramspace`
and specialized such that it can load and dump dantro classes.
"""

import copy
import io
import logging
import os
from functools import partial as _partial
from typing import Any, Union

import matplotlib as mpl
import ruamel.yaml

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

# Import yaml from paramspace and configure it
from paramspace import yaml

yaml.default_flow_style = False

# Register further classes
from ._dag_utils import (
    DAGNode,
    DAGReference,
    DAGTag,
    KeywordArgument,
    Placeholder,
    PositionalArgument,
    ResultPlaceholder,
)

yaml.register_class(Placeholder)
yaml.register_class(ResultPlaceholder)
yaml.register_class(PositionalArgument)
yaml.register_class(KeywordArgument)
yaml.register_class(DAGReference)
yaml.register_class(DAGTag)
yaml.register_class(DAGNode)

# Special constructors ........................................................
# .. For the case of a reference to the "previous" node . . . . . . . . . . . .
yaml.constructor.add_constructor("!dag_prev", lambda l, n: DAGNode(-1))


# .. Colormap and norm creation . . . . . . . . . . . . . . . . . . . . . . . .


def _cmap_constructor(loader, node) -> "matplotlib.colors.Colormap":
    """Constructs a :py:class:`matplotlib.colors.Colormap` object for use in
    plots. Uses the :py:class:`~dantro.plot.utils.color_mngr.ColorManager` and
    directly resolves the colormap object from it.
    """
    from .plot.utils import ColorManager

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        cmap_kwargs = loader.construct_mapping(node, deep=True)
    else:
        cmap_kwargs = loader.construct_scalar(node)
    cmap = ColorManager(cmap=cmap_kwargs).cmap
    cmap._original_yaml = copy.deepcopy(cmap_kwargs)

    return cmap


def _cmap_norm_constructor(loader, node) -> "matplotlib.colors.Colormap":
    """Constructs a :py:class:`matplotlib.colors.Colormap` object for use in
    plots. Uses the :py:class:`~dantro.plot.utils.color_mngr.ColorManager` and
    directly resolves the colormap object from it.
    """
    from .plot.utils import ColorManager

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        norm_kwargs = loader.construct_mapping(node, deep=True)
    else:
        norm_kwargs = loader.construct_scalar(node)
    norm = ColorManager(norm=norm_kwargs).norm

    norm._original_yaml = copy.deepcopy(norm_kwargs)
    return norm


yaml.constructor.add_constructor("!cmap", _cmap_constructor)
yaml.constructor.add_constructor("!cmap_norm", _cmap_norm_constructor)


# Special representers ........................................................

# .. Representers for colormaps and norms . . . . . . . . . . . . . . . . . . .
# If constructed from the !cmap or !cmap_norm tags, these will have the
# additional attribute `_original_yaml`, which is then used for representation.


def _from_original_yaml(representer, node, *, tag: str):
    """For objects where a ``_original_yaml`` attribute was saved."""
    if isinstance(node._original_yaml, dict):
        return representer.represent_mapping(tag, node._original_yaml)
    return representer.represent_scalar(tag, node._original_yaml)


for CmapCls in [
    getattr(mpl.colors, a) for a in dir(mpl.colors) if "Colormap" in a
]:
    yaml.representer.add_representer(
        CmapCls, _partial(_from_original_yaml, tag="!cmap")
    )

for NormCls in [
    getattr(mpl.colors, a) for a in dir(mpl.colors) if "Norm" in a
]:
    yaml.representer.add_representer(
        NormCls, _partial(_from_original_yaml, tag="!cmap_norm")
    )

# -----------------------------------------------------------------------------


def load_yml(path: str, *, mode: str = "r") -> Union[dict, Any]:
    """Deserializes a YAML file into an object.

    Uses the dantro-internal ``ruamel.yaml.YAML`` object for loading and thus
    supports all registered constructors.

    Args:
        path (str): The path to the YAML file that should be loaded. A ``~`` in
            the path will be expanded to the current user's directory.
        mode (str, optional): Read mode

    Returns:
        Union[dict, Any]: The result of the data loading. Typically, this will
            be a dict, but depending on the structure of the file, it may also
            be of another type.
    """
    path = os.path.expanduser(path)
    log.debug("Loading YAML file... mode: %s, path:\n  %s", mode, path)

    with open(path, mode) as yaml_file:
        return yaml.load(yaml_file)


def write_yml(d: Union[dict, Any], *, path: str, mode: str = "w"):
    """Serialize an object using YAML and store it in a file.

    Uses the dantro-internal ``ruamel.yaml.YAML`` object for dumping and thus
    supports all registered representers.

    Args:
        d (dict): The object to serialize and write to file
        path (str): The path to write the YAML output to. A ``~`` in the path
            will be expanded to the current user's directory.
        mode (str, optional): Write mode of the file
    """
    path = os.path.expanduser(path)
    log.debug(
        "Dumping %s to YAML file... mode: %s, path:\n  %s",
        type(d).__name__,
        mode,
        path,
    )

    # Make sure the directory is present
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, mode) as yaml_file:
        # Add the yaml '---' prefix
        yaml_file.write("---\n")

        # Now dump the rest
        yaml.dump(d, stream=yaml_file)

        # Ensure new line
        yaml_file.write("\n")


# -----------------------------------------------------------------------------
# More specific YAML-based functions


def yaml_dumps(
    obj: Any,
    *,
    register_classes: tuple = (),
    yaml_obj: ruamel.yaml.YAML = None,
    **dump_params,
) -> str:
    """Serializes the given object using a newly created YAML dumper.

    The aim of this function is to provide YAML dumping that is not dependent
    on any package configuration; all parameters can be passed here.

    In other words, his function does _not_ use the dantro._yaml.yaml object
    for dumping but each time creates a new dumper with fixed settings. This
    reduces the chance of interference from elsewhere. Compared to the time
    needed for serialization in itself, the extra time needed to create the
    new ruamel.yaml.YAML object and register the classes is negligible.

    .. note::

        To use dantro's YAML object, it needs to be passed explicitly via the
        ``yaml_obj`` argument! Otherwise a new one will be created which might
        not have the desired classes registered.

    Args:
        obj (Any): The object to dump
        register_classes (tuple, optional): Additional classes to register
        yaml_obj (ruamel.yaml.YAML, optional): If given, use this YAML object
            for dumping. If not given, will create a new one.
        **dump_params: Dumping parameters

    Returns:
        str: The output of serialization

    Raises:
        ValueError: On failure to serialize the given object
    """
    s = io.StringIO()
    if yaml_obj is not None:
        y = yaml_obj
    else:
        y = ruamel.yaml.YAML()

    # Register classes; then apply dumping parameters via object properties
    for Cls in register_classes:
        y.register_class(Cls)

    for k, v in dump_params.items():
        setattr(y, k, v)

    # Serialize
    try:
        y.dump(obj, stream=s)

    except Exception as err:
        raise ValueError(
            f"Could not serialize the given {type(obj)} object!"
        ) from err

    return s.getvalue()
