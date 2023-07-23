"""Takes care of all YAML-related imports and configuration

The ``ruamel.yaml.YAML`` object used here is imported from :py:mod:`yayaml`
and specialized such that it can load and dump dantro classes.
"""

import copy
import logging
from functools import partial as _partial
from typing import Callable, List, Tuple

import matplotlib as mpl
import ruamel.yaml
from yayaml import (
    add_yaml_error_hint,
    is_constructor,
    load_yml,
    write_yml,
    yaml,
    yaml_dumps,
)

from ._dag_utils import (
    DAGNode,
    DAGReference,
    DAGTag,
    KeywordArgument,
    Placeholder,
    PositionalArgument,
    ResultPlaceholder,
)

log = logging.getLogger(__name__)

# -- YAML configuration -------------------------------------------------------

yaml.default_flow_style = False

# -- Class registration -------------------------------------------------------
yaml.register_class(Placeholder)
yaml.register_class(ResultPlaceholder)
yaml.register_class(PositionalArgument)
yaml.register_class(KeywordArgument)
yaml.register_class(DAGReference)
yaml.register_class(DAGTag)
yaml.register_class(DAGNode)

# Special constructors ........................................................
# .. For the case of a reference to the "previous" node . . . . . . . . . . . .
def previous_DAGNode(loader, node):
    return DAGNode(-1)


yaml.constructor.add_constructor("!dag_prev", previous_DAGNode)
add_yaml_error_hint(
    lambda e: "!dag_prev" in str(e),
    "Did you include a space after the !dag_prev tag in that line?",
)

# .. Colormap and norm creation . . . . . . . . . . . . . . . . . . . . . . . .


@is_constructor("!cmap", aliases=("!colormap",))
def cmap_constructor(loader, node) -> "matplotlib.colors.Colormap":
    """Constructs a :py:class:`matplotlib.colors.Colormap` object for use in
    plots. Uses the :py:class:`~dantro.plot.utils.color_mngr.ColorManager` and
    directly resolves the colormap object from it.
    """
    from .plot.utils import ColorManager

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        cmap_kwargs = loader.construct_mapping(node, deep=True)
    else:
        cmap_kwargs = loader.construct_scalar(node)
    cm = ColorManager(cmap=cmap_kwargs)
    cmap = cm.cmap
    cmap._original_yaml = copy.deepcopy(cmap_kwargs)

    return cmap


@is_constructor("!cmap-norm", aliases=("!cmap_norm",))
def cmap_norm_constructor(loader, node) -> "matplotlib.colors.Colormap":
    """Constructs a :py:class:`matplotlib.colors.Colormap` object for use in
    plots. Uses the :py:class:`~dantro.plot.utils.color_mngr.ColorManager` and
    directly resolves the colormap object from it.
    """
    from .plot.utils import ColorManager

    if isinstance(node, ruamel.yaml.nodes.MappingNode):
        norm_kwargs = loader.construct_mapping(node, deep=True)
    else:
        norm_kwargs = loader.construct_scalar(node)
    cm = ColorManager(norm=norm_kwargs)
    norm = cm.norm
    norm._original_yaml = copy.deepcopy(norm_kwargs)

    return norm


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
        NormCls, _partial(_from_original_yaml, tag="!cmap-norm")
    )
