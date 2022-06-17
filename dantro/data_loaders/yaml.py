"""Supplies loading functions for YAML files"""

import logging

from .._yaml import load_yml
from ..containers import MutableMappingContainer, ObjectContainer
from ._tools import add_loader

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class YamlLoaderMixin:
    """Supplies functionality to load YAML files in the
    :py:class:`~dantro.data_mngr.DataManager`.
    Uses the :py:func:`dantro._yaml.load_yml` function for loading the files.
    """

    @add_loader(TargetCls=MutableMappingContainer)
    def _load_yaml(
        filepath: str, *, TargetCls: type, **load_kwargs
    ) -> MutableMappingContainer:
        """Load a YAML file from the given path and create a container to
        store that data in.
        Uses the :py:func:`dantro._yaml.load_yml` function for loading.

        Args:
            filepath (str): Where to load the YAML file from
            TargetCls (type): The class constructor
            **load_kwargs: Passed on to :py:func:`dantro._yaml.load_yml`

        Returns
            MutableMappingContainer: The loaded YAML content as a container
        """
        d = load_yml(filepath, **load_kwargs)
        return TargetCls(data=d, attrs=dict(filepath=filepath))

    @add_loader(TargetCls=ObjectContainer)
    def _load_yaml_to_object(
        filepath: str, *, TargetCls: type, **load_kwargs
    ) -> ObjectContainer:
        """Load a YAML file from the given path and create a container to
        store that data in.

        Uses the :py:func:`dantro._yaml.load_yml` function for loading.

        Args:
            filepath (str): Where to load the YAML file from
            TargetCls (type): The class constructor
            **load_kwargs: Passed on to :py:func:`dantro._yaml.load_yml`

        Returns:
            ObjectContainer: The loaded YAML content as an ObjectContainer
        """
        # NOTE Implementation is the same as above, but the TargetCls is not!
        d = load_yml(filepath, **load_kwargs)
        return TargetCls(data=d, attrs=dict(filepath=filepath))

    # Also make available under `yml` aliases
    _load_yml = _load_yaml
    _load_yml_to_object = _load_yaml_to_object
