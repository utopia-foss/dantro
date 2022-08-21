"""Defines a loader mixin to load plain text files"""

from ..containers import StringContainer
from ._tools import add_loader


class TextLoaderMixin:
    """A mixin for :py:class:`~dantro.data_mngr.DataManager` that supports
    loading of plain text files."""

    @add_loader(TargetCls=StringContainer)
    def _load_plain_text(
        filepath: str, *, TargetCls: type, **load_kwargs
    ) -> StringContainer:
        """Loads the content of a plain text file into a
        :py:class:`~dantro.containers.general.StringContainer`.

        Args:
            filepath (str): Where the plain text file is located
            TargetCls (type): The class constructor
            **load_kwargs: Passed on to :py:func:`open`

        Returns:
            StringContainer: The reconstructed StringContainer
        """
        with open(filepath, **load_kwargs) as f:
            data = f.read()

        return TargetCls(data=data, attrs=dict(filepath=filepath))

    # Also make the loader available under the ``text`` label
    _load_text = _load_plain_text
