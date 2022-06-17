"""Defines a loader mixin to load plain text objects"""

from ..containers import StringContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------


class TextLoaderMixin:
    """Supplies functionality to load plain text files"""

    @add_loader(TargetCls=StringContainer)
    def _load_plain_text(
        filepath: str, *, TargetCls: type, **load_kwargs
    ) -> StringContainer:
        """Loads the content of a plain text file back into a
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
