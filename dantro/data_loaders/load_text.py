"""Defines a loader mixin to load plain text objects"""

from ..containers import TextDataContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------

class TextLoaderMixin:
    """Supplies functionality to load plain text"""

    @add_loader(TargetCls=TextDataContainer)
    def _load_plain_text(filepath: str, *, TargetCls: type,
                         **load_kwargs) -> TextDataContainer:
        """Loads the content of a plain text file back into a TextDataContainer.
        
        Args:
            filepath (str): Where the *.txt file is located
            TargetCls (type): The class constructor
            **load_kwargs: Passed on to open, see there for kwargs
        
        Returns:
            TextDataContainer: The reconstructed TextDataContainer
        """
        with open(filepath, **load_kwargs) as f:
            data = f.read()

        return TargetCls(data=data, attrs=dict(filepath=filepath))

    # Also make it available under plain 'text'
    _load_text = _load_plain_text
