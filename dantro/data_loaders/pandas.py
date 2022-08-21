"""Defines a loader mixin to load data via :py:mod:`pandas`"""

from ..containers import PassthroughContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------


class PandasLoaderMixin:
    """Supplies functionality to load data via :py:mod:`pandas`."""

    @add_loader(TargetCls=PassthroughContainer)
    def _load_pandas_csv(
        filepath: str,
        *,
        TargetCls: type,
        **load_kwargs,
    ) -> PassthroughContainer:
        """Loads CSV data using :py:mod:`pandas.read_csv`, returning a
        :py:class:`~dantro.containers.general.PassthroughContainer`.

        .. note::

            As there is no proper equivalent of a :py:class:`pandas.DataFrame`
            in dantro (yet), and unpacking the dataframe into a dantro group
            would reduce functionality, a passthrough-container is used here.
            It behaves essentially like the object it wraps.

        Args:
            filepath (str): Where the CSV data file is located
            TargetCls (type): The class constructor
            **load_kwargs: Passed on to :py:func:`pandas.read_csv`

        Returns:
            PassthroughContainer: Payload being the loaded CSV data in form of
                a :py:class:`pandas.DataFrame`.
        """
        import pandas as pd

        df = pd.read_csv(filepath, **load_kwargs)

        return TargetCls(data=df, attrs=dict(filepath=filepath))
