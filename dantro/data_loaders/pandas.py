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
        """Loads CSV data using :py:func:`pandas.read_csv`, returning a
        :py:class:`~dantro.containers.general.PassthroughContainer`
        that contains a :py:class:`pandas.DataFrame`.

        .. note::

            As there is no proper equivalent of a :py:class:`pandas.DataFrame`
            in dantro (yet), and unpacking the dataframe into a dantro group
            would reduce functionality, a passthrough-container is used here.
            It behaves mostly like the object it wraps.

            However, in some cases, you may have to retrieve the underlying
            data using the ``.data`` property.

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

    @add_loader(TargetCls=PassthroughContainer)
    def _load_pandas_generic(
        filepath: str,
        *,
        TargetCls: type,
        reader: str,
        **load_kwargs,
    ) -> PassthroughContainer:
        """Loads data from a file using one of :py:mod:`pandas` ``read_*``
        functions, returning a :py:class:`pandas.DataFrame` wrapped into a
        :py:class:`~dantro.containers.general.PassthroughContainer`.

        The ``reader`` argument needs to match a reader function from
        `pandas IO <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`_.

        .. note::

            As there is no proper equivalent of a :py:class:`pandas.DataFrame`
            in dantro (yet), and unpacking the dataframe into a dantro group
            would reduce functionality, a passthrough-container is used here.
            It behaves mostly like the object it wraps.

            However, in some cases, you may have to retrieve the underlying
            data using the ``.data`` property.

        .. note::

            Some of pandas' reader functions require additional packages to
            have been installed.

        .. warning::

            While this in principle allows access to reader functions that are
            *not* file-based, calling those will most probably fail because the
            functions do not expect a file path as their first argument.

        Args:
            filepath (str): Where the data file is located
            TargetCls (type): The class constructor
            reader (str): The name of the reader function from pandas IO to use
            **load_kwargs: Passed on to the reader function

        Returns:
            PassthroughContainer: Payload being the loaded data in form of
                a :py:class:`pandas.DataFrame`.
        """
        import pandas as pd

        try:
            read_func = getattr(pd, f"read_{reader}")

        except AttributeError as err:
            NOT_FILE_READERS = (
                "clipboard",
                "gbq",
                "sql",
                "sql_query",
                "sql_table",
            )
            _avail = ", ".join(
                s[5:]
                for s in dir(pd)
                if s.startswith("read_") and s[5:] not in NOT_FILE_READERS
            )
            raise ValueError(
                f"Invalid pandas reader name '{reader}'!\n"
                f"Available readers:  {_avail}"
            ) from err

        df = read_func(filepath, **load_kwargs)

        return TargetCls(data=df, attrs=dict(filepath=filepath))
