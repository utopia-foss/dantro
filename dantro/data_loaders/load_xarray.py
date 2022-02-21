"""Defines a loader mixin to load xarray objects"""

from .._import_tools import LazyLoader
from ..containers import PassthroughContainer, XrDataContainer
from ._tools import add_loader

xr = LazyLoader("xarray")

# -----------------------------------------------------------------------------


class XarrayLoaderMixin:
    """Supplies functionality to load xarray objects"""

    @add_loader(TargetCls=XrDataContainer)
    def _load_xr_dataarray(
        filepath: str,
        *,
        TargetCls: type,
        load_completely: bool = False,
        engine: str = "scipy",
        **load_kwargs,
    ) -> XrDataContainer:
        """Loads an xr.DataArray from a netcdf file into an XrDataContainer.

        Args:
            filepath (str): Where the xarray-dumped netcdf file is located
            TargetCls (type): The class constructor
            load_completely (bool, optional): If true, will call .load() on the
                loaded DataArray to load it completely into memory
            engine (str, optional): Which engine to use for loading. Refer to
                the xarray documentation for available engines.
            **load_kwargs: Passed on to xr.load_dataarray, see there for kwargs

        Returns:
            XrDataContainer: The reconstructed XrDataContainer
        """
        da = xr.load_dataarray(filepath, engine=engine, **load_kwargs)

        if load_completely:
            da = da.load()

        # Create the XrDataContainer, carrying over attributes
        return TargetCls(data=da, attrs=da.attrs)

    @add_loader(TargetCls=PassthroughContainer)
    def _load_xr_dataset(
        filepath: str,
        *,
        TargetCls: type,
        load_completely: bool = False,
        engine: str = "scipy",
        **load_kwargs,
    ) -> PassthroughContainer:
        """Loads an xr.Dataset from a netcdf file into a PassthroughContainer.

        .. note::
            As there is no proper equivalent of a dataset in dantro (yet), and
            unpacking the dataset into a dantro group would reduce
            functionality, the PassthroughContainer is used here. It should
            behave almost the same as an xr.Dataset.

        Args:
            filepath (str): Where the xarray-dumped netcdf file is located
            TargetCls (type): The class constructor
            load_completely (bool, optional): If true, will call .load() on the
                loaded xr.Dataset to load it completely into memory.
            engine (str, optional): Which engine to use for loading. Refer to
                the xarray documentation for available engines.
            **load_kwargs: Passed on to xr.load_dataset, see there for kwargs

        Returns:
            PassthroughContainer: The reconstructed XrDataset, stored in a
                passthrough container.
        """
        ds = xr.load_dataset(filepath, engine=engine, **load_kwargs)

        if load_completely:
            ds = ds.load()

        # Create the PassthroughContainer, carrying over dataset attributes
        return TargetCls(data=ds, attrs=ds.attrs)
