"""Defines a loader mixin to load xarray objects"""

import xarray as xr

from ..containers import XrDataContainer, PassthroughContainer
from ._tools import add_loader

# -----------------------------------------------------------------------------

class XarrayLoaderMixin:
    """Supplies functionality to load xarray objects"""

    @add_loader(TargetCls=XrDataContainer)
    def _load_xr_dataarray(filepath: str, *, TargetCls: type,
                           load_completely: bool=False, **load_kwargs
                           ) -> XrDataContainer:
        """Loads an xr.DataArray from a netcdf file into an XrDataContainer.
        
        Args:
            filepath (str): Where the xarray-dumped netcdf file is located
            TargetCls (type): The class constructor
            load_completely (bool, optional): If true, will call .load() on the
                loaded DataArray to load it completely into memory
            **load_kwargs: Passed on to xr.load_dataarray, see there for kwargs
        
        Returns:
            XrDataContainer: The reconstructed XrDataContainer
        """
        da = xr.load_dataarray(filepath, **load_kwargs)

        if load_completely:
            da = da.load()

        # Create the XrDataContainer, carrying over attributes
        return TargetCls(data=da, attrs=da.attrs)

    @add_loader(TargetCls=PassthroughContainer)
    def _load_xr_dataset(filepath: str, *, TargetCls: type,
                         load_completely: bool=False, **load_kwargs
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
            **load_kwargs: Passed on to xr.load_dataarray, see there for kwargs
        
        Returns:
            PassthroughContainer: The reconstructed XrDataset, stored in a
                passthrough container.
        """
        ds = xr.load_dataset(filepath, **load_kwargs)

        if load_completely:
            ds = ds.load()

        # Create the PassthroughContainer, carrying over dataset attributes
        return TargetCls(data=ds, attrs=ds.attrs)
