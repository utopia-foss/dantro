
Plot Functions
==============


A Declarative Generic Plot Function
-----------------------------------

Handling, transforming, and plotting high-dimensional data is difficult and often requires specialization to use-cases. 
`dantro` provides the generic :py:func:`~dantro.plot_creators.ext_funcs.generic.facet_grid` plot function that - together with the other dantro features - allows for a declarative way of creating plots from high-dimensional data. 

The idea is that high-dimensional raw data first is transformed using the :ref:`dag_framework`. 
The :py:func:`~dantro.plot_creators.ext_funcs.generic.facet_grid` function then gets the ready-to-plot data as input and visualizes it by automatically choosing an appropriate kind of plot – if possible and not explicitely given – in a declarative way through specification of layout keywords such as for example ``col``\ ums, ``row``\ s, or ``hue``.
This approach is called `faceting <http://xarray.pydata.org/en/stable/plotting.html#faceting>`_; dantro makes use of the `excellent plotting functionality of xarray <http://xarray.pydata.org/en/stable/plotting.html>`_ for this feature.
The :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` interface then copes with the plot style and further layout.
All steps are fully configurable and optimized for the YAML-based plotting interface.
Thus, generating a plot of multidimensional data does not require touching any actual code but just specifying the desired representation in the plot configuration. 🎉

