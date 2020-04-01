
Plot Functions
==============


A Declarative Generic Plot Function
-----------------------------------

Handling, transforming, and plotting high-dimensional data is difficult and often requires specialization to use-cases. 
`dantro` provides the generic :py:func:`~dantro.plot_creators.ext_funcs.generic.facet_grid` plot function that - together with the other dantro features - allows for a declarative way of creating plots from high-dimensional data. 

The idea is that high-dimensional raw data first is transformed using the :ref:`dag_framework`. 
The :py:func:`~dantro.plot_creators.ext_funcs.generic.facet_grid` function then gets the ready-to-plot data as input and visualizes it by automatically choosing an appropriate kind of plot – if possible and not explicitely given – in a declarative way through specification of layout keywords such as for example `col`ums, `row`s, or `hue`. 
The :py:class:`~dantro.plot_creators._plot_helper.PlotHelper` interface then copes with the plot style and further layout.
All steps are fully `YAML` configurable, thus, generating the multidimensional plot does not require any actual coding but just giving the correct keywords in the right place. 🎉

The :py:func:`~dantro.plot_creators.ext_funcs.generic.facet_grid` wraps the `xarray plotting <http://xarray.pydata.org/en/stable/plotting.html>`_ functionality generating `xarray.FacetGrid <http://xarray.pydata.org/en/stable/generated/xarray.plot.FacetGrid.html`>_ objects, and incorporates it into the dantro data handling, transforming, and plotting framework.
