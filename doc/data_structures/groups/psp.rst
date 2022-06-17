.. _data_structures_psp_group:

The :py:class:`~dantro.groups.psp.ParamSpaceGroup`
=====================================================
The :py:mod:`~dantro.groups.psp.ParamSpaceGroup` is a group where each member is assumed to be a point in a multi-dimensional parameter space.

For the representation of the parameter space, the ``paramspace`` package (see `here <https://pypi.org/project/paramspace/>`_) is used.
Subsequently, a :py:class:`~dantro.groups.psp.ParamSpaceGroup` is associated with a ``paramspace.ParamSpace`` object, which maps the members of the group to states in the parameter space.

Each member of the group (i.e.: each state of the parameter space) is represented by a :py:class:`~dantro.groups.psp.ParamSpaceStateGroup`, which ensures that the name of the group is a valid state name.

.. contents::
    :local:
    :depth: 2

----

Usage Example
-------------
This usage example shows how a :py:class:`~dantro.groups.psp.ParamSpaceGroup` is populated and used.

First, let's define a parameter space, in this case a two-dimensional one that goes over the parameters ``beta`` and ``seed``.
(For more information on usage of the paramspace package, consult `its documentation <https://paramspace.readthedocs.io/>`_).

.. ipython:: python

    # Define a 2D parameter space (typically done from a YAML file)
    from paramspace import ParamSpace, ParamDim

    all_params = {
        "some_parameter": "foo",
        "more_parameters": {
            "spam": "fish",
            "beta": ParamDim(default=1., values=[.01, .03, .1, .3, 1.]),
        },
        "seed": ParamDim(default=42, range=[20])
    }

    pspace = ParamSpace(all_params)

    # What does this look like?
    print(pspace.get_info_str())

Now, let's set up a :py:mod:`~dantro.groups.psp.ParamSpaceGroup` and populate it (with some random data in this case):

.. ipython:: python

    import numpy as np
    import xarray as xr

    from dantro.groups import ParamSpaceGroup
    from dantro.containers import XrDataContainer

    pspgrp = ParamSpaceGroup(name="my_parameter_sweep", pspace=pspace)

    # Iterate over the parameter space, create a ParamSpaceState group (using
    # the state number as name), and populate it with some random data
    for params, state_no_str in pspace.iterator(with_info='state_no_str'):
        pss = pspgrp.new_group(state_no_str)
        some_data = xr.DataArray(data=np.random.random((2,3,4)),
                                 dims=('foo', 'bar', 'baz'),
                                 coords=dict(foo=[0, 1],
                                             bar=[0, 10, 20],
                                             baz=[.1, .2, .4, .8]))
        pss.add(XrDataContainer(name="some_data", data=some_data))

The ``pspgrp`` is now populated and ready to use.

.. hint::

    For instructions on how to load data *from files* into a :py:class:`~dantro.groups.psp.ParamSpaceGroup`, see the examples in the :ref:`integration guide <integrate_dantro>`.

Let's explore its properties a bit, also comparing it to the shape of the parameter space it is associated with:

.. ipython::

    In [1]: print(pspgrp.tree_condensed)

    @doctest
    In [2]: pspgrp.pspace.num_dims
    Out[2]: 2

    # The volume is the product of the dimension sizes, here: 5 * 20 = 100
    @doctest
    In [3]: pspgrp.pspace.volume
    Out[3]: 100

    @doctest
    In [4]: len(pspgrp) == pspgrp.pspace.volume
    Out[4]: True

On top of the capabilities of a regular group-like iteration, the individual members (i.e., :py:class:`~dantro.groups.psp.ParamSpaceStateGroup` objects) can query their coordinates within the parameter space via their :py:attr:`~dantro.groups.psp.ParamSpaceStateGroup.coords` property.

.. ipython:: python

    from dantro.groups import ParamSpaceStateGroup

    for pss in pspgrp.values():
        assert isinstance(pss, ParamSpaceStateGroup)
        assert 'beta' in pss.coords
        assert 'seed' in pss.coords

Furthermore, it also supplies the :py:meth:`~dantro.groups.psp.ParamSpaceGroup.select` method, with which data from the ensemble of parameter states can be combined into a higher-dimensional object.
The resulting object then has the parameter space dimensions *plus* the data dimensions:

.. ipython::

    In [1]: all_data = pspgrp.select(field="some_data")

    In [2]: print(all_data)

    # ... should now have 5 dimensions: 3 data dimensions + 2 pspace dimensions
    @doctest
    In [3]: all_data["some_data"].ndim
    Out[3]: 5

    @doctest
    In [4]: set(all_data["some_data"].coords.keys())
    Out[4]: {'bar', 'baz', 'beta', 'foo', 'seed'}

Importantly, having data available in this structure allows to conveniently create plots for each point in parameter space using the :ref:`plot creators specialized for this purpose <pcr_psp>`.


.. _universes_and_multiverses:

Universes and Multiverses
-------------------------

At this point, we would like to introduce some dantro-specific nomenclature and the motivation behind it.

dantro is meant to be used as a data processing pipeline, e.g. for simulation data (see :ref:`the Integration Example <integrate_data_gen>`).
In such a scenario, one often feeds a set of model parameters to a computer simulation, which then generates some output data (the input to the processing pipeline).
Usually, individual simulations are independent of each other and their behaviour is fully defined by the parameters it is instantiated with.

This led to the following metaphors:

    * A **Universe** refers to a self-sufficient computer simulation which requires only a set of input parameters.
    * A **Multiverse** is a set of many such universes, which are completely independent of each other.

To push it a bit more: The universes may all be goverened by the same physical laws (i.e., *have the same underlying computer model*) but the values of physical constants are different (i.e., *have different simulation parameters*).

For dantro, these terms typically refer to the *output* of such computer simulations:

    * **Universe data** is the output of a single simulation, loaded into a :py:class:`~dantro.groups.psp.ParamSpaceStateGroup`
    * **Multiverse data** is the output from *multiple* individual universes.
      As these are typically generated for points of the same parameters space, they can also be gathered into a :py:class:`~dantro.groups.psp.ParamSpaceGroup`.

Subsequently, when handling data that is structured this way, parts of dantro (most notably the :py:class:`~dantro.plot.creators.psp.MultiversePlotCreator` and :py:class:`~dantro.plot.creators.psp.UniversePlotCreator`) also use these metaphors instead of the parameter space terminology.


.. note::

    At the end of the day, these are still metaphors.
    However, in the context of simulation-based research, we hope that they simplify the vocabulary with which researchers talk about computer models and their output.

    These thoughts also inspired parts of the frontend of the `Utopia project <https://gitlab.com/utopia-project/utopia>`_, where a ``Multiverse`` object coordinates the simulation of individual *universes* using the dantro and paramspace objects showcased above.
