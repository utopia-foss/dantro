.. _data_structures_psp_group:

The :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup`
=====================================================
The :py:mod:`~dantro.groups.pspgrp.ParamSpaceGroup` is a group where each member is assumed to be a point in a multi-dimensional parameter space.

For the representation of the parameter space, the ``paramspace`` package (see `here <https://pypi.org/project/paramspace/>`_) is used.
Subsequently, a :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup` is associated with a ``paramspace.ParamSpace`` object, which maps the members of the group to states in the parameter space.

Each member of the group (i.e.: each state of the parameter space) is represented by a :py:class:`~dantro.groups.pspgrp.ParamSpaceStateGroup`, which ensures that the name of the group is a valid state name.

Usage Example
-------------
This usage example shows how a :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup` is populated and used.

First, let's define a parameter space, in this case a two-dimensional one that goes over the parameters ``beta`` and ``seed``.
(For more information on usage of the paramspace package, consult `its documentation <https://paramspace.readthedocs.io/>`_).

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_psp_define_pspace
    :end-before:  ### End ---- groups_psp_define_pspace
    :dedent: 4

Now, let's set up a :py:mod:`~dantro.groups.pspgrp.ParamSpaceGroup` and populate it (with some random data in this case):

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_psp_fill_pspgrp
    :end-before:  ### End ---- groups_psp_fill_pspgrp
    :dedent: 4

The ``pspgrp`` is now populated and ready to use.

.. hint::

    For instructions on how to load data *from files* into a :py:class:`~dantro.groups.pspgrp.ParamSpaceGroup`, see the examples in the :ref:`integration guide <integrate_dantro>`.

On top of the capabilities of a regular group like iteration, the individual members can query their coordinates within the parameter space.
Furthermore, it also supplies the :py:meth:`~dantro.groups.pspgrp.ParamSpaceGroup.select` method, with which data from the parameter states can be combined into a higher-dimensional object.
The resulting object then has the parameter space dimensions *plus* the data dimensions:

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_psp_usage
    :end-before:  ### End ---- groups_psp_usage
    :dedent: 4

Importantly, having data available in this structure allows to conveniently create plots for each point in parameter space using the :ref:`plot creators specialized for this purpose <pcr_psp>`.
