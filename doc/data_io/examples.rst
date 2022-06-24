.. _dag_examples:

Data Transformation Examples
============================

This page provides examples of :ref:`data processing <data_processing>` via the :ref:`data transformation framework <dag_framework>`.

All examples are tested.

.. contents::
   :local:
   :depth: 2

----

Curve Fitting
-------------
For fitting a curve to some data, the following operations can be combined:

* The ``lambda`` operation to define the model function, see :py:func:`~dantro.data_ops.expr_ops.generate_lambda`
* The ``curve_fit`` operation, which is an alias for ``scipy.optimize.curve_fit``; see `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_examples_curve_fitting
    :end-before:  ### End ---- dag_examples_curve_fitting
    :dedent: 4

.. note::

    The polynomial above of course doesn't *quite* achieve fitting an elephant to the data, as John von Neumann wanted to show.
    However, see `this blog post <https://www.johndcook.com/blog/2011/06/21/how-to-fit-an-elephant/>`_ on how this can actually be achieved.
