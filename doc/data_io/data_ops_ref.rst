.. _data_ops_ref:

Data Operations Reference
=========================

Below, you will find the dantro :ref:`data operations database <data_ops_db>` that is used in :ref:`data processing <data_processing>`, e.g. to :ref:`select and transform data during plotting <plot_creator_dag>`.

These pages may also be of interest:

- :ref:`General information on data operations <data_processing>`
- :ref:`Troubleshooting <data_ops_troubleshooting>`

  - :ref:`data_ops_troubleshooting_missing_op`
  - :ref:`data_ops_troubleshooting_why_fail`

- :ref:`Usage examples (as part of plots) <plot_examples>`


.. admonition:: Background info on operation definition
    :class: dropdown

    You may notice that operations are defined in one of two ways:

    - As an ad-hoc defined ``lambda``:

      - Example: The ``call`` operation simply is: ``lambda c, *a, **k: c(*a, **k)``

    - As an alias for a callable defined elsewhere:

      - Example: The ``print`` operation simply links to dantro's own :py:func:`~dantro.data_ops.ctrl_ops.print_data` function.

    This is simply to concur to a uniform interface and, in the case of the lambdas, allow operations that are averse to the kind of object they act on.
    Picking up the ``call`` example, the first positional argument can be any callable, the data operation does not care which one.

    Something similar is the case for operation names starting with a dot (like ``.mean``):
    They follow the convention that the *first positional argument* is the object on which the attribute call is made.
    The effect of these operations thus depends on the type of the object that it acts on.

----

.. literalinclude:: ../../dantro/data_ops/db.py
   :start-after: _OPERATIONS = KeyOrderedDict({
   :end-before: }) # End of default operation definitions
   :dedent: 4

Additionally, the following boolean operations are available.

.. literalinclude:: ../../dantro/data_ops/_base_ops.py
   :start-after: BOOLEAN_OPERATORS = {
   :end-before: } # End of boolean operator definitions
   :dedent: 4


.. warning::

    While the operations database should be regarded as an append-only database and changing it is highly discouraged, it *can* be changed, e.g. via the ``overwrite_existing`` argument to :py:func:`~dantro.data_ops.db_tools.register_operation` (see :ref:`register_data_ops`).

    Therefore, the database as it is shown above *might* not reflect its state during *your* use of the data operations framework.
