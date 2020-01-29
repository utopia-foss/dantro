Data Processing
===============
Through the :py:mod:`~dantro.utils.data_ops` module, dantro supplies some useful functionality to generically work with function calls.
This is especially useful for numerical operations.

The :py:mod:`~dantro.utils.data_ops` module can be used on its own, but it is certainly worth to have a look at :doc:`transform`, which wraps the application and combination of modules to further generalize the processing of dantro data.

.. contents::
    :local:
    :depth: 2

----

Overview
--------
The operations database
^^^^^^^^^^^^^^^^^^^^^^^
The core of :py:mod:`~dantro.utils.data_ops` is the operations database.
It is defined simply as a mapping from an operation name to a callable.
This makes it very easy to access a certain callable.

A basic set of python functions and numerical operations is defined per default, see :ref:`below <data_ops_available>`.


Applying operations
^^^^^^^^^^^^^^^^^^^
The task of resolving the callable from the database, passing arguments to it, and returning the result falls to the :py:func:`~dantro.utils.data_ops.apply_operation` function.
It also provides useful feedback in cases where the operation failed, e.g. by including the given arguments into the error message.


Registering operations
^^^^^^^^^^^^^^^^^^^^^^
To register additional operations, use the :py:func:`~dantro.utils.data_ops.register_operation` function.

For new operations, a name should be chosen that is not already in use.
If you are registering multiple custom operations, consider using a common prefix for them.

.. note::

    It is not necessary to register operations that are *importable*!
    Just use a combination of the ``import`` and ``call`` operations to achieve this behaviour.

    Operations should only be registered if the above does not work comfortably.


.. _data_ops_available:

Available operations
--------------------
Below, you will find a full list of operations that are available by default.

For some entries, functions defined in the :py:mod:`~dantro.utils.data_ops` module are used as callables; see there for more information.
Also, the callables are frequently defined as lambdas in order to concur to the requirement that all operations need to be callable via positional and keyword arguments.
For example, an attribute call need be wrapped to a regular function call where — by convention — the first positional argument is regarded as the object whose attribute is to be called.

To dynamically find out which operations are available, use the :py:func:`~dantro.utils.available_operations` function, which also includes the names of additionally registered operations.

.. literalinclude:: ../../dantro/utils/data_ops.py
   :start-after: _OPERATIONS = KeyOrderedDict({
   :end-before: }) # End of default operation definitions
   :dedent: 4

Additionally, the following boolean operations are available.

.. literalinclude:: ../../dantro/utils/data_ops.py
   :start-after: BOOLEAN_OPERATORS = {
   :end-before: } # End of boolean operator definitions
   :dedent: 4

.. warning::

    While the operations database should be regarded as an append-only database and changing it is highly discouraged, it *can* be changed, e.g. via the ``overwrite_existing`` argument to :py:func:`~dantro.utils.register_operation`.
    Therefore, the list above *might* not reflect the current status of the database.
