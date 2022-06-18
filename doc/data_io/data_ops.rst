.. _data_processing:

Data Processing
===============
Through the :py:mod:`~dantro.data_ops` module, dantro supplies some useful functionality to generically work with function calls.
This is especially useful for numerical operations.

The :py:mod:`~dantro.data_ops` module can be used on its own, but it is certainly worth to have a look at :doc:`transform`, which wraps the application and combination of modules to further generalize the processing of dantro data.
For practical examples, of combining data processing operations with the data transformation framework, have a look at :doc:`examples`.

.. contents::
    :local:
    :depth: 2

----

Overview
--------
The operations database
^^^^^^^^^^^^^^^^^^^^^^^
The core of :py:mod:`~dantro.data_ops` is the operations database.
It is defined simply as a mapping from an operation name to a callable.
This makes it very easy to access a certain callable.

A quite expansive set of functions and numerical operations is already defined per default, see :ref:`below <data_ops_available>`.

.. hint::

    If you want to set up your own operations database, the corresponding functions all allow to specify the database to use for registration:
    Simply pass the ``_ops`` argument to the corresponding function.


Applying operations
^^^^^^^^^^^^^^^^^^^
The task of resolving the callable from the database, passing arguments to it, and returning the result falls to the :py:func:`~dantro.data_ops.apply.apply_operation` function.
It also provides useful feedback in cases where the operation failed, e.g. by including the given arguments into the error message.


.. _register_data_ops:

Registering operations
^^^^^^^^^^^^^^^^^^^^^^
To register additional operations, use the :py:func:`~dantro.data_ops.db_tools.register_operation` function:

.. testcode:: register_operation

    from dantro.data_ops import register_operation

    # Define an operation
    def increment_data(data, *, increment = 1):
        """Applies some custom operations on the given data"""
        return data + increment

    # Register it under its own name: "increment_data"
    register_operation(increment_data)

    # Can also give it a different name
    register_operation(increment_data, name="my_ops.increment")

.. testcode:: register_operation
    :hide:

    from dantro.data_ops.db import _OPERATIONS
    assert "increment_data" in _OPERATIONS
    assert "my_ops.increment" in _OPERATIONS


For new operations, a name should be chosen that is not already in use.
If you are registering multiple custom operations, consider using a common prefix for them.

.. note::

    It is not necessary to register operations that are *importable*!
    For example, you can instead use a combination of the ``import`` and ``call`` operations to achieve this behavior.
    With the ``from_module`` operation, you can easily retrieve a function from a module; see :py:func:`~dantro._import_tools.get_from_module`.
    There are shortcuts for imports from commonly-used modules, e.g. ``np.``, ``xr.`` and ``scipy.``.

    Operations should only be registered if you have implemented a custom operation or if the above does not work comfortably.

The :py:func:`~dantro.data_ops.db_tools.is_operation` decorator
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
As an alternative to :py:func:`~dantro.data_ops.db_tools.register_operation`, the :py:func:`~dantro.data_ops.db_tools.is_operation` decorator can be used to register a function with the operations database right where its defined:

.. testcode:: register_with_decorator

    from dantro.data_ops import is_operation

    # Operation name deduced from function name
    @is_operation
    def some_operation(data, *args):
        # ... do stuff here ...
        return data

    # Custom operation name
    @is_operation("do_stuff")
    def some_operation_with_a_custom_name(foo, bar):
        pass

    # Overwriting an operation of the same name
    @is_operation("do_stuff", overwrite_existing=True)
    def actually_do_stuff(spam, fish):
        pass

.. testcode:: register_with_decorator
    :hide:

    # Check that they are actually there
    from dantro.data_ops.db import _OPERATIONS
    assert "some_operation" in _OPERATIONS
    assert "do_stuff" in _OPERATIONS


.. _customize_db_tools:

Customizing database tools
^^^^^^^^^^^^^^^^^^^^^^^^^^

There is the option to customize the tools that work with or on the operations database.
For instance, if it is desired to use a custom operations database, the toolchain can be adapted as follows:

.. toggle::

    .. code-block:: python

        from typing import Union, Callable

        # Privately import the functions that are to be adapted
        from dantro.data_ops import (
            register_operation as _register_operation,
            is_operation as _is_operation,
            available_operations as _available_operations,
            apply_operation as _apply_operation,
        )

        # Your operations database object that is used as the default database.
        MY_OPERATIONS = dict()

        # Define a registration function with `skip_existing = True` as default
        # and evaluation of the default database
        def my_reg_func(*args, skip_existing=True, _ops=None, **kwargs):
            _ops = _ops if _ops is not None else MY_OPERATIONS
            return _register_operation(*args, skip_existing=skip_existing,
                                       _ops=_ops, **kwargs)

        # Define a custom decorator that uses the custom registration function
        def my_decorator(arg: Union[str, Callable] = None, /, **kws):
            return _is_operation(arg, _reg_func=my_reg_func, **kws)

        # Adapt the remaining tool chain
        def available_operations(*args, _ops=None, **kwargs):
            _ops = _ops if _ops is not None else MY_OPERATIONS
            return _available_operations(*args, _ops=_ops, **kwargs)

        def apply_operation(*args, _ops=None, **kwargs):
            _ops = _ops if _ops is not None else MY_OPERATIONS
            return _apply_operation(*args, _ops=_ops, **kwargs)

        # Usage of the decorator or the other functions is the same:
        @my_decorator
        def some_operation(d):
            # do stuff here
            return d

        @my_decorator("my_operation_name")
        def some_other_operation(d):
            # do stuff here
            return d

        print(available_operations())

    .. warning::

        The :py:class:`~dantro.dag.TransformationDAG` does *not* automatically use the custom operations database and functions!
        This remains to be implemented.



.. _data_ops_available:

Available operations
--------------------
Below, you will find a full list of operations that are available by default.

For some entries, functions defined in the :py:mod:`~dantro.data_ops` module are used as callables; see there for more information.
Also, the callables are frequently defined as lambdas to concur with the requirement that all operations need to be callable via positional and keyword arguments.
For example, an attribute call needs to be wrapped to a regular function call where — by convention — the first positional argument is regarded as the object whose attribute is to be called.

To dynamically find out which operations are available, use the :py:func:`~dantro.data_ops.db_tools.available_operations` (importable from :py:mod:`dantro.data_ops`) function, which also includes the names of additionally registered operations.

.. literalinclude:: ../../dantro/data_ops/db.py
   :start-after: _OPERATIONS = KeyOrderedDict({
   :end-before: }) # End of default operation definitions
   :dedent: 4

Additionally, the following boolean operations are available.

.. literalinclude:: ../../dantro/data_ops/_base_ops.py
   :start-after: BOOLEAN_OPERATORS = {
   :end-before: } # End of boolean operator definitions
   :dedent: 4

.. hint::

    If you can't find a desired operation, e.g. from ``numpy`` or ``xarray``, use the ``np.`` and ``xr.`` operations to easily import a callable from those modules.
    This is also possible for ``pd.``, ``scipy.`` and ``nx.``.

    With ``from_module`` (:py:func:`~dantro._import_tools.get_from_module`), you can achieve the same for every other module.

    See :py:mod:`dantro.data_ops` for more information on function signatures.

.. warning::

    While the operations database should be regarded as an append-only database and changing it is highly discouraged, it *can* be changed, e.g. via the ``overwrite_existing`` argument to :py:func:`~dantro.data_ops.db_tools.register_operation`, importable from :py:mod:`dantro.data_ops`.
    Therefore, the list above *might* not reflect the current status of the database.
