.. _data_processing:

Data Processing
===============
Through the :py:mod:`~dantro.data_ops` module, dantro supplies some useful functionality to generically work with function calls.
This is especially useful for numerical operations.

The :py:mod:`~dantro.data_ops` module can be used on its own, but it is certainly worth to have a look at its use as part of the :ref:`dag_framework` or for :ref:`plot data selection <plot_creator_dag>`.
For practical examples, of combining data processing operations with the data transformation framework, have a look at :doc:`examples` and :ref:`plot_examples`.

.. contents::
    :local:
    :depth: 2

----

.. _data_ops_db:

The operations database
-----------------------
The core of :py:mod:`~dantro.data_ops` is the operations database.
It is defined simply as a mapping from an operation name to a callable.
This makes it very easy to access a certain callable.

A quite expansive set of functions and numerical operations is already defined per default, see :ref:`the data operations reference page <data_ops_ref>`.

.. hint::

    If you want to set up your own operations database, the corresponding functions all allow to specify the database to use for registration:
    Simply pass the ``_ops`` argument to the corresponding function.


.. _data_ops_available:

Available operations
--------------------
To dynamically find out which operations are available, use the :py:func:`~dantro.data_ops.db_tools.available_operations` (importable from :py:mod:`dantro.data_ops`) function, which also includes the names of additionally registered operations:

.. testcode:: available_operations

    from dantro.data_ops import available_operations

    # Show all available operation names
    all_ops = available_operations()

    # Search for the ten most similar ones to a certain name
    mean_ops = available_operations(match="mean", n=10)

.. testcode:: available_operations
    :hide:

    assert "import" in all_ops
    assert ".mean" in mean_ops


An up-to-date version of dantro's **default operations database** can be found :ref:`on this page <data_ops_ref>`.



.. _apply_data_ops:

Applying operations
-------------------
The task of resolving the callable from the database, passing arguments to it, and returning the result falls to the :py:func:`~dantro.data_ops.apply.apply_operation` function.
It also provides useful feedback in cases where the operation failed, e.g. by including the given arguments into the error message.

However, chances are that you will be using the data operations from within other parts of dantro, e.g. the :ref:`data transformation framework <dag_framework>` or for :ref:`plot data selection <plot_creator_dag>`.




.. _register_data_ops:

Registering operations
----------------------
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
--------------------------

There is the option to customize the tools that work with or on the operations database.
For instance, if it is desired to use a custom operations database, the toolchain can be adapted as follows:

.. toggle::

    .. testcode:: custom_db_tools

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

        print(", ".join(available_operations()))

    .. testoutput:: custom_db_tools

        some_operation, my_operation_name

    .. warning::

        The :py:class:`~dantro.dag.TransformationDAG` does *not* automatically use the custom operations database and functions!
        Being able to specify this is a task that remains to be implemented; contributions welcome.






.. _data_ops_troubleshooting:

Troubleshooting
---------------

.. _data_ops_troubleshooting_missing_op:

Missing an operation?
^^^^^^^^^^^^^^^^^^^^^
If you are missing a certain operation, there are multiple ways to go about this, either by importing it or by defining one ad-hoc.

- If it is a function call, e.g. from :py:mod:`numpy`, use the ``np.`` operation to easily import a callable (using :py:func:`~dantro._import_tools.get_from_module` under the hood).
  The same can be done for other frequently-used packages via the ``xr.``, ``pd.``, ``scipy.`` and ``nx.`` operations.
- Use the ``from_module`` (:py:func:`~dantro._import_tools.get_from_module`) or ``import`` (:py:func:`~dantro._import_tools.import_module_or_object`) operations for arbitrary imports.
- Use the ``lambda`` (:py:func:`~dantro.data_ops.expr_ops.generate_lambda`) operation to ad-hoc define a lambda.
- :ref:`Register <register_data_ops>` your own data operation.
- If you are using data operations as part of the :ref:`data transformation framework <dag_framework>`, e.g. during :ref:`plotting <plot_creator_dag>`, consider adding a :ref:`meta-operation <dag_meta_ops>`; that one will not be part of the operations database but will behave in an equivalent way.
- Make a `contribution to dantro <https://gitlab.com/utopia-project/dantro>`_ to add an operation by default.


.. _data_ops_troubleshooting_why_fail:

Why does my operation fail?
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In case you get :py:exc:`~dantro.exceptions.DataOperationFailed` or similar errors, there are a few things you can do:

- Carefully read the error message

  - Is the number and name of the given arguments correct?

- Inspect the given traceback

  - Is there something more insightful further up in the chain of errors?
  - It is worth scrolling through it a bit more, as this may be deeply nested.
  - If you do not get a traceback (e.g. when using the :py:class:`~dantro.plot_mngr.PlotManager`), make sure you are in debug mode.

- Have a look at the operation definition and docstrings

  - Many functions are merely ad-hoc defined lambdas; see :ref:`the data operations database <data_ops_ref>` for more info on how an operation is defined.
  - The implementation for dantro-based operations can be found in :py:mod:`dantro.data_ops`.

- Still stuck with an error? Might this be a bug? Consider opening an issue in the `dantro GitLab project <https://gitlab.com/utopia-project/dantro>`_.

.. hint::

    If using the data operations as part of the :ref:`data transformation framework <dag_framework>`, note that you can also :ref:`visualize the context <dag_graph_vis>` in which the operation failed.

    As part of the plotting framework, these visualization may be :ref:`automatically created <plot_creator_dag_vis>` alongside your (potentially failing) plot.
