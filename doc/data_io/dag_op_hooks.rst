.. default-domain:: dantro.data_ops.hooks

.. _dag_op_hooks:

DAG Syntax Operation Hooks
==========================
DAG syntax operation hooks (short: *operation hooks*) help to make the specification of data transformations more concise and powerful.

A hook consists of a callable that is attached to a certain operation name, e.g. ``expression``, and is invoked after the DAG syntax parser extracted all arguments.
The hook can manipulate the given ``operation``, ``args`` and ``kwargs`` arguments prior to the creation of the :py:class:`~dantro.dag.Transformation` object.

.. contents::
   :local:
   :depth: 2

For the integration into the transformation framework, see :ref:`here <dag_op_hooks_integration>`.

----

Available Hooks
---------------
The following hooks are available by default:

.. ipython::

    @suppress
    In [1]: import dantro

    @doctest
    In [2]: ", ".join(dantro.data_ops.DAG_PARSER_OPERATION_HOOKS)
    Out[2]: 'expression'

.. NOTE: Doctest is done to ensure that operation hooks are documented below!

The section titles below use the :ref:`operation name <data_ops_available>` of the hooks they are triggered by.

.. _dag_op_hook_expression:

``expression``
^^^^^^^^^^^^^^
The :py:func:`.op_hook_expression` prepares arguments for the :py:func:`~dantro.data_ops.expr_ops.expression` operation, making it more convenient to perform symbolic math operations with entities defined in the DAG.

It tries to extract the free symbols from the expression string and turns them into :py:class:`~dantro._dag_utils.DAGTag` objects of the same name.
For example, with the tags ``a``, ``b``, and ``c``:

.. literalinclude:: ../../tests/cfg/transformations.yml
    :language: yaml
    :start-after: ### Start -- dag_op_hooks_expression_basics
    :end-before:  ### End ---- dag_op_hooks_expression_basics
    :dedent: 4

The parser and the hook transform the ``expression`` operation node into:

.. code-block:: yaml

    operation: expression
    args: ["a**b / (c - 1.)"]
    kwargs:
      symbols:
        a: !dag_tag a
        b: !dag_tag b
        c: !dag_tag c

This alleviates specifying the ``kwargs.symbols`` argument manually, thus saving a lot of typing.

.. note::

    The ``define`` operation in the above example is just a trivial example of an operation; instead of defining extra DAG nodes, it would be much easier to simply add the parameters to the expression directly.

    Typically, nodes ``a``, ``b``, ``c`` would be the result of some prior, more complicated expression, e.g using any of the :ref:`other available operations <data_ops_available>`.

.. warning::

    If using the ``expression`` operation as part of a :ref:`meta-operation <dag_meta_ops>`, make sure to refer to these tags *inside* the meta-operation in some way.
    See the :ref:`dag_meta_ops_remarks` there for more information.

Furthermore, if any of the symbols are called ``prev`` or ``previous_result``, they are turned into :py:class:`~dantro._dag_utils.DAGNode` references to the previous node, similar to the ``!dag_prev`` YAML tag:

.. literalinclude:: ../../tests/cfg/transformations.yml
    :language: yaml
    :start-after: ### Start -- dag_op_hooks_expression_prev
    :end-before:  ### End ---- dag_op_hooks_expression_prev
    :dedent: 4

.. hint::

    The hook also makes the ``expression`` operation more robust in cases where ``with_previous_result`` is set.
    As the previous result is inserted as first positional argument, this would normally produce invalid syntax for the :py:func:`~dantro.data_ops.expr_ops.expression` operation.

By default, the :py:func:`~dantro.data_ops.expr_ops.expression` operation will attempt to cast the result to a floating point value, which is more compatible with other operations.
However, this default prohibits to work with the symbolic math features of `sympy <https://sympy.org>`_.
If you would like to keep symbolic expressions, specify the ``astype`` argument accordingly.

.. literalinclude:: ../../tests/cfg/transformations.yml
    :language: yaml
    :start-after: ### Start -- dag_op_hooks_expression_symbolic
    :end-before:  ### End ---- dag_op_hooks_expression_symbolic
    :dedent: 4
