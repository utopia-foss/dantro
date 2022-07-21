.. _dag_framework:

Data Transformation Framework
=============================

The uniform structure of the dantro data tree is the ideal starting point to allow more general application of transformation on data.
This page describes dantro's data transformation framework, revolving around the :py:class:`~dantro.dag.TransformationDAG` class.
It is sometimes also referred to as *DAG framework* or *data selection and transformation framework* and finds application :doc:`in the plotting framework <../plotting/plot_data_selection>`.

This page is an introduction to the DAG framework and a description of its inner workings.
To learn more about its practical usage, make sure to look at the :doc:`examples`.

.. contents::
   :local:
   :depth: 2

Related pages:

.. toctree::

    dag_op_hooks
    examples

----

Overview
--------

The purpose of the transformation framework is to be able to *generally* apply mathematical operations on data that is stored in a dantro data tree.
Specifically, it makes it possible to define transformations without touching actual Python code.
To that end, a meta language is defined that makes it possible to define almost arbitrary transformations.

In dantro terminology, a transformation is defined as a set consisting of an operation and some arguments.
Say, for example, we want to perform a simple addition of two quantities, 1 and 2, we are used to writing ``1 + 2``.
To define a transformation using the meta language, this would translate to a set consisting of the ``add`` operation and two (ordered) arguments: ``1`` and ``2``.

Now, typically transformations don't come on their own and are not nearly as trivial as the above.
You might desire to compute ``a + b``, where both ``a`` and ``b`` are results of previous transformations.

This can be represented as a *directed acyclic graph*, or short: DAG.
For the example above, the graph is rather small:

::

        a:(…)   b:(…)
          ^      ^
           \    /
            \  /
      (add, a, b)

The nodes in this graph represent transformations.
These nodes can have labels, e.g. ``a`` and ``b``, which are called *references* or *tags* in dantro terminology.
As illustrated by the example above, the tags can be used in place of arguments to denote that the result of a previous transformation (with the corresponding label) should be used.

The directed edges in the graph represent dependencies.
The *acyclic* in DAG is required such that the computation of a transformation result does not end in an infinite loop due to a circular dependency.

The :py:class:`~dantro.dag.Transformation` and :py:class:`~dantro.dag.TransformationDAG` dantro classes implement exactly this structure, making the following features available:

* Easy and generic access to data stored in an associated :py:class:`~dantro.data_mngr.DataManager`
* Definition of arbitrary DAGs via dictionary-based configurations
* Syntax optimized to make specification via YAML easy
* Shorthand notations available
* New and custom operations can be registered
* There are no restrictions on the signature of operations
* Caching of transformations is possible, avoiding re-calculation of computationally expensive transformations
* Transformations are uniquely representable by a hash


The Transformation Syntax
-------------------------
This section will guide you through the syntax used to define transformations.
It will explain the basic elements and inner workings of the mini-language created for the purpose of the DAG.

.. note::

    This explanation goes into quite some detail; and it's quite important to understand the underlying structures of the
    If you feel like you would like to jump ahead to see what awaits you, have a look at the :ref:`dag_minimal_syntax`.


The :py:class:`~dantro.dag.TransformationDAG`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The structure a user (you!) is mainly interacting with is the :py:class:`~dantro.dag.TransformationDAG` class.
It takes care to build the DAG by creating :py:class:`~dantro.dag.Transformation` objects according to the specification you provided.
In the following, all YAML examples will represent the arguments that are passed to the :py:class:`~dantro.dag.TransformationDAG` during initialization.

Basics
^^^^^^
Ok, let's start with the basics: How can transformations be defined?
For the sake of simplicity, let's only look at transformations that are fully independent of other transformations.

Explicit syntax
"""""""""""""""
The explicit syntax to define a single :py:class:`~dantro.dag.Transformation` via the :py:class:`~dantro.dag.TransformationDAG` looks like this:

.. code-block:: yaml

    transform:
      - operation: add
        args: [1, 2]
        kwargs: {}

The ``transform`` argument is the main argument to specify transformations.
It accepts a sequence of mappings.
Each entry of the sequence contains all arguments that are needed to create a *single* :py:class:`~dantro.dag.Transformation`.

As you see, the syntax is very close to the above definition of what a dantro transformation contains.

.. note::

    The ``args`` and ``kwargs`` arguments can also be left out, if no positional or keyword arguments are to be passed, respectively.
    This is equivalent to setting them to ``~`` or empty lists / dicts.

Specifying multiple transformations
"""""""""""""""""""""""""""""""""""
To specify multiple transformations, simply add more entries to the ``transform`` sequence:

.. code-block:: yaml

    transform:
      - operation: add
        args: [3, 4]
      - operation: sub
        args: [8, 2]
      - operation: mul
        args: [6, 7]


Assigning tags
""""""""""""""
Nodes of the DAG all have a unique identifier in the form of a hash string, which is a 32 character hexadecimal string.
While it can be used to identify a transformation, the easiest way to refer to it is by using a so-called *tag*.

Tags are simply plain text pointers to a specific hash, which in turn denotes a specific transformation.
To add a tag to a transformation, use the ``tag`` key.

.. code-block:: yaml

    transform:
      - operation: add
        args: [3, 4]
        tag: some_addition
      - operation: sub
        args: [8, 2]
        tag: some_substraction
      - operation: mul
        args: [6, 7]
        tag: the_answer

.. note::

    No two transformations can have the same tag.

.. _dag_referencing:

Advanced Referencing
^^^^^^^^^^^^^^^^^^^^
In the examples above, all transformations were independent of each other.
Having completely independent and disconnected nodes, of course, defeats the purpose of having a DAG structure.

Now let's look at proper, non-trivial DAGs, where individual transformations use the results of other transformations.


Referencing other Transformations
"""""""""""""""""""""""""""""""""
Other transformations can be referenced in three ways, each with a corresponding Python class and an associated YAML tag:

* :py:class:`~dantro._dag_utils.DAGReference` and ``!dag_ref``: This is the most basic and most explicit reference, using the transformations' **hash** to identify a reference.
* :py:class:`~dantro._dag_utils.DAGTag` and ``!dag_tag``: References by tag are the preferred references. They use the plain text name specified via the ``tag`` key.
* :py:class:`~dantro._dag_utils.DAGNode` and ``!dag_node``: Uses the ID of the node within the DAG. Mostly for internal usage!

.. note::

    When the DAG is built, all references are brought into the most explicit format: :py:class:`~dantro._dag_utils.DAGReference` s.
    Thus, internally, the transformation framework works *only* with hash references.

The **best way** to refer to other transformations is **by tag**: there is no ambiguity, it is easy to define, and it allows you to easily build a DAG tree structure.
A simple example with three nodes would be the following:

.. code-block:: yaml

    transform:
      - operation: add
        args: [3, 4]
        tag: some_addition
      - operation: sub
        args: [8, 2]
        tag: some_substraction
      - operation: mul
        args:
          - !dag_tag some_addition
          - !dag_tag some_substraction
        tag: the_answer

Which is equivalent to:

::

    some_addition = 3 + 4
    some_substraction = 8 - 2
    the_answer = some_addition * some_substraction

References can appear within the positional and the keyword arguments of a transformation.
As you see, they behave quite a bit like variables behave in programming languages; the only difference being: you can't reassign a tag and you should not form circular dependencies.

Using the result of the previous transformation
"""""""""""""""""""""""""""""""""""""""""""""""
When chaining multiple transformations to each other and not being interested in the intermediate results, it is tedious to always define tags:

.. code-block:: yaml

    transform:
      - operation: mul
        args: [1, 2]
        tag: f2
      - operation: mul
        args: [!dag_tag f2, 3]
        tag: f3
      - operation: mul
        args: [!dag_tag f3, 4]
        tag: f4
      - operation: mul
        args: [!dag_tag f4, 5]
        tag: f5

Let's say, we're only interested in ``f5``.
The only thing we want is that the result from the previous transformation is carried on to the next one.
The ``with_previous_result`` feature can help in this case: It adds as the first positional argument a reference to the *previous* node.
Thus, it is no longer necessary to define a tag.

.. code-block:: yaml

    transform:
      - operation: mul
        args: [1, 2]
      - operation: mul
        args: [3]
        with_previous_result: true
      - operation: mul
        args: [4]
        with_previous_result: true
      - operation: mul
        args: [5]
        with_previous_result: true
        tag: f5

Note that the ``args``, in that case, specify one fewer positional argument.

.. warning::

    Using ``!dag_node`` in your specifications is **not** recommended.
    Use it only if you really know what you're doing.

In case the result of the previous transformation should not be used in place of the first positional argument but somewhere else, there is the ``!dag_prev`` YAML tag, which creates a node reference to the previous node:

.. code-block:: yaml

    transform:
      - operation: define
        args: [10]
      - operation: sub
        args: [0, !dag_prev ]
      - operation: div
        args: [1, !dag_prev ]
      - operation: power
        args: [10, !dag_prev ]
        tag: my_result

.. note::

    Notice the space behind ``!dag_prev``.
    The YAML parser might complain about a character directly following the tag, like ``…, !dag_prev]``.


Computing Results
^^^^^^^^^^^^^^^^^
To compute the results of the DAG, invoke the :py:class:`~dantro.dag.TransformationDAG`\ 's :py:meth:`~dantro.dag.TransformationDAG.compute` method.

It can be called without any arguments, in which case the result of all *tagged* transformations will be computed and returned as a dict.
If only the result of a subset of tags should be computed, they can also be specified.

Computing results works as follows:

1. Each tagged :py:class:`~dantro.dag.Transformation` is visited and its own :py:meth:`~dantro.dag.Transformation.compute` method is invoked
2. A cache lookup occurs, attempting to read the result from a memory or file cache.
3. The transformations resolve potential references in their arguments: If a :py:class:`~dantro._dag_utils.DAGReference` is encountered, the corresponding :py:class:`~dantro.dag.Transformation` is resolved and that transformation's :py:meth:`~dantro.dag.TransformationDAG.compute` method is invoked. This traverses all the way up the DAG until reaching the root nodes which contain only basic data types (that need no computation).
4. Having resolved all references into results, the arguments are assembled, the operation callable is resolved, and invoked by passing the arguments.
5. The result is kept in a memory cache. It *can* additionally be stored in a file cache to persist to later invocations.
6. The result object is returned.


.. note::

    *Only* nodes that are tagged *can* be part of the results.
    Intermediate results still need to be computed, but it will not be part of the results dict.
    If you want an intermediate result to be available there, add a tag to it.

    This also means: If there are parts of the DAG that are not tagged *at all*, they will not be reached by any recursive argument lookup.

.. hint::

    Use the ``compute_only`` argument of :py:meth:`~dantro.dag.TransformationDAG.compute` to specify which tags are to be computed.
    If not given, all tags will be computed, *unless* they start with a ``.`` or ``_`` (these are so-called "private" tags).

    To compute private tags directly, include them in ``compute_only``.

.. hint::

    To learn which parts of the computation require the most time, e.g. in order to evaluate whether to :ref:`cache the result <dag_file_cache>`, inspecting the DAG profile statistics can be useful.
    The :py:class:`~dantro.dag.TransformationDAG`\ 's ``verbosity`` attribute controls how extensively statistics are written to the log output.
    By default (verbosity ``1``), only per-node statistics are emitted.
    For levels ``>= 2``, per-operation statistics are shown alongside.


.. _dag_operations:

Resolving and applying operations
"""""""""""""""""""""""""""""""""
Let's have a brief look into how the ``operation`` argument is actually resolved and how the operation is then applied.

This feature is not specific to the DAG, but the DAG uses the :py:mod:`~dantro.data_ops` module, which implements a database of available operations and the :py:func:`~dantro.data_ops.apply.apply_operation` function to apply an operation.
Basically, this is a thin wrapper around a function lookup and its invocation.

For a full list of available data operations, see :ref:`here <data_ops_ref>`.

.. hint::

    You can also use the ``import`` operation to retrieve a callable (or any other object) via a Python import and then use the ``call`` operation to invoke it.
    These two operations are combined in the ``import_and_call`` operation:

    .. code-block:: yaml

        transform:
          - operation: import_and_call
            args: [numpy.random, randint]
            kwargs:
              low: 0
              high: 10
              size: [2, 3, 4]

    To specifically register additional operations, use the :py:func:`~dantro.data_ops.db_tools.register_operation` function.
    This should only be done for operations that are not easily usable via the ``import`` and ``call`` operations.


.. _dag_select:

Selecting from the :py:class:`~dantro.data_mngr.DataManager`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The above examples are trivial in that they do not use any actual data but define some dummy values.
This section shows how data can be selected from the :py:class:`~dantro.data_mngr.DataManager` that is associated with the :py:class:`~dantro.dag.TransformationDAG`.

The process of selecting data is not different than other transformations.
It makes use of the ``getitem`` operation that would also be used for regular item access, and it uses the fact that the data manager is available via the ``dm`` tag.

.. note::

    The :py:class:`~dantro.data_mngr.DataManager` is also identified by a hash, which is computed from its name and its associated data directory path.
    Thus, managers for different data directories have different hashes.


The ``select`` interface
""""""""""""""""""""""""
As selecting data from the :py:class:`~dantro.data_mngr.DataManager` is a common use case, the :py:class:`~dantro.dag.TransformationDAG` supports the ``select`` argument besides the ``transform`` argument.

The ``select`` argument expects a mapping of tags to either strings (the path within the data tree) or further mappings (where more configurations are possible):

.. code-block:: yaml

    select:
      some_data: path/to/some_data
      more_data:
        path: path/to/more_data
        # ... potentially more kwargs
    transform: ~

The results dict will then have two tags, ``some_data`` and ``more_data``, each of which is the selected object from the data tree.

.. note::

    The above example is translated into the following basic transformation specifications:

    .. code-block:: yaml

        transform:
          - operation: getitem
            args: [!dag_tag dm, path/to/more_data]
            tag: more_data
          - operation: getitem
            args: [!dag_tag dm, path/to/some_data]
            tag: some_data

    Note that the order of operations is sorted alphabetically by the tag specified under the ``select`` key.


Directly transforming selected data
"""""""""""""""""""""""""""""""""""
Often, it is desired to apply some sequential transformations to selected data before working with it.
As part of the ``select`` interface, this is also possible:

.. code-block:: yaml

    select:
      square_increment:
        path: path/to/some_data
        with_previous_result: true
        transform:
          - operation: squared
          - operation: increment

      some_sum:
        path: path/to/more_data
        transform:
          - operation: getattr
            args: [!dag_prev , data]
          - operation: sub
            args: [0, !dag_prev ]
          - operation: .sum
            args: [!dag_prev ]
    transform:
      - operation: add
        args: [!dag_tag square_increment, !dag_tag some_sum ]
        tag: my_result

Notice the difference between ``square_increment``, where the result is carried over, and ``some_sum``, where the reference has to be specified explicitly.
As visible there, within the ``select`` interface, the ``with_previous_result`` option can also be specified such that it applies to a sequence of transformations that are based on some selection from the data manager.

.. note::

    The parser expands this syntax into a sequence of basic transformations.

    It does so *before* any other transformations from the ``transform`` argument are evaluated. Thus, whichever tags are defined there are not available from within ``select``!

Changing the selection base
"""""""""""""""""""""""""""
By default, selection happens from the associated :py:class:`~dantro.data_mngr.DataManager`, tagged ``dm``.
This option can be controlled via the ``select_base`` property, which can be set both as argument to ``__init__`` and afterward via the property.
The property expects either a ``DAGReference`` object or a valid tag string.

If set, all following ``select`` arguments are using that reference as the basis, leading to ``getitem`` operations on that object rather than on the data manager.

As the ``select`` arguments are evaluated before any transform operations, only the default tags are available during initialization.
To widen the possibilities, the :py:class:`~dantro.dag.TransformationDAG` allows the ``base_transform`` argument during initialization; this is just a sequence of transform specifications, which are applied *before* the ``select`` argument is evaluated, thus allowing to select some object, tag it, and use that tag for the ``select_base`` argument.

.. note::

    The ``select_path_prefix`` argument offers similar functionality, but merely prepends a path to the argument.
    If possible, the ``select_base`` functionality should be preferred over ``select_path_prefix`` as it reduces lookups and cooperates more nicely with the file caching features.

.. admonition:: Background Information

    Internally, when the ``select`` specification is evaluated, it is set to select against a special tag ``select_base``; by default, this is the same as the ``dm`` special tag.

    Effectively, the ``select`` feature always selects starting from the object the :py:attr:`~dantro.dag.TransformationDAG.select_base` property points to *at the time the nodes are added to the DAG*.
    In other words, if the ``select_base`` is changed *after* the nodes were added, this will not have any effect.

    For :ref:`meta-operations <dag_meta_ops>` this means that the base of selection is not relevant *at definition* of the meta-operations; the base gets evaluated when the meta-operation is *used*.



.. _dag_define:

The ``define`` interface
^^^^^^^^^^^^^^^^^^^^^^^^
So far, we have seen two ways to add transformation nodes to the DAG: via ``transform`` or via ``select``.
These are based either on directly adding the nodes, giving full control, or adding transformations based on a selection of data.

The ``define`` interface is a combintion of these two approaches: same as ``select``, it revolves around the final tag that's meant to be attached to the definition, but it does not require a data selection like ``select`` does.

Let's look at an example that combines all these ways of adding transformations:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_define
    :end-before:  ### End ---- dag_define
    :dedent: 4

Here, the ``exponent`` as well as some conversion factor tags are defined not ad-hoc but *separately* via the ``define`` interface.
As can be seen in the example, there are two ways to do this:

* If providing a ``list`` or ``tuple`` type, it is interpreted as a sequence of transformations, accepting the same syntax as ``transform``.
  After the final transformation, another node is added that sets the specified tag, ``days_to_seconds_factor`` in this example.
* If prodiving *any* other type, it is interpreted directly as a definition, adding a single transformation node that holds the given argument, the integer ``4`` in the case of the ``exponent`` tag.

.. note::

    The ``define`` argument is evaluated *before* the other two.
    Subsequently, tags defined via ``define`` *can* be used within ``select`` or ``transform``, but not the other way around.

.. hint::

    In the context of **plotting**, the ``define`` interface has an important benefit over the ``select`` and ``transform`` syntax for adding nodes to the DAG:
    It is *dictionary-based*, which makes it very easy to recursively update its content, which is very useful for :ref:`plot_cfg_inheritance`.



Individually adding nodes
^^^^^^^^^^^^^^^^^^^^^^^^^
Nodes can be added to :py:class:`~dantro.dag.TransformationDAG` during initialization; all the examples above are written in that way.
However, transformation nodes can also be added after initialization using the following two methods:

- :py:meth:`~dantro.dag.TransformationDAG.add_node` adds a single node and returns its reference.
- :py:meth:`~dantro.dag.TransformationDAG.add_nodes` adds multiple nodes, allowing the ``define``, ``select``, and ``transform`` arguments in the same syntax as during initialization.
  Internally, this parses the arguments and calls :py:meth:`~dantro.dag.TransformationDAG.add_node`.


.. _dag_minimal_syntax:

Minimal Syntax
^^^^^^^^^^^^^^
To make the definition a bit less verbose, there is a so-called *minimal syntax*, which is internally translated into the explicit and verbose one documented above.
This can make DAG specification much easier:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_minimal_syntax
    :end-before:  ### End ---- dag_minimal_syntax
    :dedent: 4

This DAG will have three custom tags defined: ``some_data``, ``more_data`` and ``my_result``.
Computation of the ``my_result`` tag is equivalent to:

::

    my_result = ((some_data + more_data) + 1) ** 4

As can be seen above, the minimal syntax gets rid of the ``operation``, ``args`` and ``kwargs`` keys by allowing to specify it as ``<operation name>: <args or kwargs>`` or even as just a string ``<operation name>``, without further arguments.

With arguments, ``<operation name>: <args or kwargs>``
""""""""""""""""""""""""""""""""""""""""""""""""""""""
When passing a sequence (e.g. ``[foo, bar]``) the arguments are interpreted as positional arguments; when passing a mapping (e.g. ``{foo: bar}``), they are treated as keyword arguments.

.. hint::

    In this shorthand notation it is still possible to specify the respective "other" types of arguments using the ``args`` or ``kwargs`` keys.
    For example:

    .. code-block:: yaml

        transform:
          - my_operation: [foo, bar]
            kwargs: { some: more, keyword: arguments }
          - my_other_operation: {foo: bar}
            args: [some, positional, arguments]

Without arguments, ``<operation name>``
"""""""""""""""""""""""""""""""""""""""
When specifying only the name of the operation as a string (e.g. ``increment`` and ``print``), it is assumed that the operation accepts *only* a single *positional* argument and no other arguments.
That argument is automatically filled with a reference to the result of the *previous* transformation, i.e.: the result is carried over.

For example, the above transformation with the ``increment`` operation would be translated to:

.. code-block:: yaml

    operation: increment
    args: [!dag_prev ]
    kwargs: {}
    tag: ~


.. _dag_op_hooks_integration:

Operation Hooks
^^^^^^^^^^^^^^^
The DAG syntax parser allows attaching additional parsing functions to operations, which can help to supply a more concise syntax.
These so-called *operation hooks* are described in more detail :ref:`here <dag_op_hooks>`.
As an example, the ``expression`` operation can be specified much more conveniently with the use of its hook.
Taking the example from :ref:`above <dag_minimal_syntax>`, the same can be expressed as:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_op_hooks_expression_minimal
    :end-before:  ### End ---- dag_op_hooks_expression_minimal
    :dedent: 4

In this case, the hook automatically extracts the free symbols (``some_data`` and ``more_data``) and translates them to the corresponding :py:class:`~dantro._dag_utils.DAGTag` objects.
Effectively, it parses the above to:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_op_hooks_expression_expanded_minimal
    :end-before:  ### End ---- dag_op_hooks_expression_expanded_minimal
    :dedent: 4

If you care to **deactivate a hook**, set the ``ignore_hooks`` flag for the operation:

.. code-block:: yaml

    operation: some_hooked_operation
    args: [foo, bar]
    ignore_hooks: true

.. warning::

    Failing operation hooks will emit a logger warning, informing about the error; they do not raise an exception.
    While this might not lead to a failure during *parsing*, it might lead to an error during computation, e.g. when you are relying on the hook to have adjusted the operation arguments.

    Depending on the operation arguments, there can be cases where the hook will not be *able* to perform its function because it lacks information that is only available after a computation.
    In such cases, it's best to deactivate the hook as described above.



.. _dag_graph_vis:

Graph representation and visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:class:`~dantro.dag.TransformationDAG` has the ability to represent the internally used directed acyclic graph as a :py:class:`networkx.classes.digraph.DiGraph`.
By calling the :py:meth:`~dantro.dag.TransformationDAG.generate_nx_graph` method, the :py:class:`~dantro.dag.Transformation` objects are added to a graph and the dependencies between these transformations are added as directed edges.

This can help to better understand the generated DAG and is useful not only for debugging but also for optimization, as it allows to show the associated profiling information.

.. hint::

    It can be configured whether the edges should represent the "flow" of results through the DAG (edges pointing towards the node that *requires* a certain result) or whether they should point towards a node's *dependency*.

    By default, :py:meth:`~dantro.dag.TransformationDAG.generate_nx_graph` has ``edges_as_flow`` set to ``True``, thus having edges point in the effective direction of computation.


Visualization
"""""""""""""
In addition to generating the graph object, the :py:meth:`~dantro.dag.TransformationDAG.visualize` method can generate a visual output of the DAG:

.. image:: ../_static/_gen/dag_vis/doc_examples_select_with_transform.pdf
   :target: ../_static/_gen/dag_vis/doc_examples_select_with_transform.pdf
   :width: 100%
   :alt: DAG visualization

In this example, the ``- my_result -`` node is tagged at the bottom and the arrows come from the transformations that these operations depend on.
Effectively, calculation starts at the top, with data being read from the ``dm`` node, the associated :py:class:`~dantro.data_mngr.DataManager`, then following the arrows towards the ``my_result`` node and applying the specified operations like ``squared``, ``increment`` and so on.

The circles in the background show the status of the computation, green meaning that a node's result was computed as expected; other colors and their corresponding status are detailed in the legend.
The node status can indicate where in a DAG computation routine an error occurred.
To control this, have a look at the ``show_node_status`` argument and the ``annotation_kwargs``, where the legend can be controlled.

.. note::

    Operation arguments cannot easily be shown as it would quickly become too cluttered.
    For that reason, the visualization typically restricts itself to showing the operation name, the result (if computed), and the tag (if set).

    See :py:meth:`~dantro.dag.TransformationDAG.visualize` for more info.

.. hint::

    If using the data transformation framework for :ref:`plot data selection <plot_creator_dag>`, visualization is deeply integrated there; see :ref:`plot_creator_dag_vis`.

.. hint::

    DAG visualization works much better with `pygraphviz <https://pygraphviz.github.io>`_ installed, because it gives access to more capable layouting algorithms.

Export
""""""
To post-process the DAG data elsewhere, use the standalone :py:func:`~dantro.utils.nx.export_graph` function.




.. _dag_transform_full_syntax_spec:

Full syntax specification of a single :py:class:`~dantro.dag.Transformation`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To illustrate the possible arguments to a :py:class:`~dantro.dag.Transformation`, the following block contains a full specification of available keys and arguments.

Note that this is the *explicit* representation, which is a bit verbose.
Except for ``operation``, ``args``, ``kwargs`` and ``tag``, all entries are set to default values.

.. code-block:: yaml

    operation: some_operation       # The name of the operation
    args:                           # Positional arguments
      - !dag_tag some_result        # Reference to another result
      - second_arg
    kwargs:                         # Keyword arguments
      one_kwarg: 123
      another_kwarg: foobar
    salt: ~                         # Is included in the hash; set a value here
                                    # if you would like to provoke a cache miss
    fallback: ~                     # May only be given if ``allow_failure`` is
                                    # also set, in which case it specifies a
                                    # fallback value (or reference) to use
                                    # instead of the operation result.

    # All arguments _below_ are NOT taken into account when computing the hash
    # of this transformation. Two transformations that differ _only_ in the
    # arguments given below are considered equal to each other.

    tag: my_result                  # The tag of this transformation. Optional.
    allow_failure: ~                # Whether to allow this transformation to
                                    # fail during computation or resolution of
                                    # the arguments (i.e.: upstream error).
                                    # Special options are: log, warn, silent

    file_cache:                     # File cache options
      read:                         # Read-related options
        enabled: false              # Whether to read from the file cache
        load_options: {}            # Passed on to DataManager.load
      write:                        # Write-related options
        enabled: false              # Whether to write to the file cache

        # If writing is enabled, the following options determine whether a
        # cache file should actually be written (does not always make sense)
        always: false               # If true, forces writing
        allow_overwrite: false      # If false, will not write if a cache file
                                    # already exists
        min_size: ~                 # If given, the result needs to have at
                                    # least this size (in bytes) for it to be
                                    # written to a cache file.
        max_size: ~                 # Like min_size, but upper boundary
        min_compute_time: ~         # If given, a cache file is only written
                                    # if the computation time of this node on
                                    # its own, i.e. without the computation
                                    # time of the dependencies exceeded this
                                    # value.
        min_cumulative_compute_time: ~  # Like min_compute_time, but actually
                                    # taking into account the time it took to
                                    # compute results of the dependencies.

        # Options used when storing a result in the cache
        storage_options:
          raise_on_error: false     # Whether to raise if saving failed
          attempt_pickling: true    # Whether to attempt pickling if saving
                                    # via a specific save function failed
          pkl_kwargs: {}            # Passed on to pkl.dumps
          ignore_groups: true       # Whether to attempt storing dantro groups
          # ... additional arguments passed on to the specific saving function

.. note::

    This does not reflect any arguments made available by the DAG parser!
    Features like the :ref:`minimal syntax <dag_minimal_syntax>` or the :ref:`operation hooks <dag_op_hooks_integration>` are handled *prior* to the initialization of a :py:class:`~dantro.dag.Transformation` object.

.. hint::

    Often the easiest way to learn is by example.
    Make sure to check out the :doc:`examples` page, where you will find practical examples that go beyond what is shown here.



.. _dag_meta_ops:

Meta-Operations
---------------
In essence, the transformation framework, as described above, can be used to define a sequence of data operations, just like a sequential program code could do.
Now, what if parts of these operations are used multiple times?
In a typical computer program, one would define a *function* to modularize part of the program.
The *equivalent* construct in the data transformation framework is a so-called **meta-operation**, which can be characterized in the following way:

* It can have input *arguments* that define which objects it should work on
* It consists of a number of operations that transform the arguments in the desired way
* It has one (and only one) output, the *return value*

**How are meta-operations defined?**
Meta-operations can be defined in just the same way as regular transformations are defined, with some additional syntax for defining positional arguments (``args``) and keyword arguments (``kwargs``).
Let's look at an example:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_simple_def
    :end-before:  ### End ---- dag_meta_ops_simple_def
    :dedent: 4

This defines two meta-operations: ``square_plus_one`` (with one positional argument) and ``select_and_compute_mean`` (with the ``to_select`` keyword argument).
During initialization of :py:class:`~dantro.dag.TransformationDAG`, these can be passed using the ``meta_operations`` argument.

**How are meta-operations used?**
In exactly the same way as all regular data operations: simply define their name as the ``operation`` argument of a transformation.

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_simple_use
    :end-before:  ### End ---- dag_meta_ops_simple_use
    :dedent: 4

Here, one of the meta-operations is used to compute two mean values from a selection; these are then added together via the regular ``add`` operation; finally, the other meta-operation is applied to that sum, yielding the result.
While the individual meta-operations are not complex in themselves, this illustrates how repeatedly invoked transformations can be modularized.

.. note::

    The examples in these sections use the ``meta_operations`` top-level entry to illustrate the *definition* of meta-operations.
    The ``transform`` and/or ``select`` top-level entries are used to denote how meta-operations can be invoked (in the same way as regular operations).

As a brief summary:

* Meta-operations are defined via the ``meta_operations`` argument of :py:class:`~dantro.dag.TransformationDAG`, using the same syntax as for other transformations.
* They can specify positional and keyword arguments and have a return value.
* They can be used for the ``operation`` argument of *any* transformation, same as other :ref:`available data operations <data_ops_ref>`.
* Meta-operations allow modularization and thereby simplify the definition of data transformations.


.. hint::

    To use meta-operations for :ref:`plot data selection <plot_creator_dag>`, define them under the ``dag_options.meta_operations`` key of a plot configuration.


Defining meta-operations
^^^^^^^^^^^^^^^^^^^^^^^^
The example above already gave a glimpse into how to define meta-operations.
In many ways, this works exactly the same as defining transformations, e.g. under the ``transform`` argument.

Specifying arguments
""""""""""""""""""""
Like Python functions, meta-operations can have two kinds of arguments:

* Positional arguments, defined using the ``!arg <position>`` YAML tag
* Keyword arguments, defined using the ``!kwarg <name>`` YAML tag

These can be used anywhere inside the meta-operation specification and serve as placeholders for expected arguments.
Let's look at an example:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_arg_def
    :end-before:  ### End ---- dag_meta_ops_arg_def
    :dedent: 4

When the meta-operation gets translated into nodes, the corresponding positional and keyword arguments are replaced with the values from the ``args`` and ``kwargs`` of the :ref:`transformation specification <dag_transform_full_syntax_spec>`.

Some remarks:

* Positional and keyword arguments can be mixed
* Arguments can be referred to multiple times within a meta-operation definition
* The set of positional arguments, if specified, needs to include all integers between zero and the highest defined ``!arg <position>``.
* *Optional* arguments and *variable* positional or keyword arguments are not supported (yet).


Return values
"""""""""""""
Meta-operations *always* have *one and only one* return value: the last defined transformation.

.. hint::

    To have "multiple" return values, e.g. to return an intermediate result, aggregate objects into a ``dict`` that can then be unpacked outside of the meta-operation.
    For an example, see :ref:`dag_meta_ops_aggregate_return_values`.


Using ``select`` within meta-operations
"""""""""""""""""""""""""""""""""""""""
The definitions inside ``meta_operations`` can have two possible formats:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_definition_formats
    :end-before:  ### End ---- dag_meta_ops_definition_formats
    :dedent: 4

As can be seen above, the dict-based definition supports using the :ref:`select interface <dag_select>`.
Importantly, this supports parametrization: simply use ``!arg`` or ``!kwarg``  inside the ``select`` specification, e.g. to make the path of the to-be-selected object an argument to the meta-operation.


Internal tags
"""""""""""""
When defining simple meta-operations, passing the output of the previous operation through to the next one using ``!dag_prev`` usually suffices to connect operations.
Such meta-operations are essentially linear DAGs.

However, to define non-linear meta-operations (or: general DAGs), it needs to be possible to use the result of *any* previously specified transformation.
For that purpose, the ``tag`` entry and the ``!dag_tag`` YAML tag can be used, same as in the :ref:`usual specification of references between transformations <dag_referencing>`:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_tagging
    :end-before:  ### End ---- dag_meta_ops_tagging
    :dedent: 4

*Internal* tags are all ``tag`` definitions inside the ``meta_operation`` definition.
These tags are solely accessible *within* the meta-operation and will not be available as results later on (only the return value will).
In the above example, the ``left``, ``right``, and ``top`` tags are internal tags and they are referenced using the already-known ``!dag_tag`` YAML tag.

This is in contrast to the regular ``tag`` definitions (the ``result`` tag in the example), which is a regular tag.
Effectively, the regular tag is attached to the last transformation of the meta-operation, here being the ``div`` operation.

.. note::

    In order to avoid silent errors and reduce unexpected behaviour, *all* internally defined tags *need* to be used within the meta-operation.


.. _dag_meta_op_default_arguments:

Argument default values
"""""""""""""""""""""""
Meta operation arguments ``!arg`` and ``!kwarg`` can also have default values.
These are defined by passing a list of length 2 to the YAML tags (instead of a scalar number for positional arguments or name for keyword arguments).

For instance, if you want an optional keyword argument ``foo``, define it as:

.. code-block:: yaml

    !kwarg [foo, my_default_value]

Equivalently for positional arguments:

.. code-block:: yaml

    !arg [0, my_default_value]

Let's look at an example where the ``my_increment`` meta-operation would increment by one per default or by some other value, if desired:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_args_with_defaults
    :end-before:  ### End ---- dag_meta_ops_args_with_defaults
    :dedent: 4

The above meta-operation is equivalent to the following Python function with one required positional-only argument and one optional positional-only argument:

.. testcode::

    def my_increment(x, delta = 1, /):
        return x + delta

    one = my_increment(0)
    two = my_increment(one)
    ten = my_increment(two, 8)

For a larger example that is using keyword arguments, see :ref:`below <dag_meta_ops_gauss>`.

.. hint::

    Default values need not be scalar, they can be anything — as long as they do not contain any :py:class:`~dantro._dag_utils.Placeholder` objects like tags, references, or other argument definitions.

.. warning::

    To clearly distinguish which arguments are optional and which ones are required, make sure that any ``!arg`` or ``!kwarg`` with a default value has a default value for *all* those occurrences of the arguments in your meta-operation:

    There should never be a ``!arg [0, 42]`` and ``!arg 0`` in your meta-operation at the same time.




Examples
^^^^^^^^
``prime_multiples``
"""""""""""""""""""
The following example performs operations on the arguments and then uses internal tags (``!dag_tag``) to connect their output to a result.

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_prime_multiples
    :end-before:  ### End ---- dag_meta_ops_prime_multiples
    :dedent: 4

As can be seen in the following plot, the meta-operation is unpacked into individual transformation nodes:

.. image:: ../_static/_gen/dag_vis/doc_examples_meta_ops_prime_multiples.pdf
   :target: ../_static/_gen/dag_vis/doc_examples_meta_ops_prime_multiples.pdf
   :width: 100%
   :alt: DAG visualization

.. hint::

    The :ref:`DAG visualization <dag_graph_vis>` also shows which operation
    originated from which meta-operation (in parentheses below the operation name).
    Here, all originate from ``prime_multiples``.

.. _dag_meta_ops_aggregate_return_values:

Aggregate return values
"""""""""""""""""""""""
In this example, a ``dict`` operation is used to return multiple results from a meta-operation.

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_multiple_return_values
    :end-before:  ### End ---- dag_meta_ops_multiple_return_values
    :dedent: 4

Note that by aggregating results into an object, the DAG will not be able to discern whether a branch of the ``compute_stats`` meta-operation is *actually* needed, thus *potentially* computing more results than required.
In order to avoid computing more nodes than necessary, aggregated return values should be used sparingly; ideally, use them only to return an *intermediate* result.

This packing and unpacking can also be observed in the DAG plot:

.. image:: ../_static/_gen/dag_vis/doc_examples_meta_ops_multiple_return_values.pdf
   :target: ../_static/_gen/dag_vis/doc_examples_meta_ops_multiple_return_values.pdf
   :width: 100%
   :alt: DAG visualization


.. _dag_meta_ops_gauss:

``my_gauss``
""""""""""""
This example shows how to define a mathematical expression (also see: :ref:`operation hooks <dag_op_hooks_integration>`) and exposing its symbols as arguments of the meta-operation:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_gauss_simple
    :end-before:  ### End ---- dag_meta_ops_gauss_simple
    :dedent: 4

For this case, it makes a lot of sense to use :ref:`default values for meta-operation arguments <dag_meta_op_default_arguments>`, thus reducing the number of keyword arguments that *need* to be specified:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_meta_ops_gauss_with_defaults
    :end-before:  ### End ---- dag_meta_ops_gauss_with_defaults
    :dedent: 4


.. hint::

    If you do not want to define default arguments, e.g. because you want to control the shared defaults via some YAML-based logic, you can also reduce the number of repeated arguments using YAML anchors and inheritance:

    .. literalinclude:: ../../tests/cfg/dag.yml
        :language: yaml
        :start-after: ### Start -- dag_meta_ops_anchor_usage
        :end-before:  ### End ---- dag_meta_ops_anchor_usage
        :dedent: 4



.. _dag_meta_ops_remarks:

Remarks & Caveats
^^^^^^^^^^^^^^^^^
Note the following remarks regarding the definition and use of meta-operations:

* Inside meta-operations, no outside tags except the "special" tags (``dag``, ``dm``, ``select_base``) can be used.
  Further inputs should be handled by adding *arguments* to the meta-operation as described above.
* When using the ``select`` syntax in the definition of a meta-operation and aiming to define an argument, note that the *long* syntax needs to be used:

    .. code-block:: yaml

        select:
          # Correct
          some_data:
            path: !kwarg some_data_path

          # WRONG! Will not work.
          other_data: !kwarg other_data_path

* When defining a meta-operation and using an operation that makes use of an :ref:`operation hook <dag_op_hooks_integration>`, the tags created by the hook need to be *explicitly* exposed as arguments, otherwise there will be an ``Unused tags ...`` error.
  To expose them, there are two ways:

  * Use them internally by adding a ``define``, ``dict``, or ``list`` operation prior to the operation that uses the hook; then explicitly specify them as arguments there.
  * In the case of the ``expression`` operation hook, use the ``kwargs.symbols`` entry to directly define them as arguments, as done in the ``my_gauss`` example above.

* A meta-operation always adds a so-called "result node", which uses the ``pass`` operation to make the result of the meta-operation available.
  When using a meta-operation, the arguments ``tag`` and ``file_cache`` (see :ref:`below <dag_file_cache>`) as well as any :ref:`error handling arguments <dag_error_handling>` are added only to this result node.
  For all other transformation nodes of a meta-operation, the following holds:

  * They may have only *internal* tags attached
  * They may define their own ``file_cache`` behavior; if they do not, the :ref:`default values for file caching <dag_file_cache_defaults>` are used.
  * They are free to define their own error handling behavior.



.. _dag_error_handling:

Error Handling
--------------
Operations are not always guaranteed to succeed.
To define more robust operations, some form of error handling is required, akin to ``try-except`` blocks in Python.

In the data transformation framework, the ``allow_failure`` option handles failing data operations and allows to specify a ``fallback`` value that should be used as result in case the operation failed.
Let's have a look:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_error_handling_example01
    :end-before:  ### End ---- dag_error_handling_example01
    :dedent: 4

Here, the ``ZeroDivisionError`` is avoided and, instead, the value of the previous node (which defines a float infinity value) is used.
Subsequently, the ``result`` will be the Python floating-point ``inf``.


.. note::

    The ``allow_failure`` argument also accepts a few string-like values which control the verbosity of the output in case of failure:

        * ``log`` does the same as ``True``: print a prominent log message that informs about the failed operation and the use of the fallback.
        * ``warn``: emits a Python warning
        * ``silent``: suppresses the message altogether

    Example:

    .. literalinclude:: ../../tests/cfg/dag.yml
        :language: yaml
        :start-after: ### Start -- dag_error_handling_example02
        :end-before:  ### End ---- dag_error_handling_example02
        :dedent: 4

    For debugging, make sure to **not** use ``silent``.

.. hint::

    The ``fallback`` argument accepts not only scalars, but also sequences or mappings, which in turn may contain ``!dag_tag`` references.


Upstream errors
^^^^^^^^^^^^^^^
Sometimes, an error only becomes relevant in a later operation and it makes sense to defer error handling to that point.
The analogy to Python exception handling would be to handle the error not directly where it occurs but in an outside scope.

This is also possible within the error handling framework, because ``allow_failure`` pertains to *both* the computation of the specified operation as well as the resolution of its arguments.
As the resolution of arguments triggers the computation of dependent nodes (and their dependencies, and so forth), an upstream error may also be caught in a downstream node:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_error_handling_example03
    :end-before:  ### End ---- dag_error_handling_example03
    :dedent: 4

In this example, the nodes tagged ``log10_value`` and ``pi_over_some_other_value`` are both problematic but do not specify any error handling.
However, we may only be interested in ``my_result``, which depends on those two transformation results.
Let's say, we specified ``compute_only: [my_result]``.
What would happen in such a case?

* The transformation tagged ``my_result`` is looked up in the DAG.
* The transformation's arguments are recursively resolved, triggering lookup of the dependencies ``log10_value`` and ``pi_over_some_other_value``.
* The referenced transformations would in turn look up *their* arguments and finally lead to the application of the problematic operations (``div`` and ``math.log10``), which will fail for the arguments in this example.
* An error is raised during those operations.
* The error propagates back to the ``my_result`` transformation.
* With ``allow_failure: true``, the error is caught and the fallback value is used instead.

.. warning::

    The above example only works with ``compute_only: [my_result]``.
    If the problematic tags were to be computed *directly*, e.g. via ``compute_only: all``, they would raise an error because they do not specify any error handling themselves.

.. note::

    This example is purely for illustration!
    Typically, one would define these operations using numpy and they would not raise exceptions but issue a ``RuntimeWarning`` and use ``nan`` as result.


Error handling within ``select``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``select`` operation may also specify a fallback.
This fallback will *only* be applied to the ``getitem`` operation which is used to look up the ``path`` from the specified selection base:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_error_handling_select_simple_fallback
    :end-before:  ### End ---- dag_error_handling_select_simple_fallback
    :dedent: 4

.. hint::

    The ``transform`` elements can of course again specify their own fallbacks.


Limitations
"""""""""""
There are some **limitations** to using ``allow_failure`` within ``select``.
Mainly, specifying a fallback may be difficult in practice because other tags may not be available yet at the time where the DAG is populated with the ``select`` arguments.

The tags specified by ``select`` are added in *alphabetical order* and before any transformations from ``transform`` are added to the DAG.
Subsequently, lookups within one ``select`` field are only possible from within ``select`` and for fields that appeared *sooner* in that alphabetical order.
(See `this issue <https://gitlab.com/utopia-project/dantro/-/issues/265>`_ for a potential improvement to this behavior.)

Using a tagged reference in the ``fallback`` works in the following example because ``'_some_fallback_data' < 'mean_data'``:

.. literalinclude:: ../../tests/cfg/dag.yml
    :language: yaml
    :start-after: ### Start -- dag_error_handling_select_tagged_fallback
    :end-before:  ### End ---- dag_error_handling_select_tagged_fallback
    :dedent: 4

.. hint::

    We advise to not build overly complex fallback structures within ``select``, e.g. using tagged fallbacks which in turn have tagged fallbacks and so forth.
    While possible, it may easily becomes tedious to build or maintain.

    If you require more advanced error handling for certain operations, consider wrapping them into your own data operation.
    See :ref:`dag_operations` for more information.




.. _dag_file_cache:

The File Cache
--------------
Caching of already computed results is a powerful feature of the :py:class:`~dantro.dag.TransformationDAG` class.
The idea is, that if some specific computationally expensive transformation already took place previously, it should not be necessary to compute it again.

Background
^^^^^^^^^^
To understand the file cache, it's first necessary to understand the internal handling of transformations.

Within the DAG, each transformation is fully identified by its hash.
If the hashes of two transformations are the same it means the operation is the same and all arguments are the same.

All :py:class:`~dantro.dag.Transformation` objects are stored in an :py:attr:`~dantro.dag.TransformationDAG.objects` database, which maps a hash to a Python object.
In effect, there is one *and only one* :py:class:`~dantro.dag.Transformation` object associated with a certain hash.

Say, a DAG contains two nodes, N1 and N2, with the same hash.
Then the object database contains a single transformation T, which is used in place of *both* nodes N1 and N2.
Thus, if the result of one of the nodes is computed, the other should already know the result and not need to re-compute it.

That is what is called the **memory cache**: once a result is computed, it stays in memory, such that it need not be recomputed again.
This is useful not only in the above situation but also when doing DAG traversal during computation.

The **file cache** is not much different than the memory cache: it aims to make computation results persist to reduce computational resources.
With the file cache, the results can persist over multiple invocations of the transformations framework.


Configuration
^^^^^^^^^^^^^
Cache directory
"""""""""""""""
Cache files need to be written in some place.
This can be specified via the ``cache_dir`` argument during the initialization of a :py:class:`~dantro.dag.TransformationDAG`; see there for details.

By default, the cache directory is called ``.cache`` and is located inside the data directory associated with the DAG's DataManager.
It is created once it is needed.

.. _dag_file_cache_defaults:

Default file cache arguments
""""""""""""""""""""""""""""
File cache behavior can be configured separately for each :py:class:`~dantro.dag.Transformation`, as can be seen from the full syntax specification above.

However, it's often useful to have default values defined that all transformations share.
To do so, pass a dict to the ``file_cache_defaults`` argument.
In the simplest form, it looks like this:

.. code-block:: yaml

    file_cache_defaults:
      read: true
      write: true
    transform:
      - # ...

This enables both reading from the cache directory and writing to it.
When passing booleans, to ``read`` and ``write``, the default behavior is used.
To more specifically configure the behavior, again see the full syntax specification above.

When specifying additional ``file_cache`` arguments within ``transform``, the values specified there recursively update the ones given by ``file_cache_defaults``.

.. note::

    The ``getitem`` operations defined via the ``select`` interface always have caching disabled; it makes no sense to cache objects that have been looked up directly from the data tree.

.. warning::

    The file cache arguments are not taken into account for computation of the transformations' hash.
    Thus, if there are two transformations with the same hash, only the additional file cache arguments given to the *first* one are taken into account; the second ones have no effect because the second transformation object is discarded altogether.

.. warning::

    If it is desired to have two transformations with different file cache options, the ``salt`` can be used to perturb its hash and thus force the use of the additional file cache arguments.


Reading from the file cache
"""""""""""""""""""""""""""
Generally, the best computation is the one you don't need to make.
If there is no result in memory and reading from cache is enabled, the cache directory is searched for a file that has as its basename the hash of the transformation that is to be computed.

If that is the case, the DataManager is used to load the data into the data tree *and* set the memory cache.
(Note that this is Python, i.e. it's not a *copy* but the memory cache is a reference to the object in the data tree.)

By default, it is *not* attempted to read from the cache directory.
See above on how to enable it.

.. note::

    When desiring to use the caching feature of the transformation framework, the employed :py:class:`~dantro.data_mngr.DataManager` needs to be able to load numerical data.
    If you are not already using the :py:class:`~dantro.data_loaders.AllAvailableLoadersMixin`, consider adding :py:class:`~dantro.data_loaders.numpy.NumpyLoaderMixin`, :py:class:`~dantro.data_loaders.xarray.XarrayLoaderMixin`, and :py:class:`~dantro.data_loaders.pickle.PickleLoaderMixin` to your :py:class:`~dantro.data_mngr.DataManager` specialization.


Writing to the file cache
"""""""""""""""""""""""""
After a computation result was either looked up from the cache or computed, it can be stored in the file cache.
By default, writing to the cache is *not* enabled, either. See above on how to enable it.

When writing a cache file, many options can trigger that a transformation's result is written to a file.
For example, it might make sense to store only results that took a very long time to compute or that are very large.

Once it is decided that a result is to be written to a cache file, the corresponding storage function is invoked.
It creates the cache directory, if it does not already exist, and then attempts to save the result object using a set of different storage functions.

There are specific storage functions for numerical data: numpy arrays are stored via the ``numpy.save`` function, which is also used to store :py:class:`~dantro.containers.numeric.NumpyDataContainer` objects.
Another specific storage function takes care of ``xarray.DataArray`` and :py:class:`~dantro.containers.xr.XrDataContainer` objects.

If there is no specific storage function available, it is attempted to pickle the object.

.. note::

    It is not currently possible to store :py:class:`~dantro.base.BaseDataGroup`-derived objects in the file cache.


Remarks
^^^^^^^
* The structure of the DAG -- a Merkle tree, or: hash tree -- ensures that each node's hash depends on all parent nodes' hashes. Thus, all downstream hashes will change if some early operation's arguments are changed.
* The transformation framework can not distinguish between arguments that are relevant for the result and those who might not; all arguments are taken into account in computing the hash.
* It might not always make sense to read from or write to the cache, depending on how long it took to compute, how much data is to be stored and loaded and how long that takes.
* Dividing up large transformations into many small transformations will increase the possibility of cache hits; however, this also increases the memory footprint of the DAG by potentially requiring more memory for intermediate objects and more read/write operations to the file cache.
* There may never be more than one file in the cache directory that has the same basename (i.e.: hash) as another file. Such situations need to be resolved manually by deleting all but one of the corresponding files.
* There is no harm in just deleting the cache directory, e.g. when it gets too large.
