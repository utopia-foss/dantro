Data Transformations
====================

The uniform structure of the dantro data tree is the ideal starting point to allow more general application of transformation on data.
This page describes dantro's data transformation framework, revolving around the :py:class:`~dantro.dag.TransformationDAG` class.

.. contents::
   :local:
   :depth: 2

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
    If you feel like you would like to jump ahead to see what awaits you, have a look at the :ref:`dag-minimal-syntax`.


The :py:class:`~dantro.dag.TransformationDAG`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The structure a user (you!) is mainly interacting with is the :py:class:`~dantro.dag.TransformationDAG` class.
It takes care to build the DAG by creating :py:class:`~dantro.dag.Transformation` objects according to the specification you provided.
In the following, all YAML examples will represent the arguments that are passed to the :py:class:`~dantro.dag.TransformationDAG` during initialization.

Basics
^^^^^^
Ok, let's start with the basics: How can transformations be defined?
For the sake of simplicity, let's only look at transformations that are fully independent from other transformations.

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
Nodes of the DAG all have a unique identifier in form of a hash string, which is a 32 character hexadecimal string.
While it can be used to identify a transformation, the easiest way to refer to it is using a so-called *tag*.

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


Advanced Referencing
^^^^^^^^^^^^^^^^^^^^
In the examples above, all transformations were independent from each other.
Having completely independent and disconnected nodes, of course, defeats the purpose of having a DAG structure.

Now let's look at proper, non-trivial DAGs, where individual transformations use the results of other transformations.


Referencing other Transformations
"""""""""""""""""""""""""""""""""
Other transformations can be referenced in three ways, each with a corresponding Python class and an associated YAML tag:

* :py:class:`~dantro.dag.DAGReference` and ``!dag_ref``: This is the most basic and most explicit reference, using the transformations' **hash** to identify a reference.
* :py:class:`~dantro.dag.DAGTag` and ``!dag_tag``: References by tag are the preferred references. They use the plain text name specified via the ``tag`` key.
* :py:class:`~dantro.dag.DAGNode` and ``!dag_node``: Uses the ID of the node within the DAG. Mostly for internal usage!

.. note::

    When the DAG is built, all references are brought into the most explicit format: :py:class:`~dantro._dag_utils.DAGReference` s.
    Thus, internally, the transformation framework works *only* with hash references.

The **best way** to refer to other transformations is **by tag**: there is no ambiguity, it is easy to define, and it allows to easily built a DAG tree structure.
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

Note that the ``args`` in that case specify one fewer positional argument.

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
To compute the results of the DAG, invoke the :py:class:`~dantro.dag.TransformationDAG`s :py:meth:`~dantro.dag.TransformationDAG.compute` method.

It can be called without any arguments, in which case the result of all *tagged* transformations will be computed and returned as a dict.
If only the result of a subset of tags should be computed, they can also be specified.

Computing results works as follows:

1. Each tagged :py:class:`~dantro.dag.Transformation` is visited and its own :py:meth:`~dantro.dag.Transformation.compute` method is invoked
2. A cache lookup occurs, attempting to read the result from a memory or file cache.
3. The transformations resolve potential references in their arguments: If a :py:class:`~dantro.dag.DAGReference` is encountered, the corresponding :py:class:`~dantro.dag.Transformation` is resolved and that transformation's :py:meth:`~dantro.dag.TransformationDAG.compute` method is invoked. This traverses all the way up the DAG until reaching the root nodes which contain only basic data types (that need no computation).
4. Having resolved all references into results, the arguments are assembled, the operation callable is resolved, and invoked by passing the arguments.
5. The result is kept in a memory cache. It *can* additionally be stored in a file cache to persist to later invocations.
6. The result object is returned.


.. note::

    *Only* nodes that are tagged *can* be part of the results.
    Intermediate results still need to be computed, but it will not be part of the results dict.
    If you want an intermediate result to be available there, add a tag to it.

    This also means: If there are parts of the DAG that are not tagged *at all*, they will not be reached by any recursive computation.



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

The ``select`` arguments expects a mapping of tags to either strings (the path within the data tree) or further mappings (where more configurations are possible):

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
This option can be controlled via the ``select_base`` property, which can be set both as argument to ``__init__`` and afterwards via the property.
The property expects either a ``DAGReference`` object or a valid tag string.

If set, all following ``select`` arguments are using that reference as the basis, leading to ``getitem`` operations on that object rather than on the data manager.

As the ``select`` arguments are evaluated prior to any transform operations, only the default tags are available during initialization.
To widen the possibilities, the :py:class:`~dantro.dag.TransformationDAG` allows the ``base_transform`` argument during initialization; this is just a sequence of transform specifications, which are applied *before* the ``select`` argument is evaluated, thus allowing to select some object, tag it, and use that tag for the ``select_base`` argument.

.. note::

    The ``select_path_prefix`` argument offers similar functionality, but merely prepends a path to the argument.
    If possible, the ``select_base`` functionality should be preferred over ``select_path_prefix`` as it reduces lookups and cooperates more nicely with the file caching features.


Individually adding nodes
^^^^^^^^^^^^^^^^^^^^^^^^^
Nodes can be added to :py:class:`~dantro.dag.TransformationDAG` during initialization; all the examples above are written in such a way.
However, transformation nodes can also be added after initialization using the following two methods:

- :py:meth:`~dantro.dag.TransformationDAG.add_node` adds a single node and returns its reference.
- :py:meth:`~dantro.dag.TransformationDAG.add_nodes` adds multiple nodes, allowing the both the ``select`` and ``transform`` arguments in the same syntax as during initialization.
  Internally, this parses the arguments and calls :py:meth:`~dantro.dag.TransformationDAG.add_node`.


.. _dag-minimal-syntax:
Minimal Syntax
^^^^^^^^^^^^^^
To make definition a bit less verbose, there is a so-called *minimal syntax*, which is translated into the explicit and verbose one:

.. code-block:: yaml

    select:
      some_data: path/to/some_data
      more_data: path/to/more_data
    transform:
      - add: [!dag_tag some_data, !dag_tag more_data]
      - increment
      - print
      - power: [!dag_prev , 4]
        tag: my_result

This DAG will have three custom tags defined: ``some_data``, ``more_data`` and ``my_result``.
Computation of the ``my_result`` tag is equivalent to:

::

    my_result = [(some_data + more_data) + 1]^4

As can be seen above, the minimal syntax gets rid of the ``operation``, ``args`` and ``kwargs`` keys by allowing to specify it as ``<operation name>: <args or kwargs>`` or even as just a string ``<operation name>``, without further arguments.

With arguments, ``<operation name>: <args or kwargs>``
""""""""""""""""""""""""""""""""""""""""""""""""""""""
By passing a sequence (e.g. ``[foo, bar]``) the arguments are interpreted as positional arguments; by passing a mapping (e.g. ``{foo: bar}``), they are treated as keyword arguments.

.. warning::

    When using the minimal syntax, it is not allowed to *additionally* specify the ``args``, ``kwargs`` and/or ``operation`` keys.

Without arguments, ``<operation name>``
"""""""""""""""""""""""""""""""""""""""
When specifying only the name of the operation as a string (e.g. ``increment`` and ``print``), it is assumed that the operation accepts only a single positional argument.
That argument is automatically filled with a reference to the result of the *previous* transformation, i.e.: the result is carried over.

For example, the transformation with the ``increment`` operation would be translated to:

.. code-block:: yaml

    operation: increment
    args: [!dag_prev ]
    kwargs: {}
    tag: ~


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
    
    # All arguments _below_ are NOT taken into account when computing the hash
    # of this transformation. Two transformations that differ _only_ in the
    # arguments given below are considered equal to each other.

    tag: my_result                  # The tag of this transformation. Optional.
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
                                    # time of the dependencies, exceeded this
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
This is useful not only in the above situation, but also when doing DAG traversal during computation.

The **file cache** is not much different than the memory cache: it aims to make computation results persist in order to reduce computational resources.
With the file cache, the results can persist over multiple invocations of the transformations framework.


Configuration
^^^^^^^^^^^^^
Cache directory
"""""""""""""""
Cache files need to be written some place.
This can be specified via the ``cache_dir`` argument during initialization of a :py:class:`~dantro.dag.TransformationDAG`; see there for details.

By default, the cache directory is called ``.cache`` and is located inside the data directory associated with the DAG's DataManager.
It is created once it is needed.

Default file cache arguments
""""""""""""""""""""""""""""
File cache behaviour can be configured separately for each :py:class:`~dantro.dag.Transformation`, as can be seen from the full syntax specification above.

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
When passing booleans, to ``read`` and ``write``, default behaviour is used.
To more specifically configure the behaviour, again see the full syntax specification above.

When specifying additional ``file_cache`` arguments within ``transform``, the values specified there recursively update the ones given by ``file_cache_defaults``.

.. note::

    The ``getitem`` operations defined via the ``select`` interface always have caching disabled; it makes no sense to cache objects that have been looked up directly from the data tree.

.. warning::

    The file cache arguments are not taken into account for computation of the transformations' hash.
    Thus, if there are two transformations with the same hash, only the additional file cache arguments given to the *first* one are taken into account; the second have no effect, because the second transformation object is discarded altogether.

.. warning::

    If it is desired to have two transformations with different file cache options, the ``salt`` can be used to perturb its hash and thus force the use of the additional file cache arguments.


Reading from the file cache
"""""""""""""""""""""""""""
Generally, the best computation is the one you don't need to make.
If there is no result in memory and reading from cache is enabled, the cache directory is searched for a file that has as its basename the hash of the transformation that is to be computed.

If that is the case, the DataManager is used to load the data into the data tree *and* set the memory cache. (Note that this is Python, i.e. it's not a *copy* but the memory cache is a reference to the object in the data tree.) 

By default, it is *not* attempted to read from the cache directory.
See above on how to enable it.


Writing to the file cache
"""""""""""""""""""""""""
After a computation result was either looked up from the cache or computed, it can be stored in the file cache.
By default, writing to the cache is *not* enabled, either. See above on how to enable it.

When writing a cache file, there are many options that can trigger that a transformation's result is written to a file.
For example, it might make sense to store only results that took a very long time to compute or that are very large.

Once it is decided that a result is to be written to a cache file, the corresponding storage function is invoked.
It creates the cache directory, if it does not already exist, and then attempts to save the result object using a set of different storage functions.

There are specific storage functions for numerical data: numpy array are stored via the ``numpy.save`` function, which is also used to store :py:class:`~dantro.containers.NumpyDataContainer` objects.
Another specific storage function takes care of ``xarray.DataArray`` and :py:class:`~dantro.containers.XrDataContainer` objects.

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
