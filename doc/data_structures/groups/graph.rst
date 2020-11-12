.. _data_structures_graph_group:

The :py:class:`~dantro.groups.graph.GraphGroup`
===============================================
The :py:class:`~dantro.groups.graph.GraphGroup` is specialized on managing and handling graph-related data.
The group defines a graph via data groups and data containers that store nodes, edges, and optionally their properties.

.. contents::
    :local:
    :depth: 2

----


Creating a GraphGroup
---------------------

The :py:class:`~dantro.groups.graph.GraphGroup` holds the following, customizable variables that describe in which containers or attributes to find the info on the nodes and edges:

- ``_GG_node_container = "nodes"``: The name of the container or group (``node_container``) containing the node data
- ``_GG_edge_container = "edges"``: The name of the container or group (``edge_container``) containing the edge data
- ``_GG_attr_directed = "directed"``: The :py:class:`~dantro.groups.graph.GraphGroup` attribute (boolean) describing whether the graph is directed or not
- ``_GG_attr_parallel = "parallel"``: The :py:class:`~dantro.groups.graph.GraphGroup` attribute (boolean) describing whether the graph allows for parallel edges or not
- ``_GG_attr_edge_container_is_transposed = "edge_container_is_transposed"``: The :py:class:`~dantro.groups.graph.GraphGroup` attribute (boolean) describing whether the edge container is transposed, i.e., has the shape ``(edge-tuple-size, edge-number)``
- ``_GG_attr_keep_dim = "keep_dim"``: The :py:class:`~dantro.groups.graph.GraphGroup` attribute (iterable) describing which dimensions are not to be squeezed during data selection.

If you do not change anything, the default values are taken.

The :py:class:`~dantro.groups.graph.GraphGroup` only holds and manages graph data but itself is not a graph, in the sense that no functionality such as finding the neighborhood of a node is implemented there.
Instead, the :py:class:`~dantro.groups.graph.GraphGroup` uses the `networkx <https://networkx.github.io>`_ library and its interface for creating ``Graph`` objects.
In the following, an overview is given how graphs can be created from a :py:class:`~dantro.groups.graph.GraphGroup`.


Creating Graphs from a GraphGroup
---------------------------------

The :py:class:`~dantro.groups.graph.GraphGroup` contains the :py:meth:`~dantro.groups.graph.GraphGroup.create_graph` method that creates a graph from the containers and groups that are part of the :py:class:`~dantro.groups.graph.GraphGroup` together with information provided by the :py:class:`~dantro.groups.graph.GraphGroup` attributes.
However, the function also allows you to *explicitly* set graph properties such as whether the graph should be directed or allow for parallel edges.
The function returns a `networkx graph object <https://networkx.github.io/documentation/stable/reference/classes/index.html>`_ corresponding to the provided data and information.

The :py:meth:`~dantro.groups.graph.GraphGroup.create_graph` function further allows you to optionally set node and edge properties by specifying ``node_props`` or ``edge_props`` lists.

Any data can be pre-selected using the ``sel`` and ``isel`` arguments.
The selectors are applied to all data involved, i.e., to node, edge, as well as property data.
It is also possible to provide both ``sel`` and ``isel`` as long as the intersection of both key-sets is empty.

.. warning::

    **Invalid keys in** ``sel`` **and** ``isel`` **are ignored silently.** This means that the node, edge, and property data need not have the same set of dimensions in order to apply a selection.
    Moreover, all dimensions of size 1 are squeezed, hence no selection has to be specified in such scenarios, i.e. when the selection is unambiguous.

If you have node/edge data that changes over time, you can select along the ``time`` dimension directly via the ``at_times`` or the ``at_time_idx`` argument.
This sets or overwrites the respective entry in the ``sel`` or ``isel`` dicts.

The following example demonstrates the graph creation: Let us assume that we have a graph with static nodes and dynamic edges, each with dynamic properties.
The dynamic data, stored as :py:class:`~dantro.groups.time_series.TimeSeriesGroup`, is given for two points in time.
The resulting data tree looks as follows:

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_graphgroup_datatree
    :end-before:  ### End ---- groups_graphgroup_datatree
    :dedent: 4

Let's now create a graph from the :py:class:`~dantro.groups.graph.GraphGroup`:

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_graphgroup_create_graph
    :end-before:  ### End ---- groups_graphgroup_create_graph
    :dedent: 4

.. hint::

    Graph creation might fail for graphs with a single node (edge) due to the node (edge) dimension (of size 1) being squeezed out. It is therefore strongly recommended to specify the node and edge dimension names in the ``_GG_attr_keep_dim`` group attribute. Alternatively, they can be specified via the ``keep_dim`` argument in :py:meth:`~dantro.groups.graph.GraphGroup.create_graph`, :py:meth:`~dantro.groups.graph.GraphGroup.set_node_property`, and :py:meth:`~dantro.groups.graph.GraphGroup.set_edge_property`.


Setting Graph Properties
------------------------

If you already have a ``networkx`` graph object, you can set node or edge properties using the :py:meth:`~dantro.groups.graph.GraphGroup.set_node_property` or :py:meth:`~dantro.groups.graph.GraphGroup.set_edge_property` function.
Properties can be added from:

- data stored *inside* the :py:class:`~dantro.groups.graph.GraphGroup`: Here, the ``name`` argument specifies the name of the data container or group which stores the property.
- *external* data: see :ref:`Loading External Data as Graph Property <loading_ext_prop_data>`

In both cases, ``name`` will be the name of the node or edge property in the ``networkx`` graph.

Again, the data can be pre-selected using the ``sel``, ``isel``, ``at_time``, and ``at_time_idx`` arguments.

In the example below, the ``other_edge_prop`` data stored inside the graph group is added as edge property.
Note that time specification is required here, even though ``other_edge_prop`` is one-dimensional, because ``edge_container`` contains edge data for multiple times.

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_graphgroup_set_properties
    :end-before:  ### End ---- groups_graphgroup_set_properties
    :dedent: 4

Node properties can be added analogously.

.. note::

    Node (edge) properties can only be added for those nodes (edges) that are available in the ``node_container`` (``edge_container``).
    By default, property data is assumed to be aligned with the :py:class:`~dantro.groups.graph.GraphGroup`\ s node (edge) data.
    However, it can be aligned with the latter via `xarray.align <http://xarray.pydata.org/en/stable/generated/xarray.align.html>`_ by setting ``align`` to ``True`` in :py:meth:`~dantro.groups.graph.GraphGroup.set_node_property` (:py:meth:`~dantro.groups.graph.GraphGroup.set_edge_property`).
    The indexes of the ``node_container`` (``edge_container``) are used for the alignment in each dimension.
    If the class variable ``_GG_WARN_UPON_BAD_ALIGN`` is set to ``True`` (default: ``True``), warnings on possible pitfalls are given.


.. _loading_ext_prop_data:

Loading External Data as Graph Property
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to add a graph property from data that is *not* stored inside the :py:class:`~dantro.groups.graph.GraphGroup` (e.g., you have pre-processed some data), this can be realized in two different ways:

- Making use of the :py:attr:`~dantro.groups.graph.GraphGroup.property_maps`.
    After registering external data with a key using :py:meth:`~dantro.groups.graph.GraphGroup.register_property_map`, it will be permanently available via the provided key, i.e., the key can be passed as ``name`` in the ``set_*_property`` functions.
- Loading the external data *directly* by passing it via the ``data`` argument in the respective ``set_*_property`` function.
    The ``name`` argument then *sets* the name of the property.

Have a look at a small example where some external data ``ext_data`` is added to the graph as a node property:

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_graphgroup_property_maps
    :end-before:  ### End ---- groups_graphgroup_property_maps
    :dedent: 4
