Graph Group
===========
The :py:class:`~dantro.groups.graph.GraphGroup` is specialized on managing and handling graph-related data. 
The group defines a graph via data groups and data containers that store node ids, edge ids, and optionally their properties.

Creating a GraphGroup
---------------------

The ``GraphGroup`` holds the following, customizable variables that describe in which containers or attributes to find the info on the nodes and edges:

- ``_GG_node_container = "nodes"``: The name of the container or group containing the node ids
- ``_GG_edge_container = "edges"``: The name of the container or group containing the edge ids
- ``_GG_attr_directed = "directed"``: The `GraphGroup` attribute (boolean) describing whether the graph is directed or not
- ``_GG_attr_parallel = "parallel"``: The `GraphGroup` attribute (boolean) describing whether the graph allows for parallel edges or not
- ``_GG_attr_node_prop = "node_prop"``: The attribute (boolean) of a group or container within the ``GraphGroup`` classifying it as a node property
- ``_GG_attr_edge_prop = "edge_prop"``: The attribute (boolean) of a group or container within the ``GraphGroup`` classifying it as an edge property

If you do not change anything the default values are taken.

The ``GraphGroup`` only manages graph data but itself is not a graph, in the sense that no functionality such as finding the neighborhood of a node is implemented here. A widely used interface for graphs is provided by the `NetworkX <https://networkx.github.io>`_ library. In the following, we will see how to create such a graph.

Creating Graphs from a GraphGroup
---------------------------------

The ``GraphGroup`` contains the :py:meth:`~dantro.groups.graph.GraphGroup.create_graph` method that creates a graph from the containers and groups that are part of the ``GraphGroup`` together create_graphwith information provided by the ``GraphGroup`` attributes. However, the function also allows you to *explicitly* set graph properties such as whether the graph should be directed or allow for parallel edges. The function returns a `NetworkX <https://networkx.github.io>`_ graph object corresponding to the selected graph properties.

The :py:meth:`~dantro.groups.graph.GraphGroup.create_graph` function further allows you to optionally set node and edge properties by specifying ``node_props`` or ``edge_props`` lists.

If you have node/edge data that changes over time, you can select a specific time either by value via the ``at_times`` argument or through its index via ``at_time_idx``. Here, your data can either be stored as ``TimeSeriesGroup`` or as 2D data container with one time dimension.

The following example demonstrates the graph creation: Let us assume that we have a graph with static nodes and dynamic edges, each with dynamic properties. The dynamic data is written for two points in time. The resulting data tree looks as follows:

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_graphgroup_datatree
    :end-before:  ### End ---- groups_graphgroup_datatree
    :dedent: 4

Let's now create a graph from the ``GraphGroup``:

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_graphgroup_create_graph
    :end-before:  ### End ---- groups_graphgroup_create_graph
    :dedent: 4

.. note::

    For one-dimensional node/edge data or node/edge property data the ``at_times`` and ``at_time_idx`` specifier are ignored in the :py:meth:`~dantro.groups.graph.GraphGroup.create_graph` function.
    This means that static data can conveniently be stored next to other data in the ``GraphGroup`` that changes over time.

Setting Graph Properties
------------------------

If you already have a ``NetworkX`` graph object you can set node and edge properties through the :py:meth:`~dantro.groups.graph.GraphGroup.set_node_property` or :py:meth:`~dantro.groups.graph.GraphGroup.set_edge_property` function respectively. Here, the ``name`` argument specifies the name of the data container or group which stores the property and a specific time can be selected directly through ``at_time`` or via index with ``at_time_idx``.

In the example below the ``other_node_prop`` node property is added to the graph. Edge properties can be added analogously.

.. literalinclude:: ../../../tests/test_doc_examples.py
    :language: python
    :start-after: ### Start -- groups_graphgroup_set_properties
    :end-before:  ### End ---- groups_graphgroup_set_properties
    :dedent: 4