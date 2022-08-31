.. default-domain:: dantro.plot.utils.color_mngr

.. _color_mngr:

The ``ColorManager``
====================

The :py:class:`.ColorManager` can take care of setting up a colormap, a corresponding normalization, and assists in drawing colorbars.
Its aim is to make the :py:mod:`matplotlib.colors` module accessible via the configuration.

For instance, specifying a colormap, norm and ``vmin``/``vmax`` can be done like this:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- extremes_and_norm
    :end-before:  ### END ---- extremes_and_norm
    :dedent: 2

In the corresponding output plot, data points that go beyond ``vmax`` are shown in white, and those that cannot be represented with the custom ``LogNorm`` are shown as "bad" values in red:

.. image:: ../_static/_gen/color_mngr/extremes_and_norm.pdf
    :target: ../_static/_gen/color_mngr/extremes_and_norm.pdf
    :width: 100%
    :alt: ColorManager output example

.. contents::
    :local:
    :depth: 2

----


Integration
-----------
In order to make full use of the ``ColorManager``'s capabilities, it needs to be integrated in the following way:

* Depending on your use case, divert the ``cmap``, ``norm``, ``vmin``, ``vmax`` and colorbar label arguments to initialize a ``ColorManager``.
* Where you would typically use the ``cmap`` and ``norm`` arguments passed through, use the corresponding properties :py:attr:`.ColorManager.cmap` and :py:attr:`.ColorManager.norm` instead.
* Instead of creating the colorbar yourself, use the :py:meth:`.ColorManager.create_cbar` method, which knows about the custom colorbar labels.

This may look as follows:

.. literalinclude:: ../../tests/plot/utils/test_color_mngr.py
    :language: python
    :start-after: ### START -- ColorManager_integration
    :end-before:  ### END ---- ColorManager_integration
    :dedent: 4

.. note::

    If you are not in control of the colorbar — or are not using one — you can also use :py:func:`.parse_cmap_and_norm_kwargs`, which will extract the relevant arguments to initialize a :py:class:`.ColorManager` and return the *resolved* colormap and norm object.

    In that case, the ``labels`` argument will not have any effect.


YAML tags
^^^^^^^^^
Furthermore, YAML *tags* can be used to generate colormaps or norms in places where the ``ColorManager`` cannot be integrated but a corresponding :py:class:`matplotlib.colors.Colormap` object or norm object is accepted or even required:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- yaml_constructors
    :end-before:  ### END ---- yaml_constructors
    :dedent: 2


Examples
--------
The following examples use a YAML representation for the parameters.

Specifying colormap and ``vmin``/``vmax``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Directly specifying a colormap by name and the boundaries for the norm (falling back to the default :py:class:`~matplotlib.colors.Normalize`):

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- minimal_vmin_vmax
    :end-before:  ### END ---- minimal_vmin_vmax
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/minimal_vmin_vmax.pdf
    :target: ../_static/_gen/color_mngr/minimal_vmin_vmax.pdf
    :width: 100%
    :alt: ColorManager output example


Extreme values
^^^^^^^^^^^^^^
Can also specify colormap extreme values and a custom norm:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- extremes_and_norm
    :end-before:  ### END ---- extremes_and_norm
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/extremes_and_norm.pdf
    :target: ../_static/_gen/color_mngr/extremes_and_norm.pdf
    :width: 100%
    :alt: ColorManager output example

Note how the "bad" values are shown in red while values that go beyond ``vmax`` are shown in white.
The ``extend`` argument for :py:meth:`.ColorManager.create_cbar` can be used to control whether these colors are shown at the top or bottom of the colorbar.
By default, the method inspects whether extreme values are set in the colormap and shows them and selects the fitting visualization automatically.


Custom norm
^^^^^^^^^^^
The ``norm`` argument can be used to select a custom :py:class:`~matplotlib.colors.Normalize`-derived normalization class.
Arguments can simply be passed through.

Available norms are specified in :py:data:`.NORMS`.

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- norm_TwoSlopeNorm
    :end-before:  ### END ---- norm_TwoSlopeNorm
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/norm_TwoSlopeNorm.pdf
    :target: ../_static/_gen/color_mngr/norm_TwoSlopeNorm.pdf
    :width: 100%
    :alt: ColorManager output example


Segmented colormaps
^^^^^^^^^^^^^^^^^^^
It is easy to specify a :py:class:`~matplotlib.colors.ListedColormap` using the ``cmap.from_values`` argument; the keys of that dictionary specify the positions of the segments.

The segments can either be centered around the values specified as keys or be given explicitly as the left and right edges of the respective segments.

Inferred edges
""""""""""""""
When using scalar keys (``0, 1, 4`` here), the bin edges are inferred automatically:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- from_values_and_labels
    :end-before:  ### END ---- from_values_and_labels
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/from_centers.pdf
    :target: ../_static/_gen/color_mngr/from_centers.pdf
    :width: 100%
    :alt: ColorManager output example

.. note::

    If using irregular distances between bins (as done above), the label position will not appear centered.
    However, the label position can be adjusted separately via the keys of ``labels``.


Custom bin edges
""""""""""""""""
A segmented colormap with custom bin edges, achieved by passing the boundaries via 2-tuple-like keys:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- from_intervals
    :end-before:  ### END ---- from_intervals
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/from_intervals.pdf
    :target: ../_static/_gen/color_mngr/from_intervals.pdf
    :width: 100%
    :alt: ColorManager output example

.. note::

    If passing ``labels`` here as well, they would still need *scalar* keys for their position.


Labels and colors (shorthand syntax)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There is a shorthand syntax for specifying labels and the corresponding colors via a *single* mapping (see the ``cmap`` argument ``labels_and_colors`` below).
Here, the corresponding values are inferred from the size of the mapping and ``vmin`` and/or ``vmax``.

There

Implicit syntax
"""""""""""""""
.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- labels_and_colors_implicit
    :end-before:  ### END ---- labels_and_colors_implicit
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/cat_labels_and_colors_implicit.pdf
    :target: ../_static/_gen/color_mngr/cat_labels_and_colors_implicit.pdf
    :width: 100%
    :alt: ColorManager output example

Explicit syntax
"""""""""""""""
To avoid name clashes between labels and valid colormap arguments (those specified in :py:attr:`.ColorManager._POSSIBLE_CMAP_KWARGS`), a more explicit syntax can be used:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- labels_and_colors_explicit
    :end-before:  ### END ---- labels_and_colors_explicit
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/cat_labels_and_colors_explicit.pdf
    :target: ../_static/_gen/color_mngr/cat_labels_and_colors_explicit.pdf
    :width: 100%
    :alt: ColorManager output example

Custom ``vmin``/``vmax``
""""""""""""""""""""""""
In cases where the inferred integer range of the labels should not start at zero, simply define ``vmin`` and/or ``vmax``.

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- labels_and_colors_shifted
    :end-before:  ### END ---- labels_and_colors_shifted
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/cat_labels_and_colors_shifted.pdf
    :target: ../_static/_gen/color_mngr/cat_labels_and_colors_shifted.pdf
    :width: 100%
    :alt: ColorManager output example

Skipping values
"""""""""""""""
If some values should not be associated with a color, they can be skipped:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- skip_vals_and_labels
    :end-before:  ### END ---- skip_vals_and_labels
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/cat_skip_vals_and_labels.pdf
    :target: ../_static/_gen/color_mngr/cat_skip_vals_and_labels.pdf
    :width: 100%
    :alt: ColorManager output example

.. note::

    In order for the labels to not be shown at all, use spaces as the key of the ``labels_and_colors`` mapping.
    Take care to use *unique* mapping keys, e.g. by using different numbers of spaces.


Continuous colormaps
^^^^^^^^^^^^^^^^^^^^
A continous colormap with linearly interpolated colors (:py:class:`~matplotlib.colors.LinearSegmentedColormap`) can also easily be specified:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- continuous_default_norm
    :end-before:  ### END ---- continuous_default_norm
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/continuous.pdf
    :target: ../_static/_gen/color_mngr/continuous.pdf
    :width: 100%
    :alt: ColorManager output example


With boundaries
"""""""""""""""
This can still be combined with the :py:class:`matplotlib.colors.BoundaryNorm`:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- continuous_with_boundaries
    :end-before:  ### END ---- continuous_with_boundaries
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/continuous_with_boundaries.pdf
    :target: ../_static/_gen/color_mngr/continuous_with_boundaries.pdf
    :width: 100%
    :alt: ColorManager output example

.. note::

    The :py:class:`matplotlib.colors.BoundaryNorm` does *not* include the upper boundary.


Alpha values for extremes
^^^^^^^^^^^^^^^^^^^^^^^^^
Specifying color *and* ``alpha`` for the extremes is also possible:

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- color_syntax_advanced
    :end-before:  ### END ---- color_syntax_advanced
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/color_syntax_advanced.pdf
    :target: ../_static/_gen/color_mngr/color_syntax_advanced.pdf
    :width: 100%
    :alt: ColorManager output example


Seaborn color maps
^^^^^^^^^^^^^^^^^^
It is also possible to use seaborn colormaps, which opens up `many new possibilities <https://seaborn.pydata.org/tutorial/color_palettes.html>`_ to define colormaps.
These fall into three categories:

* Named colormaps like ``icefire`` that are available after seaborn was imported (which it is when the ``ColorManager`` is invoked).
* Color palettes defined via :py:func:`seaborn.color_palette`.
* Divering colormaps constructed via :py:func:`seaborn.diverging_palette`.

The latter two use a prefix syntax for the ``name`` argument: To use those modes, use a ``name`` argument that starts with ``color_palette::`` or ``diverging::``, respectively, followed by the corresponding arguments.
More information and examples below.

Color palettes
""""""""""""""

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- seaborn_color_palette_example
    :end-before:  ### END ---- seaborn_color_palette_example
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/seaborn_color_palette_example.pdf
    :target: ../_static/_gen/color_mngr/seaborn_color_palette_example.pdf
    :width: 100%
    :alt: ColorManager output example

More examples:

.. code-block:: text

    color_palette::YlOrBr
    color_palette::icefire
    color_palette::icefire_r          # reversed
    color_palette::light:b            # white -> blue
    color_palette::dark:b             # black -> blue
    color_palette::light:#69d         # custom color
    color_palette::light:#69d_r       # custom color reversed
    color_palette::dark:salmon_r      # named colormap reversed
    color_palette::ch:s=-.2,r=.6      # cubehelix with parameters

.. hint::

    You may have to put these prefixed strings into quotes to avoid YAML interpreting them as mappings.


Diverging palettes
""""""""""""""""""
By default, :py:func:`seaborn.diverging_palette` does *not* offer the expanded string-based syntax that :py:func:`seaborn.color_palette` supports.
However, the ``ColorManager`` has that functionality.

.. literalinclude:: ../../tests/cfg/color_manager_cfg.yml
    :language: yaml
    :start-after: ### START -- seaborn_diverging_example
    :end-before:  ### END ---- seaborn_diverging_example
    :dedent: 2

.. image:: ../_static/_gen/color_mngr/seaborn_diverging_example.pdf
    :target: ../_static/_gen/color_mngr/seaborn_diverging_example.pdf
    :width: 100%
    :alt: ColorManager output example

More examples:

.. code-block:: text

    diverging::220,20
    diverging::145,300,s=60
    diverging::250, 30, l=65, center=dark



----

API Reference
-------------
Below, an excerpt from the :py:class:`.ColorManager` API is shown.

``ColorManager.__init__``
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: dantro.plot.utils.color_mngr.ColorManager.__init__
    :noindex:

``ColorManager.create_cbar``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automethod:: dantro.plot.utils.color_mngr.ColorManager.create_cbar
    :noindex:

``parse_cmap_and_norm_kwargs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: dantro.plot.utils.color_mngr.parse_cmap_and_norm_kwargs
    :noindex:
