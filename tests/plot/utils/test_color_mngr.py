"""Test the utopya.eval.plots_mpl module"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pkg_resources import resource_filename

from dantro.plot.utils import ColorManager, parse_cmap_and_norm_kwargs
from dantro.tools import load_yml

from ..._fixtures import tmpdir_or_local_dir

COLOR_MANAGER_CFG = resource_filename("tests", "cfg/color_manager_cfg.yml")

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------


def test_ColorManager(tmpdir_or_local_dir):
    """Tests the ColorManager class."""
    out_dir = tmpdir_or_local_dir

    # The configurations to test
    test_configurations = load_yml(COLOR_MANAGER_CFG)

    # Test initializing the default ColorManager
    colormanager = ColorManager()
    assert colormanager.cmap.name == "viridis"
    assert isinstance(colormanager.norm, mpl.colors.Normalize)
    assert colormanager.labels is None

    for name, cfg in test_configurations.items():
        cbar_kwargs = cfg.pop("cbar_kwargs", {})

        # Test the failing cases explicitly
        if name == "invalid_norm":
            with pytest.raises(
                ValueError, match="Received invalid norm specifier:"
            ):
                ColorManager(**cfg)
            continue

        elif name == "invalid_cmap":
            with pytest.raises(
                ValueError, match="Received invalid colormap name:"
            ):
                ColorManager(**cfg)
            continue

        elif name == "disconnected_intervals":
            with pytest.raises(
                ValueError, match="Received disconnected intervals:"
            ):
                ColorManager(**cfg)
            continue

        elif name == "decreasing_boundaries":
            with pytest.raises(
                ValueError, match="Received decreasing boundaries:"
            ):
                ColorManager(**cfg)
            continue

        elif name == "single_bin_center":
            with pytest.raises(
                ValueError, match="At least 2 bin centers must be given"
            ):
                ColorManager(**cfg)
            continue

        # Initialize the ColorManager and retrieve the created colormap, norm,
        # and colorbar labels.
        colormanager = ColorManager(**cfg)

        cmap = colormanager.cmap
        norm = colormanager.norm
        labels = colormanager.labels

        assert isinstance(cmap, mpl.colors.Colormap)
        assert isinstance(norm, mpl.colors.Normalize)
        assert labels is None or isinstance(labels, dict)

        # Test the `ColorManager.create_cbar` method
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            np.arange(10), np.arange(10), c=np.arange(10), cmap=cmap, norm=norm
        )

        cb = colormanager.create_cbar(
            scatter,
            label="my_label",
            tick_params=dict(size=7),
            **cbar_kwargs,
        )

        assert isinstance(cb, mpl.colorbar.Colorbar)
        assert cb.norm == norm
        assert cb.cmap == cmap
        if "labels" in cfg and isinstance(cfg["labels"], dict):
            assert (cb.get_ticks() == list(cfg["labels"].keys())).all()

        # Save and close the plot
        plt.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.close()

        # Test the `ColorManager.map_to_color` method
        colors = colormanager.map_to_color(42.0)
        assert mpl.colors.is_color_like(colors)
        colors = colormanager.map_to_color(np.linspace(2.1, 5.3, 10))
        assert all([mpl.colors.is_color_like(c) for c in colors])

        # NOTE Some explicit color checks. If it fails check the respective
        #      configurations.
        if name == "from_intervals":
            assert cmap(1) == mpl.colors.to_rgba("w")

        if name == "shortcut_categorical":
            assert cmap(0) == mpl.colors.to_rgba("g")

    # Test ColorManager when passing norm and cmap (as mpl object) directly
    colormanager = ColorManager(
        cmap=mpl.colors.ListedColormap(["r", "b"]),
        norm=mpl.colors.BoundaryNorm([-2, 1, 2], ncolors=2),
    )
    assert colormanager.cmap(0) == mpl.colors.to_rgba("r")


def test_ColorManager_yaml():
    """Tests ColorManager yaml representation"""
    import matplotlib as mpl

    from dantro._yaml import yaml, yaml_dumps
    from dantro.plot.utils.color_mngr import NORMS

    read_yaml = lambda s: yaml.load(s)
    dump_yaml = lambda s: yaml_dumps(s, yaml_obj=yaml)
    roundtrip = lambda o: read_yaml(dump_yaml(o))
    rev_roundtrip = lambda s: dump_yaml(read_yaml(s))

    # Simple case
    cmap = read_yaml("!cmap viridis")
    assert isinstance(cmap, mpl.colors.Colormap)
    assert "!cmap viridis" in dump_yaml(cmap)

    # Representation works for all registered types
    rev_roundtrip("!cmap_norm NoNorm")
    rev_roundtrip("!cmap_norm Normalize")
    rev_roundtrip("!cmap_norm CenteredNorm")
    rev_roundtrip("!cmap_norm LogNorm")

    rev_roundtrip("!cmap_norm {name: Normalize}")
    rev_roundtrip(
        "!cmap_norm {name: BoundaryNorm, boundaries: [0, 1], ncolors: 123}"
    )
    rev_roundtrip("!cmap_norm {name: CenteredNorm}")
    rev_roundtrip("!cmap_norm {name: LogNorm}")
    rev_roundtrip("!cmap_norm {name: PowerNorm, gamma: 2}")
    rev_roundtrip("!cmap_norm {name: SymLogNorm, linthresh: 1.e-3}")
    rev_roundtrip("!cmap_norm {name: TwoSlopeNorm, vcenter: -123}")

    # Not testable via YAML and also (probably) not representable
    # rev_roundtrip("!cmap_norm {name: FuncNorm, functions: …}")


def test_parse_cmap_and_norm_kwargs():
    import matplotlib as mpl

    parse = parse_cmap_and_norm_kwargs

    # Passthrough because there were no relevant arguments
    d = dict()
    d_out = parse(**d)
    assert d_out == d

    # Colormap set
    d = dict(cmap="inferno", ignored="foo")
    d_out = parse(**d)
    assert d_out != d
    assert d_out["ignored"] == "foo"
    assert isinstance(d_out["cmap"], mpl.colors.Colormap)

    # Passthrough because colormanager was set to be ignored
    d_out = parse(use_color_manager=False, **d)
    assert d_out == d

    # Custom key map, here leading to `cmap` key not being parsed
    d_out = parse(_key_map=dict(cmap="my_cmap"), **d)
    assert d_out == d

    # Can also use dict-based ColorManager features, even when mapped
    d["my_cmap"] = dict(name="inferno", bad="r", under="k", over="w")
    d["my_norm"] = dict(name="SymLogNorm", linthresh=1.0e-3)
    d["vmin"] = 0
    d["vmax"] = 1
    d_out = parse(_key_map=dict(cmap="my_cmap", norm="my_norm"), **d)
    assert isinstance(d_out["my_cmap"], mpl.colors.Colormap)
    assert isinstance(d_out["my_norm"], mpl.colors.SymLogNorm)
    assert d_out["cmap"] == "inferno"  # ignored because it was not key-mapped
    assert d_out["vmin"] == 0
    assert d_out["vmax"] == 1

    assert (d_out["my_cmap"].get_over() == np.array([1, 1, 1, 1])).all()
    assert (d_out["my_cmap"].get_under() == np.array([0, 0, 0, 1])).all()
