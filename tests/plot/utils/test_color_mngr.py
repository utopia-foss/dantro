"""Test the utopya.eval.plots_mpl module"""

import contextlib
import os
from builtins import *
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dantro._import_tools import get_resource_path
from dantro.exceptions import *
from dantro.plot.utils import ColorManager, parse_cmap_and_norm_kwargs
from dantro.tools import load_yml

from ..._fixtures import *

COLOR_MANAGER_CFG = get_resource_path("tests", "cfg/color_manager_cfg.yml")

# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------


def test_ColorManager(out_dir):
    """Tests the ColorManager class."""
    # Test initializing the default ColorManager
    cm = ColorManager()
    assert cm.cmap.name == "viridis"
    assert isinstance(cm.norm, mpl.colors.Normalize)
    assert cm.labels is None
    assert cm.vmin is None
    assert cm.vmax is None

    # Test passing norm and cmap (as mpl objects) directly
    cm = ColorManager(
        cmap=mpl.colors.ListedColormap(["r", "b"]),
        norm=mpl.colors.BoundaryNorm([-2, 1, 2], ncolors=2),
    )
    assert cm.cmap(0) == mpl.colors.to_rgba("r")

    # Test other cases via configuration
    for name, cfg in load_yml(COLOR_MANAGER_CFG).items():
        print(f"\n\n--- Test case: {name} ---")
        cbar_kwargs = cfg.pop("cbar_kwargs", {})
        _raises = cfg.pop("_raises", None)
        _match = cfg.pop("_match", None)
        _test_cmap = cfg.pop("_test_cmap", {})
        _test_norm = cfg.pop("_test_norm", {})
        _test_attrs = cfg.pop("_test_attrs", {})
        title = cfg.pop("_title", name.replace("_", " "))

        if _raises is not None:
            ctx = pytest.raises(globals()[_raises], match=_match)
        else:
            ctx = contextlib.nullcontext()

        # Initialize the ColorManager
        with ctx:
            cm = ColorManager(**cfg)

        if _raises:
            print("Raised as expected.")
            continue

        # Check the general interface
        print("vmin:         ", cm.vmin)
        print("vmax:         ", cm.vmax)
        print("_cmap_kwargs: ", cm._cmap_kwargs)
        print("_norm_kwargs: ", cm._norm_kwargs)
        print("cmap:         ", cm.cmap)
        print("norm:         ", cm.norm)
        print("labels:       ", cm.labels)

        assert isinstance(cm.cmap, mpl.colors.Colormap)
        assert isinstance(cm.norm, mpl.colors.Normalize)
        assert cm.labels is None or isinstance(cm.labels, dict)

        # Test the `ColorManager.create_cbar` method by creating a dummy plot
        fig, ax = plt.subplots()
        ax.grid(linewidth=0.1, zorder=-10)
        ax.axhline(0.0, linewidth=0.8, color="black", alpha=0.5, zorder=-10)
        ax.axvline(0.0, linewidth=0.8, color="black", alpha=0.5, zorder=-10)
        vals = np.linspace(-4, 12, 17)
        scatter = ax.scatter(
            vals,
            vals,
            c=vals,
            edgecolor="k",
            linewidth=0.2,
            cmap=cm.cmap,
            norm=cm.norm,
            zorder=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        cb = cm.create_cbar(
            scatter,
            **cbar_kwargs,
        )
        plt.savefig(os.path.join(out_dir, f"{name}.pdf"))
        plt.close()

        # Check the Colorbar object
        assert isinstance(cb, mpl.colorbar.Colorbar)
        assert cb.norm == cm.norm
        assert cb.cmap == cm.cmap
        if "labels" in cfg and isinstance(cfg["labels"], dict):
            assert (cb.get_ticks() == list(cfg["labels"].keys())).all()
            # TODO Parametrise via config?

        # Test the `ColorManager.map_to_color` method
        colors = cm.map_to_color(42.0)
        assert mpl.colors.is_color_like(colors)
        colors = cm.map_to_color(np.linspace(2.1, 5.3, 10))
        assert all([mpl.colors.is_color_like(c) for c in colors])

        # Some explicit attribute, colormap, and norm checks
        if _test_attrs or _test_cmap or _test_norm:
            print("\nChecking attributes, colormap, and norm ...")
            for attr_name, expected in _test_attrs.items():
                print(f"  cm.{attr_name} == {expected}")
                assert getattr(cm, attr_name) == expected

            for cmap_val, color in _test_cmap.items():
                print(f"  cmap({cmap_val}) == 'to_rgba({color})'")
                assert cm.cmap(cmap_val) == mpl.colors.to_rgba(color)

            for val, expected in _test_norm.items():
                print(f"  norm({val}) == {expected}")
                assert cm.norm(val) == expected


def test_ColorManager_integration(out_dir):
    import matplotlib.pyplot as plt

    # fmt: off
    ### START -- ColorManager_integration
    from dantro.plot import ColorManager

    def my_scatter_func(
        x,
        y,
        *,
        c,
        ax,
        cmap=None,
        norm=None,
        vmin: float = None,
        vmax: float = None,
        cbar_labels: Union[list, dict] = None,
        cbar_kwargs: dict = {},
        **scatter_kwargs,
    ):
        """This plot function illustrates ColorManager integration"""

        # Set up the ColorManager
        cm = ColorManager(
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, labels=cbar_labels
        )

        # Now plot, passing the corresponding cmap and norm arguments ...
        scatter = ax.scatter(
            x,
            y,
            c=c,
            cmap=cm.cmap,
            norm=cm.norm,
            **scatter_kwargs,
        )

        # ... and let the ColorManager create the colorbar
        cbar = cm.create_cbar(
            scatter,
            **cbar_kwargs,
        )

        return scatter, cbar
    ### END ---- ColorManager_integration
    # fmt: on
    fig, ax = plt.subplots()
    data = np.linspace(-10, 10, 21)
    my_scatter_func(
        data,
        data**2,
        c=data,
        ax=ax,
        cmap=dict(name="PiYG", under="k", over="w"),
    )
    plt.savefig(os.path.join(out_dir, f"integration_test.pdf"))
    plt.close()


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
