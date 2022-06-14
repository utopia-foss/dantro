"""A module containing tools for generating plot configurations"""

import copy
import logging
from collections import OrderedDict
from difflib import get_close_matches as _get_close_matches
from itertools import chain as _chain
from typing import Dict, List, Sequence, Tuple, Union

from paramspace import ParamSpace

from ..exceptions import PlotConfigError
from ..tools import make_columns, recursive_update

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def _check_visited(
    visited: Sequence[Tuple[str, str]], *, next_visit: Tuple[str, str]
) -> Sequence[Tuple[str, str]]:
    """Performs cycle detection on the sequence of visited entries and raises
    an error if there will be a cycle. Otherwise, returns the new visiting
    sequence by appending the ``next_visit`` to the given sequence of
    ``visited`` entries.
    """
    if next_visit in visited:
        _loop = " <- ".join(
            [f"{b}::{p}" for b, p in _chain(visited, (next_visit,))]
        )
        raise PlotConfigError(
            f"While resolving the plot configuration for plot "
            f"'{visited[0][1]}', detected a circular dependency:  {_loop}  "
            "(with arrows denoting dependency and, plot configurations "
            "labelled in the form <base config label>::<plot name>). "
            "Check the `based_on` entries of the involved plot configurations "
            "and make sure that the determined configurations come from the "
            "*intended* base configuration."
        )
    return tuple(visited) + (next_visit,)


def _find_in_pool(
    name: str,
    *,
    base_pools: OrderedDict,
    skip: Sequence[Tuple[str, str]] = (),
) -> Tuple[str, dict, Tuple[str, dict]]:
    """Looks up a plot configuration in the given pool and returns the name
    of the pool, the found configuration, and the subset of pools that were not
    yet looked up.

    With ``skip``, certain entries can be skipped, e.g. the entry from which
    the current ``based_on`` is resolved from.
    """
    for i, (pool_name, pool_cfgs) in enumerate(reversed(base_pools.items())):
        if (pool_name, name) in skip or pool_cfgs is None:
            log.debug("Skipping '%s::%s'...", pool_name, name)
            continue

        try:
            pcfg = pool_cfgs[name]
            log.debug("Found '%s' in pool '%s'.", name, pool_name)
            break
        except KeyError:
            pass

    else:
        # Failed to find one. Generate a useful error message
        all_names = set(
            _chain(
                *[pc.keys() for _, pc in base_pools.items() if pc is not None]
            )
        )
        matches = _get_close_matches(name, all_names, n=5)
        _dym = ""
        if matches:
            _dym = f"Did you mean: {', '.join(matches)} ?\n"

        _pools = "\n".join(
            [
                f"--- Pool '{name}'\n{make_columns(pc.keys())}"
                for name, pc in reversed(base_pools.items())
                if pc is not None
            ]
        )
        raise PlotConfigError(
            f"Did not find a base plot configuration named '{name}' in the "
            f"pool of available base configurations! {_dym}Check that an "
            "entry with that name is part of at least one of the specified "
            f"pools:\n{_pools}"
        )

    # Found the desired configuration by searching the last i entries.
    # Reduce the base_pools to the subset that was not yet searched and the
    # currently used one. Then have all information ready to return ...
    _base_pools = (
        OrderedDict(list(base_pools.items())[:-i]) if i > 0 else base_pools
    )
    return pool_name, pcfg, _base_pools


def _resolve_based_on(
    pcfg: Union[dict, ParamSpace],
    *,
    base_pools: OrderedDict,
    _visited: Sequence[Tuple[str, str]],
) -> Union[dict, ParamSpace]:
    """Assembles a single plot's configuration by recursively resolving its
    ``based_on`` entry from a pool of available base plot configurations.

    This function *always* works on a deep copy of ``pcfg`` and will remove any
    potentially existing ``based_on`` entry on the root level of ``pcfg``.

    Furthermore, it accepts ``ParamSpace`` objects for plot configuration
    entries, recursively updating their dict representation and again creating
    a ``ParamSpace`` object from them afterwards.
    """
    # Need a few helper functions to handle ParamSpace objects
    # ... this COULD be avoided if ParamSpace would behave more dict-like,
    #     which it currently does not, so we have no choice but to unpack and
    #     repack it all the time. Should not be a big issue though, compared to
    #     all the plotting operations.
    _from_pspace = isinstance(pcfg, ParamSpace)
    _generate_return_value = lambda d: d if not _from_pspace else ParamSpace(d)
    _unpack_pspace = lambda d: d if not isinstance(d, ParamSpace) else d._dict
    # FIXME Should not have to use private API!

    # Prepare the given configuration
    pcfg = _unpack_pspace(copy.deepcopy(pcfg))

    based_on = pcfg.pop("based_on", None)
    if not based_on:
        return _generate_return_value(pcfg)

    elif isinstance(based_on, str):
        based_on = (based_on,)

    # Aggregate the based_on entries
    _pcfg = dict()
    for _based_on in based_on:
        log.debug("Resolving based_on: '%s' ...", _based_on)

        pool_name, base_cfg, sub_base_pools = _find_in_pool(
            _based_on, base_pools=base_pools, skip=(_visited[-1],)
        )

        # Might need to recursively resolve entries on that one ...
        # NOTE This also ensures that `base_cfg` is a deep copy, removing any
        #      possible mutability side effects from the recursive_update below
        base_cfg = _resolve_based_on(
            base_cfg,
            base_pools=sub_base_pools,
            _visited=_check_visited(
                _visited, next_visit=(pool_name, _based_on)
            ),
        )
        base_cfg = _unpack_pspace(base_cfg)

        # ... now apply to existing configuration
        _pcfg = recursive_update(_pcfg, base_cfg)

    # Finally, apply the given top level configuration
    pcfg = recursive_update(_pcfg, pcfg)
    return _generate_return_value(pcfg)


def resolve_based_on(
    plots_cfg: Dict[str, Union[dict, ParamSpace]],
    *,
    label: str,
    base_pools: Union[OrderedDict, Sequence[Tuple[str, dict]]],
) -> Dict[str, dict]:
    """Resolves the ``based_on`` entries of *all* plot configurations in the
    given plot configurations dictionary ``plots_cfg``.

    The procedure is as follows:

        - Iterate over root-level entries in the ``plots_cfg`` dict
        - For each entry, check if a ``based_on`` entry is present and needs
          to be resolved.
        - If so, recursively resolve the configuration entry, starting from the
          first entry in the ``based_on`` sequence and recursively updating it
          with content of the following elements. The final recursive update
          is that of the plot configuration given in ``plots_cfg``.

    Lookups happen from a pool of plot configurations: the ``base_pools``,
    combined with the given ``plots_cfg`` itself. The ``based_on`` entries are
    looked up by name using the following rules:

        - Other plot configurations within ``plots_cfg`` have highest
          precedence.
        - If no name is found there, lookups happen from within ``base_pools``,
          iterating over it *in reverse*, meaning that entries later in the
          ordered dict take precedence over those earlier.
        - If entries in a base pool are again using ``based_on``, these will be
          looked up using the same rules, but with the pool restricted to
          entries *with lower precedence* than that pool.
        - Lookups within the same pool will exclude the name of the currently
          updated plot configuration.
          Example: ``some_plot: {based_on: some_plot}`` will look for
          ``some_plot`` in some lower-precedence pool.

    The resolution of plot configurations works on deep copies of the given
    ``plots_cfg`` and all the ``based_on`` entries to avoid mutability issues
    between parts of these highly nested dictionaries.

    For integrated use of this functionality, see :ref:`plot_cfg_inheritance`.

    Args:
        plots_cfg (dict): A dict with multiple plot configurations to resolve
            the ``based_on`` entries of. Root-level keys are assumed to
            correspond to individual plot configurations.
            If this argument evaluates to False, will silently assume an empty
            plots configuration.
        label (str): The label to use for the given plots configuration when
            adding it to the base configuration pool.
        base_pools (Union[OrderedDict, Sequence[Tuple[str, dict]]]): The base
            configuration pools to look up the ``based_on`` entries in. This
            needs to be an OrderedDict or a type that can be converted into
            one. Keys will be used as labels for the individual pools.
            The order of this pool is relevant, see above.

    Raises:
        PlotConfigError: Upon missing ``based_on`` values or dependency loops.
    """
    if not isinstance(base_pools, OrderedDict):
        base_pools = OrderedDict(list(base_pools))

    plots_cfg = plots_cfg if plots_cfg else {}
    for pcfg_name, pcfg in plots_cfg.items():
        plots_cfg[pcfg_name] = _resolve_based_on(
            pcfg,
            base_pools=OrderedDict(
                tuple(base_pools.items()) + ((label, plots_cfg),)
            ),
            _visited=[(label, pcfg_name)],
        )

    return plots_cfg


def resolve_based_on_single(
    *,
    name: str,
    based_on: Union[str, Sequence[str]],
    plot_cfg: dict,
    **resolve_based_on_kwargs,
) -> dict:
    """Wrapper for :py:func:`~dantro.plot._cfg.resolve_based_on` for cases of
    single independent plot configurations.

    Args:
        name (str): The name of the single plot
        based_on (Union[str, Sequence[str]]): The *extracted* ``based_on``
            argument.
        plot_cfg (dict): The rest of the single plot's configuration. This
            may not include ``based_on``! If this argument evaluates to False,
            will silently assume an empty plots configuration.
        **resolve_based_on_kwargs: Passed on

    """
    return resolve_based_on(
        {name: dict(based_on=based_on, **(plot_cfg if plot_cfg else {}))},
        **resolve_based_on_kwargs,
    )[name]
