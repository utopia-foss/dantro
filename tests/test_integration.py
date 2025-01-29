"""A test file that makes sure dantro can be integrated as expected

Many parts of this test module are literal-included in doc/integrating.rst for
a step-by-step instructions (similar to the other documentation examples).
"""

from typing import Callable

import pytest

from dantro._import_tools import get_resource_path

# Local Variables and test fixtures -------------------------------------------

INTEGRATION_CFG_PATH = get_resource_path("tests", "cfg/integration.yml")
PLOTS_CFG_PATH = get_resource_path("tests", "cfg/integration_plots.yml")


# -----------------------------------------------------------------------------
# -- INCLUDES START BELOW -----------------------------------------------------
# -----------------------------------------------------------------------------
# NOTE Important! Turn off black formatting for everything that is included...
# fmt: off



# -----------------------------------------------------------------------------
# Other definitions

### Start -- generate_and_store_data_function
import os
from functools import reduce

import h5py as h5
import numpy as np

from dantro.tools import write_yml


class Agent:
    """A simple agent class, emulating an ABM"""
    def __init__(self, *, energy: float):
        """Sets up an agent with some initial energy value"""
        self.energy = energy

    def iterate(self, *, p_death: float, p_eat: float,
                dE_eat: float, dE_live: float) -> "Agent":
        """Iterates the agent state: deduces life costs, evaluates probability
        of eating and random death.

        Note that negative energy will lead to the agent not being regarded
        as alive any longer.
        """
        self.energy += dE_live

        if np.random.random() < p_eat:
            self.energy += dE_eat

        if p_death > 0. and np.random.random() < p_death:
            self.energy = 0
        return self

    def is_alive(self) -> bool:
        """Whether the agent is still alive, i.e. has positive energy"""
        return self.energy > 0.


def generate_and_store_data(out_dir: str, *, seed: int, **params) -> dict:
    """Generate the simulation data using the given parameters and store the
    results in a file inside ``out_dir``.

    .. note::

        In practice, this will be your own data-generating module or project.
        This example function aims to show different aspects of what's possible
        to do with dantro.

    Args:
        out_dir (str): Path to the directory to store data files in
        **params: The data generation parameters
    """
    def perform_random_walk(*, num_steps: int, initial_state: float,
                            max_step_size: float) -> np.ndarray:
        """Performs a 1D random walk, returns an array of size (num_steps+1)"""
        rand_nums = np.random.uniform(-max_step_size, max_step_size,
                                      size=(num_steps + 1,))
        rand_nums[0] = initial_state
        return np.cumsum(rand_nums)

    def iterate_abm(agents, **iter_kwargs) -> list:
        """Iterates the ABM and returns an updated list of agents"""
        agents = [a.iterate(**iter_kwargs) for a in agents]
        return [a for a in agents if a.is_alive()]

    def write_agent_data(agents, *, step: int, base_group: h5.Group,
                         mean_energy: h5.Dataset, num_agents: h5.Dataset):
        """Stores agent data in the given base group"""
        energy = [a.energy for a in agents]
        base_group.create_dataset(f"energy/{step}", data=energy)

        mean_energy[step] = np.mean(energy if energy else [np.nan])
        num_agents[step] = len(agents)

        # Label the group accordingly
        base_group["energy"].attrs["content"] = "time_series"

    # -- Preparations
    # Emulate a logger for this example. In a real example, this would be a
    # proper logger, configured to write directly to a file ...
    log = []

    # Set up output directory
    log.append(f"Creating output directory {out_dir} ...")
    os.makedirs(out_dir, exist_ok=True)

    # Seed the RNG
    log.append(f"Setting PRNG seed to {seed} ...")
    np.random.seed(seed)

    # Set up HDF5 file to write most of the output to
    log.append("Opening HDF5 output file ...")
    f = h5.File(os.path.join(out_dir, "data.h5"), mode='w')


    # -- Generate data and store it into the HDF5 file
    log.append("Generating and storing data now ...")

    f.create_dataset("random_walk",
                     data=perform_random_walk(**params["random_walk"]))
    log.append("Stored random walk data.")

    log.append("Setting up simple ABM ...")
    num_steps = params["abm"]["num_steps"]
    num_agents = params["abm"]["num_agents"]

    g = f.create_group("abm")
    mean_energy_ds = g.create_dataset("mean_energy", shape=(num_steps+1,),
                                      dtype='float64', fillvalue=np.nan)
    num_agents_ds = g.create_dataset("num_agents", shape=(num_steps+1,),
                                     dtype='uint32', fillvalue=0)

    agents = [Agent(**params["abm"]["init"]) for _ in range(num_agents)]
    write_agent_data(agents, step=0, base_group=g,
                     mean_energy=mean_energy_ds, num_agents=num_agents_ds)

    for i in range(num_steps):
        agents = iterate_abm(agents, **params["abm"]["iterate"])
        write_agent_data(agents, step=i+1, base_group=g,
                         mean_energy=mean_energy_ds, num_agents=num_agents_ds)
        if not agents:
            break

    # -- Finish up data writing
    # The hierarchically structured data
    log.append("Closing HDF5 file ...")
    f.close()

    # The parameters as a YAML file
    log.append("Storing simulation parameters ...")
    write_yml(dict(seed=seed, **params),
              path=os.path.join(out_dir, "params.yml"))

    # Lastly, the log output as a text file
    log.append("Storing log output ... good bye!")
    with open(os.path.join(out_dir, "sim.log"), 'w') as f:
        f.write("\n".join(log))
        f.write("\n")
### End ---- generate_and_store_data_function


# -----------------------------------------------------------------------------
# The actual integration routine

def test_integration(tmpdir):
    """Tests the full integration procedure"""

    ### Start -- data_generation_00
    # -- Step 0: Some basic imports and definitions
    import os

    import paramspace as psp

    import dantro as dtr
    from dantro.tools import load_yml, write_yml

    sim_cfg_path = "~/path/to/sim_cfg.yml"             # simulation parameters
    base_out_path = "~/my_output_directory"            # where to store output
    project_cfg_path = "~/my_project/project_cfg.yml"  # project configuration
    plots_cfg_path = "~/path/to/plot_cfg.yml"          # plot configurations
    # NOTE: In practice, you might want to define these paths not in this
    #       absolute fashion but via the `importlib.resources` module
    ### End ---- data_generation_00

    # To not have side effects, yet allow readable examples, overwrite the
    # values given above with existing paths or tmpdir paths, respectively.
    sim_cfg_path = INTEGRATION_CFG_PATH  # to reduce files, sim config is here
    base_out_path = tmpdir.join("out")
    project_cfg_path = INTEGRATION_CFG_PATH
    plots_cfg_path = PLOTS_CFG_PATH


    ### Start -- data_generation_01
    # -- Step 1: Load a configuration file
    sim_cfg = load_yml(sim_cfg_path)

    # ... and extract the simulation parameters (an iterable ParamSpace object)
    pspace = sim_cfg["parameter_space"]
    assert isinstance(pspace, psp.ParamSpace)
    ### End ---- data_generation_01


    ### Start -- data_generation_02
    # -- Step 2: Prepare the output directory path for *this* simulation
    sim_name = "my_simulation"  # ... probably want a time stamp here
    sim_out_dir = os.path.join(base_out_path, sim_name)

    # Store the parameter space there, for reproducibility
    write_yml(pspace, path=os.path.join(sim_out_dir, "pspace.yml"))
    ### End ---- data_generation_02


    ### Start -- data_generation_03
    # -- Step 3: Use the parameter space to generate and store the output data
    for params, state_str in pspace.iterator(with_info="state_no_str"):
        # `params` is now a dict that contains the set of parameters for this
        # specific instantiation.
        # `state_str` is a string identifying this point of the parameter space

        # Create the path to the output directory for _this_ simulation
        this_out_dir = os.path.join(sim_out_dir, "sim_" + state_str)

        # Generate the data, using all those parameters
        print(f"Running simulation for parameters '{state_str}' … ", end="\t")
        generate_and_store_data(this_out_dir, **params)
        print("Done.")

    print(f"All data generated and stored at {sim_out_dir} .")
    ### End ---- data_generation_03


    ### Start -- data_loading_01
    # -- Step 4: Define a custom DataManager
    import dantro.data_loaders
    import dantro.groups

    class MyDataManager(dtr.data_loaders.AllAvailableLoadersMixin,
                        dtr.DataManager):
        """MyDataManager can load HDF5, YAML, Text, Pickle, ..."""
        # Register known group types
        _DATA_GROUP_CLASSES = dict(ParamSpaceGroup=dtr.groups.ParamSpaceGroup,
                                   TimeSeriesGroup=dtr.groups.TimeSeriesGroup)

        # Specify a content mapping for loading HDF5 data
        _HDF5_GROUP_MAP = dict(time_series=dtr.groups.TimeSeriesGroup)
    ### End ---- data_loading_01

    ### Start -- data_loading_02
    # -- Step 5: Load the project configuration
    project_cfg = load_yml(project_cfg_path)

    # ... and extract the initialization parameters for MyDataManager
    dm_cfg = project_cfg["data_manager"]


    # -- Step 6: Set up the DataManager & associate it with the data directory.
    dm = MyDataManager(sim_out_dir, name=sim_name, **dm_cfg)

    # The data tree is still empty (except for the `simulations` group).
    # Let's check:
    print(dm.tree)
    # Will print:
    #   Tree of MyDataManager 'my_simulation', 1 member, 0 attributes
    #    └─ simulations           <ParamSpaceGroup, 0 members, 0 attributes>
    ### End ---- data_loading_02

    ### Start -- data_loading_03
    # -- Step 7: Load data using the load configuration given at initialisation
    dm.load_from_cfg(print_tree="condensed")
    # Will load the data and then print a condensed tree overview:
    # Tree of MyDataManager 'my_simulation', 1 member, 0 attributes
    #  └─ simulations             <ParamSpaceGroup, 30 members, 1 attribute>
    #     └┬ 12                   <ParamSpaceStateGroup, 3 members, 0 attributes>
    #        └┬ params            <MutableMappingContainer, 1 attribute>
    #         ├ data              <OrderedDataGroup, 2 members, 0 attributes>
    #           └┬ abm            <OrderedDataGroup, 2 members, 0 attributes>
    #              └┬ energy      <TimeSeriesGroup, 31 members, 1 attribute>
    #                 └┬ 0        <XrDataContainer, float64, (dim_0: 42), 0 attributes>
    #                  ├ 1        <XrDataContainer, float64, (dim_0: 42), 0 attributes>
    #                  ├ ...          ... (27 more) ...
    #                  ├ 29       <XrDataContainer, float64, (dim_0: 4), 0 attributes>
    #                  └ 30       <XrDataContainer, float64, (dim_0: 0), 0 attributes>
    #               ├ mean_energy <NumpyDataContainer, float64, shape (101,), 0 attributes>
    #               └ num_agents  <NumpyDataContainer, uint32, shape (101,), 0 attributes>
    #            └ random_walk    <NumpyDataContainer, float64, shape (1024,), 0 attributes>
    #         └ log               <StringContainer, str stored, 1 attribute>
    #      ├ 13                   <ParamSpaceStateGroup, 3 members, 0 attributes>
    #        └┬ params            <MutableMappingContainer, 1 attribute>
    #         ├ data              <OrderedDataGroup, 2 members, 0 attributes>
    #           └┬ abm            <OrderedDataGroup, 2 members, 0 attributes>
    #              └┬ energy      <TimeSeriesGroup, 35 members, 1 attribute>
    #                 └┬ 0        <XrDataContainer, float64, (dim_0: 42), 0 attributes>
    #                  ├ ...          ... (33 more) ...
    #                  └ 34       <XrDataContainer, float64, (dim_0: 0), 0 attributes>
    #               ├ mean_energy <NumpyDataContainer, float64, shape (101,), 0 attributes>
    #               └ num_agents  <NumpyDataContainer, uint32, shape (101,), 0 attributes>
    #            └ random_walk    <NumpyDataContainer, float64, shape (1024,), 0 attributes>
    #         └ log               <StringContainer, str stored, 1 attribute>
    #      ├ 14                   <ParamSpaceStateGroup, 3 members, 0 attributes>
    #      ...
    ### End ---- data_loading_03

    ### Start -- data_loading_04
    # To access data, can use the dict interface and paths
    for sim in dm["simulations"].values():
        num_steps = sim["params"]["abm"]["num_steps"]
        extinct_after = np.argmin(sim["data/abm/num_agents"])

        print(f"In simulation '{sim.name}', agents got extinct after "
              f"{extinct_after} / {num_steps} iterations.")
    ### End ---- data_loading_04
        assert extinct_after > 0

    ### Start -- data_transformation_01
    # -- Step 8: Add a module where additional data operations can be defined
    """This module can be used to register project-specific data operations"""
    from dantro.data_ops import register_operation

    def do_something(data):
        """Given some data, does something."""
        # ... do something here ...
        return data

    register_operation(name="do_something", func=do_something)
    ### End ---- data_transformation_01


    ### Start -- data_viz_01
    # -- Step 9: Specialize a PlotManager
    class MyPlotManager(dtr.PlotManager):
        """My custom PlotManager"""
        pass

        # If plot creators are customized, specify them here
        # CREATORS = dict(custom=MyCustomPlotCreator)
    ### End ---- data_viz_01

    ### Start -- data_viz_02
    # -- Step 10: Initialize MyPlotManager from the project configuration
    pm_cfg = project_cfg["plot_manager"]

    pm = MyPlotManager(dm=dm, **pm_cfg)
    ### End ---- data_viz_02

    ### Start -- data_viz_03
    # -- Step 11: Invoke the plots specified in a configuration file
    pm.plot_from_cfg(plots_cfg=plots_cfg_path)
    ### End ---- data_viz_03
