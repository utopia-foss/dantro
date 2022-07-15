"""Utility functions for the figure generation package"""

import os
import sys


def import_from_dantro_tests(mod_file_path: str):
    """Imports a specific module from within the dantro tests package.

    Because the package is not installed, need to do this from a file path.
    """
    from dantro._import_tools import import_module_from_file

    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../tests")
    )
    parent_dir = os.path.dirname(base_dir)
    modstr = os.path.splitext(mod_file_path)[0].replace("/", ".")
    modstr = f"tests.{modstr}"

    print(f"Importing module from dantro test file {mod_file_path} ...")
    print(f"  modstr:    {modstr}")
    print(f"  base_dir:  {base_dir}\n")

    # Additionally add it to sys.path; needed for pkg_resources to work
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    return import_module_from_file(
        mod_file_path,
        base_dir=base_dir,
        mod_name_fstr=modstr,
    )
