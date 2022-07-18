"""Subpackage that organizes general plotting utilities."""

from .color_mngr import ColorManager, parse_cmap_and_norm_kwargs
from .mpl import figure_leak_prevention
from .plot_func import PlotFuncResolver, is_plot_func
