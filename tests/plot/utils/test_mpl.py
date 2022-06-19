"""Tests the dantro.plot.utils.mpl module"""

import matplotlib.pyplot as plt
import pytest

from dantro.plot.utils import figure_leak_prevention

# -----------------------------------------------------------------------------


def test_figure_leak_prevention():
    """Tests the figure_leak_prevention context manager"""
    # After a fresh start, open some figures
    plt.close("all")
    figs = [plt.figure() for i in range(3)]
    assert plt.get_fignums() == [1, 2, 3]

    with figure_leak_prevention():
        # Open some more. These should be closed when exiting.
        figs = [plt.figure() for i in range(3)]
        assert plt.get_fignums() == [1, 2, 3, 4, 5, 6]
        assert plt.gcf().number == 6

    assert plt.get_fignums() == [1, 2, 3, 6]

    # Once more, now with an exception, which should lead to the current fig
    # not surviving beyond the context
    with pytest.raises(Exception):
        with figure_leak_prevention(close_current_fig_on_raise=True):
            figs = [plt.figure() for i in range(2)]
            assert plt.get_fignums() == [1, 2, 3, 6, 7, 8]
            assert plt.gcf().number == 8

            raise Exception()

    assert plt.get_fignums() == [1, 2, 3, 6]
