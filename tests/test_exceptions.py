"""Tests the exceptions module"""

import pytest

from dantro.exceptions import *

# -----------------------------------------------------------------------------


class MockObj:
    """...for the ItemAccessError interface"""

    def __init__(self, *, path, logstr):
        self.path = path
        self.logstr = logstr


class MockGroup(MockObj):
    """...for the ItemAccessError interface"""

    def __init__(self, *, keys, **kwargs):
        super().__init__(**kwargs)
        self._keys = keys

    def keys(self):
        return self._keys

    def __len__(self) -> int:
        return len(self._keys)


# -----------------------------------------------------------------------------


def test_raise_improved_exception():
    """Tests dantro.exceptions.raise_improve_exception"""
    # Need a function to test this more easily
    def test_improved_raise(
        ExcType: type = Exception,
        msg: str = "no error message",
        ExceptType: type = Exception,
        **kwargs,
    ):
        try:
            raise ExcType(msg)
        except ExceptType as exc:
            raise_improved_exception(exc, **kwargs)

    # Here we go ...
    with pytest.raises(Exception, match="no error message"):
        test_improved_raise()

    # Without active exception to re-raise
    with pytest.raises(Exception, match="No active exception"):
        raise_improved_exception(Exception(""))

    # Now with hints
    with pytest.raises(Exception, match="some hint"):
        test_improved_raise(hints=[(lambda _: True, "some hint")])


def test_ItemAccessError():
    """Tests the error that is raised upon bad item access"""
    with pytest.raises(ItemAccessError, match="No item 'key'"):
        raise ItemAccessError(MockObj(path="foo", logstr="bar"), key="key")

    # Shows available keys
    with pytest.raises(ItemAccessError, match="No item.*key1"):
        raise ItemAccessError(
            MockGroup(path="foo", logstr="bar", keys=("key1", "key2")),
            key="key",
        )

    # Shows "Did you mean"
    with pytest.raises(ItemAccessError, match="Did you mean: 12"):
        raise ItemAccessError(
            MockGroup(
                path="foo", logstr="bar", keys=[str(i) for i in range(100)]
            ),
            key="012",
        )

    # Says that it was empty
    with pytest.raises(ItemAccessError, match="object is empty"):
        raise ItemAccessError(
            MockGroup(path="foo", logstr="bar", keys=()),
            key="some key",
        )

    # Initialization errors
    with pytest.raises(TypeError, match="dantro data tree"):
        ItemAccessError("not a good object", key="fasd")

    with pytest.raises(TypeError, match="needs to be a string"):
        ItemAccessError(MockObj(path="foo", logstr="bar"), key=None)
