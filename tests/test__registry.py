"""Tests the ObjectRegistry construct"""

import pytest

from dantro._registry import ObjectRegistry
from dantro.exceptions import *

# -----------------------------------------------------------------------------


class TestClass:
    pass


# -----------------------------------------------------------------------------


def test_init():
    reg = ObjectRegistry()
    assert len(reg) == 0


def test_register():
    reg = ObjectRegistry()

    obj = TestClass()
    with pytest.raises(MissingNameError, match="Need `name`"):
        reg.register(obj)

    obj.__name__ = "some_name"
    reg.register(obj)
    assert len(reg) == 1

    reg.register(obj, name="custom_name")
    assert len(reg) == 2

    assert obj in reg
    assert obj.__name__ in reg
    assert "custom_name" in reg

    assert reg["custom_name"] is obj
    assert reg["some_name"] is obj

    # Skipping and overwriting
    # … identical object is skipped without error
    reg.register(obj, name="custom_name")
    assert reg["custom_name"] is obj

    # … but will raise for a new object
    new_obj = TestClass()
    with pytest.raises(RegistryEntryExists):
        reg.register(new_obj, name="custom_name")

    reg.register(new_obj, name="custom_name", skip_existing=True)
    assert reg["custom_name"] is obj
    assert reg["custom_name"] is not new_obj

    reg.register(new_obj, name="custom_name", overwrite_existing=True)
    assert reg["custom_name"] is not obj
    assert reg["custom_name"] is new_obj

    # Check type of object and name
    reg._EXPECTED_TYPE = (int,)
    with pytest.raises(InvalidRegistryEntry, match="Expected type"):
        reg.register("not an int")

    with pytest.raises(TypeError, match="needs to be a string"):
        reg.register(123, name=123)


def test_dict_interface():
    reg = ObjectRegistry()

    assert len(reg) == 0
    reg.register(TestClass)
    reg.register(TestClass, name="custom_name")
    assert len(reg) == 2

    reg["TestClass"] is TestClass
    reg["custom_name"] is TestClass

    assert reg.items()
    assert reg.values()
    assert reg.keys()

    with pytest.raises(MissingRegistryEntry, match="Missing object named"):
        reg["invalid_key"]

    # cannot set or delete
    with pytest.raises(TypeError):
        reg["foo"] = "bar"

    with pytest.raises(TypeError):
        del reg["foo"]


def test_decoration():
    reg = ObjectRegistry()

    @reg._decorator
    class SomeTestClass:
        pass

    @reg._decorator("custom_name")
    class AnotherTestClass:
        pass

    @reg._decorator(name="custom_name2")
    class AnotherTestClass2:
        pass

    @reg._decorator()
    class YetAnotherTestClass:
        pass

    assert SomeTestClass in reg
    assert SomeTestClass.__name__ in reg
    assert "custom_name" in reg

    # With already existing name, should raise
    with pytest.raises(RegistryEntryExists):

        @reg._decorator("custom_name")
        class OneMoreTestClass:
            pass

    @reg._decorator("custom_name", overwrite_existing=True)
    class OneMoreTestClass:
        pass
