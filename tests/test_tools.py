"""Test the tools module"""

import numpy as np
import pytest

import dantro
import dantro.tools as t

# Local constants


# Fixtures --------------------------------------------------------------------


# Tests -----------------------------------------------------------------------


def test_recursive_getitem():
    """Tests the recursive_getitem tool function"""
    rgi = t.recursive_getitem
    d = dict(s=0, foo=dict(bar=dict(baz="spam"), some_list=[0, dict(fish=42)]))

    assert rgi(d, ("s",)) == 0
    assert rgi(d, ("foo", "bar", "baz")) == "spam"
    assert rgi(d, ("foo", "some_list", 0)) == 0
    assert rgi(d, ("foo", "some_list", 1, "fish")) == 42

    # index and key errors are both raised as ValueErrors
    with pytest.raises(ValueError, match="key 'FOO'.*KeyError"):
        assert rgi(d, ("FOO",))
    with pytest.raises(ValueError, match="key 'FOO'.*KeyError"):
        assert rgi(
            d,
            ("foo", "FOO"),
        )
    with pytest.raises(ValueError, match="index '2'.*IndexError"):
        assert rgi(d, ("foo", "some_list", 2))
    with pytest.raises(ValueError, match="key '0'.*KeyError"):
        assert rgi(d, (0, "some", "more", "keys"))


def test_fill_line():
    """Tests the fill_line and center_in_line methods"""
    # Shortcut for setting number of columns
    fill = lambda *args, **kwargs: t.fill_line(*args, num_cols=10, **kwargs)

    # Check that the expected number of characters are filled at the right spot
    assert fill("foo") == "foo" + 7 * " "
    assert fill("foo", fill_char="-") == "foo" + 7 * "-"
    assert fill("foo", align="r") == 7 * " " + "foo"
    assert fill("foo", align="c") == "   foo    "
    assert fill("foob", align="c") == "   foob   "

    with pytest.raises(ValueError, match="length 1"):
        fill("foo", fill_char="---")

    with pytest.raises(ValueError, match="align argument 'bar' not supported"):
        fill("foo", align="bar")

    # The center_in_line method has a fill_char given and adds a spacing
    assert t.center_in_line("foob", num_cols=10) == "·· foob ··"  # cdot!
    assert t.center_in_line("foob", num_cols=10, spacing=2) == "·  foob  ·"


def test_make_columns():
    """Tests wrapping a string into columns"""
    make_cols = lambda *a, **kws: t.make_columns(*a, **kws, wrap_width=20)

    assert make_cols([]) == ""
    assert make_cols(["foo", "bar"]) == "  foo    bar  \n"
    assert (
        make_cols(["one", "two", "seven", "eight"]) == "  one      two    \n"
        "  seven    eight  \n"
    )
    assert (
        make_cols(["some", "strings", "that are longer"])
        == "  some             \n"
        "  strings          \n"
        "  that are longer  \n"
    )

    # custom fstr, right-aligned, fewer spaces
    assert (
        make_cols(["foo", "spam", "fishzzz"], fstr=" {item:>{width:}s} ")
        == "     foo     spam \n"
        " fishzzz \n"
    )


def test_is_iterable():
    """Tests the is_iterable function"""
    assert t.is_iterable("foo")
    assert t.is_iterable([1, 2, 3])
    assert not t.is_iterable(123)


def test_is_hashable():
    """Tests the is_hashable function"""

    class Foo:
        def __init__(self, *, allow_hash: bool):
            self.allow_hash = allow_hash

        def __hash__(self):
            if not self.allow_hash:
                raise ValueError()
            return hash(self.allow_hash)

    assert t.is_hashable("foo")
    assert t.is_hashable((1, 2, 3))
    assert t.is_hashable(123)
    assert not t.is_hashable([123, 456])
    assert not t.is_hashable(Foo(allow_hash=False))
    assert t.is_hashable(Foo(allow_hash=True))


def test_decode_bytestrings():
    """Tests the decode bytestrings function"""
    decode = t.decode_bytestrings

    foob = bytes("foo", "utf8")
    barb = bytes("bar", "utf8")

    # Nothing happens for regular data
    assert decode(1) == 1
    assert decode("foo") == "foo"
    assert (decode(np.array([123, 345])) == np.array([123, 345])).all()

    # Bytes get decoded to utf8
    assert decode(foob) == "foo"

    # Numpy string arrays get their dtype changed
    assert decode(np.array(["foo", "bar"], dtype="S")).dtype == np.dtype("<U3")
    assert decode(np.array(["foo", "bar"], dtype="a")).dtype == np.dtype("<U3")

    # Object arrays get their items converted
    assert decode(np.array([foob, barb], dtype="O"))[0] == "foo"
    assert decode(np.array([foob, barb], dtype="O"))[1] == "bar"
    assert decode(np.array([foob, "bar"], dtype="O"))[1] == "bar"


def test_format_bytesize():
    """Tests byte size formatting"""
    fmt = t.format_bytesize

    assert fmt(1) == "1 B"
    assert fmt(-1) == "-1 B"
    assert fmt(-1, precision=3) == "-1 B"
    assert fmt(1023) == "1023 B"
    assert fmt(1023, precision=0) == "1023 B"
    assert fmt(1023, precision=3) == "1023 B"

    assert fmt(1024) == "1.0 kiB"
    assert fmt(1024, precision=3) == "1.000 kiB"

    assert fmt(1024 ** 2 - 1) == "1.0 MiB"
    assert fmt(1024 ** 2 - 1, precision=3) == "1023.999 kiB"

    assert fmt(1024 ** 8) == "1.0 YiB"
    assert fmt(1024 ** 9) == "1024.0 YiB"


# Tests of package-private modules --------------------------------------------


def test_yaml_dumps():
    """Test the _yaml.yaml_dumps function for string dumps.

    This only tests the functionaltiy provided by the dantro implementation; it
    does not test the behaviour of the ruamel.yaml.dump function itself!
    """
    dumps = dantro._yaml.yaml_dumps

    # Basics
    assert "foo: bar" in dumps(dict(foo="bar"))

    # Passing additional parameters has an effect
    assert "'foo': 'bar'" in dumps(dict(foo="bar"), default_style="'")
    assert '"foo": "bar"' in dumps(dict(foo="bar"), default_style='"')

    # Custom classes
    class CannotSerializeThis:
        """A class that cannot be serialized"""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class CanSerializeThis(CannotSerializeThis):
        """A class that _can_ be serialized"""

        yaml_tag = "!my_custom_tag"

        @classmethod
        def from_yaml(cls, constructor, node):
            return cls(**constructor.construct_mapping(node.kwargs))

        @classmethod
        def to_yaml(cls, representer, node):
            return representer.represent_mapping(cls.yaml_tag, node.kwargs)

    # Without registering it, it should not work
    with pytest.raises(ValueError, match="Could not serialize"):
        dumps(CannotSerializeThis(foo="bar"))

    with pytest.raises(ValueError, match="Could not serialize"):
        dumps(CanSerializeThis(foo="bar"))

    # Now, register it
    assert "!my_custom_tag" in dumps(
        CanSerializeThis(foo="bar"), register_classes=(CanSerializeThis,)
    )
