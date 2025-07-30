"""Tests the dantro.tools module"""

import copy
import os

import numpy as np
import pytest

import dantro
import dantro.tools as t

# -----------------------------------------------------------------------------


def test_update_terminal_info(monkeypatch):
    """Tests updating of TERMINAL_INFO dict"""
    initial_info = copy.copy(t.TERMINAL_INFO)

    # This should work
    assert t.update_terminal_info() == initial_info  # ran in the same session

    # This should fail and not result in a change, but also not raise
    with monkeypatch.context() as m:
        print(m.delattr.__doc__)
        m.delattr(t, "_get_terminal_size", raising=True)

        t.TERMINAL_INFO["lines"] = "will not be overwritten"
        t.update_terminal_info()
        assert t.TERMINAL_INFO["lines"] == "will not be overwritten"

    # This should work again
    t.update_terminal_info()
    assert isinstance(t.TERMINAL_INFO["lines"], int)


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


def test_is_unpickleable_function():
    class Foo:
        bar = 123

    def local_function(foo):
        return "bar"

    def outer():
        def inner():
            return 1

        return inner

    def make_func(x):
        def inner():
            return x

        return inner

    # no function types
    assert not t.is_unpickleable_function("bar")
    assert not t.is_unpickleable_function(123)
    assert not t.is_unpickleable_function(dict(foo="bar"))
    assert not t.is_unpickleable_function(Foo())

    # importable functions should be fine
    assert not t.is_unpickleable_function(test_is_hashable)
    assert not t.is_unpickleable_function(t.is_unpickleable_function)
    assert not t.is_unpickleable_function(t.update_terminal_info)

    # lambdas, local functions, nested functions, and closures: not pickleable!
    assert t.is_unpickleable_function(lambda foo: "bar")
    assert t.is_unpickleable_function(local_function)
    assert t.is_unpickleable_function(outer())
    assert t.is_unpickleable_function(make_func(123))

    # NOTE cannot check for __main__ functions


def test_try_conversion():
    """Tests the try_conversion function"""
    conv = t.try_conversion

    assert conv("true") is True
    assert conv("True") is True
    assert conv("tRuE") is True

    assert conv("false") is False
    assert conv("False") is False
    assert conv("fAlSe") is False

    assert conv("~") is None
    assert conv("None") is None
    assert conv("none") is None

    assert conv("0") == 0
    assert conv("1") == 1
    assert conv("1.0") == 1.0
    assert conv("1.0e0") == 1.0e0
    assert conv("1+0j") == 1 + 0j
    assert isinstance(conv("0"), int)
    assert isinstance(conv("1"), int)
    assert isinstance(conv("1.0"), float)
    assert isinstance(conv("1+0j"), complex)

    assert conv("not a number") == "not a number"


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

    assert fmt(1024**2 - 1) == "1.0 MiB"
    assert fmt(1024**2 - 1, precision=3) == "1023.999 kiB"

    assert fmt(1024**8) == "1.0 YiB"
    assert fmt(1024**9) == "1024.0 YiB"


def test_format_time():
    """Test the time formatting method"""
    fmt = t.format_time

    assert fmt(10) == "10s"
    assert fmt(1) == "1s"
    assert fmt(-1) == "- 1s"
    assert fmt(-10) == "- 10s"

    assert fmt(0.995) == "< 1s"
    assert fmt(0.1) == "< 1s"
    assert fmt(0) == "0s"
    assert fmt(-0.1) == "> -1s"

    assert fmt(10.1) == "10s"
    assert fmt(-10.1) == "- 10s"
    assert fmt(10.1, ms_precision=2) == "10.10s"
    assert fmt(-10.1, ms_precision=2) == "- 10.10s"

    assert fmt(0.1, ms_precision=2) == "0.10s"
    assert fmt(-0.1, ms_precision=2) == "- 0.10s"
    assert fmt(59.127, ms_precision=2) == "59.13s"
    assert fmt(-59.127, ms_precision=2) == "- 59.13s"

    assert fmt(60.127, ms_precision=2) == "1m"
    assert fmt(61.127, ms_precision=2) == "1m 1s"
    assert fmt(-61.127, ms_precision=2) == "- 1m 1s"

    assert fmt(123) == "2m 3s"
    assert fmt(123, ms_precision=2) == "2m 3s"

    _d = 60 * 60 * 24
    _h = 60 * 60
    _m = 60
    assert fmt(1 * _d + 2 * _h + 3 * _m + 4 + 0.5) == "1d 2h 3m 4s"
    assert fmt(1 * _d + 2 * _h + 3 * _m + 4) == "1d 2h 3m 4s"
    assert fmt(1 * _d + 2 * _h + 3 * _m + 0) == "1d 2h 3m"
    assert fmt(1 * _d + 2 * _h + 0 * _m + 4) == "1d 2h 4s"
    assert fmt(5 * _d + 2 * _h + 0 * _m + 4) == "5d 2h 4s"

    assert fmt(-(1 * _d + 2 * _h + 3 * _m + 4 + 0.5)) == "- 1d 2h 3m 4s"
    assert fmt(-(1 * _d + 2 * _h + 3 * _m + 4)) == "- 1d 2h 3m 4s"
    assert fmt(-(1 * _d + 2 * _h + 3 * _m + 0)) == "- 1d 2h 3m"
    assert fmt(-(1 * _d + 2 * _h + 0 * _m + 4)) == "- 1d 2h 4s"
    assert fmt(-(5 * _d + 2 * _h + 0 * _m + 4)) == "- 5d 2h 4s"

    assert fmt(1 * _d + 2 * _h + 3 * _m + 4, max_num_parts=3) == "1d 2h 3m"
    assert fmt(1 * _d + 2 * _h + 3 * _m + 4, max_num_parts=2) == "1d 2h"
    assert fmt(-(1 * _d + 2 * _h + 3 * _m + 4), max_num_parts=2) == "- 1d 2h"


def test_glob_paths(tmpdir):
    """Tests the glob_paths function"""
    glob_paths = t.glob_paths

    for i in range(10):
        subdir = tmpdir.mkdir(f"dir{i}")
        f = subdir.join(f"foo{i}.abc")
        f.ensure(file=True)  # touch

    p0 = glob_paths("*", base_path=tmpdir)
    assert len(p0) == 10

    p1 = glob_paths("**", base_path=tmpdir)
    assert len(p1) == 1 + 10 + 10

    p1s = glob_paths("**", base_path=tmpdir, sort=True)
    assert len(p1s) == len(p1)
    assert sorted(p1) == p1s

    p1b = glob_paths("**", base_path=tmpdir, recursive=False)
    assert len(p1b) == 10

    p2 = glob_paths("*", ignore=["*"], base_path=tmpdir)
    assert not p2

    p3 = glob_paths(
        "*",
        ignore=[
            "*",
            "**",
            "i_do_not_exist",
        ],
        base_path=tmpdir,
    )
    assert not p3

    # no base path: use cwd
    p4 = glob_paths("*")
    assert p4
    cwd = os.getcwd()
    assert all(p.startswith(cwd) for p in p4)

    # only directories
    p5 = glob_paths("**", base_path=tmpdir, include_files=False)
    assert not any(p.endswith(".abc") for p in p5)

    # only files
    p6 = glob_paths("**", base_path=tmpdir, include_directories=False)
    assert all(p.endswith(".abc") for p in p6)


# Tests of package-private modules --------------------------------------------


def test_load_yml(tmpdir):
    """Tests _yaml.load_yml function; this is actually implemented in yayaml,
    but it is included here to make finding errors easier."""
    from ruamel.yaml.parser import ParserError

    load_yml = dantro._yaml.load_yml

    # Some regular file, returning a dict
    with open(tmpdir.join("works.yml"), "x") as f:
        f.write("---\n{foo: bar, baz: 123, nested: {spam: fish}}\n")

    d = load_yml(tmpdir.join("works.yml"))
    assert d == dict(foo="bar", baz=123, nested=dict(spam="fish"))

    # An empty file, returning None
    with open(tmpdir.join("empty.yml"), "x") as f:
        f.write("---\n")

    rv = load_yml(tmpdir.join("empty.yml"))
    assert rv is None

    # Loading fails
    with open(tmpdir.join("fails.yml"), "x") as f:
        f.write("---\nsome, !bad, syntax :: }")

    with pytest.raises(ParserError):
        load_yml(tmpdir.join("fails.yml"))


def test_load_yml_hints(tmpdir):
    """Tests the YAML error hints"""
    from ruamel.yaml.constructor import ConstructorError
    from ruamel.yaml.parser import ParserError

    load_yml = dantro._yaml.load_yml

    # Loading fails, but a !dag_prev -specific hint is shown
    with open(tmpdir.join("fails.yml"), "x") as f:
        f.write("---\n")
        f.write("bar: baz\n")
        f.write("transform:\n")
        f.write("  - [zero, !dag_prev, one, two]\n")
        f.write("  - !dag_prev\n")
        f.write("spam: fish\n")

    with pytest.raises(ConstructorError, match="Did you include a space"):
        load_yml(tmpdir.join("fails.yml"))

    # Without hints
    with pytest.raises(ConstructorError) as exc_no_hints:
        load_yml(tmpdir.join("fails.yml"), improve_errors=False)
    assert "Hint(s)" not in str(exc_no_hints)

    # Another scenario
    with open(tmpdir.join("fails2.yml"), "x") as f:
        f.write("---\n")
        f.write("bar: baz\n")
        f.write("transform: [foo: !dag_prev]\n")

    with pytest.raises(ParserError, match=r"include a space after"):
        load_yml(tmpdir.join("fails2.yml"))
