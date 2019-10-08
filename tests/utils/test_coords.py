"""Test the utils.coords module"""

import pytest

import numpy as np

import dantro
import dantro.utils
import dantro.utils.coords

from dantro.containers import ObjectContainer, XrDataContainer


# -----------------------------------------------------------------------------
# NOTE Partly tested by containers using this, e.g. XrDataContainer ...

def test_coord_extractor_functions():
    """Tests the coordinate extractor functions"""
    extr = dantro.utils.coords.COORD_EXTRACTORS

    # Values; just returns the given ones, no type change
    assert extr['values']([1,2,3]) == [1,2,3]
    assert isinstance(extr['values']((1,2,3)), tuple)

    # Range
    assert extr['range']([10]) == list(range(10))
    assert extr['range']([2, 10, 2]) == list(range(2, 10, 2))

    # np.arange
    assert (extr['arange']([0, 10]) == np.arange(0, 10)).all()
    assert (extr['arange']([2, 10, 2]) == np.arange(2, 10, 2)).all()
    
    # np.linspace
    assert (extr['linspace']([0, 10, 10]) == np.linspace(0, 10, 10)).all()
    assert (extr['linspace']([2, 10, 2]) == np.linspace(2, 10, 2)).all()
    
    # np.logspace
    assert (extr['logspace']([0, 10, 11]) == np.logspace(0, 10, 11)).all()
    assert (extr['logspace']([2, 10, 2]) == np.logspace(2, 10, 2)).all()

    # start and step
    assert extr['start_and_step']([0, 1], data_shape=(2,3,4),
                                  dim_num=2) == [0, 1, 2, 3]
    assert extr['start_and_step']([10, 2], data_shape=(5,),
                                  dim_num=0) == [10, 12, 14, 16, 18]

    # trivial
    assert extr['trivial'](None, data_shape=(2,3,4), dim_num=2) == [0, 1, 2, 3]
    assert extr['trivial'](123, data_shape=(40,), dim_num=0) == list(range(40))

    # scalar
    assert extr['scalar'](1) == [1]
    assert extr['scalar']([1]) == [1]
    assert extr['scalar'](np.array([1])) == [1]
    assert extr['scalar']((1,)) == [1]
    assert isinstance(extr['scalar']((1,)), list)

    with pytest.raises(ValueError, match="Expected scalar coordinate"):
        extr['scalar']([1, 2, 3])

    # linked
    class C:
        """A Mock class for creating a Link object"""
        logstr = "object of class C"

    assert isinstance(extr['linked']("foo/bar",
                                     link_anchor_obj=C()),
                      dantro.utils.coords.Link)
    assert isinstance(extr['linked'](np.array(["foo/bar"]),
                                     link_anchor_obj=C()),
                      dantro.utils.coords.Link)


def test_extract_coords_from_name():
    """Tests .utils.coords.extract_coords_from_name"""
    extract = dantro.utils.coords.extract_coords_from_name
    Cont = lambda name: ObjectContainer(name=name, data=None)

    assert extract(Cont('123;456;789'), dims=('foo', 'bar', 'baz'),
                   separator=';'
                   ) == dict(foo=[123], bar=[456], baz=[789])
    assert extract(Cont('123;456;789'), dims=('foo', 'bar', 'baz'),
                   attempt_conversion=False, separator=';'
                   ) == dict(foo=['123'], bar=['456'], baz=['789'])

    # Conversion
    kws = dict(dims=('foo',), separator=';')
    assert extract(Cont('1'), **kws)['foo'] == [1]
    assert extract(Cont('1.'), **kws)['foo'] == [1.]
    assert isinstance(extract(Cont('1.'), **kws)['foo'][0], float)
    assert extract(Cont('1.+1j'), **kws)['foo'] == [1.+1j]
    assert extract(Cont('stuff'), **kws)['foo'] == ['stuff']
    assert isinstance(extract(Cont('stuff'), **kws)['foo'][0], str)

    # Error messages
    with pytest.raises(ValueError,
                       match="Number of coordinates .* does not match"):
        extract(Cont('1;2'), **kws)

    with pytest.raises(ValueError, match="One or more .* were empty!"):
        extract(Cont('1;;3'), dims=('foo', 'bar', 'baz'), separator=';')


def test_extract_coords():
    """This is mostly tested already by the XrDataContainer test"""
    extract = dantro.utils.coords.extract_coords

    # Invalid mode
    with pytest.raises(ValueError, match="Invalid extraction mode 'bad_mode'"):
        extract(XrDataContainer(name='test', data=[1,2,3]),
                mode='bad_mode', dims=('foo',))

    # Caching
    with pytest.raises(NotImplementedError):
        extract(XrDataContainer(name='test', data=[1,2,3]),
                mode='name', dims=('foo',), use_cache=True)
