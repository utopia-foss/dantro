"""Test the utils.ordereddict module"""

import logging
import random
import sys
import uuid
from typing import Callable

import numpy as np
import pytest

import dantro
import dantro.groups
import dantro.utils
import dantro.utils.coords

logging.getLogger("dantro.utils.ordereddict").setLevel(dantro.logging.TRACE)

# Fixtures --------------------------------------------------------------------


def random_kv_pairs(
    max_num: int = 100, *, key_kind="int", key_sort_func: Callable = None
) -> tuple:
    """Returns a randomly shuffled list of key-value pairs and a sequence
    of the ordered keys.
    """
    # First, build a set of random keys, i.e.: without collisions
    if key_kind == "int":
        keys = {random.randint(0, max_num) for _ in range(max_num)}

    elif key_kind == "str":
        keys = {random.randint(0, max_num) for _ in range(max_num)}
        keys = [str(k) for k in keys]

    else:
        raise ValueError(key_kind)

    # Now, build the key-value pair list and shuffle it (in place)
    l = [(k, uuid.uuid4().hex) for k in keys]
    random.shuffle(l)

    # Also generate a sequence of ordered keys and return it alongside
    return l, sorted((k for k, _ in l), key=key_sort_func)


# Tests -----------------------------------------------------------------------


def test_KeyOrderedDict():
    """Tests the KeyOrderedDict, a subclass of OrderedDict maintaining key
    order rather than insertion order
    """

    KOD = dantro.utils.KeyOrderedDict

    # Simple test
    kod = KOD()

    # Insert elements one by one
    kv_pairs, sorted_keys = random_kv_pairs()

    for k, v in kv_pairs:
        kod[k] = v

    print("\n--- Initial test")
    print(f"Length: {len(kod)}, expected {len(sorted_keys)}")
    print("Keys (expected):", ", ".join([str(k) for k in sorted_keys]))
    print("Keys:           ", ", ".join([str(k) for k in kod]))

    # Check that keys are ordered correctly
    assert all([k1 == k2 for k1, k2 in zip(kod.keys(), sorted_keys)])

    # Reverse iteration should also work
    assert all(
        [k1 == k2 for k1, k2 in zip(reversed(kod), reversed(sorted_keys))]
    )

    # Custom comparator, here: for reverse ordering
    comp_reversed = lambda k: -k
    kv_pairs, sorted_keys = random_kv_pairs(key_sort_func=comp_reversed)
    kod = KOD(kv_pairs, key=comp_reversed)
    assert all([k1 == k2 for k1, k2 in zip(kod.keys(), sorted_keys)])

    # Custom insert method, with hints
    kod.insert(123, "foo")
    kod.insert(124, "bar", hint_after=123)

    # With str-cast keys but integer sorting
    comp_reversed = lambda k: int(k)
    kv_pairs, sorted_keys = random_kv_pairs(
        key_kind="str", key_sort_func=comp_reversed
    )
    kod = KOD(kv_pairs, key=comp_reversed)
    assert all([k1 == k2 for k1, k2 in zip(kod.keys(), sorted_keys)])

    # Test insertion complexity ...............................................
    # Comparison: out-of-order
    kv_pairs, sorted_keys = random_kv_pairs()
    kod_ooo = KOD(kv_pairs)
    assert all([k1 == k2 for k1, k2 in zip(kod_ooo.keys(), sorted_keys)])

    # In order, ascending: constant, no search
    kod_ioa = KOD(sorted(kv_pairs))
    assert all([k1 == k2 for k1, k2 in zip(kod_ioa.keys(), sorted_keys)])
    assert kod_ioa._num_comparisons < kod_ooo._num_comparisons
    assert kod_ioa._num_comparisons <= len(kod_ioa) + 10

    # In order, descending: constant, no search
    kod_iod = KOD(reversed(sorted(kv_pairs)))
    assert all([k1 == k2 for k1, k2 in zip(kod_iod.keys(), sorted_keys)])
    assert kod_iod._num_comparisons < kod_ooo._num_comparisons
    assert kod_iod._num_comparisons <= 2 * len(kod_iod) + 10

    # Insertion hints reduce number of required iterations
    # To replicate this, use a contiguous sequence of even numbers
    kod_cont = KOD((i, i) for i in range(0, 100, 2))
    assert all([k1 == k2 for k1, k2 in zip(kod_cont, range(0, 100, 2))])
    n0 = kod_cont._num_comparisons
    assert n0 == 50 - 1

    # Inserting in the middle is costly without hints
    kod_cont.insert(51, "foo")
    n1 = kod_cont._num_comparisons
    assert n1 == n0 + (51 // 2 + 1) + 2  # +2 is the constant search offset

    # ... but cheap, with hints!
    kod_cont.insert(49, "foo", hint_after=48)
    n2 = kod_cont._num_comparisons
    assert n2 == n1 + 2  # yay :)

    # If the hint is bad, it is discarded
    kod_cont.insert(79, "foo", hint_after=80)  # sic
    n3 = kod_cont._num_comparisons
    assert n3 == n2 + (79 // 2 + 1 + 2) + 2
    #                                ^-- for additionally added 49 and 51

    # Test remaining OrderedDict functionality ................................
    # Do not use a custom comparator for that
    kv_pairs, sorted_keys = random_kv_pairs()
    kod = KOD(kv_pairs)

    print("\n--- Dict functionality")
    print(f"Length: {len(kod)}, expected {len(sorted_keys)}")
    print("Keys (expected):", ", ".join([str(k) for k in sorted_keys]))
    print("Keys:           ", ", ".join([str(k) for k in kod]))
    print("Items:\n ", "\n  ".join([str(i) for i in kod.items()]))

    # String representation, pickling, copy
    KOD.__repr__(None)
    repr(kod)

    kod.__reduce__()

    kod.copy()

    # Comparison
    assert kod == kod.copy()
    assert kod == {k: v for k, v in kod.items()}

    # Iteration methods
    assert all([p1 == p2 for p1, p2 in zip(kod.items(), sorted(kv_pairs))])
    assert all([v == kod[k] for v, k in zip(kod.values(), sorted_keys)])

    # Size and any kind of delete operations
    size = sys.getsizeof
    s1 = size(kod)

    del kod[sorted_keys[0]]
    del kod[sorted_keys[1]]
    s2 = size(kod)
    assert s2 < s1

    kv3 = kod[sorted_keys[2]]
    assert kv3 is kod.pop(sorted_keys[2])
    s3 = size(kod)
    assert None is kod.pop(sorted_keys[2], None)
    assert size(kod) == s3
    assert s3 < s2

    kod.clear()
    assert size(kod) < s3

    # Fill it again, this time from the classmethod and without values
    kod = KOD.fromkeys([k for k, _ in kv_pairs], value="foobar")

    # Set with default
    print("\n--- setdefault")
    print(f"Length: {len(kod)}, expected {len(sorted_keys)}")
    print("Keys (expected):", ", ".join([str(k) for k in sorted_keys]))
    print("Keys:           ", ", ".join([str(k) for k in kod]))
    assert sorted_keys[3] in kod
    assert kod.setdefault(sorted_keys[3]) == "foobar"
    assert -42 not in kod
    assert kod.setdefault(-42) is None
    assert kod[-42] is None

    # Exceptions
    with pytest.raises(TypeError, match="expected at most 1 arguments, got"):
        KOD("foo", "bar")

    with pytest.raises(KeyError, match="-123"):
        kod.pop(-123)

    kod._key = lambda k: int(k)
    with pytest.raises(ValueError, match="Could not apply key transformation"):
        kod["foo"] = None

    kod._key = lambda k: None
    with pytest.raises(ValueError, match="Failed comparing 'None'"):
        kod["foo"] = "bar"
