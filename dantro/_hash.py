"""This module implements a deterministic hash function to use within dantro.

It is mainly used for all things related to the TransformationDAG.
"""

import hashlib
from typing import Union, Any

# The length of shortened hashes
SHORT_HASH_LENGTH = 8  # For hexdigest, 16^8 = 2^32 > 4*10^9 -> plenty.

def _hash(s: Union[str, Any]) -> str:
    """Returns a deterministic hash of the given string or object. If the
    given object is not already a string, it is tried to be serialized.

    This uses the hashlib.md5 algorithm which returns a 32 character string.
    Note that the hash is used for a checksum, not for security purposes.
    """
    return hashlib.md5(s.encode('utf-8')).hexdigest()
