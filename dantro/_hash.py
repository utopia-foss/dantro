"""This module implements a deterministic hash function to use within dantro.

It is mainly used for all things related to the TransformationDAG.
"""

import hashlib

# The full hash length of an MD5 hexdigest
FULL_HASH_LENGTH = 32

# The length of shortened hashes
SHORT_HASH_LENGTH = 8  # For hexdigest, 16^8 = 2^32 > 4*10^9 -> plenty.


def _hash(s: str) -> str:
    """Returns a deterministic hash of the given string.
    
    This uses the hashlib.md5 algorithm which returns a hexadecimal digest of
    length 32.
    
    .. note::
    
        This hash is meant to be used as a checksum, not for security.
    
    Args:
        s (str): The string to create the hash of
    
    Returns:
        str: The 32 character hexadecimal md5 hash digest
    """
    return hashlib.md5(s.encode('utf-8')).hexdigest()
