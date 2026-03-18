"""
Secure memory wiping for Python objects containing sensitive cryptographic material.

Python's immutable types (int, bytes, str) cannot be zeroed through normal means —
the garbage collector eventually frees the memory, but the secret data lingers in
the heap until overwritten by unrelated allocations. This module uses ctypes.memset
to directly zero the underlying C memory of CPython objects.

Usage:
    from crypto.secure_wipe import wipe, wipe_all

    try:
        scalar = int.from_bytes(private_key, 'big')
        raw = some_crypto_operation(scalar)
        ...
    finally:
        wipe(scalar)
        wipe(raw)

    # Or wipe multiple at once:
    finally:
        wipe_all(scalar, raw, intermediate_bytes)

    # For lists of bytearrays (e.g., Argon2 memory blocks):
    finally:
        wipe_list(memory_blocks)

Limitations:
    - CPython only (silently no-ops on PyPy, GraalPy, etc.)
    - Small ints (-5 to 256) are cached singletons — wiping them would corrupt the
      interpreter, so they are skipped. Private keys should never be this small.
    - Copies may exist in CPU registers, compiler temporaries, or Python's internal
      free lists. This eliminates the primary threat (long-lived heap objects).
    - Objects must not be referenced elsewhere after wiping — the object becomes
      a zeroed husk that will crash or return wrong values if used again.
"""

import ctypes
import sys

# Detect CPython — ctypes.memset on id() only works on CPython
_IS_CPYTHON = hasattr(sys, "getrefcount")

# Pre-compute header sizes (stable across a single CPython version)
_INT_HEADER = sys.getsizeof(0) if _IS_CPYTHON else 0
_BYTES_HEADER = (sys.getsizeof(b"") - 1) if _IS_CPYTHON else 0  # -1 for null terminator


def wipe(obj) -> None:
    """Best-effort secure wipe of a Python int, bytes, bytearray, or str.

    For bytearray: zeros every byte in-place (always works, any Python).
    For bytes: zeros the internal buffer via ctypes (CPython only).
    For int: zeros the digit array via ctypes (CPython only).
    For str: zeros the character data via ctypes (CPython only).

    Safe to call on None, 0, empty objects, or non-CPython — silently no-ops.
    """
    if obj is None:
        return

    if isinstance(obj, bytearray):
        for i in range(len(obj)):
            obj[i] = 0
        return

    if not _IS_CPYTHON:
        return

    try:
        if isinstance(obj, int):
            # Skip cached small ints (singletons in CPython)
            if -5 <= obj <= 256:
                return
            size = sys.getsizeof(obj) - _INT_HEADER
            if size > 0:
                ctypes.memset(id(obj) + _INT_HEADER, 0, size)

        elif isinstance(obj, bytes):
            if not obj:
                return
            ctypes.memset(id(obj) + _BYTES_HEADER, 0, len(obj))

        elif isinstance(obj, str):
            if not obj:
                return
            # CPython str: compact ASCII uses 1 byte/char, UCS-1 = 1, UCS-2 = 2, UCS-4 = 4
            # sys.getsizeof(s) - sys.getsizeof("") gives the total data bytes
            data_bytes = sys.getsizeof(obj) - sys.getsizeof("")
            if data_bytes > 0:
                ctypes.memset(id(obj) + sys.getsizeof(""), 0, data_bytes)

    except Exception:
        pass  # Non-CPython, or layout changed — fail silently


def wipe_all(*objs) -> None:
    """Wipe multiple objects. Convenience wrapper for finally blocks.

    Usage:
        finally:
            wipe_all(scalar, key_bytes, seed_int, hex_str)
    """
    for obj in objs:
        wipe(obj)


def wipe_list(lst) -> None:
    """Wipe all elements in a list, then clear the list.

    Handles lists of bytearrays (e.g., Argon2 memory blocks),
    lists of ints (e.g., polynomial coefficients), or mixed types.

    Usage:
        memory = [bytearray(1024) for _ in range(blocks)]
        # ... use memory ...
        wipe_list(memory)
    """
    if lst is None:
        return
    try:
        for item in lst:
            if isinstance(item, (list, bytearray)):
                # Nested list (e.g., Argon2 block = list of ints)
                if isinstance(item, list):
                    for i in range(len(item)):
                        wipe(item[i])
                        item[i] = 0
                else:
                    wipe(item)
            else:
                wipe(item)
        lst.clear()
    except Exception:
        pass
