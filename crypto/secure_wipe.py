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
    - Immutable-object wiping requires a validated CPython object layout.  On
      PyPy, GraalPy, or an unrecognized CPython layout, bytearray wiping still
      works but bytes/int/str wipe attempts fail closed with
      SecureWipeUnsupportedError instead of silently no-oping.
    - Small ints (-5 to 256) are cached singletons — wiping them would corrupt the
      interpreter, so they are skipped. Private keys should never be this small.
    - Copies may exist in CPU registers, compiler temporaries, or Python's internal
      free lists. This eliminates the primary threat (long-lived heap objects).
    - Immutable objects are wiped only when their refcount indicates there are
      no aliases beyond the active call path.  Shared/interned/cached objects
      are skipped instead of corrupting unrelated runtime state.
    - Objects must not be referenced elsewhere after wiping — the object becomes
      a zeroed husk that will crash or return wrong values if used again.
"""

import ctypes
import sys

# Detect CPython — ctypes.memset on id() only works on CPython
_IS_CPYTHON = hasattr(sys, "getrefcount")

# Pre-compute data offsets (stable across a single CPython version)
_INT_HEADER = 0
_BYTES_HEADER = (sys.getsizeof(b"") - 1) if _IS_CPYTHON else 0  # -1 for null terminator

# Refcount ceiling: skip immutable objects likely interned/cached by the runtime.
# getrefcount(obj) itself adds 1, the caller's variable adds 1, and
# wipe()'s parameter adds 1 = 3 baseline.  Allow 1 extra for transient
# references (e.g., passed through a wrapper).  Anything above 4 almost
# certainly means the runtime has cached/interned the object.
_REFCOUNT_MAX = 4

# ── Import-time layout validation ────────────────────────────────────
#
# Create canary objects, wipe them via our computed offsets, and verify
# the data was actually zeroed.  If the offsets are wrong (e.g., a new
# CPython version changed the struct layout) we disable ctypes wiping
# entirely rather than risk writing to wrong memory.

_LAYOUT_OK = False

if _IS_CPYTHON:
    try:
        # -- int canary: 0xDEAD is outside the small-int cache
        _canary_int = int.from_bytes(b"\xde\xad", "big")
        _int_mem = ctypes.string_at(id(_canary_int), sys.getsizeof(_canary_int))
        _digit_offset = _int_mem.find((0xDEAD).to_bytes(4, "little"))
        if _digit_offset < 0:
            raise RuntimeError("could not locate PyLong digit offset")
        _INT_HEADER = _digit_offset
        _ci_size = sys.getsizeof(_canary_int) - _INT_HEADER
        if _ci_size > 0:
            ctypes.memset(id(_canary_int) + _INT_HEADER, 0, _ci_size)
            if _canary_int == 0:
                # -- bytes canary: len > 1 to avoid interned singletons
                _canary_bytes = bytes(b"\xab\xcd\xef")
                ctypes.memset(id(_canary_bytes) + _BYTES_HEADER, 0, len(_canary_bytes))
                if _canary_bytes == b"\x00\x00\x00":
                    _LAYOUT_OK = True
        del _canary_int, _canary_bytes, _ci_size, _int_mem, _digit_offset
    except Exception:
        _LAYOUT_OK = False


class SecureWipeUnsupportedError(RuntimeError):
    """Raised when immutable-object wiping cannot be performed safely."""


def secure_wipe_supported() -> bool:
    """Return True when immutable bytes/int/str wiping is available."""
    return bool(_IS_CPYTHON and _LAYOUT_OK)


def _require_immutable_wipe_supported(kind: str) -> None:
    if secure_wipe_supported():
        return
    runtime = "CPython" if _IS_CPYTHON else sys.implementation.name
    raise SecureWipeUnsupportedError(
        f"secure wipe for {kind} requires a validated CPython memory layout; "
        f"current runtime/layout is unsupported ({runtime})"
    )


def wipe(obj) -> bool:
    """Best-effort secure wipe of a Python int, bytes, bytearray, or str.

    For bytearray: zeros every byte in-place (always works, any Python).
    For bytes: zeros the internal buffer via ctypes (CPython only).
    For int: zeros the digit array via ctypes (CPython only).
    For str: zeros the character data via ctypes (CPython only).

    Returns True if the object was wiped, False if skipped because it is empty,
    cached, unsupported, or has no known wipe strategy.  Raises
    SecureWipeUnsupportedError when asked to wipe an immutable secret on an
    interpreter/layout where that cannot be done safely.
    """
    if obj is None:
        return False

    if isinstance(obj, bytearray):
        for i in range(len(obj)):
            obj[i] = 0
        return True

    if isinstance(obj, int):
        if -5 <= obj <= 256:
            return False
        _require_immutable_wipe_supported("int")
        refcount = sys.getrefcount(obj)
        if refcount > _REFCOUNT_MAX:
            import logging
            logging.getLogger(__name__).debug(
                "wipe(): int object has refcount %d > %d, cannot wipe safely",
                refcount, _REFCOUNT_MAX)
            return False
        size = sys.getsizeof(obj) - _INT_HEADER
        if size > 0:
            ctypes.memset(id(obj) + _INT_HEADER, 0, size)
            return True
        return False

    if isinstance(obj, bytes):
        if len(obj) <= 1:
            return False
        _require_immutable_wipe_supported("bytes")
        refcount = sys.getrefcount(obj)
        if refcount > _REFCOUNT_MAX:
            import logging
            logging.getLogger(__name__).debug(
                "wipe(): bytes object (len=%d) has refcount %d > %d, cannot wipe safely",
                len(obj), refcount, _REFCOUNT_MAX)
            return False
        ctypes.memset(id(obj) + _BYTES_HEADER, 0, len(obj))
        return True

    if isinstance(obj, str):
        if not obj:
            return False
        _require_immutable_wipe_supported("str")
        refcount = sys.getrefcount(obj)
        if refcount > _REFCOUNT_MAX:
            import logging
            logging.getLogger(__name__).debug(
                "wipe(): str object (len=%d) has refcount %d > %d, cannot wipe safely",
                len(obj), refcount, _REFCOUNT_MAX)
            return False
        data_bytes = sys.getsizeof(obj) - sys.getsizeof("")
        if data_bytes > 0:
            ctypes.memset(id(obj) + sys.getsizeof(""), 0, data_bytes)
            return True
        return False

    return False


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
    for item in lst:
        if isinstance(item, list):
            for i in range(len(item)):
                wipe(item[i])
                item[i] = 0
        else:
            wipe(item)
    lst.clear()
