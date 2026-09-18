# Copyright (c) 2026 Lock.com — PolyForm Shield License 1.0.0

"""Pure-Python X25519 Diffie-Hellman key exchange (RFC 7748).

Montgomery curve: y^2 = x^3 + 486662*x^2 + x  over GF(2^255 - 19).

Uses the Montgomery ladder for scalar multiplication (x-coordinate only),
producing a 32-byte shared secret from a 32-byte private key and a 32-byte
public key (u-coordinate).

Sizes:
    Private key: 32 bytes (clamped scalar)
    Public key:  32 bytes (u-coordinate of [sk] * basepoint)
    Shared secret: 32 bytes

Secret operations (keygen, Diffie-Hellman, public-key recovery) require a
native constant-time backend: PyNaCl (libsodium) first, then cryptography
(OpenSSL). Without one they raise RuntimeError instead of degrading.

The pure-Python Montgomery ladder below is branch-free at the Python level
(XOR-mask conditional swaps, no data-dependent branches), but CPython big
integers still leak secret-dependent timing and allocation patterns, so it
is reference code for RFC 7748 test vectors only. Set
UQS_ALLOW_PURE_PYTHON_SECRETS=1 to run it deliberately (see
crypto/native_backend.py).
"""

import hmac

# ── Constant-time backend ──────────────────────────────────────
# pynacl (libsodium) provides constant-time X25519 operations.
# When available, keygen/DH delegate to C for side-channel resistance.
# cryptography (OpenSSL) is the second native backend. Pure Python
# internals are retained for test vectors only.
_HAS_NACL = False
try:
    import nacl.bindings
    _HAS_NACL = True
except ImportError:
    pass

_HAS_CRYPTOGRAPHY_X25519 = False
try:
    from cryptography.hazmat.primitives.asymmetric import x25519 as _crypto_x25519
    from cryptography.hazmat.primitives import serialization as _crypto_serialization
    _HAS_CRYPTOGRAPHY_X25519 = True
except ImportError:
    _crypto_x25519 = None
    _crypto_serialization = None

try:
    from .native_backend import require_native_backend
except ImportError:
    from crypto.native_backend import require_native_backend

# ── Secure memory utilities (libsodium-backed) ────────────────
_HAS_SODIUM = False
try:
    from nacl._sodium import ffi as _ffi, lib as _lib
    _HAS_SODIUM = True
except ImportError:
    pass

# Pure-stdlib ctypes fallback for fast zeroing when PyNaCl is unavailable.
# Single libc memset call — much faster than a Python loop on large buffers
# (Argon2 state, polynomials). Skipped on implementations without ctypes
# (e.g., MicroPython), which fall through to the Python loop below.
_HAS_CTYPES = False
try:
    import ctypes as _ctypes
    _HAS_CTYPES = True
except ImportError:
    pass


def _secure_zero(buf):
    """Securely wipe a mutable buffer (bytearray / memoryview)."""
    if not isinstance(buf, (bytearray, memoryview)):
        return
    n = len(buf)
    if n == 0:
        return
    if _HAS_SODIUM:
        _lib.sodium_memzero(_ffi.from_buffer(buf), n)
    elif _HAS_CTYPES:
        _ctypes.memset(_ctypes.addressof(_ctypes.c_char.from_buffer(buf)), 0, n)
    else:
        for i in range(n):
            buf[i] = 0


_ZERO_32 = b'\x00' * 32

# ── Exported Size Constants ────────────────────────────────────────
X25519_SK_SIZE = 32   # clamped scalar
X25519_PK_SIZE = 32   # u-coordinate of [sk] * basepoint
X25519_SS_SIZE = 32   # shared secret

_P = 2**255 - 19  # Field prime (same as Ed25519)
_A24 = 121665     # (A - 2) / 4 where A = 486662 (RFC 7748 Section 5)
_MASK256 = (1 << 256) - 1  # 256-bit mask for branchless operations


def _ct_cswap_int(a, b, swap):
    """Constant-time conditional swap of two integers.

    If swap=1, returns (b, a). If swap=0, returns (a, b).
    Uses XOR-mask technique: mask = -swap (all-zeros or all-ones),
    then XOR the masked difference into both operands.
    """
    mask = -(swap & 1) & _MASK256
    x = mask & (a ^ b)
    return a ^ x, b ^ x


def _clamp(k_bytes):
    """Apply RFC 7748 scalar clamping."""
    k = bytearray(k_bytes)
    k[0] &= 248
    k[31] &= 127
    k[31] |= 64
    return bytes(k)


def _decode_u(u_bytes):
    """Decode 32-byte little-endian u-coordinate, masking top bit."""
    u = bytearray(u_bytes)
    u[31] &= 127  # Mask bit 255 per RFC 7748
    return int.from_bytes(bytes(u), 'little')


def _encode_u(u):
    """Encode u-coordinate as 32 bytes little-endian."""
    return (u % _P).to_bytes(32, 'little')


def _x25519_raw(k_bytes, u_bytes):
    """Core X25519 scalar multiplication (RFC 7748 Section 5).

    Montgomery ladder operating on projective (X : Z) coordinates.
    Returns the u-coordinate of [k] * (u, ...) as an integer mod p.
    """
    k = int.from_bytes(_clamp(k_bytes), 'little')
    u = _decode_u(u_bytes)

    # Montgomery ladder with projective coordinates
    x_2, z_2 = 1, 0  # Represents point at infinity
    x_3, z_3 = u, 1  # Represents (u, ...)

    # Branchless conditional swap using XOR-mask technique (constant-time).
    swap = 0
    for t in range(254, -1, -1):
        k_t = (k >> t) & 1
        swap ^= k_t
        x_2, x_3 = _ct_cswap_int(x_2, x_3, swap)
        z_2, z_3 = _ct_cswap_int(z_2, z_3, swap)
        swap = k_t

        A = (x_2 + z_2) % _P
        AA = (A * A) % _P
        B = (x_2 - z_2) % _P
        BB = (B * B) % _P
        E = (AA - BB) % _P
        C = (x_3 + z_3) % _P
        D = (x_3 - z_3) % _P
        DA = (D * A) % _P
        CB = (C * B) % _P

        x_3 = pow(DA + CB, 2, _P)
        z_3 = (u * pow(DA - CB, 2, _P)) % _P
        x_2 = (AA * BB) % _P
        z_2 = (E * (AA + _A24 * E)) % _P

    # Final conditional swap (branchless)
    x_2, x_3 = _ct_cswap_int(x_2, x_3, swap)
    z_2, z_3 = _ct_cswap_int(z_2, z_3, swap)

    # Convert from projective: result = x_2 * z_2^(p-2) mod p
    return (x_2 * pow(z_2, _P - 2, _P)) % _P


def x25519_keygen(seed):
    """Generate X25519 keypair from 32-byte seed.

    Args:
        seed: 32-byte random seed (used directly as private scalar after clamping).

    Returns:
        (sk_bytes, pk_bytes) tuple.
        sk_bytes: 32-byte private key (clamped scalar).
        pk_bytes: 32-byte public key (u-coordinate of [sk] * basepoint 9).
    """
    if len(seed) != 32:
        raise ValueError(f"X25519 seed must be 32 bytes, got {len(seed)}")

    # Clamp the seed to produce the private scalar.  _x25519_raw also
    # clamps internally — the double-clamp is intentional: clamping is
    # idempotent, and always clamping in the low-level function provides
    # defence-in-depth if _x25519_raw is ever called with an unclamped
    # scalar from another path.
    sk = _clamp(seed)

    _require_native_backend()
    if _HAS_NACL:
        # libsodium: constant-time scalar * basepoint
        pk = nacl.bindings.crypto_scalarmult_base(sk)
        return sk, pk
    if _HAS_CRYPTOGRAPHY_X25519:
        return sk, _cryptography_public_key_from_private(sk)

    # Reference ladder: reachable only with UQS_ALLOW_PURE_PYTHON_SECRETS=1.
    basepoint = (9).to_bytes(32, 'little')
    u = _x25519_raw(sk, basepoint)
    pk = _encode_u(u)
    return sk, pk


def _require_dh_lengths(sk, pk):
    if len(sk) != 32:
        raise ValueError(f"X25519 sk must be 32 bytes, got {len(sk)}")
    if len(pk) != 32:
        raise ValueError(f"X25519 pk must be 32 bytes, got {len(pk)}")


def _reject_low_order_shared_secret(result):
    if hmac.compare_digest(result, _ZERO_32):
        raise ValueError("X25519: low-order input point (all-zero shared secret)")


def _require_native_backend():
    """Fail closed before a secret scalar reaches the Python ladder."""
    require_native_backend(
        _HAS_NACL or _HAS_CRYPTOGRAPHY_X25519,
        "X25519 secret operations",
        "PyNaCl (libsodium) or cryptography (OpenSSL)",
    )


def _cryptography_public_key_from_private(sk):
    """[sk] * basepoint through OpenSSL; the scalar copy is zeroed after use."""
    sk_buf = bytearray(sk)
    try:
        private_key = _crypto_x25519.X25519PrivateKey.from_private_bytes(sk_buf)
        return private_key.public_key().public_bytes(
            encoding=_crypto_serialization.Encoding.Raw,
            format=_crypto_serialization.PublicFormat.Raw,
        )
    finally:
        _secure_zero(sk_buf)


def _cryptography_exchange(sk, pk):
    """X25519 Diffie-Hellman through OpenSSL; raises on a low-order peer key."""
    sk_buf = bytearray(sk)
    try:
        private_key = _crypto_x25519.X25519PrivateKey.from_private_bytes(sk_buf)
        peer_key = _crypto_x25519.X25519PublicKey.from_public_bytes(bytes(pk))
        return private_key.exchange(peer_key)
    finally:
        _secure_zero(sk_buf)


def x25519(sk, pk):
    """Compute X25519 shared secret.

    Args:
        sk: 32-byte private key.
        pk: 32-byte peer's public key (u-coordinate).

    Returns:
        32-byte shared secret.

    Raises:
        ValueError: If the result is the all-zero point (low-order input).
    """
    _require_dh_lengths(sk, pk)

    _require_native_backend()
    if _HAS_NACL:
        # libsodium: constant-time scalar multiplication
        result = nacl.bindings.crypto_scalarmult(sk, pk)
        _reject_low_order_shared_secret(result)
        return result
    if _HAS_CRYPTOGRAPHY_X25519:
        result = _cryptography_exchange(sk, pk)
        _reject_low_order_shared_secret(result)
        return result

    # Reference ladder: reachable only with UQS_ALLOW_PURE_PYTHON_SECRETS=1.
    u = _x25519_raw(sk, pk)
    result = _encode_u(u)
    _reject_low_order_shared_secret(result)
    return result


def x25519_pk_from_sk(sk):
    """Compute X25519 public key from secret key.

    Constant-time when pynacl (libsodium) is available.
    Used by hybrid_kem to recover the public key during decapsulation.

    Args:
        sk: 32-byte private key (clamped or unclamped).

    Returns:
        32-byte public key (u-coordinate of [sk] * basepoint 9).
    """
    _require_native_backend()
    if _HAS_NACL:
        return nacl.bindings.crypto_scalarmult_base(sk)
    if _HAS_CRYPTOGRAPHY_X25519:
        return _cryptography_public_key_from_private(sk)
    # Reference ladder: reachable only with UQS_ALLOW_PURE_PYTHON_SECRETS=1.
    return _encode_u(_x25519_raw(sk, (9).to_bytes(32, 'little')))


def _x25519_raw_bytes_no_reject(sk, pk):
    """Compute raw X25519 DH for implicit-rejection protocols.

    Always returns 32 bytes. Returns all-zeros for low-order inputs
    instead of raising. Used by hybrid KEM for constant-time decapsulation
    (IND-CCA2: must not branch on validity of the classical component).

    libsodium and OpenSSL both refuse to return an all-zero shared secret
    (the RFC 7748 low-order check) by raising: libsodium as
    nacl.exceptions.RuntimeError, OpenSSL as ValueError. Only that refusal
    is mapped back to the all-zero result the caller expects. Any other
    backend error (a TypeError for a scalar that is not ``bytes``, say)
    propagates, because mapping it to zeros would hand the caller a silent
    wrong secret. The secret scalar is never routed through the Python
    ladder on a native error: a peer who can choose ``pk`` could otherwise
    trigger variable-time arithmetic on demand.
    """
    _require_dh_lengths(sk, pk)
    pk = bytes(pk)  # public value; libsodium's binding accepts bytes only
    if _HAS_NACL:
        try:
            return nacl.bindings.crypto_scalarmult(sk, pk)
        except RuntimeError:
            pass
    if _HAS_CRYPTOGRAPHY_X25519:
        try:
            return _cryptography_exchange(sk, pk)
        except ValueError:
            pass
    if _HAS_NACL or _HAS_CRYPTOGRAPHY_X25519:
        return _ZERO_32
    _require_native_backend()
    # Reference ladder: reachable only with UQS_ALLOW_PURE_PYTHON_SECRETS=1.
    return _encode_u(_x25519_raw(sk, pk))


def _x25519_raw_bytes(sk, pk):
    """Compute raw X25519 DH and reject low-order all-zero outputs."""
    result = _x25519_raw_bytes_no_reject(sk, pk)
    _reject_low_order_shared_secret(result)
    return result


def _x25519_raw_bytes_into(sk, pk, out):
    """Like _x25519_raw_bytes but writes into a mutable bytearray."""
    out[:] = _x25519_raw_bytes(sk, pk)
    return out
