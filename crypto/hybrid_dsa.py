# Copyright (c) 2026 Lock — MIT License

"""Hybrid Ed25519 + ML-DSA-65 digital signature scheme.

AND-composition: both algorithms must independently verify for the hybrid
signature to be valid. Security holds as long as *either* Ed25519 or
ML-DSA-65 remains unbroken.

Ed25519 provides classical (pre-quantum) security (~128-bit).
ML-DSA-65 provides post-quantum security (NIST Level 3, ~192-bit).

Stripping resistance: BOTH component signatures are domain-separated so
neither can be extracted and used as a valid standalone signature:
    - Ed25519 signs: b"hybrid-dsa-v1" || len(ctx) || ctx || message
    - ML-DSA signs the same domain-prefixed message with empty FIPS context,
      allowing the pqcrypto backend while preserving stripping resistance.

This means ML-DSA signatures produced by the hybrid scheme are NOT valid
standalone ML-DSA-65 signatures on the same (ctx, message) pair.

Context strings are limited to 255 bytes.

Sizes:
    Secret key:  4,096 bytes  (Ed25519 sk 64B + ML-DSA-65 sk 4,032B)
    Public key:  1,984 bytes  (Ed25519 pk 32B + ML-DSA-65 pk 1,952B)
    Signature:   3,373 bytes  (Ed25519 sig 64B + ML-DSA-65 sig 3,309B)

Best-effort constant-time: both component verifications are always
evaluated (no short-circuit on first failure). Component algorithms
(Ed25519, ML-DSA-65) use constant-time scalar multiplication and
branchless arithmetic internally.
"""

from .ed25519 import ed25519_keygen, ed25519_sign, ed25519_verify
from .ml_dsa import ml_keygen, ml_sign, ml_verify, _pk_from_sk as _ml_pk_from_sk

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


def _mlock(buf):
    """Lock memory pages to prevent swapping to disk."""
    if _HAS_SODIUM and isinstance(buf, (bytearray, memoryview)) and len(buf):
        _lib.sodium_mlock(_ffi.from_buffer(buf), len(buf))


def _munlock(buf):
    """Unlock memory pages (also zeros the region)."""
    if _HAS_SODIUM and isinstance(buf, (bytearray, memoryview)) and len(buf):
        _lib.sodium_munlock(_ffi.from_buffer(buf), len(buf))

# Component sizes
_ED25519_SK = 64
_ED25519_PK = 32
_ED25519_SIG = 64
_ML_DSA_SK = 4032
_ML_DSA_PK = 1952
_ML_DSA_SIG = 3309

# Hybrid sizes (exported for external validation)
HYBRID_DSA_SK_SIZE = _ED25519_SK + _ML_DSA_SK    # 4,096
HYBRID_DSA_PK_SIZE = _ED25519_PK + _ML_DSA_PK    # 1,984
HYBRID_DSA_SIG_SIZE = _ED25519_SIG + _ML_DSA_SIG  # 3,373
HYBRID_DSA_COMPONENT_ALGORITHMS = ("Ed25519", "ML-DSA-65")

# Domain prefix to prevent signature stripping attacks
_DOMAIN = b"hybrid-dsa-v1"
HYBRID_DSA_VERSION = 1
SUPPORTED_HYBRID_DSA_VERSIONS = (HYBRID_DSA_VERSION,)
_VERSION_DOMAINS = {HYBRID_DSA_VERSION: _DOMAIN}


def normalize_hybrid_dsa_version(version=HYBRID_DSA_VERSION) -> int:
    """Normalize and validate a hybrid-DSA wire-format version."""
    if version is None or version == "":
        return HYBRID_DSA_VERSION
    if isinstance(version, str):
        raw = version.strip().lower()
        if raw.startswith("v"):
            raw = raw[1:]
        if not raw.isdecimal():
            raise ValueError(f"Unsupported hybrid DSA version: {version!r}")
        version_i = int(raw, 10)
    else:
        version_i = int(version)
    if version_i not in _VERSION_DOMAINS:
        raise ValueError(f"Unsupported hybrid DSA version: {version_i}")
    return version_i


def get_supported_hybrid_dsa_versions() -> tuple[int, ...]:
    """Return supported hybrid-DSA wire-format versions."""
    return SUPPORTED_HYBRID_DSA_VERSIONS


def _domain_for_version(version=HYBRID_DSA_VERSION) -> bytes:
    return _VERSION_DOMAINS[normalize_hybrid_dsa_version(version)]


def _ed25519_message(message, ctx, *, version=HYBRID_DSA_VERSION):
    """Build domain-bound message for the Ed25519 component.

    Format: b"hybrid-dsa-v1" + len(ctx) [1 byte] + ctx + message

    This ensures the Ed25519 signature cannot be stripped from the hybrid
    and presented as a valid standalone Ed25519 signature.
    """
    domain = _domain_for_version(version)
    return domain + len(ctx).to_bytes(1, 'big') + ctx + message


def _ml_dsa_message(message, ctx, *, version=HYBRID_DSA_VERSION):
    """Build the domain-bound message for the ML-DSA component.

    Signs with empty FIPS context so the pqcrypto C backend can sign the
    component.  The domain and caller context are still bound into the signed
    bytes, so the ML-DSA signature remains non-portable outside the hybrid
    scheme.
    """
    domain = _domain_for_version(version)
    return domain + len(ctx).to_bytes(1, "big") + ctx + message


def hybrid_dsa_keygen(seed):
    """Generate hybrid Ed25519 + ML-DSA-65 keypair.

    Args:
        seed: 64-byte seed material.
              First 32 bytes -> Ed25519 keygen.
              Last 32 bytes -> ML-DSA-65 keygen.

    Returns:
        (sk_bytes, pk_bytes) tuple.
        sk_bytes: 4,096-byte hybrid secret key.
        pk_bytes: 1,984-byte hybrid public key.
    """
    if len(seed) != 64:
        raise ValueError(f"Hybrid DSA seed must be 64 bytes, got {len(seed)}")

    seed_view = memoryview(seed)
    ed_seed = bytearray(seed_view[:32])
    ml_seed = bytearray(seed_view[32:])
    _mlock(ed_seed)
    _mlock(ml_seed)
    try:
        ed_sk, ed_pk = ed25519_keygen(ed_seed)
        ml_sk, ml_pk = ml_keygen(ml_seed)

        return ed_sk + ml_sk, ed_pk + ml_pk
    finally:
        _munlock(ed_seed)
        _secure_zero(ed_seed)
        _munlock(ml_seed)
        _secure_zero(ml_seed)


def hybrid_dsa_sign(message, sk_bytes, ctx=b"", *, version=HYBRID_DSA_VERSION,
                    verify_pk_bytes=None):
    """Sign with both Ed25519 and ML-DSA-65.

    Both algorithms sign the message with domain-separated contexts for
    stripping resistance: neither component signature is usable standalone.

    Memory hardening: secret key copies are locked in RAM and securely wiped.
    Fault injection countermeasure: verifies the hybrid signature before returning.
    (Component sign functions also perform their own verify-after-sign internally.)

    Args:
        message: Arbitrary-length message bytes.
        sk_bytes: 4,096-byte hybrid secret key.
        ctx: Context bytes (0-241 bytes, default empty).
             Bound into both Ed25519 and ML-DSA signing contexts.
        version: Hybrid-DSA wire-format version. Version 1 preserves the
             original ``hybrid-dsa-v1`` domain exactly.
        verify_pk_bytes: Optional hybrid public key for verify-after-sign.
             When given, both the ML-DSA component check and the composite
             check verify against this independently-supplied key instead of
             one re-derived from sk_bytes. When omitted, it is derived from
             the secret key.

    Returns:
        3,373-byte hybrid signature.

    Raises:
        RuntimeError: If verify-after-sign detects a fault.
    """
    if len(sk_bytes) != HYBRID_DSA_SK_SIZE:
        raise ValueError(
            f"Hybrid DSA sk must be {HYBRID_DSA_SK_SIZE} bytes, got {len(sk_bytes)}"
        )

    if len(ctx) > 255:
        raise ValueError(
            f"Context string must be 0-255 bytes for hybrid DSA, got {len(ctx)}"
        )
    version = normalize_hybrid_dsa_version(version)
    if verify_pk_bytes is not None and len(verify_pk_bytes) != HYBRID_DSA_PK_SIZE:
        raise ValueError(
            f"verify_pk_bytes must be {HYBRID_DSA_PK_SIZE} bytes, "
            f"got {len(verify_pk_bytes)}"
        )

    # Copy secret keys into mutable buffers for secure wiping
    ed_sk_buf = bytearray(sk_bytes[:_ED25519_SK])
    ml_sk_buf = bytearray(sk_bytes[_ED25519_SK:])
    _mlock(ed_sk_buf)
    _mlock(ml_sk_buf)

    try:
        # Ed25519 signs domain-prefixed message (stripping resistance)
        ed_sig = ed25519_sign(
            _ed25519_message(message, ctx, version=version),
            bytes(ed_sk_buf),
        )

        # ML-DSA signs a domain-prefixed message with empty FIPS context so
        # pqcrypto can provide the production signing backend.
        ml_verify_pk = (
            bytes(verify_pk_bytes[_ED25519_PK:])
            if verify_pk_bytes is not None
            else None
        )
        ml_sig = ml_sign(
            _ml_dsa_message(message, ctx, version=version),
            bytes(ml_sk_buf),
            ctx=b"",
            verify_pk_bytes=ml_verify_pk,
        )

        sig = ed_sig + ml_sig

        # Composite verify-after-sign (fault injection countermeasure)
        # Component functions already verify internally, but this catches
        # faults in the concatenation or in hybrid-level logic.
        if verify_pk_bytes is not None:
            pk_bytes = bytes(verify_pk_bytes)
        else:
            ed_pk = bytes(ed_sk_buf[32:])  # pk is embedded in ed25519 sk
            ml_pk = _ml_pk_from_sk(bytes(ml_sk_buf))
            pk_bytes = ed_pk + ml_pk
        if not hybrid_dsa_verify(message, sig, pk_bytes, ctx=ctx, version=version):
            raise RuntimeError("Hybrid DSA verify-after-sign failed (fault detected)")

        return sig
    finally:
        _munlock(ed_sk_buf)
        _secure_zero(ed_sk_buf)
        _munlock(ml_sk_buf)
        _secure_zero(ml_sk_buf)


def hybrid_dsa_verify(message, sig_bytes, pk_bytes, ctx=b"", *, version=HYBRID_DSA_VERSION):
    """Verify hybrid Ed25519 + ML-DSA-65 signature.

    BOTH component signatures must independently verify. If either
    fails, the hybrid signature is rejected.

    Args:
        message: Arbitrary-length message bytes.
        sig_bytes: 3,373-byte hybrid signature.
        pk_bytes: 1,984-byte hybrid public key.
        ctx: Context bytes (0-241 bytes, must match what was used during signing).
        version: Hybrid-DSA wire-format version. Unsupported versions fail
             closed and return ``False``.

    Returns:
        True only if both Ed25519 AND ML-DSA-65 verify.
    """
    if len(sig_bytes) != HYBRID_DSA_SIG_SIZE:
        return False
    if len(pk_bytes) != HYBRID_DSA_PK_SIZE:
        return False

    if len(ctx) > 255:
        return False
    try:
        version = normalize_hybrid_dsa_version(version)
    except (TypeError, ValueError):
        return False

    ed_sig = sig_bytes[:_ED25519_SIG]
    ml_sig = sig_bytes[_ED25519_SIG:]
    ed_pk = pk_bytes[:_ED25519_PK]
    ml_pk = pk_bytes[_ED25519_PK:]

    # Both must verify — always evaluate both (no short-circuit on first failure)
    ed_ok = ed25519_verify(
        _ed25519_message(message, ctx, version=version),
        ed_sig,
        ed_pk,
    )
    ml_ok = ml_verify(
        _ml_dsa_message(message, ctx, version=version),
        ml_sig,
        ml_pk,
        ctx=b"",
    )
    return ed_ok and ml_ok
