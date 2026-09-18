# Copyright (c) 2026 Lock.com — PolyForm Shield License 1.0.0

"""Secret operations fail closed without their native constant-time backend.

Pins the policy in crypto/native_backend.py: with the native package
unavailable and UQS_ALLOW_PURE_PYTHON_SECRETS unset, every operation that
touches a secret raises RuntimeError before the secret reaches pure-Python
arithmetic. Verification of public data keeps working, and the override
still runs the pure-Python reference against its known-answer vectors.
"""

import hashlib
import importlib
import sys
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

# crypto/__init__.py re-exports functions named like their modules
# (crypto.x25519 is the DH function), so fetch the real modules.
aes_gcm = importlib.import_module("crypto.aes_gcm")
argon2 = importlib.import_module("crypto.argon2")
ed25519 = importlib.import_module("crypto.ed25519")
hybrid_dsa = importlib.import_module("crypto.hybrid_dsa")
hybrid_kem = importlib.import_module("crypto.hybrid_kem")
ml_dsa = importlib.import_module("crypto.ml_dsa")
ml_kem = importlib.import_module("crypto.ml_kem")
x25519 = importlib.import_module("crypto.x25519")
from crypto.native_backend import (  # noqa: E402
    PURE_PYTHON_SECRETS_ENV,
    pure_python_secrets_allowed,
    require_native_backend,
)

_BASEPOINT = (9).to_bytes(32, "little")

# RFC 7748 Section 6.1
_ALICE_SK = bytes.fromhex(
    "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
_ALICE_PK = bytes.fromhex(
    "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")
_BOB_SK = bytes.fromhex(
    "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
_BOB_PK = bytes.fromhex(
    "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")
_X25519_SHARED = bytes.fromhex(
    "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")

# RFC 8032 Section 7.1, test 1
_ED_SEED = bytes.fromhex(
    "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
_ED_PK = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
_ED_SIG = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")

# NIST SP 800-38D test case 14 (AES-256)
_GCM_KEY = bytes(32)
_GCM_NONCE = bytes(12)
_GCM_PT = bytes(16)
_GCM_CT = bytes.fromhex(
    "cea7403d4d606b6e074ec5d3baf39d18d0d1c8a799996bf0265b98b5d48ab919")

# argon2-cffi reference vector (minimal parameters)
_ARGON2_VECTOR = dict(
    password=b"\x01" * 32, salt=b"\x02" * 16, time_cost=1, memory_cost=8,
    parallelism=1, hash_len=32,
    expected="27094ba97a2f6ad140cfbb1ffb6c8f2508b68f5e32272544c77cdd7b6994faef",
)


def _fixture(label: str, size: int) -> bytes:
    return hashlib.shake_256(label.encode("ascii")).digest(size)


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.delenv(PURE_PYTHON_SECRETS_ENV, raising=False)


@pytest.fixture
def override(monkeypatch):
    monkeypatch.setenv(PURE_PYTHON_SECRETS_ENV, "1")


@pytest.fixture
def no_x25519_native(monkeypatch):
    monkeypatch.setattr(x25519, "_HAS_NACL", False)
    monkeypatch.setattr(x25519, "_HAS_CRYPTOGRAPHY_X25519", False)


@pytest.fixture
def no_ed25519_native(monkeypatch):
    monkeypatch.setattr(ed25519, "_HAS_NACL", False)


@pytest.fixture
def no_pqcrypto(monkeypatch):
    monkeypatch.setattr(ml_kem, "_HAS_PQCRYPTO", False)
    monkeypatch.setattr(ml_dsa, "_HAS_PQCRYPTO", False)


# ── policy ─────────────────────────────────────────────────────


@pytest.mark.parametrize("value,allowed", [
    ("1", True), ("true", True), ("YES", True), (" on ", True),
    ("0", False), ("", False), ("false", False), ("maybe", False),
])
def test_override_parsing(monkeypatch, value, allowed):
    monkeypatch.setenv(PURE_PYTHON_SECRETS_ENV, value)
    assert pure_python_secrets_allowed() is allowed


def test_require_native_backend_message_names_operation_and_package(no_override):
    require_native_backend(True, "op", "pkg")
    with pytest.raises(RuntimeError) as info:
        require_native_backend(False, "X25519 secret operations", "PyNaCl")
    text = str(info.value)
    assert text.startswith("X25519 secret operations requires PyNaCl")
    assert PURE_PYTHON_SECRETS_ENV in text


# ── X25519 ─────────────────────────────────────────────────────


def test_x25519_secret_operations_fail_closed(no_override, no_x25519_native, monkeypatch):
    # The reference ladder must never run when the gate refuses.
    monkeypatch.setattr(x25519, "_x25519_raw", lambda *a: pytest.fail("ladder reached"))
    with pytest.raises(RuntimeError, match="X25519 secret operations require"):
        x25519.x25519_keygen(_fixture("x25519 seed", 32))
    with pytest.raises(RuntimeError, match="X25519 secret operations require"):
        x25519.x25519(_fixture("x25519 sk", 32), _BASEPOINT)
    with pytest.raises(RuntimeError, match="X25519 secret operations require"):
        x25519.x25519_pk_from_sk(_fixture("x25519 pk sk", 32))
    with pytest.raises(RuntimeError, match="X25519 secret operations require"):
        x25519._x25519_raw_bytes_no_reject(_fixture("x25519 raw sk", 32), _BASEPOINT)
    with pytest.raises(RuntimeError, match="X25519 secret operations require"):
        hybrid_kem.hybrid_kem_keygen(_fixture("hybrid kem seed", 96))


def test_x25519_override_runs_reference_ladder(override, no_x25519_native):
    alice_sk, alice_pk = x25519.x25519_keygen(_ALICE_SK)
    bob_sk, bob_pk = x25519.x25519_keygen(_BOB_SK)
    assert alice_pk == _ALICE_PK
    assert bob_pk == _BOB_PK
    assert x25519.x25519(alice_sk, bob_pk) == _X25519_SHARED
    assert x25519.x25519_pk_from_sk(bob_sk) == _BOB_PK


@pytest.mark.skipif(not x25519._HAS_NACL, reason="PyNaCl not installed")
@pytest.mark.skipif(not x25519._HAS_CRYPTOGRAPHY_X25519, reason="cryptography not installed")
def test_x25519_openssl_backend_matches_libsodium(no_override, monkeypatch):
    sk_nacl, pk_nacl = x25519.x25519_keygen(_ALICE_SK)
    shared_nacl = x25519.x25519(sk_nacl, _BOB_PK)
    monkeypatch.setattr(x25519, "_HAS_NACL", False)
    monkeypatch.setattr(x25519, "_x25519_raw", lambda *a: pytest.fail("ladder reached"))
    sk_ssl, pk_ssl = x25519.x25519_keygen(_ALICE_SK)
    assert (sk_ssl, pk_ssl) == (sk_nacl, pk_nacl) == (x25519._clamp(_ALICE_SK), _ALICE_PK)
    assert x25519.x25519(sk_ssl, _BOB_PK) == shared_nacl == _X25519_SHARED
    assert x25519.x25519_pk_from_sk(sk_ssl) == _ALICE_PK
    assert x25519._x25519_raw_bytes_no_reject(sk_ssl, _BOB_PK) == _X25519_SHARED


@pytest.mark.parametrize("backend", ["libsodium", "openssl"])
def test_x25519_native_low_order_rejection_never_reaches_python(
        no_override, monkeypatch, backend):
    if backend == "libsodium":
        if not x25519._HAS_NACL:
            pytest.skip("PyNaCl not installed")
        monkeypatch.setattr(x25519, "_HAS_CRYPTOGRAPHY_X25519", False)
    else:
        if not x25519._HAS_CRYPTOGRAPHY_X25519:
            pytest.skip("cryptography not installed")
        monkeypatch.setattr(x25519, "_HAS_NACL", False)
    monkeypatch.setattr(x25519, "_x25519_raw", lambda *a: pytest.fail("ladder reached"))
    sk = x25519._clamp(_fixture("x25519 low order sk", 32))
    low_order_pk = bytes(32)
    assert x25519._x25519_raw_bytes_no_reject(sk, low_order_pk) == bytes(32)
    with pytest.raises(ValueError, match="low-order"):
        x25519._x25519_raw_bytes(sk, low_order_pk)
    with pytest.raises(Exception):
        x25519.x25519(sk, low_order_pk)


@pytest.mark.skipif(not x25519._HAS_NACL, reason="PyNaCl not installed")
def test_x25519_no_reject_takes_buffer_public_keys_on_libsodium_alone(no_override, monkeypatch):
    # libsodium's binding accepts bytes only. A bytearray or memoryview
    # public key (a ciphertext slice) must still yield the real shared
    # secret, never a silent all-zero result that hybrid decapsulation
    # would turn into the implicit-rejection secret.
    monkeypatch.setattr(x25519, "_HAS_CRYPTOGRAPHY_X25519", False)
    monkeypatch.setattr(x25519, "_x25519_raw", lambda *a: pytest.fail("ladder reached"))
    sk = x25519._clamp(_ALICE_SK)
    assert x25519._x25519_raw_bytes_no_reject(sk, bytearray(_BOB_PK)) == _X25519_SHARED
    assert x25519._x25519_raw_bytes_no_reject(sk, memoryview(_BOB_PK)) == _X25519_SHARED
    if ml_kem._HAS_PQCRYPTO:
        ek, dk = hybrid_kem.hybrid_kem_keygen(_fixture("hybrid kem buffer seed", 96))
        ct, ss = hybrid_kem.hybrid_kem_encaps(ek)
        assert hybrid_kem.hybrid_kem_decaps(dk, bytearray(ct)) == ss


@pytest.mark.skipif(not x25519._HAS_NACL, reason="PyNaCl not installed")
def test_x25519_no_reject_propagates_unexpected_backend_errors(no_override, monkeypatch):
    import nacl.bindings

    # Only the low-order refusal maps to zeros; any other backend failure
    # must surface instead of becoming a silent wrong secret.
    monkeypatch.setattr(x25519, "_HAS_CRYPTOGRAPHY_X25519", False)
    monkeypatch.setattr(x25519, "_x25519_raw", lambda *a: pytest.fail("ladder reached"))
    sk = x25519._clamp(_ALICE_SK)
    with pytest.raises(TypeError):
        x25519._x25519_raw_bytes_no_reject(bytearray(sk), _BOB_PK)

    def _broken(*_args):
        raise MemoryError("backend failure")

    monkeypatch.setattr(nacl.bindings, "crypto_scalarmult", _broken)
    with pytest.raises(MemoryError):
        x25519._x25519_raw_bytes_no_reject(sk, _BOB_PK)


# ── Ed25519 ────────────────────────────────────────────────────


def test_ed25519_secret_operations_fail_closed(no_override, no_ed25519_native, monkeypatch):
    monkeypatch.setattr(
        ed25519, "_scalar_mult_base", lambda *a: pytest.fail("scalar ladder reached"))
    with pytest.raises(RuntimeError, match="Ed25519 secret operations require"):
        ed25519.ed25519_keygen(_ED_SEED)
    with pytest.raises(RuntimeError, match="Ed25519 secret operations require"):
        ed25519._public_key_from_seed(_ED_SEED)
    with pytest.raises(RuntimeError, match="Ed25519 secret operations require"):
        ed25519.ed25519_sign(b"", _ED_SEED + _ED_PK)
    with pytest.raises(RuntimeError, match="Ed25519 secret operations require"):
        hybrid_dsa.hybrid_dsa_keygen(_fixture("hybrid dsa seed", 64))


def test_ed25519_verify_stays_available_without_native(no_override, no_ed25519_native):
    assert ed25519.ed25519_verify(b"", _ED_SIG, _ED_PK)
    assert not ed25519.ed25519_verify(b"x", _ED_SIG, _ED_PK)


def test_ed25519_override_runs_reference_vectors(override, no_ed25519_native):
    sk, pk = ed25519.ed25519_keygen(_ED_SEED)
    assert pk == _ED_PK
    assert ed25519.ed25519_sign(b"", sk) == _ED_SIG


# ── AES-GCM ────────────────────────────────────────────────────


def test_aes_gcm_fails_closed(no_override, monkeypatch):
    monkeypatch.setattr(aes_gcm, "_HAS_CRYPTO", False)
    monkeypatch.setattr(aes_gcm, "_key_expansion", lambda *a: pytest.fail("AES reached"))
    with pytest.raises(RuntimeError, match="AES-GCM requires the cryptography package"):
        aes_gcm.aes_gcm_encrypt(_GCM_KEY, _GCM_NONCE, _GCM_PT)
    with pytest.raises(RuntimeError, match="AES-GCM requires the cryptography package"):
        aes_gcm.aes_gcm_decrypt(_GCM_KEY, _GCM_NONCE, _GCM_CT)


def test_aes_gcm_override_runs_reference_vector(override, monkeypatch):
    monkeypatch.setattr(aes_gcm, "_HAS_CRYPTO", False)
    assert aes_gcm.aes_gcm_encrypt(_GCM_KEY, _GCM_NONCE, _GCM_PT) == _GCM_CT
    assert aes_gcm.aes_gcm_decrypt(_GCM_KEY, _GCM_NONCE, _GCM_CT) == _GCM_PT


# ── Argon2id ───────────────────────────────────────────────────


def test_argon2_fails_closed(no_override, monkeypatch):
    monkeypatch.setattr(argon2, "_cffi_hash", None)
    monkeypatch.setattr(argon2, "_cffi_lib", None)
    monkeypatch.setattr(argon2, "_argon2id_pure", lambda *a: pytest.fail("pure Argon2 reached"))
    vec = dict(_ARGON2_VECTOR)
    vec.pop("expected")
    with pytest.raises(RuntimeError, match="Argon2id key derivation requires argon2-cffi"):
        argon2.hash_secret_raw(secret=vec.pop("password"), type=2, **vec)
    vec = dict(_ARGON2_VECTOR)
    vec.pop("expected")
    with pytest.raises(RuntimeError, match="Argon2id key derivation requires argon2-cffi"):
        argon2.argon2id(*vec.values())


def test_argon2_override_runs_reference_vector(override, monkeypatch):
    monkeypatch.setattr(argon2, "_cffi_hash", None)
    monkeypatch.setattr(argon2, "_cffi_lib", None)
    vec = dict(_ARGON2_VECTOR)
    expected = vec.pop("expected")
    assert argon2.argon2id(*vec.values()).hex() == expected


# ── ML-KEM-768 ─────────────────────────────────────────────────


def test_ml_kem_gated_operations_fail_closed(no_override, no_pqcrypto, monkeypatch):
    ek, dk = ml_kem.ml_kem_keygen(_fixture("ml-kem seed", 64))  # seeded keygen stays available
    monkeypatch.setattr(ml_kem, "_k_pke_encrypt", lambda *a: pytest.fail("encrypt reached"))
    monkeypatch.setattr(ml_kem, "_k_pke_decrypt", lambda *a: pytest.fail("decrypt reached"))
    with pytest.raises(RuntimeError, match="ML-KEM decapsulation requires the pqcrypto package"):
        ml_kem.ml_kem_decaps(dk, bytes(ml_kem.ML_KEM_CT_SIZE))
    with pytest.raises(RuntimeError, match="ML-KEM encapsulation requires the pqcrypto package"):
        ml_kem.ml_kem_encaps(ek)
    # Caller-supplied randomness is the shared secret's preimage: gated too.
    with pytest.raises(RuntimeError, match="ML-KEM encapsulation requires the pqcrypto package"):
        ml_kem.ml_kem_encaps(ek, randomness=_fixture("ml-kem m", 32))
    with pytest.raises(ValueError, match="randomness must be 32 bytes"):
        ml_kem.ml_kem_encaps(ek, randomness=b"short")
    with pytest.raises(RuntimeError, match="ML-KEM random key generation requires"):
        ml_kem.ml_kem_keygen()
    hybrid_ek, hybrid_dk = hybrid_kem.hybrid_kem_keygen(_fixture("hybrid kem seed 2", 96))
    with pytest.raises(RuntimeError, match="requires the pqcrypto package"):
        hybrid_kem.hybrid_kem_decaps(hybrid_dk, bytes(hybrid_kem.HYBRID_KEM_CT_SIZE))


def test_ml_kem_override_round_trips(override, no_pqcrypto):
    ek, dk = ml_kem.ml_kem_keygen(_fixture("ml-kem seed", 64))
    ct, ss = ml_kem.ml_kem_encaps(ek, randomness=_fixture("ml-kem m", 32))
    assert ml_kem.ml_kem_decaps(dk, ct) == ss


@pytest.mark.skipif(not ml_kem._HAS_PQCRYPTO, reason="pqcrypto not installed")
def test_ml_kem_caller_randomness_refused_with_pqcrypto(no_override, monkeypatch):
    # pqcrypto draws its own randomness, so explicit randomness could only
    # run the pure-Python reference: refused before m reaches it.
    monkeypatch.setattr(ml_kem, "_k_pke_encrypt", lambda *a: pytest.fail("encrypt reached"))
    ek, dk = ml_kem.ml_kem_keygen(_fixture("ml-kem seed", 64))
    with pytest.raises(RuntimeError, match="cannot use the pqcrypto backend"):
        ml_kem.ml_kem_encaps(ek, randomness=_fixture("ml-kem m", 32))
    hybrid_ek, _hybrid_dk = hybrid_kem.hybrid_kem_keygen(_fixture("hybrid kem seed 3", 96))
    with pytest.raises(RuntimeError, match="cannot use the pqcrypto backend"):
        hybrid_kem.hybrid_kem_encaps(hybrid_ek, randomness=_fixture("hybrid m", 64))
    ct, ss = ml_kem.ml_kem_encaps(ek)  # fresh randomness: pqcrypto path
    assert ml_kem.ml_kem_decaps(dk, ct) == ss


# ── ML-DSA-65 ──────────────────────────────────────────────────


def test_ml_dsa_signing_fails_closed(no_override, no_pqcrypto, monkeypatch):
    sk, pk = ml_dsa.ml_keygen(_fixture("ml-dsa seed", 32))  # seeded keygen stays available
    monkeypatch.setattr(ml_dsa, "_ml_sign_internal", lambda *a, **k: pytest.fail("signer reached"))
    with pytest.raises(RuntimeError, match="ML-DSA signing requires the pqcrypto package"):
        ml_dsa.ml_sign(b"message", sk)
    with pytest.raises(RuntimeError, match="ML-DSA signing requires the pqcrypto package"):
        ml_dsa.ml_sign(b"message", sk, ctx=b"ctx")
    with pytest.raises(RuntimeError, match="ML-DSA signing requires the pqcrypto package"):
        ml_dsa.ml_sign(b"message", sk, deterministic=True)


@pytest.mark.skipif(not ml_dsa._HAS_PQCRYPTO, reason="pqcrypto not installed")
def test_ml_dsa_modes_pqcrypto_cannot_serve_are_refused(no_override, monkeypatch):
    sk, pk = ml_dsa.ml_keygen(_fixture("ml-dsa seed", 32))
    monkeypatch.setattr(ml_dsa, "_ml_sign_internal", lambda *a, **k: pytest.fail("signer reached"))
    with pytest.raises(RuntimeError, match="cannot use the pqcrypto backend"):
        ml_dsa.ml_sign(b"message", sk, ctx=b"ctx")
    with pytest.raises(RuntimeError, match="cannot use the pqcrypto backend"):
        ml_dsa.ml_sign(b"message", sk, deterministic=True)
    with pytest.raises(RuntimeError, match="cannot use the pqcrypto backend"):
        ml_dsa.ml_sign(b"message", sk, rnd=bytes(32))
    sig = ml_dsa.ml_sign(b"message", sk)  # empty context: pqcrypto path
    assert ml_dsa.ml_verify(b"message", sig, pk)


# ── seed.py integration ────────────────────────────────────────


def test_generate_quantum_keypair_surfaces_the_gate(no_override, no_ed25519_native):
    import seed

    master_key = bytes(range(64))
    with pytest.raises(RuntimeError, match="Ed25519 secret operations require"):
        seed.generate_quantum_keypair(master_key, "hybrid-dsa-65")
