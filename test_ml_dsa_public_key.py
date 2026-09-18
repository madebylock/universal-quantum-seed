# Copyright (c) 2026 Lock.com — PolyForm Shield License 1.0.0

"""ML-DSA verify-after-sign takes the public key from the tr-keyed table.

Re-deriving the public key from the secret key on every signature decodes
s1 and s2 and runs the Python NTT over them; CPython's integer paths differ
for zero, positive and negative coefficients, which a co-resident cache
probe can read. The public key is remembered at keygen (keyed by tr, a
public value), a caller-supplied verify_pk_bytes must bind to the secret
key's tr, and a key this process has never seen is derived once.
"""

import hashlib
import importlib
import sys
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

hybrid_dsa = importlib.import_module("crypto.hybrid_dsa")
ml_dsa = importlib.import_module("crypto.ml_dsa")


def _fixture(label: str, size: int) -> bytes:
    return hashlib.shake_256(label.encode("ascii")).digest(size)


def test_ml_keygen_remembers_public_key_by_tr():
    sk, pk = ml_dsa.ml_keygen(_fixture("ml-dsa pk table seed", 32))

    assert ml_dsa._tr_of_sk(sk) == hashlib.shake_256(pk).digest(64)
    assert ml_dsa._cached_public_key(ml_dsa._tr_of_sk(sk)) == pk
    assert ml_dsa.ml_public_key_for(sk) == pk
    with pytest.raises(ValueError, match="secret key must be"):
        ml_dsa.ml_public_key_for(bytes(sk)[:-1])


@pytest.mark.skipif(not ml_dsa._HAS_PQCRYPTO, reason="pqcrypto not installed")
def test_ml_sign_never_rederives_public_key_from_secret_after_keygen(monkeypatch):
    real_pk_from_sk = ml_dsa._pk_from_sk
    sk, pk = ml_dsa.ml_keygen(_fixture("ml-dsa no rederive seed", 32))
    hsk, hpk = hybrid_dsa.hybrid_dsa_keygen(_fixture("hybrid no rederive seed", 64))
    monkeypatch.setattr(
        ml_dsa,
        "_pk_from_sk",
        lambda _sk: pytest.fail("ML-DSA public key was re-derived from the secret key"),
    )

    for message in (b"first", b"second"):
        sig = ml_dsa.ml_sign(message, sk)
        assert ml_dsa.ml_verify(message, sig, pk)
        hsig = hybrid_dsa.hybrid_dsa_sign(message, hsk, ctx=b"ctx")
        assert hybrid_dsa.hybrid_dsa_verify(message, hsig, hpk, ctx=b"ctx")

    # A key generated elsewhere is derived once, then remembered.
    fresh_sk, fresh_pk = ml_dsa.ml_keygen(_fixture("ml-dsa fresh seed", 32))
    with ml_dsa._PUBLIC_KEYS_LOCK:
        ml_dsa._PUBLIC_KEYS.pop(ml_dsa._tr_of_sk(fresh_sk), None)
    derivations = []

    def _counting_pk_from_sk(secret):
        derivations.append(1)
        return real_pk_from_sk(secret)

    monkeypatch.setattr(ml_dsa, "_pk_from_sk", _counting_pk_from_sk)
    for message in (b"third", b"fourth"):
        sig = ml_dsa.ml_sign(message, fresh_sk)
        assert ml_dsa.ml_verify(message, sig, fresh_pk)
    assert derivations == [1]


def test_ml_sign_rejects_verify_pk_not_bound_to_sk(monkeypatch):
    sk, pk = ml_dsa.ml_keygen(_fixture("ml-dsa bound seed", 32))
    _other_sk, other_pk = ml_dsa.ml_keygen(_fixture("ml-dsa foreign seed", 32))
    flipped = bytearray(pk)
    flipped[100] ^= 0x01
    flipped = bytes(flipped)
    hsk, hpk = hybrid_dsa.hybrid_dsa_keygen(_fixture("hybrid bound seed", 64))
    hed_pk = hpk[:hybrid_dsa._ED25519_PK]

    if ml_dsa._HAS_PQCRYPTO:
        sig = ml_dsa.ml_sign(b"message", sk, verify_pk_bytes=pk)
        assert ml_dsa.ml_verify(b"message", sig, pk)
        hsig = hybrid_dsa.hybrid_dsa_sign(b"message", hsk, verify_pk_bytes=hpk)
        assert hybrid_dsa.hybrid_dsa_verify(b"message", hsig, hpk)

    # Force the pqcrypto branch and make every signer fail loudly: a foreign
    # or tampered public key must be refused before a signature exists.
    monkeypatch.setattr(ml_dsa, "_HAS_PQCRYPTO", True)
    monkeypatch.setattr(
        ml_dsa,
        "_c_dsa_sign",
        lambda *_args: pytest.fail("a signature was produced for an unbound public key"),
        raising=False,
    )
    monkeypatch.setattr(
        ml_dsa,
        "_ml_sign_internal",
        lambda *_args, **_kwargs: pytest.fail(
            "a signature was produced for an unbound public key"
        ),
    )
    for bad_pk in (other_pk, flipped):
        with pytest.raises(RuntimeError, match="not the public key of this secret key"):
            ml_dsa.ml_sign(b"message", sk, verify_pk_bytes=bad_pk)
        with pytest.raises(RuntimeError, match="not the public key of this secret key"):
            hybrid_dsa.hybrid_dsa_sign(b"message", hsk, verify_pk_bytes=hed_pk + bad_pk)
    with pytest.raises(ValueError, match="verify_pk_bytes must be"):
        ml_dsa.ml_sign(b"message", sk, verify_pk_bytes=pk[:-1])
    assert ml_dsa._cached_public_key(ml_dsa._tr_of_sk(sk)) == pk

    monkeypatch.setattr(ml_dsa, "_HAS_PQCRYPTO", False)
    monkeypatch.setenv(ml_dsa.PURE_PYTHON_SECRETS_ENV, "1")
    with pytest.raises(RuntimeError, match="not the public key of this secret key"):
        ml_dsa.ml_sign(b"message", sk, deterministic=True, verify_pk_bytes=other_pk)
