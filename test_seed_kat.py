# Copyright (c) 2026 Lock.com — PolyForm Shield License 1.0.0

import hashlib
import json
import sys
from pathlib import Path

# Support both `pip install -e .` (`import uqs`) and running pytest directly
# from the repo root, where the package is importable by file path.
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

try:
    import uqs
    from uqs import get_fingerprint, get_profile, get_seed, verify_checksum
except ImportError:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "uqs", _PACKAGE_DIR / "__init__.py"
    )
    uqs = importlib.util.module_from_spec(spec)
    sys.modules["uqs"] = uqs
    spec.loader.exec_module(uqs)
    from uqs import get_fingerprint, get_profile, get_seed, verify_checksum


def test_wordlist_integrity_hash_uses_canonical_lf_bytes():
    data = b"\xef\xbb\xbfalpha\r\nbeta\r\n"

    assert hashlib.sha256(uqs._canonical_wordlist_bytes(data)).hexdigest() == (
        hashlib.sha256(b"alpha\nbeta\n").hexdigest()
    )


def test_wordlist_integrity_error_is_import_error_subclass():
    assert issubclass(uqs.WordlistIntegrityError, ImportError)


def test_wordlist_integrity_hash_is_pinned_in_code():
    words_path = Path(uqs.__file__).with_name("words.py")
    sidecar_path = Path(uqs.__file__).with_name("words.py.sha256")
    sidecar_hash = sidecar_path.read_text(encoding="utf-8").split()[0].lower()
    actual_hash = hashlib.sha256(
        uqs._canonical_wordlist_bytes(words_path.read_bytes())
    ).hexdigest()

    assert uqs._TRUSTED_WORDLIST_SHA256 == sidecar_hash
    assert uqs._TRUSTED_WORDLIST_SHA256 == actual_hash


def _canonical_json_bytes(data: bytes) -> bytes:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def test_seed_v1_known_answer_vectors():
    kat_path = Path(__file__).with_name("kat") / "seed_v1.json"
    sidecar_path = kat_path.with_suffix(kat_path.suffix + ".sha256")
    kat_bytes = kat_path.read_bytes()
    actual_hash = hashlib.sha256(_canonical_json_bytes(kat_bytes)).hexdigest()
    sidecar_hash = sidecar_path.read_text(encoding="utf-8").split()[0].lower()

    # Pinning the test vectors against drift: any future edit to
    # seed_v1.json must also update the sidecar.
    assert sidecar_hash == actual_hash, (
        "KAT sidecar hash does not match seed_v1.json contents"
    )

    kat = json.loads(kat_bytes.decode("utf-8"))
    assert kat["version"] == 1
    assert kat["domain"] == "universal-seed-v1"

    for vector in kat["vectors"]:
        indexes = vector["indexes"]

        assert len(indexes) == vector["word_count"], vector["id"]
        # Vectors flagged ``expect_invalid_checksum`` exist to lock the
        # negative path of verify_checksum; they don't have derived seed
        # material in the file.
        if vector.get("expect_invalid_checksum"):
            assert not verify_checksum(indexes), vector["id"]
            continue
        assert verify_checksum(indexes), vector["id"]

        master = get_seed(indexes, vector["passphrase"])
        assert master.hex() == vector["master_seed_hex"], vector["id"]
        assert get_profile(master, "").hex() == vector["default_profile_hex"], vector["id"]
        assert (
            get_profile(master, vector["profile"]).hex()
            == vector["named_profile_hex"]
        ), vector["id"]
        assert (
            get_fingerprint(indexes, vector["passphrase"]) == vector["fingerprint"]
        ), vector["id"]


def test_spec_v1_test_vectors_match_reference_encoding():
    # spec/v1/test-vectors.json is the normative, human-facing vector set
    # (regenerated from kat/seed_v1.json). Pin every vector to the reference
    # encoder/decoder and KDF so the published spec vectors cannot rot.
    vectors_path = Path(__file__).with_name("spec") / "v1" / "test-vectors.json"
    spec_vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    assert spec_vectors["domain"] == "universal-seed-v1"

    decode = uqs.seed._decode_seed_indexes
    encode = uqs.seed._encode_seed_indexes
    for vector in spec_vectors["vectors"]:
        indexes = vector["indexes"]
        word_count = vector["word_count"]

        assert len(indexes) == word_count, vector["id"]
        if vector.get("expect_invalid"):
            assert not verify_checksum(indexes), vector["id"]
            assert decode(indexes) is None, vector["id"]
            continue
        assert verify_checksum(indexes), vector["id"]

        entropy = decode(indexes)
        assert entropy is not None, vector["id"]
        assert entropy.hex() == vector["entropy_hex"], vector["id"]
        assert (
            encode(bytes.fromhex(vector["entropy_hex"]), word_count=word_count)
            == indexes
        ), vector["id"]
        if "master_seed_hex" in vector:
            master = get_seed(indexes, vector["passphrase"])
            assert master.hex() == vector["master_seed_hex"], vector["id"]


def test_spec_v1_quantum_seed_vectors_match_reference_derivation():
    """Pin spec/v1/test-vectors.json quantum_seed_derivation to the code.

    Three of these vectors had silently drifted from get_quantum_seed because
    nothing read the file; every published post-quantum seed must now match
    the derivation exactly (36-word provenance is required by the in-app copy).
    """
    vectors_path = Path(__file__).with_name("spec") / "v1" / "test-vectors.json"
    spec = json.loads(vectors_path.read_bytes().decode("utf-8"))
    assert spec["quantum_seed_derivation"], "no quantum seed vectors"
    for vector in spec["quantum_seed_derivation"]:
        derived = uqs.get_quantum_seed(
            bytes.fromhex(vector["master_key_hex"]),
            vector["algorithm"],
            vector["key_index"],
            _word_count=36,
        )
        assert derived.hex() == vector["seed_hex"], vector["id"]
        assert len(derived) == vector["seed_length"], vector["id"]


def test_seed_v1_quantum_seed_vectors():
    """Pin post-quantum seed derivation to the shared KAT, in lockstep with
    the other implementations: every (master_key, algorithm, key_index) must
    derive exactly the published seed. (Three spec vectors had silently
    drifted because nothing read them.)"""
    kat_path = Path(__file__).with_name("kat") / "seed_v1.json"
    kat = json.loads(kat_path.read_bytes().decode("utf-8"))
    vectors = kat.get("quantum_seed_derivation") or []
    assert vectors, "shared KAT carries no quantum seed vectors"
    for vector in vectors:
        derived = uqs.get_quantum_seed(
            bytes.fromhex(vector["master_key_hex"]),
            vector["algorithm"],
            vector["key_index"],
            _word_count=36,
        )
        assert derived.hex() == vector["seed_hex"], vector["id"]
        assert len(derived) == vector["seed_length"], vector["id"]
