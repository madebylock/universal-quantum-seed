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
