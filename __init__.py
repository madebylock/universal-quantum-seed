# Copyright (c) 2026 Lock — MIT License

"""Universal Quantum Seed — 256 visual icons, 42 languages, 272-bit entropy.

Drop-in package: copy or symlink this folder as `modules/seed/` in lock
and all existing imports work unchanged.
"""

import hashlib
from pathlib import Path

_TRUSTED_WORDLIST_SHA256 = (
    "f24ab526b5484934d766738421eb4ea6a8199c4a69d6c450aca8a4ab48f9bc46"
)


def _canonical_wordlist_bytes(data: bytes) -> bytes:
    # Git stores this generated Python wordlist as LF text, while some Windows
    # checkouts expand it to CRLF. Hash the canonical repository form.
    return data.replace(b"\r\n", b"\n")


def _verify_wordlist_integrity():
    here = Path(__file__).resolve().parent
    words_path = here / "words.py"
    digest_path = here / "words.py.sha256"
    expected = _TRUSTED_WORDLIST_SHA256
    expected_line = digest_path.read_text(encoding="utf-8").strip()
    sidecar_expected = expected_line.split()[0].lower()
    if sidecar_expected != expected:
        raise ImportError(
            "UQS wordlist integrity check failed: sidecar hash is not trusted")
    actual = hashlib.sha256(
        _canonical_wordlist_bytes(words_path.read_bytes())
    ).hexdigest()
    if actual.lower() != expected:
        raise ImportError(
            "UQS wordlist integrity check failed: words.py hash mismatch")


_verify_wordlist_integrity()

try:
    from .seed import (
        generate_words,
        generate_seed,
        get_seed,
        get_profile,
        get_quantum_seed,
        generate_quantum_keypair,
        get_fingerprint,
        get_entropy_bits,
        verify_checksum,
        validate_seed,
        _compute_checksum,
        resolve,
        search,
        verify_randomness,
        mouse_entropy,
        kdf_info,
        get_languages,
        canonical_word,
        normalize_seed_version,
        get_supported_versions,
        UQS_VERSION,
        SUPPORTED_UQS_VERSIONS,
        DARK_VISUALS,
    )
except ImportError:
    from seed import (
        generate_words,
        generate_seed,
        get_seed,
        get_profile,
        get_quantum_seed,
        generate_quantum_keypair,
        get_fingerprint,
        get_entropy_bits,
        verify_checksum,
        validate_seed,
        _compute_checksum,
        resolve,
        search,
        verify_randomness,
        mouse_entropy,
        kdf_info,
        get_languages,
        canonical_word,
        normalize_seed_version,
        get_supported_versions,
        UQS_VERSION,
        SUPPORTED_UQS_VERSIONS,
        DARK_VISUALS,
    )

__all__ = [
    "generate_words",
    "generate_seed",
    "get_seed",
    "get_profile",
    "get_quantum_seed",
    "generate_quantum_keypair",
    "get_fingerprint",
    "get_entropy_bits",
    "verify_checksum",
    "validate_seed",
    "_compute_checksum",
    "resolve",
    "search",
    "verify_randomness",
    "mouse_entropy",
    "kdf_info",
    "get_languages",
    "canonical_word",
    "normalize_seed_version",
    "get_supported_versions",
    "UQS_VERSION",
    "SUPPORTED_UQS_VERSIONS",
    "DARK_VISUALS",
]
