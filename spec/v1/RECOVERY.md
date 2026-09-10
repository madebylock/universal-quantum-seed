# Universal Quantum Seed — Recovery Guide v1

**Spec version:** 1.0
**Domain separator:** `universal-seed-v1`

> This document describes exactly how to recover keys from a Universal Quantum Seed
> backup using only standard cryptographic primitives. No proprietary code is needed.

---

## What You Need

1. Your **24 or 36 words** (or icon indexes 0–255)
2. Your **passphrase** (if one was set; empty string if none)
3. A system that can compute:
   - Arbitrary-precision integer arithmetic (a 190- or 284-bit integer; Python `int` is enough)
   - HMAC-SHA-256 (checksum verification)
   - HMAC-SHA-512 (HKDF-Extract, HKDF-Expand, profiles)
   - PBKDF2-SHA-512
   - Argon2id
4. The **icon-to-index mapping** (see SPEC.md Section 2, or `words.py`)

---

## Step-by-Step Recovery

### Step 1: Resolve Words to Indexes

Each word maps to an index 0–255. The canonical list is in SPEC.md Section 2.

```
eye=0, ear=1, nose=2, mouth=3, tongue=4, bone=5, ...
```

If you wrote your seed in another language, resolve each word through the
42-language lookup table in `words.py`. A valid v1 phrase never contains the
same icon twice — if you see a repeat, one of the words is misread. The checksum
(Step 3) will catch any other misresolution.

**Result:** A list of N distinct indexes (N = 24 or 36).

### Step 2: Rank the Phrase to an Integer

A v1 phrase does not carry separate "checksum words". The N distinct icons are a
mixed-radix number: position 0 is the least-significant digit (radix 256), position 1
the next (radix 255), …, position N−1 the most-significant (radix 256−N+1). The
*digit* at a position is the rank of its icon among the icons **not yet used** by
earlier positions (0 = the smallest unused index).

```python
LAYOUT = {24: (22, 14), 36: (34, 12)}   # word_count: (entropy_bytes E, checksum_bits C)

N = len(indexes)
E, C = LAYOUT[N]                        # any other length is not a v1 phrase

unused = list(range(256))
digits = []
for icon in indexes:
    assert 0 <= icon <= 255, "not an icon index"
    assert icon in unused,   "repeated icon: not a valid v1 phrase"
    d = unused.index(icon)              # how many unused icons are smaller
    digits.append(d)
    del unused[d]

V = 0
for pos in range(N - 1, -1, -1):        # Horner: last position first
    V = V * (256 - pos) + digits[pos]

assert (V >> C) < (1 << (8 * E)), "value in the unrank headroom: not a valid v1 phrase"
```

By hand: the digit at position i equals `icon_i` minus the number of icons at
positions 0..i−1 that are smaller than `icon_i`. The headroom check rejects
packed values that encoding can never produce (the mixed radix holds slightly
more than 2^(8E+C) values).

### Step 3: Split Off and Verify the Checksum

The low C bits of V are the checksum; everything above them is the entropy.
The checksum is the top C bits of the first 16 bits of HMAC-SHA-256 over the
**entropy bytes**:

```python
checksum = V & ((1 << C) - 1)                    # low C bits
entropy  = (V >> C).to_bytes(E, "big")           # E bytes, big-endian

key      = b"universal-seed-v1-checksum"
digest   = HMAC-SHA256(key, entropy)
expected = int.from_bytes(digest[0:2], "big") >> (16 - C)   # top C bits of first 16

assert checksum == expected, "checksum mismatch: transcription error"
```

If the checksum doesn't match, you have a transcription error. Fix it before
proceeding — incorrect data will derive a wrong (and useless) key.

**Result:** `entropy` — 22 bytes (24 words) or 34 bytes (36 words). Only these
entropy bytes enter the key derivation pipeline; the icon indexes themselves
never do.

#### Complete Steps 2–3 as one runnable function

```python
import hmac, hashlib

LAYOUT = {24: (22, 14), 36: (34, 12)}   # word_count: (entropy_bytes, checksum_bits)

def recover_entropy(indexes):
    N = len(indexes)
    E, C = LAYOUT[N]                                    # KeyError => not 24/36 words
    unused = list(range(256))
    digits = []
    for icon in indexes:
        assert 0 <= icon <= 255, "not an icon index"
        assert icon in unused, "repeated icon: not a valid v1 phrase"
        d = unused.index(icon)
        digits.append(d)
        del unused[d]
    V = 0
    for pos in range(N - 1, -1, -1):
        V = V * (256 - pos) + digits[pos]
    assert (V >> C) < (1 << (8 * E)), "unrank headroom: not a valid v1 phrase"
    checksum = V & ((1 << C) - 1)
    entropy = (V >> C).to_bytes(E, "big")
    digest = hmac.new(b"universal-seed-v1-checksum", entropy, hashlib.sha256).digest()
    expected = int.from_bytes(digest[:2], "big") >> (16 - C)
    assert checksum == expected, "checksum mismatch: transcription error"
    return entropy

# Quick self-check (the all-zero 24-word vector below):
assert recover_entropy([91, 42] + list(range(22))) == bytes(22)
```

### Step 4: Length-Prefixed Payload

Build a versioned, domain-separated payload from the **entropy bytes** recovered
in Step 3 (not the icon indexes), with explicit length prefixes on every
variable-length field. Each field is length- or domain-tagged so the boundary
between the entropy region and the passphrase is unambiguous:

```python
import struct, unicodedata

# NFKC normalization prevents cross-platform fund loss from different
# Unicode representations (macOS NFD vs Windows NFC).
passphrase_bytes = (
    unicodedata.normalize("NFKC", passphrase).encode("utf-8") if passphrase else b""
)

payload  = b"universal-seed-v1-seed-payload-v1"   # domain + version
payload += struct.pack("<H", len(entropy))        # uint16 LE: 22 or 34
for pos, byte in enumerate(entropy):
    payload += struct.pack("<BB", pos, byte)      # (pos, entropy byte) pairs
payload += b"\x01passphrase"                       # field tag
payload += struct.pack("<I", len(passphrase_bytes))  # uint32 LE pp length
payload += passphrase_bytes
```

The domain prefix, entropy-length prefix, field tag, and passphrase-length prefix
together ensure no two distinct `(entropy, passphrase)` inputs share a payload
— including across the 24-word and 36-word formats. An empty passphrase `""`
produces the same result as no passphrase.

### Step 5: HKDF-Extract (RFC 5869)

Collapse the payload into a 64-byte pseudorandom key using HMAC-SHA-512:

```python
prk = HMAC-SHA512(key=b"universal-seed-v1", message=payload)
```

- Key: `b"universal-seed-v1"` (17 bytes)
- Message: positional payload + optional passphrase bytes
- Output: 64 bytes (512 bits)

### Step 6: PBKDF2-SHA-512

Stretch the PRK through PBKDF2:

```python
stage1 = PBKDF2-SHA512(
    password = prk,
    salt     = b"universal-seed-v1-stretch-pbkdf2",
    rounds   = 600000,
    dklen    = 64
)
```

### Step 7: Argon2id

Further harden through Argon2id:

```python
stage2 = Argon2id(
    secret      = stage1,
    salt        = b"universal-seed-v1-stretch-argon2id",
    time_cost   = 3,
    memory_cost = 65536,    # 64 MiB
    parallelism = 4,
    hash_len    = 64,
    type        = Argon2id
)
```

### Step 8: HKDF-Expand (RFC 5869)

Derive the final 64-byte master key:

```python
def hkdf_expand(prk, info, length):
    from math import ceil
    n = ceil(length / 64)
    okm = b""
    prev = b""
    for i in range(1, n + 1):
        prev = HMAC-SHA512(key=prk, message=prev + info + bytes([i]))
        okm += prev
    return okm[:length]

master_key = hkdf_expand(stage2, b"universal-seed-v1-master", 64)
```

### Step 9: Done

`master_key` is your 64-byte (512-bit) master seed.

- First 32 bytes: 256-bit encryption key
- Last 32 bytes: 256-bit authentication key
- Or use the full 64 bytes as a master seed for further derivation

---

## Profile Recovery

If you used a hidden profile password:

```python
def get_profile(master_key, profile_password):
    if not profile_password:
        return master_key   # empty = default profile
    payload = b"universal-seed-v1-profile" + profile_password.encode("utf-8")
    return HMAC-SHA512(key=master_key, message=payload)
```

Each profile password produces an independent 64-byte key. Without the password,
the profile's existence cannot be detected.

---

## Cryptographic Key Recovery

If you derived keypairs, the seeds are derived from the master key via HKDF-Expand:

```python
_QUANTUM_SEED_SIZES = {
    "ml-dsa-65": 32,            # xi seed for FIPS 204 KeyGen
    "slh-dsa-shake-128s": 48,   # SK.seed(16) + SK.prf(16) + PK.seed(16)
    "ml-kem-768": 64,           # d (32B) || z (32B) for FIPS 203 KeyGen
    "hybrid-dsa-65": 64,        # Ed25519 seed (32B) + ML-DSA-65 seed (32B)
    "hybrid-kem-768": 96,       # X25519 seed (32B) + ML-KEM-768 seed (64B d||z)
}

def get_quantum_seed(master_key, algorithm, key_index=0):
    import struct
    size = _QUANTUM_SEED_SIZES[algorithm]
    info = b"universal-seed-v1-quantum-" + algorithm.encode("ascii") + struct.pack("<I", key_index)
    return hkdf_expand(master_key, info, size)
```

Feed the resulting seed into the appropriate keygen:

### Post-quantum algorithms

- **ML-DSA-65 (FIPS 204):** 32-byte seed -> `KeyGen(xi)` -> (sk: 4,032 B, pk: 1,952 B)
- **SLH-DSA-SHAKE-128s (FIPS 205):** 48-byte seed -> `slh_keygen(seed)` -> (sk: 64 B, pk: 32 B)
- **ML-KEM-768 (FIPS 203):** 64-byte seed (d||z) -> `KeyGen(d, z)` -> (ek: 1,184 B, dk: 2,400 B)

### Hybrid algorithms

- **Hybrid-DSA-65 (Ed25519 + ML-DSA-65):**
  64-byte seed -> first 32B to Ed25519 keygen, last 32B to ML-DSA-65 keygen
  -> sk: 4,096 B (Ed25519 sk 64B + ML-DSA sk 4,032B)
  -> pk: 1,984 B (Ed25519 pk 32B + ML-DSA pk 1,952B)

- **Hybrid-KEM-768 (X25519 + ML-KEM-768):**
  96-byte seed -> first 32B to X25519 keygen, last 64B to ML-KEM-768 keygen (d||z)
  -> ek: 1,216 B (X25519 pk 32B + ML-KEM ek 1,184B)
  -> dk: 2,432 B (X25519 sk 32B + ML-KEM dk 2,400B)

### Hybrid DSA verification

Both Ed25519 AND ML-DSA-65 must independently verify. The component signatures use
domain separation to prevent stripping:
- Ed25519 verifies: `b"hybrid-dsa-v1" || len(ctx) [1 byte] || ctx || message`
- ML-DSA-65 verifies with context: `b"hybrid-dsa-v1\x00" || ctx`

### Hybrid KEM shared secret recovery

The two component shared secrets are combined via HKDF:
```python
salt = SHA-256(x25519_ct || ml_kem_ct)
PRK  = HMAC-SHA256(salt, x25519_ss || ml_kem_ss)
info = b"hybrid-kem-v1" || SHA-256(x25519_pk || ml_kem_ek) || 0x01
SS   = HMAC-SHA256(PRK, info)
```

---

## Fingerprint Verification

To verify you've recovered correctly, compute the fingerprint from the
master seed (always runs full KDF):

```python
master_key = get_seed(full_seed, passphrase)   # full recovery (Steps 4-8)
# Default fingerprint is 8 hex chars (32 bits). Use bits=64/128/256 for
# longer/audit-strength fingerprints.
fingerprint = SHA-256(master_key)[0:4].hex().upper()  # e.g. "3F6FEE12"
```

Compare this fingerprint against your saved fingerprint. If it matches, recovery
was successful.

---

## Quick Reference — All Domain Strings

| Stage | String | Usage |
|:---|:---|:---|
| Checksum | `b"universal-seed-v1-checksum"` | HMAC-SHA-256 key over the entropy bytes; top 14 (24w) / 12 (36w) bits of digest[0:2] |
| HKDF-Extract | `b"universal-seed-v1"` | HMAC-SHA-512 key |
| PBKDF2 salt | `b"universal-seed-v1-stretch-pbkdf2"` | PBKDF2-SHA-512 salt |
| Argon2id salt | `b"universal-seed-v1-stretch-argon2id"` | Argon2id salt |
| HKDF-Expand | `b"universal-seed-v1-master"` | info string |
| Profile | `b"universal-seed-v1-profile"` | HMAC-SHA-512 message prefix |
| ML-DSA-65 | `b"universal-seed-v1-quantum-ml-dsa-65"` + index | HKDF-Expand info |
| SLH-DSA | `b"universal-seed-v1-quantum-slh-dsa-shake-128s"` + index | HKDF-Expand info |
| ML-KEM-768 | `b"universal-seed-v1-quantum-ml-kem-768"` + index | HKDF-Expand info |
| Hybrid-DSA-65 | `b"universal-seed-v1-quantum-hybrid-dsa-65"` + index | HKDF-Expand info |
| Hybrid-KEM-768 | `b"universal-seed-v1-quantum-hybrid-kem-768"` + index | HKDF-Expand info |
| Hybrid DSA domain | `b"hybrid-dsa-v1"` | Stripping resistance prefix |
| Hybrid KEM domain | `b"hybrid-kem-v1"` | HKDF info prefix |
| Hybrid KEM fail | `b"hybrid-kem-x25519-fail"` | X25519 implicit rejection |

---

## Test Vectors (Minimal)

### All-zero entropy, 24 words, no passphrase

```
Entropy:       22 zero bytes (hex 00 x 22)
HMAC-SHA256(b"universal-seed-v1-checksum", entropy)[0:2] = a9 6f
Checksum:      0xA96F >> 2 = 0x2A5B = 10843          (top 14 bits)
V:             (0 << 14) | 10843 = 10843

Unrank (position 0 first):
  pos 0: 10843 mod 256 = 91, V = 42   -> unused[91] = 91
  pos 1:    42 mod 255 = 42, V = 0    -> unused[42] = 42
  pos 2..23: digit 0                   -> smallest unused: 0, 1, 2, ..., 21

Phrase indexes: [91, 42, 0, 1, 2, 3, ..., 21]   (24 distinct icons)
Passphrase:     "" (empty)

PRK (HKDF-Extract only, Step 5):
  681a638c85c930b36d8cd4596e80737347fbd27a403105c3cc65ee93856663fc
  39579b312927303d25ed4ea1eb5c7112364bd68c099db59e0d25509fc376c9ea

Fingerprint:    6C1075D0
```

Rank check by hand: digits = [91, 42, 0, 0, …] (icon 0 at position 2 has no
smaller unused icon, and each later icon is the smallest remaining), so
V = 42·256 + 91 = 10843, entropy = 10843 >> 14 = 0, checksum = 10843.

### Sequential entropy, 24 words, no passphrase

```
Entropy:        00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15
digest[0:2]:    ef 78  ->  checksum = 0xEF78 >> 2 = 0x3BDE = 15326
Phrase indexes: [222, 53, 187, 179, 67, 73, 232, 167, 154, 189, 2, 214,
                 21, 35, 10, 93, 104, 81, 118, 224, 226, 161, 0, 1]
Fingerprint:    0960B7F0
```

(Position 2 illustrates the rank: its digit is 186 — icon 187 minus the one
earlier icon, 53, that is smaller than it.)

See `test-vectors.json` for the full set (entropy, phrase, master seed and
fingerprint for every known-answer vector, plus vectors that MUST be rejected)
and `kat/seed_v1.json` for the pinned known-answer file.

---

## Compatibility

This guide describes v1 of the Universal Quantum Seed. The compatibility contract:

> **v1 seeds MUST always derive the same outputs forever.**
> No parameter may be changed within v1. If parameters change, a new version
> with a new domain separator and spec folder MUST be created.

The distinct-icon encoding (Steps 2–3) replaced the pre-release "N−2 data
words + 2 checksum words" layout in place, before any official release. Phrases
written down by pre-release builds are not valid v1 phrases and there is no
legacy decode path; the KDF (Steps 4–8) and every domain string are unchanged,
so the master seed for a given entropy is identical under both layouts.
