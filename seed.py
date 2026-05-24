# Copyright (c) 2026 Lock — MIT License

__version__ = "1.0"

"""Seed generation for the Universal Quantum Seed.

Generates cryptographically secure seeds using 256 visual icons (8 bits each).
- 24 words = 22 random + 2 checksum = 176 bits of entropy
  (accepted for recovery/classical compatibility, not recommended for new
  long-term seeds)
- 36 words = 34 random + 2 checksum = 272 bits of entropy
  (recommended default and required for post-quantum derivation)

The last 2 words of every seed are a 16-bit checksum (HMAC-SHA-256 based)
that detects transcription errors with 1-in-65,536 false-positive rate.

Entropy is gathered from multiple independent sources and mixed through
SHA-512 (a cryptographic randomness extractor). This ensures that even if
one source is weak or compromised, the output remains uniformly random
as long as *any* source provides real entropy.

Key derivation is hardened with multiple layers:
    1. Positional binding — each icon is hashed with its position
    2. Passphrase mixing — optional second factor mixed into input,
       influences every downstream step
    3. HKDF-Extract — collapses seed + passphrase into a PRK (RFC 5869)
    4. Chained KDF — PBKDF2-SHA512 (600,000 rounds) then Argon2id
       (64 MiB, 3 iterations) — defense in depth, resists GPU/ASIC/FPGA
    5. HKDF-Expand — derives final key with domain separation

Entropy sources (defense in depth — OS CSPRNG is sufficient alone):
    1. secrets.token_bytes  — OS CSPRNG (CryptGenRandom / /dev/urandom)
    2. os.urandom           — separate OS CSPRNG call
    3. time.perf_counter_ns — timer state mixed as additional input
    4. os.getpid            — process-level uniqueness
    5. CPU jitter            — instruction timing variance (cache/TLB/branch)
    6. Thread scheduling     — OS scheduler nondeterminism (context switches)
    7. Hardware RNG          — BCryptGenRandom / platform HWRNG (RDRAND/RDSEED)

Usage:
    from seed import generate_words, get_seed, get_profile, get_fingerprint, generate_quantum_keypair, resolve, search
    words  = generate_words(36)                       # [(idx, "word"), ...] — 34 random + 2 checksum
    seed   = get_seed(words)                          # 64-byte master seed
    seed   = get_seed(words, "passphrase")            # with passphrase (second factor)
    prof   = get_profile(seed, "personal")            # hidden profile — independent 64-byte key
    fp     = get_fingerprint(words)                   # 8-char visual fingerprint
    sk, pk = generate_quantum_keypair(seed, "ml-dsa-65")  # post-quantum keypair
    idx   = resolve("dog")                          # 15
    idxs, errs = resolve(["dog", "sun", "key"])     # ([15, 63, 136], [])
    matches = search("do")                          # [("dog", 15), ...]
"""

import bisect
import hashlib
import hmac
import os
import re
import secrets
import struct
import threading
import time
import unicodedata

try:
    from .crypto.argon2 import hash_secret_raw, Type as _Argon2Type
except ImportError:
    from crypto.argon2 import hash_secret_raw, Type as _Argon2Type

try:
    from .crypto.secure_wipe import wipe, wipe_all
except ImportError:
    try:
        from crypto.secure_wipe import wipe, wipe_all
    except ImportError:
        # Minimal fallback if secure_wipe is not available
        def wipe(obj):
            if isinstance(obj, bytearray):
                for i in range(len(obj)):
                    obj[i] = 0
        def wipe_all(*objs):
            for o in objs:
                wipe(o)

# UQS v1 compatibility boundary:
# Do not reorder, replace, or "clean up" these words in-place. Seed words
# resolve to numeric indexes, and existing backups depend on each index keeping
# the same English alias. Some aliases are visually or phonetically close, but
# positional binding plus the two-word checksum catches transcription errors;
# changing the vocabulary would create a larger recovery risk. A new vocabulary
# must be a versioned UQS v2 format with legacy aliases/migration.
#
# 256 base English words - one per icon position (0-255)
_BASE_WORDS = (
    "eye", "ear", "nose", "mouth", "tongue", "bone", "tooth", "skull",
    "heart", "brain", "baby", "foot", "muscle", "hand", "leg", "dog",
    "cat", "horse", "cow", "pig", "goat", "rabbit", "mouse", "tiger",
    "wolf", "bear", "deer", "elephant", "bat", "camel", "zebra", "giraffe",
    "fox", "lion", "monkey", "panda", "llama", "squirrel", "chicken", "bird",
    "duck", "penguin", "peacock", "owl", "eagle", "snake", "frog", "turtle",
    "crocodile", "lizard", "fish", "octopus", "crab", "whale", "dolphin", "shark",
    "snail", "ant", "bee", "butterfly", "worm", "spider", "scorpion", "sun",
    "moon", "star", "earth", "fire", "water", "snow", "cloud", "rain",
    "rainbow", "wind", "thunder", "volcano", "tornado", "comet", "wave", "desert",
    "island", "mountain", "rock", "diamond", "feather", "tree", "cactus", "flower",
    "leaf", "mushroom", "wood", "mango", "apple", "banana", "grape", "orange",
    "melon", "peach", "strawberry", "pineapple", "cherry", "lemon", "coconut", "cucumber",
    "seed", "corn", "carrot", "onion", "potato", "pepper", "tomato", "garlic",
    "peanut", "bread", "cheese", "egg", "meat", "rice", "cake", "snack",
    "sweet", "honey", "milk", "coffee", "tea", "wine", "beer", "juice",
    "salt", "fork", "spoon", "bowl", "knife", "bottle", "soup", "pan",
    "key", "lock", "bell", "hammer", "axe", "gear", "magnet", "sword",
    "bow", "shield", "bomb", "compass", "hook", "thread", "needle", "scissors",
    "pencil", "house", "castle", "temple", "bridge", "factory", "door", "window",
    "tent", "beach", "bank", "tower", "statue", "wheel", "boat", "train",
    "car", "bike", "plane", "rocket", "helicopter", "ambulance", "fuel", "track",
    "map", "drum", "guitar", "violin", "piano", "paint", "book", "music",
    "mask", "camera", "microphone", "headset", "movie", "dress", "coat", "pants",
    "glove", "shirt", "shoes", "hat", "flag", "cross", "circle", "triangle",
    "square", "check", "alert", "sleep", "magic", "message", "blood", "repeat",
    "dna", "germ", "pill", "doctor", "microscope", "galaxy", "flask", "atom",
    "satellite", "battery", "telescope", "tv", "radio", "phone", "bulb", "keyboard",
    "chair", "bed", "candle", "mirror", "ladder", "basket", "vase", "shower",
    "razor", "soap", "computer", "trash", "umbrella", "money", "prayer", "toy",
    "crown", "ring", "dice", "piece", "coin", "calendar", "boxing", "swimming",
    "game", "soccer", "ghost", "alien", "robot", "angel", "dragon", "clock",
)
_BASE = {i: w for i, w in enumerate(_BASE_WORDS)}

# ── Language support (loaded from words.py at module level below) ──
_LANGUAGES = {}  # populated after words.py is loaded


def _load_language(code):
    """Return {index: first_word} for a language code."""
    lang = _LANGUAGES.get(code)
    if lang is None:
        raise ValueError(f"Unknown language: {code!r}")
    return {int(k): v for k, v in lang["words"].items()}


def get_languages():
    """Return available seed languages as a list of (code, label) tuples.

    Example::

        get_languages()
        # [("english", "English"), ("arabic", "العربية"), ("french", "Français"), ...]
    """
    results = [("english", "English")]
    for code in sorted(_LANGUAGES):
        if code == "english":
            continue
        results.append((code, _LANGUAGES[code]["label"]))
    return results


def canonical_word(index, language=None) -> str:
    """Return the strict-resolvable display word for an icon index."""
    idx = int(index)
    if idx < 0 or idx > 255:
        raise ValueError(f"word index out of range: {idx}")
    if language and language != "english":
        return _load_language(language)[idx]
    return _BASE[idx]


# Domain separator — ensures keys from this system can never collide
# with keys derived by other systems using the same hash functions.
_DOMAIN = b"universal-seed-v1"
UQS_VERSION = 1
SUPPORTED_UQS_VERSIONS = (UQS_VERSION,)
_VERSION_DOMAINS = {UQS_VERSION: _DOMAIN}

# UQS v1 derivation compatibility boundary:
# Keep these parameters stable for v1. Raising UQS_ARGON2_MEMORY_KIB changes the
# deterministic master seed for every existing phrase. The 64 MiB, t=3, p=4
# profile matches RFC 9106's memory-constrained Argon2id recommendation and is
# above OWASP's minimum guidance; heavier profiles must be introduced only as a
# versioned UQS v2 KDF.
UQS_ARGON2_TIME_COST = 3         # iterations
UQS_ARGON2_MEMORY_KIB = 65536    # 64 MiB
UQS_ARGON2_PARALLELISM = 4       # lanes
UQS_ARGON2_HASH_LENGTH = 64      # output bytes

# PBKDF2 parameters — first stage of chained KDF
UQS_PBKDF2_ITERATIONS = 600_000

# Backward-compat private aliases for internal callers.
_ARGON2_TIME = UQS_ARGON2_TIME_COST
_ARGON2_MEMORY = UQS_ARGON2_MEMORY_KIB
_ARGON2_PARALLEL = UQS_ARGON2_PARALLELISM
_ARGON2_HASHLEN = UQS_ARGON2_HASH_LENGTH
_PBKDF2_ITERATIONS = UQS_PBKDF2_ITERATIONS


def normalize_seed_version(version=UQS_VERSION) -> int:
    """Return a supported UQS seed-format version.

    UQS v1 is locked by published KATs. Future formats must be added as
    separate versions; they must not mutate this v1 domain or KDF contract.
    """
    if version is None or version == "":
        return UQS_VERSION
    if isinstance(version, str):
        value = version.strip().lower()
        if value.startswith("uqs-"):
            value = value[4:]
        if value.startswith("v"):
            value = value[1:]
        if not value.isdigit():
            raise ValueError(f"unsupported UQS seed version: {version!r}")
        version_i = int(value, 10)
    else:
        try:
            version_i = int(version)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported UQS seed version: {version!r}") from exc
    if version_i not in _VERSION_DOMAINS:
        raise ValueError(f"unsupported UQS seed version: {version!r}")
    return version_i


def get_supported_versions() -> tuple[int, ...]:
    """Return UQS seed-format versions supported by this implementation."""
    return SUPPORTED_UQS_VERSIONS


def _domain_for_version(version=UQS_VERSION) -> bytes:
    return _VERSION_DOMAINS[normalize_seed_version(version)]

# ── Word lookup data ──────────────────────────────────────────────
try:
    from .words import LOOKUP as _LOOKUP, LANGUAGES as _LANGUAGES, DARK_VISUALS
except ImportError:
    from words import LOOKUP as _LOOKUP, LANGUAGES as _LANGUAGES, DARK_VISUALS

_SORTED_KEYS = sorted(_LOOKUP.keys())
_INDEX_TO_BASE = _BASE  # index -> base English word

# Inner-word index for multi-word entries (e.g. "bàn tay" → searchable by "tay")
_INNER_WORDS = []  # sorted list of (inner_word, full_key)
for _k in _LOOKUP:
    _parts = _k.split()
    if len(_parts) > 1:
        for _p in _parts[1:]:
            _INNER_WORDS.append((_p, _k))
_INNER_WORDS.sort()
_INNER_WORD_KEYS = [_iw for _iw, _ in _INNER_WORDS]

DEBUG = False


def _debug_trace(kind, event, t0=None, *, count=None):
    """Emit non-secret diagnostics only when local debugging is enabled."""
    if not DEBUG:
        return
    parts = [f"  [{kind}] {event}"]
    if count is not None:
        parts.append(f"count={int(count)}")
    if t0 is not None:
        parts.append(f"elapsed={(time.perf_counter() - t0) * 1000:.2f}ms")
    print("  ".join(parts))


# Zero-width and invisible characters to strip
_INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\u00ad\u034f\u061c"
    "\ufeff\u2060\u2061\u2062\u2063\u2064\u180e]"
)

# Definite-article suffixes for Latin-script languages (longest first)
# Icelandic: -inn (masc), -in (fem), -ið (neut)
# Romanian:  -ul (masc), -le (masc after vowels)
# Norwegian/Danish: -en (common), -et (neuter), -a (fem, Norwegian)
_ARTICLE_SUFFIXES = ("inn", "ið", "ul", "in", "le", "en", "et", "a")

# Scripts where stripping combining marks is safe
_SAFE_STRIP_SCRIPTS = {"latin", "greek", "arabic", "hebrew", "cyrillic"}


def _normalize(word):
    """Normalize a word for lookup.

    Strips whitespace, removes invisible chars, NFKC normalizes
    (full-width -> regular, ligatures -> letters), lowercases.
    """
    w = word.strip()
    w = _INVISIBLE_CHARS.sub("", w)
    w = unicodedata.normalize("NFKC", w)
    return w.lower()


def _detect_script(word):
    """Detect the primary script of a word."""
    script_counts = {}
    for c in word:
        if not c.isalpha():
            continue
        name = unicodedata.name(c, "")
        if "LATIN" in name:
            script_counts["latin"] = script_counts.get("latin", 0) + 1
        elif "GREEK" in name:
            script_counts["greek"] = script_counts.get("greek", 0) + 1
        elif "CYRILLIC" in name:
            script_counts["cyrillic"] = script_counts.get("cyrillic", 0) + 1
        elif "ARABIC" in name:
            script_counts["arabic"] = script_counts.get("arabic", 0) + 1
        elif "HEBREW" in name:
            script_counts["hebrew"] = script_counts.get("hebrew", 0) + 1
    if not script_counts:
        return "other"
    return max(script_counts, key=script_counts.get)


def _strip_diacritics(word):
    """Remove optional diacritics based on script.

    Safe for: Latin (accents, ss, o, etc.), Greek (tonos),
    Arabic (tashkeel/harakat), Hebrew (niqqud), Cyrillic.
    NOT applied to Thai, Devanagari, Bengali, Tamil, Telugu, Gurmukhi.
    """
    script = _detect_script(word)
    if script not in _SAFE_STRIP_SCRIPTS:
        return word

    result = word

    if script == "latin":
        for old, new in {"\u00df": "ss", "\u00f8": "o", "\u00e6": "ae", "\u0153": "oe",
                         "\u00f0": "d", "\u00fe": "th", "\u0142": "l", "\u0111": "d"}.items():
            result = result.replace(old, new)

    if script == "cyrillic":
        result = result.replace("\u0451", "\u0435").replace("\u0401", "\u0415")

    nfkd = unicodedata.normalize("NFKD", result)
    stripped = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _normalize_emoji(text):
    """Normalize an emoji by stripping variation selectors."""
    e = text.strip()
    e = e.replace("\ufe0e", "").replace("\ufe0f", "")
    e = _INVISIBLE_CHARS.sub("", e)
    return e


def _resolve_one(word, strict=False):
    """Resolve a single word/emoji to its visual index (0-255), or None.

    strict=False (default): tries fallbacks (diacritics, articles, suffixes)
    strict=True: exact normalized lookup only — no fuzzy fallbacks
    """
    t0 = time.perf_counter()
    key = _normalize(word)

    # Numeric index (0-255)
    if key.isdigit():
        n = int(key)
        if 0 <= n <= 255:
            _debug_trace("resolve", "numeric input accepted", t0)
            return n

    result = _LOOKUP.get(key)
    if result is not None:
        _debug_trace("resolve", "exact match", t0)
        return result

    # Try emoji-normalized (strip variation selectors)
    e_key = _normalize_emoji(word)
    if e_key and e_key != key:
        result = _LOOKUP.get(e_key)
        if result is not None:
            _debug_trace("resolve", "emoji match", t0)
            return result

    if strict:
        _debug_trace("resolve", "no match in strict mode", t0)
        return None

    # Fallback: try diacritic-stripped version
    stripped = _strip_diacritics(key)
    if stripped != key:
        result = _LOOKUP.get(stripped)
        if result is not None:
            _debug_trace("resolve", "diacritic fallback match", t0)
            return result

    # Fallback: strip Arabic definite article "ال" (al-) prefix
    candidate = stripped if stripped != key else key
    if candidate.startswith("\u0627\u0644"):
        bare = candidate[2:]
        result = _LOOKUP.get(bare)
        if result is not None:
            _debug_trace("resolve", "arabic article fallback match", t0)
            return result

    # Fallback: strip Hebrew definite article "ה" (ha-) prefix
    if candidate.startswith("\u05d4"):
        bare = candidate[1:]
        result = _LOOKUP.get(bare)
        if result is not None:
            _debug_trace("resolve", "hebrew article fallback match", t0)
            return result

    # Fallback: strip French/Italian "l'" article contraction
    for apo in ("'", "\u2019", "\u02bc"):
        prefix = "l" + apo
        if candidate.startswith(prefix):
            bare = candidate[len(prefix):]
            result = _LOOKUP.get(bare)
            if result is not None:
                _debug_trace("resolve", "latin article fallback match", t0)
                return result
            break

    # Fallback: strip definite-article suffixes (Scandinavian, Romanian, Icelandic)
    if _detect_script(candidate) == "latin" and len(candidate) > 3:
        for suffix in _ARTICLE_SUFFIXES:
            if candidate.endswith(suffix) and len(candidate) - len(suffix) >= 2:
                bare = candidate[:-len(suffix)]
                result = _LOOKUP.get(bare)
                if result is not None:
                    _debug_trace("resolve", "suffix fallback match", t0)
                    return result

    _debug_trace("resolve", "no match", t0)
    return None


def resolve(words, strict=False):
    """Resolve one or more words (any language) or emoji to visual indexes.

    Accepts a single word (string) or a list of words.

    strict=False (default): tries fuzzy fallbacks (diacritics, articles, suffixes)
    strict=True: exact normalized lookup only — no fuzzy fallbacks

    Single word:
        resolve("dog")      → 15
        resolve("unknown")  → None

    Multiple words:
        resolve(["dog", "sun", "key"])  → ([15, 63, 136], [])
        resolve(["dog", "???", "key"])  → ([15, 136], [(1, "???")])

    Returns:
        str input:  int (0-255) or None
        list input: (indexes, errors) where errors is [(position, word), ...]
    """
    if isinstance(words, str):
        return _resolve_one(words, strict=strict)

    indexes = []
    errors = []
    for i, word in enumerate(words):
        idx = _resolve_one(word, strict=strict)
        if idx is not None:
            indexes.append(idx)
        else:
            errors.append((i, word))
    return indexes, errors


def _strip_article_prefix(key):
    """Strip definite article prefixes for search. Returns stripped key or None."""
    # French/Italian l' contraction
    for apo in ("'", "\u2019", "\u02bc"):
        prefix = "l" + apo
        if key.startswith(prefix):
            return key[len(prefix):]
    # Arabic ال
    if key.startswith("\u0627\u0644"):
        return key[2:]
    # Hebrew ה
    if key.startswith("\u05d4") and len(key) > 1:
        return key[1:]
    return None


def _search_sorted(sorted_keys, lookup, key, limit, seen_indexes):
    """Binary-search sorted_keys for entries starting with key."""
    lo = bisect.bisect_left(sorted_keys, key)
    results = []
    for i in range(lo, len(sorted_keys)):
        if len(results) >= limit:
            break
        k = sorted_keys[i]
        if not k.startswith(key):
            break
        idx = lookup[k]
        if idx in seen_indexes:
            continue
        seen_indexes.add(idx)
        results.append((k, idx))
    return results


def search(prefix, limit=10):
    """Suggest words matching a prefix, for search/autocomplete.

    Returns a list of (word, index) tuples sorted alphabetically,
    up to `limit` unique indexes. Words mapping to the same index are
    deduplicated (first alphabetical match wins).
    """
    t0 = time.perf_counter()
    key = _normalize(prefix)
    if not key:
        return []

    # Numeric prefix: match indexes whose string starts with the typed digits
    if key.isdigit():
        results = []
        for idx in range(256):
            if str(idx).startswith(key):
                base = _INDEX_TO_BASE.get(idx, str(idx))
                results.append((base, idx))
                if len(results) >= limit:
                    break
        _debug_trace("search", "numeric prefix search", t0, count=len(results))
        return results

    # Collect English base words that match the prefix first
    english_first = []
    seen_indexes = set()
    for idx, base in _INDEX_TO_BASE.items():
        if base.lower().startswith(key):
            english_first.append((base.lower(), idx))
            seen_indexes.add(idx)
    english_first.sort()
    if len(english_first) > limit:
        english_first = english_first[:limit]

    results = list(english_first)
    remaining = limit - len(results)

    # Primary: binary search on full keys
    results += _search_sorted(_SORTED_KEYS, _LOOKUP, key, remaining, seen_indexes)
    remaining = limit - len(results)

    # Article prefix stripping (l'oeil → oeil, الأيدي → أيدي, etc.)
    if remaining > 0:
        alt_key = _strip_article_prefix(key)
        if alt_key:
            results += _search_sorted(_SORTED_KEYS, _LOOKUP, alt_key, remaining, seen_indexes)
            remaining = limit - len(results)

    # Inner-word matching for multi-word entries (e.g. "tay" finds "bàn tay")
    if remaining > 0:
        lo = bisect.bisect_left(_INNER_WORD_KEYS, key)
        for i in range(lo, len(_INNER_WORD_KEYS)):
            if remaining <= 0:
                break
            iw = _INNER_WORD_KEYS[i]
            if not iw.startswith(key):
                break
            full_key = _INNER_WORDS[i][1]
            idx = _LOOKUP[full_key]
            if idx in seen_indexes:
                continue
            seen_indexes.add(idx)
            results.append((full_key, idx))
            remaining -= 1

    # Substring matching — find entries containing the search term anywhere
    if remaining > 0 and len(key) >= 2:
        for full_key in _SORTED_KEYS:
            if remaining <= 0:
                break
            if key in full_key:
                idx = _LOOKUP[full_key]
                if idx in seen_indexes:
                    continue
                seen_indexes.add(idx)
                results.append((full_key, idx))
                remaining -= 1

    _debug_trace("search", "prefix search", t0, count=len(results))
    return results




class mouse_entropy:
    """Collects entropy from mouse movement samples.

    Each sample is (x, y, timestamp_ns). Entropy comes from:
    - Sub-pixel timing jitter in nanosecond timestamps
    - Unpredictable micro-movements in cursor position

    Conservative estimate: ~2 bits per unique sample.
    Samples are continuously hashed into a SHA-512 pool.

    Usage:
        pool = mouse_entropy()
        pool.add_sample(x, y)   # call on each mouse move
        pool.bits_collected      # check progress
        extra = pool.digest()    # extract entropy bytes
        seed = generate_words(36, extra_entropy=extra)
    """

    def __init__(self):
        self._hasher = hashlib.sha512()
        self._hasher.update(_DOMAIN + b"-mouse-entropy")
        self._samples = 0
        self._last_x = None
        self._last_y = None
        self._last_t = None

    def add_sample(self, x, y):
        """Add a mouse position sample with high-resolution timing.

        Returns True if the sample was new (position changed), False if skipped.
        """
        t = time.perf_counter_ns()

        # Skip duplicate positions (no movement = no entropy)
        if x == self._last_x and y == self._last_y:
            return False

        # Pack absolute position + timing
        self._hasher.update(struct.pack("<iiQ", x, y, t))

        # Hash deltas too — micro-movements carry extra entropy
        if self._last_x is not None:
            dx = x - self._last_x
            dy = y - self._last_y
            dt = t - self._last_t
            self._hasher.update(struct.pack("<iiQ", dx, dy, dt))

        self._last_x = x
        self._last_y = y
        self._last_t = t
        self._samples += 1
        return True

    @property
    def bits_collected(self):
        """Conservative estimate of entropy bits collected.

        ~2 bits per unique sample (position delta + timing jitter).
        """
        return self._samples * 2

    @property
    def sample_count(self):
        return self._samples

    def digest(self):
        """Extract collected entropy as bytes (64 bytes / 512 bits).

        Returns a copy — the pool can continue collecting after this.
        """
        return self._hasher.copy().digest()

    def reset(self):
        """Clear the pool and start fresh."""
        self.__init__()


def _cpu_jitter_entropy():
    """Collect entropy from CPU execution timing jitter.

    Runs tight loops of mixed operations and measures nanosecond-level
    variations caused by cache misses, branch prediction, TLB eviction,
    pipeline stalls, and speculative execution. Similar to the jitterentropy
    library used by the Linux kernel's /dev/random.

    Conservative estimate: ~1 bit per sample (64 samples = ~64 bits).
    """
    h = hashlib.sha512()
    h.update(_DOMAIN + b"-cpu-jitter")
    for _ in range(64):
        t1 = time.perf_counter_ns()
        # Mixed operations to trigger cache/TLB/branch-predictor jitter
        x = 0
        for j in range(100):
            x ^= (x << 3) ^ (j * 7) ^ (x >> 5)
            x &= 0xFFFFFFFFFFFFFFFF
        t2 = time.perf_counter_ns()
        h.update(struct.pack("<QQ", t2 - t1, t2))
    return h.digest()



def _thread_jitter_entropy():
    """Collect entropy from OS thread scheduling jitter.

    Spawns short-lived threads and measures round-trip timing. Entropy
    comes from nondeterministic OS scheduler decisions: context switches,
    CPU core migration, priority inheritance, and interrupt coalescing.

    Conservative estimate: ~2 bits per thread (32 threads = ~64 bits).
    """
    h = hashlib.sha512()
    h.update(_DOMAIN + b"-thread-jitter")
    results = []

    def worker(idx):
        t = time.perf_counter_ns()
        x = 0
        for i in range(50):
            x = (x + time.perf_counter_ns()) & 0xFFFFFFFF
        t2 = time.perf_counter_ns()
        results.append(struct.pack("<BQQ", idx, t, t2))

    for _batch in range(4):
        results.clear()
        threads = []
        t0 = time.perf_counter_ns()
        for i in range(8):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        t1 = time.perf_counter_ns()
        h.update(struct.pack("<QQ", t0, t1))
        for r in results:
            h.update(r)
    return h.digest()


def _hardware_rng_entropy():
    """Collect entropy from platform hardware RNG (RDRAND/RDSEED).

    Windows: BCryptGenRandom — separate API from CryptGenRandom
             (used by os.urandom), provides defense-in-depth.
             Feeds from Intel RDRAND/RDSEED and/or TPM when available.
    Linux:   getrandom() via os.urandom (already covered above).

    Also mixes in heap ASLR addresses and thread IDs for uniqueness.
    """
    h = hashlib.sha512()
    h.update(_DOMAIN + b"-hardware-rng")

    if os.name == 'nt':
        try:
            import ctypes
            bcrypt = ctypes.windll.bcrypt
            buf = ctypes.create_string_buffer(64)
            # BCRYPT_USE_SYSTEM_PREFERRED_RNG = 2
            status = bcrypt.BCryptGenRandom(None, buf, 64, 2)
            if status == 0:
                h.update(buf.raw)
        except Exception:
            pass
    else:
        # On Linux/macOS, read from /dev/random (hardware-backed)
        try:
            with open('/dev/random', 'rb') as f:
                h.update(f.read(32))
        except Exception:
            pass

    # Mix in ASLR heap addresses + thread ID for uniqueness
    h.update(struct.pack("<QQ",
        id(object()),
        threading.current_thread().ident or 0,
    ))
    return h.digest()


def _collect_entropy(n_bytes, extra_entropy=None):
    """Collect entropy from multiple sources and mix via SHA-512.

    SHA-512 acts as a randomness extractor — it uniformly distributes
    entropy across its output regardless of how the input is structured.
    The result has at least as much entropy as the strongest single source.

    Args:
        n_bytes: Number of random bytes to return.
        extra_entropy: Optional bytes to mix in (e.g. from mouse_entropy).

    Returns exactly n_bytes of cryptographic-quality random data.
    """
    pool = bytearray()

    # Source 1: OS CSPRNG via secrets (primary source)
    pool.extend(secrets.token_bytes(64))

    # Source 2: OS CSPRNG via os.urandom (separate syscall)
    pool.extend(os.urandom(64))

    # Source 3: High-resolution timing jitter
    # The LSBs of perf_counter_ns contain hardware clock noise that is
    # unpredictable even to an attacker who controls the OS CSPRNG
    for _ in range(32):
        pool.extend(struct.pack("<Q", time.perf_counter_ns()))

    # Source 4: Process-level uniqueness
    pool.extend(struct.pack("<I", os.getpid()))

    # Source 5: CPU execution timing jitter (cache/TLB/branch predictor)
    pool.extend(_cpu_jitter_entropy())

    # Source 6: Thread scheduling jitter (OS scheduler nondeterminism)
    pool.extend(_thread_jitter_entropy())

    # Source 7: Hardware RNG (RDRAND/RDSEED via BCrypt or /dev/random)
    pool.extend(_hardware_rng_entropy())

    # Source 8: User-supplied entropy (mouse movements, etc.)
    if extra_entropy:
        pool.extend(extra_entropy)

    # Mix everything through SHA-512
    h = hashlib.sha512(pool)

    # Wipe the entropy pool now that it's been hashed
    wipe(pool)

    # Fold in one final secrets call keyed on the digest
    # This ensures the output is at minimum as strong as secrets alone
    h.update(secrets.token_bytes(32))

    digest = h.digest()  # 64 bytes = 512 bits

    if n_bytes <= 64:
        return digest[:n_bytes]

    # For sizes > 64 bytes (unlikely), use HKDF-style expand
    out = bytearray()
    counter = 1
    prev = b""
    while len(out) < n_bytes:
        prev = hashlib.sha512(prev + digest + struct.pack("B", counter)).digest()
        out.extend(prev)
        counter += 1
    return bytes(out[:n_bytes])


_VALIDATION_SAMPLE_SIZE = 1024  # bytes — large enough for statistical tests
_ENTROPY_HEALTH_SAMPLES = 5
_ENTROPY_HEALTH_MIN_PASSES = 3


def _entropy_tests_pass(tests):
    return all(t["pass"] for t in tests.values())


def _validate_entropy_pipeline(extra_entropy=None):
    """Run a bounded RNG health check without retrying until success.

    Statistical tests naturally produce occasional outliers, so seed creation
    uses a fixed batch and majority decision instead of conditioning generation
    on the first sample that happens to pass. A broken source fails closed.
    """
    samples = []
    pass_count = 0
    for sample_index in range(_ENTROPY_HEALTH_SAMPLES):
        test_sample = _collect_entropy(_VALIDATION_SAMPLE_SIZE, extra_entropy)
        tests = _test_entropy(test_sample)
        sample_passed = _entropy_tests_pass(tests)
        if sample_passed:
            pass_count += 1
        samples.append({
            "sample": sample_index,
            "pass": sample_passed,
            "tests": tests,
        })

    return {
        "pass": pass_count >= _ENTROPY_HEALTH_MIN_PASSES,
        "passed": pass_count,
        "required": _ENTROPY_HEALTH_MIN_PASSES,
        "total": _ENTROPY_HEALTH_SAMPLES,
        "samples": samples,
    }


def generate_words(word_count=36, extra_entropy=None, language=None):
    """Generate a cryptographically secure seed with 16-bit checksum.

    The last 2 words are a checksum derived from the random words via
    HMAC-SHA-256 with domain separation. This provides 16-bit error detection
    (1-in-65,536 false positive rate).

    The entropy pipeline is validated before use with a fixed batch of
    1024-byte samples drawn from the same sources and tested with four
    statistical tests (monobit, chi-squared, runs, autocorrelation). The
    batch must pass by majority before the actual seed entropy is drawn.

    Args:
        word_count: 24 (176-bit, 22 random + 2 checksum; compatibility) or
                    36 (272-bit, 34 random + 2 checksum; recommended).
        extra_entropy: Optional bytes to mix in (e.g. from mouse_entropy.digest()).
        language: Optional language code (e.g. "french", "arabic").
                  None or "english" returns English words.

    Returns:
        List of (index, word) tuples. Last 2 are checksum words.

    Raises:
        ValueError: If word_count is not 24 or 36, or language is unknown.
        RuntimeError: If the entropy health-check batch fails validation
                      (indicates a compromised or broken RNG).
    """
    if word_count not in (24, 36):
        raise ValueError("word_count must be 24 or 36")

    data_count = word_count - 2  # random words (22 or 34)

    # Resolve word map for requested language
    if language and language != "english":
        word_map = _load_language(language)
    else:
        word_map = _BASE

    health = _validate_entropy_pipeline(extra_entropy)
    if not health["pass"]:
        raise RuntimeError(
            "Entropy failed validation "
            f"({health['passed']}/{health['total']} samples passed; "
            f"{health['required']} required) — RNG source may be compromised. "
            "Do NOT generate seeds on this system."
        )

    entropy = _collect_entropy(data_count, extra_entropy)
    indexes = list(entropy)
    # Append 2 checksum words
    indexes.extend(_compute_checksum(indexes))
    return [(idx, word_map[idx]) for idx in indexes]


def generate_seed(word_count=36, extra_entropy=None, language=None):
    """Compatibility wrapper returning display words only.

    Console and older callers historically consumed a plain list of words,
    while the core UQS API returns ``(index, word)`` pairs so callers can keep
    indexes and display labels together.  Keep the public wrapper thin so both
    paths share the same entropy health checks and checksum generation.
    """
    return [
        word for _idx, word in generate_words(
            word_count=word_count,
            extra_entropy=extra_entropy,
            language=language,
        )
    ]


def _compute_checksum(indexes, *, version=UQS_VERSION):
    """Compute 2 checksum indexes from a list of random seed indexes."""
    # Intentional: this checksum detects transcription mistakes; it is not an
    # authenticity check. Two 8-bit words give 16 bits of error detection,
    # stronger than BIP39's 24-word checksum, while preserving the seed format.
    # Widening it would remove entropy words and break existing seed recovery
    # unless introduced as a separate versioned format.
    domain = _domain_for_version(version)
    digest = hmac.new(domain + b"-checksum", bytes(indexes), hashlib.sha256).digest()
    return [digest[0], digest[1]]


def verify_checksum(seed, *, version=UQS_VERSION):
    """Verify the last 2 words are valid checksum words.

    Args:
        seed: List of (index, word) tuples, plain indexes, or words.

    Returns:
        True if the checksum is valid, False otherwise.
    """
    indexes = _to_indexes(seed)
    if len(indexes) not in (24, 36):
        return False
    data = indexes[:-2]
    expected = _compute_checksum(data, version=version)
    return hmac.compare_digest(bytes(indexes[-2:]), bytes(expected))


def validate_seed(seed, *, version=UQS_VERSION) -> bool:
    """Compatibility wrapper returning whether a UQS phrase is checksum-valid."""
    try:
        return verify_checksum(seed, version=version)
    except Exception:
        return False


def _hkdf_expand(prk, info, length):
    """HKDF-Expand (RFC 5869) using HMAC-SHA512.

    Returns a ``bytearray`` so callers can wipe it after consumption.
    All intermediate buffers (``prev``, ``msg``) are wiped before the
    next iteration so derived blocks don't linger in the heap.
    """
    n = (length + 63) // 64  # SHA-512 = 64-byte blocks
    okm = bytearray()
    prev = bytearray()
    try:
        for i in range(1, n + 1):
            msg = bytearray(prev)
            try:
                msg.extend(info)
                msg.append(i)
                next_prev = _hmac_digest_bytearray(prk, msg, hashlib.sha512)
            finally:
                wipe(msg)
            wipe(prev)
            prev = next_prev
            okm.extend(prev)
        return okm[:length]
    finally:
        wipe(prev)
        # Note: returned slice aliases okm's storage in CPython; callers
        # are expected to wipe the returned slice instead of okm here.


def _hmac_digest_bytearray(key, msg, digestmod):
    """HMAC then return the digest in a wipeable bytearray.

    The ``hmac.HMAC.digest()`` method returns an immutable ``bytes``
    object — we copy into a ``bytearray`` and wipe the original so the
    secret digest isn't left in the heap.
    """
    digest = hmac.new(key, msg, digestmod).digest()
    try:
        return bytearray(digest)
    finally:
        try:
            wipe(digest)
        except Exception:
            pass


def _stretch(prk, *, version=UQS_VERSION):
    """Chained key stretching: PBKDF2-SHA512 → Argon2id (defense in depth).

    Two independent KDFs run in series — the output of PBKDF2 feeds into
    Argon2id. An attacker must break both to recover the key:
    - PBKDF2:   600,000 rounds of SHA-512 ≈ 1 sec per guess
    - Argon2id: 64 MiB memory × 3 iterations × 4 lanes ≈ 1 sec per guess
    """
    domain = _domain_for_version(version)
    salt = domain + b"-stretch"

    # Stage 1: PBKDF2-SHA512. Copy the immutable result into a wipeable
    # bytearray, then wipe the original so the heap doesn't hold the
    # stage-1 secret for longer than the call.
    stage1_raw = None
    stage1 = None
    try:
        stage1_raw = hashlib.pbkdf2_hmac(
            "sha512",
            prk,
            salt + b"-pbkdf2",
            iterations=_PBKDF2_ITERATIONS,
            dklen=64,
        )
        stage1 = bytearray(stage1_raw)
        try:
            wipe(stage1_raw)
        except Exception:
            pass

        # Stage 2: Argon2id on top of PBKDF2 output. ``return_bytearray``
        # gives the caller a wipeable result + zeros the intermediate
        # CFFI / bytes copy inside hash_secret_raw.
        return hash_secret_raw(
            secret=stage1,
            salt=salt + b"-argon2id",
            time_cost=_ARGON2_TIME,
            memory_cost=_ARGON2_MEMORY,
            parallelism=_ARGON2_PARALLEL,
            hash_len=_ARGON2_HASHLEN,
            type=_Argon2Type.ID,
            return_bytearray=True,
        )
    finally:
        if stage1 is not None:
            wipe(stage1)


def _to_indexes(seed):
    """Convert a seed to a list of integer indexes.

    Uses strict resolution (no fuzzy fallbacks) to prevent
    accidental misresolution during key derivation.

    Accepts:
        - List of (int, str) tuples: output of generate_words() — indexes extracted
        - List of ints (0-255): returned as-is
        - List of strings: each word is resolved via resolve(strict=True)

    Raises ValueError if any word fails to resolve.
    """
    if not seed:
        raise ValueError("seed must not be empty")
    first = seed[0]
    if isinstance(first, tuple):
        return [idx for idx, _ in seed]
    if isinstance(first, int):
        return seed
    # Resolve words → indexes (strict: no fuzzy fallbacks)
    indexes, errors = resolve(list(seed), strict=True)
    if errors:
        bad = ", ".join(f"'{w}' (pos {i})" for i, w in errors)
        raise ValueError(f"could not resolve: {bad}")
    return indexes


def _passphrase_to_bytes(passphrase) -> bytearray:
    """NFKC-normalize a passphrase and return its UTF-8 bytes.

    Returns a ``bytearray`` so the caller can wipe it after use; the
    intermediate Python ``str`` is unreachable but the encoded UTF-8 is
    held in mutable memory.

    NFKC prevents cross-platform fund loss from different Unicode
    representations of the same visual characters (macOS NFD vs Windows NFC).
    """
    if not passphrase:
        return bytearray()
    normalized = unicodedata.normalize("NFKC", str(passphrase))
    return bytearray(normalized.encode("utf-8"))


def _build_seed_payload(indexes, passphrase="", *, version=UQS_VERSION) -> bytearray:
    """Build the length-prefixed, domain-separated UQS v1 seed payload.

    Layout: domain + word-count (uint16 LE) + per-position (pos, idx) pairs
    + b"\\x01passphrase" tag + passphrase length (uint32 LE) + passphrase bytes.
    Each field is length- or domain-tagged so the boundary between the index
    region and the passphrase is unambiguous (prevents cross-length collisions).

    Returns a ``bytearray`` so the caller (``get_seed``) can wipe it
    after derivation. The passphrase bytes are wiped here even on
    failure since they're a copy local to this function.
    """
    passphrase_bytes = _passphrase_to_bytes(passphrase)
    payload = bytearray()
    try:
        domain = _domain_for_version(version)
        payload.extend(domain + b"-seed-payload-v1")
        payload.extend(struct.pack("<H", len(indexes)))
        for pos, idx in enumerate(indexes):
            payload.extend(struct.pack("<BB", pos, idx))
        payload.extend(b"\x01passphrase")
        payload.extend(struct.pack("<I", len(passphrase_bytes)))
        payload.extend(passphrase_bytes)
        return payload
    except Exception:
        wipe(payload)
        raise
    finally:
        wipe(passphrase_bytes)


def get_seed(words, passphrase="", *, version=UQS_VERSION):
    """Derive a 64-byte master seed from words + optional passphrase.

    Only the data words (first 22 or 34) enter the KDF — the 2 checksum
    words are verified and then stripped so the derived seed depends solely
    on the random entropy.

    Security layers:
        1. Checksum verification — rejects corrupted words before derivation
        2. Positional binding — each data icon is tagged with its slot index
        3. Passphrase mixing — optional second factor mixed into input
           with explicit length-prefixing and field/domain separation
        4. HKDF-Extract — collapses payload into a pseudorandom key (RFC 5869)
        5. Chained KDF — PBKDF2-SHA512 (600k rounds) then Argon2id (64 MiB)
        6. HKDF-Expand — derives final 64-byte seed with domain separation

    The output is 64 bytes (512 bits) which can be split into:
        - First 32 bytes: 256-bit encryption key
        - Last 32 bytes:  256-bit authentication key
    Or used whole as a master seed for further derivation.

    The optional passphrase acts as a second factor. Same words with
    different passphrases produce completely unrelated seeds. An empty
    passphrase is valid and produces a deterministic seed.

    Args:
        words: List of icon indexes (ints 0-255) or words (strings in any language).
        passphrase: Optional passphrase string (second factor).
        version: UQS seed-format version. Only v1 is supported in this
            implementation; future versions must preserve v1 unchanged.

    Returns:
        64 bytes of derived seed material.

    Raises:
        ValueError: If the word count is invalid or checksum fails.
    """
    domain = _domain_for_version(version)
    indexes = _to_indexes(words)

    # Step 0: Enforce valid length and verify checksum
    if len(indexes) not in (24, 36):
        raise ValueError(f"seed must be 24 or 36 words, got {len(indexes)}")
    data = indexes[:-2]
    if not hmac.compare_digest(
        bytes(indexes[-2:]),
        bytes(_compute_checksum(data, version=version)),
    ):
        raise ValueError("invalid seed checksum")
    indexes = data

    # Step 1-2: Build a versioned, position-tagged, length-prefixed payload.
    # All intermediate secrets are bytearrays and wiped in the finally
    # block so the heap doesn't retain derivation state after return.
    payload = None
    prk = None
    stretched = None
    master = None
    try:
        payload = _build_seed_payload(indexes, passphrase, version=version)

        # Step 3: HKDF-Extract — collapse payload + passphrase into fixed PRK.
        prk = _hmac_digest_bytearray(domain, payload, hashlib.sha512)

        # Step 4: Chained KDF stretching (PBKDF2 → Argon2id).
        stretched = _stretch(prk, version=version)

        # Step 5: HKDF-Expand — derive output seed with domain separation.
        master = _hkdf_expand(stretched, domain + b"-master", 64)
        return bytes(master)
    finally:
        for secret in (payload, prk, stretched, master):
            if secret is not None:
                wipe(secret)


def get_profile(master_key, profile_password):
    """Derive a profile-specific key from a master key.

    Allows multiple independent accounts from a single seed. Each profile
    password produces a completely unrelated 64-byte key. Without the
    password, the profile's existence cannot be detected (plausible
    deniability).

    The derivation is a single HMAC-SHA512 — instant, no KDF needed
    since the master key is already hardened.

    Args:
        master_key: 64-byte master seed from get_seed().
        profile_password: Profile password string. Empty/missing returns
            master_key unchanged — this IS the default profile.

    Returns:
        64 bytes of profile-specific key material.
    """
    # ── DO NOT CHANGE — CANONICAL BEHAVIOR (locked by KAT v1) ──
    # Empty profile_password MUST return master_key unchanged.
    # This is the spec for the "default profile" across all UQS
    # implementations (this Python, signer's vendored Python,
    # universal-quantum-seed-js). The KAT vectors at kat/seed_v1.json
    # in the JS repo encode this: for every vector, `default_profile_hex`
    # == `master_seed_hex`. If you "fix" this to always derive, you will
    # break cross-platform key compatibility AND every KAT will fail.
    if not profile_password:
        return master_key
    # NFKC normalization prevents cross-platform derivation differences
    # (e.g., macOS NFD vs Windows NFC for accented characters)
    profile_password = unicodedata.normalize("NFKC", profile_password)
    payload = _DOMAIN + b"-profile" + profile_password.encode("utf-8")
    return hmac.new(master_key, payload, hashlib.sha512).digest()


# ── Quantum Key Derivation ───────────────────────────────────────

_QUANTUM_SEED_SIZES = {
    "ml-dsa-65": 32,            # xi seed for FIPS 204 KeyGen
    "slh-dsa-shake-128s": 48,   # SK.seed(16) + SK.prf(16) + PK.seed(16) for FIPS 205 (n=16)
    "ml-kem-768": 64,           # d (32B) || z (32B) for FIPS 203 KeyGen
    # Hybrid classical + post-quantum (defense in depth)
    "hybrid-dsa-65": 64,        # Ed25519 seed (32B) + ML-DSA-65 seed (32B)
    "hybrid-kem-768": 96,       # X25519 seed (32B) + ML-KEM-768 seed (64B d||z)
}


def _validate_quantum_key_index(key_index) -> int:
    if isinstance(key_index, bool) or not isinstance(key_index, int):
        raise ValueError("key_index must be an integer in [0, 4294967295]")
    if not 0 <= key_index <= 0xFFFFFFFF:
        raise ValueError("key_index must be an integer in [0, 4294967295]")
    return key_index


def get_quantum_seed(master_key, algorithm="ml-dsa-65", key_index=0, _word_count=None):
    """Derive post-quantum seed material from a master key via HKDF-Expand.

    Uses algorithm-specific domain separation to ensure complete independence
    between classical BIP32/SLIP10 keys and each post-quantum algorithm's keys.
    Deterministic: same inputs always produce the same output.

    The 36-word seed format (272-bit) is required for quantum-safe derivation.
    A 24-word seed (176-bit) does not provide sufficient entropy for post-quantum
    security — ML-DSA-65 (Level 3) requires at least 192-bit entropy.

    Args:
        master_key: 64-byte master seed from get_seed().
        algorithm: Algorithm identifier string.
            "ml-dsa-65"          -> ML-DSA (Dilithium) FIPS 204, Level 3
            "slh-dsa-shake-128s" -> SLH-DSA (SPHINCS+) FIPS 205, Level 1
            "ml-kem-768"         -> ML-KEM (Kyber) FIPS 203, Level 3
            "hybrid-dsa-65"      -> Ed25519 + ML-DSA-65 hybrid signature
            "hybrid-kem-768"     -> X25519 + ML-KEM-768 hybrid KEM
        key_index: Instance index for multiple keys (default 0).
            Allows deriving independent keypairs per algorithm.
        _word_count: Internal — word count of the source seed for entropy validation.

    Returns:
        bytes — seed material for keygen:
            ML-DSA-65:          32 bytes (xi seed)
            SLH-DSA-SHAKE-128s: 48 bytes (SK.seed || SK.prf || PK.seed, n=16 each)
            ML-KEM-768:         64 bytes (d || z)
            Hybrid-DSA-65:      64 bytes (Ed25519 seed 32B + ML-DSA seed 32B)
            Hybrid-KEM-768:     96 bytes (X25519 seed 32B + ML-KEM seed 64B)

    Raises:
        ValueError: If master_key is not 64 bytes, algorithm is unknown,
            or seed entropy is insufficient (24-word seed).
    """
    if len(master_key) != 64:
        raise ValueError(f"master_key must be 64 bytes, got {len(master_key)}")
    if _word_count is not None and _word_count < 36:
        raise ValueError(
            f"Quantum key derivation requires a 36-word seed (272-bit entropy). "
            f"A {_word_count}-word seed ({(_word_count - 2) * 8}-bit) does not provide "
            f"sufficient post-quantum security."
        )
    size = _QUANTUM_SEED_SIZES.get(algorithm)
    if size is None:
        raise ValueError(
            f"Unknown quantum algorithm: {algorithm!r}. "
            f"Supported: {', '.join(sorted(_QUANTUM_SEED_SIZES))}"
        )
    key_index = _validate_quantum_key_index(key_index)
    info = _DOMAIN + b"-quantum-" + algorithm.encode("ascii") + struct.pack("<I", key_index)
    return _hkdf_expand(master_key, info, size)


def generate_quantum_keypair(master_key, algorithm="ml-dsa-65", key_index=0, _word_count=None):
    """Generate a post-quantum keypair from a master key.

    Derives quantum seed material via get_quantum_seed(), then runs the
    appropriate keygen algorithm to produce a full keypair.

    The 36-word seed format (272-bit) is required for quantum-safe derivation.

    Args:
        master_key: 64-byte master seed from get_seed().
        algorithm: Algorithm identifier string.
            "ml-dsa-65"          -> (4032B sk, 1952B pk)
            "slh-dsa-shake-128s" -> (64B sk, 32B pk)
            "ml-kem-768"         -> (2400B dk, 1184B ek)
            "hybrid-dsa-65"      -> (4096B sk, 1984B pk)  Ed25519 + ML-DSA-65
            "hybrid-kem-768"     -> (2432B dk, 1216B ek)  X25519 + ML-KEM-768
        key_index: Instance index for multiple keys (default 0).
        _word_count: Internal — word count of the source seed for entropy validation.

    Returns:
        (secret_key, public_key) tuple with sizes per algorithm above.

    Raises:
        ValueError: If master_key is not 64 bytes, algorithm is unknown,
            or seed entropy is insufficient (24-word seed).
    """
    quantum_seed = get_quantum_seed(master_key, algorithm, key_index, _word_count)
    if algorithm == "ml-dsa-65":
        from crypto.ml_dsa import ml_keygen
        return ml_keygen(quantum_seed)
    elif algorithm == "slh-dsa-shake-128s":
        from crypto.slh_dsa import slh_keygen
        return slh_keygen(quantum_seed)
    elif algorithm == "ml-kem-768":
        from crypto.ml_kem import ml_kem_keygen
        ek, dk = ml_kem_keygen(quantum_seed)
        return dk, ek  # (secret=dk, public=ek) to match (sk, pk) convention
    elif algorithm == "hybrid-dsa-65":
        from crypto.hybrid_dsa import hybrid_dsa_keygen
        return hybrid_dsa_keygen(quantum_seed)
    elif algorithm == "hybrid-kem-768":
        from crypto.hybrid_kem import hybrid_kem_keygen
        ek, dk = hybrid_kem_keygen(quantum_seed)
        return dk, ek  # (secret=dk, public=ek) to match (sk, pk) convention
    raise ValueError(f"Unknown quantum algorithm: {algorithm!r}")


_FINGERPRINT_BITS = (32, 64, 128, 256)


def get_fingerprint(seed, passphrase="", *, bits=32):
    """Compute a visual fingerprint for verification.

    Derives the full master seed via get_seed() and returns the leading
    ``bits`` of SHA-256(master_seed) as an uppercase hex string. This
    matches regardless of import format — the same master key always
    produces the same fingerprint.

    Runs the full PBKDF2 + Argon2id pipeline (both with and without
    passphrase), so this is NOT instant. Callers should run it in a
    background thread.

    Args:
        seed: List of icon indexes (ints 0-255) or words (strings in any language).
        passphrase: Optional passphrase (if set, fingerprint changes).
        bits: Output strength. One of 32, 64, 128, 256.
            32  = 8 hex chars  — short, easy to scan, BIP-32-style typo
                  detection. Default.
            64  = 16 hex chars — middle ground.
            128 = 32 hex chars — collision-resistant for audit comparison.
            256 = 64 hex chars — full SHA-256.

    Returns:
        Uppercase hex string of length bits/4.
    """
    if bits not in _FINGERPRINT_BITS:
        raise ValueError(
            f"bits must be one of {_FINGERPRINT_BITS}, got {bits!r}"
        )
    key = get_seed(seed, passphrase)
    try:
        return hashlib.sha256(key).hexdigest()[: bits // 4].upper()
    finally:
        try:
            wipe(key)
        except Exception:
            pass

def get_entropy_bits(word_count, passphrase=""):
    """Calculate total entropy in bits from seed words + passphrase.

    Seed entropy: (word_count - 2) × 8 bits. The last 2 words are checksum
    and don't contribute additional entropy.

    Passphrase entropy is estimated from its character set:
        - Digits only (0-9):            ~3.32 bits/char
        - Lowercase only (a-z):         ~4.70 bits/char
        - Mixed case (a-z, A-Z):        ~5.70 bits/char
        - Mixed + digits:               ~5.95 bits/char
        - Full printable (symbols too):  ~6.55 bits/char
        - Unicode (non-ASCII):           ~7.00 bits/char (conservative)

    This measures the keyspace an attacker must search if they know
    the character classes used but not the actual characters.

    Args:
        word_count: Number of seed words (24 or 36, includes 2 checksum).
        passphrase: Passphrase string.

    Returns:
        Estimated total entropy as a float (e.g. 272.0, 305.3).
    """
    seed_bits = (word_count - 2) * 8  # checksum words don't add entropy

    if not passphrase:
        return float(seed_bits)

    import math

    has_lower = any(c.islower() for c in passphrase)
    has_upper = any(c.isupper() for c in passphrase)
    has_digit = any(c.isdigit() for c in passphrase)
    has_symbol = any(not c.isalnum() and c.isascii() for c in passphrase)
    has_unicode = any(ord(c) > 127 for c in passphrase)

    pool = 0
    if has_lower:
        pool += 26
    if has_upper:
        pool += 26
    if has_digit:
        pool += 10
    if has_symbol:
        pool += 33  # printable ASCII symbols
    if has_unicode:
        pool += 100  # conservative estimate

    if pool == 0:
        return float(seed_bits)

    bits_per_char = math.log2(pool)
    pp_bits = bits_per_char * len(passphrase)

    return seed_bits + pp_bits


def kdf_info():
    """Return a string describing the chained KDF pipeline."""
    return (f"PBKDF2-SHA512 ({_PBKDF2_ITERATIONS:,} rounds) "
            f"+ Argon2id (mem={_ARGON2_MEMORY}KB, t={_ARGON2_TIME}, p={_ARGON2_PARALLEL})")


def _test_entropy(data):
    """Run statistical tests on raw bytes and return per-test results.

    This is the core testing engine used by both generate_words (to validate
    entropy before use) and verify_randomness (diagnostic UI).

    Returns a dict of {test_name: {pass, detail, ...}} for the four tests.
    """
    import math

    n_bits = len(data) * 8
    bits = []
    for byte in data:
        for bit_pos in range(7, -1, -1):
            bits.append((byte >> bit_pos) & 1)

    results = {}

    # ── Test 1: Monobit (frequency) test ─────────────────────
    ones = sum(bits)
    s = abs(2 * ones - n_bits) / math.sqrt(n_bits)
    monobit_pass = s < 2.576
    results["monobit"] = {
        "pass": monobit_pass,
        "ones_ratio": ones / n_bits,
        "z_score": round(s, 4),
        "threshold": 2.576,
        "detail": f"{ones}/{n_bits} ones ({ones/n_bits:.4f}), z={s:.4f}",
    }

    # ── Test 2: Chi-squared byte frequency ───────────────────
    observed = [0] * 256
    for byte in data:
        observed[byte] += 1
    expected = len(data) / 256.0
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    chi2_pass = chi2 < 310.5
    results["chi_squared"] = {
        "pass": chi2_pass,
        "chi2": round(chi2, 2),
        "threshold": 310.5,
        "expected_per_bin": round(expected, 2),
        "detail": f"chi2={chi2:.2f} (threshold 310.5), expected/bin={expected:.2f}",
    }

    # ── Test 3: Runs test ────────────────────────────────────
    pi = ones / n_bits
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n_bits):
        runs_pass = False
        runs_z = float("inf")
    else:
        runs = 1
        for i in range(1, n_bits):
            if bits[i] != bits[i - 1]:
                runs += 1
        expected_runs = 2.0 * n_bits * pi * (1 - pi) + 1
        std_runs = 2.0 * math.sqrt(2.0 * n_bits) * pi * (1 - pi)
        if std_runs == 0:
            runs_z = float("inf")
        else:
            runs_z = abs(runs - expected_runs) / std_runs
        runs_pass = runs_z < 2.576
    results["runs"] = {
        "pass": runs_pass,
        "z_score": round(runs_z, 4) if runs_z != float("inf") else "inf",
        "threshold": 2.576,
        "detail": f"z={runs_z:.4f}" if runs_z != float("inf") else "z=inf (degenerate)",
    }

    # ── Test 4: Autocorrelation ──────────────────────────────
    # Bonferroni correction: 16 offsets at family-wise alpha=0.01
    _AUTOCORR_Z = 3.42
    autocorr_pass = True
    worst_z = 0.0
    worst_offset = 0
    for d in range(1, 17):
        matches = sum(1 for i in range(n_bits - d) if bits[i] == bits[i + d])
        total = n_bits - d
        z = abs(2 * matches - total) / math.sqrt(total)
        if z > worst_z:
            worst_z = z
            worst_offset = d
        if z >= _AUTOCORR_Z:
            autocorr_pass = False
    results["autocorrelation"] = {
        "pass": autocorr_pass,
        "worst_z": round(worst_z, 4),
        "worst_offset": worst_offset,
        "threshold": _AUTOCORR_Z,
        "detail": f"worst z={worst_z:.4f} at offset {worst_offset}",
    }

    return results


def verify_randomness(sample_bytes=None, sample_size=2048, num_samples=5):
    """Test randomness quality of the entropy source to detect weak RNG.

    Runs four statistical tests based on NIST SP 800-22 methodology:
        1. Monobit — proportion of 1-bits should be ~50%
        2. Chi-squared byte frequency — all 256 values roughly uniform
        3. Runs — transitions between 0/1 bits (detects stuck patterns)
        4. Autocorrelation — checks for bit-level correlations at offsets 1-16

    Each test returns pass/fail and a score. A healthy RNG should pass all four.
    If any test fails, the entropy source may be weak or compromised.

    Aggregation uses majority voting: a test is only marked as failed if
    more than half the samples fail it. This eliminates false positives
    from statistical noise while still catching genuinely weak entropy.

    Args:
        sample_bytes: Optional raw bytes to test. If None, samples are
                      generated from _collect_entropy.
        sample_size:  Bytes per sample when auto-generating (default 2048).
        num_samples:  Number of independent samples to generate and test
                      (default 5). Ignored if sample_bytes is provided.

    Returns:
        dict with:
            "pass": bool — True if all tests passed across all samples
            "tests": list of per-test results
            "summary": human-readable summary string

    Usage:
        result = verify_randomness()
        print(result["summary"])
        if not result["pass"]:
            raise RuntimeError("Weak randomness detected!")
    """
    samples = []
    if sample_bytes is not None:
        samples = [sample_bytes]
    else:
        for _ in range(num_samples):
            samples.append(_collect_entropy(sample_size))

    all_results = []
    for si, data in enumerate(samples):
        sample_tests = _test_entropy(data)
        all_results.append({"sample": si, "tests": sample_tests})

    # ── Aggregate with majority voting ───────────────────────────
    # A test fails only if more than half the samples fail it.
    # With 5 samples at ~4% per-sample false-positive rate,
    # P(3+ fail by chance) ≈ 0.003% — virtually eliminates noise.
    test_names = ["monobit", "chi_squared", "runs", "autocorrelation"]
    overall_pass = True
    test_summary = []
    for name in test_names:
        failed_count = sum(1 for r in all_results if not r["tests"][name]["pass"])
        majority_failed = failed_count > len(all_results) / 2
        if majority_failed:
            overall_pass = False
        status = "PASS" if not majority_failed else f"FAIL ({failed_count}/{len(all_results)} samples)"
        test_summary.append({"test": name, "pass": not majority_failed, "status": status})

    lines = []
    lines.append(f"Randomness verification: {'PASS' if overall_pass else 'FAIL'}")
    lines.append(f"Samples: {len(samples)}, Size: {len(samples[0])} bytes each")
    lines.append("")
    for ts in test_summary:
        mark = "+" if ts["pass"] else "!"
        lines.append(f"  [{mark}] {ts['test']:<20s} {ts['status']}")
    if not overall_pass:
        lines.append("")
        lines.append("WARNING: Weak randomness detected. Do NOT use for seed generation.")
    summary = "\n".join(lines)

    return {
        "pass": overall_pass,
        "tests": test_summary,
        "samples": all_results,
        "summary": summary,
    }
