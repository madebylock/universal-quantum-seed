# Copyright (c) 2026 Lock.com — PolyForm Shield License 1.0.0

"""Native backend policy for secret operations.

Every primitive in this package keeps a readable pure-Python implementation
for verification of public data and for known-answer tests. Operations that
touch a secret (private scalars, signing keys, decapsulation keys, AES keys,
passwords) must run in a native constant-time library: CPython big integers,
``bytes`` table lookups and small-integer caching all leak secret-dependent
timing, allocation and memory-access patterns to a co-resident process.
When the native package is missing, the secret operation raises
``RuntimeError`` instead of silently degrading to the Python code.

``UQS_ALLOW_PURE_PYTHON_SECRETS=1`` switches the gate off. Set it only for
deterministic test vectors and development on a machine that holds no real
secrets; never in a wallet, a server or any process that handles user keys.
"""

import os

PURE_PYTHON_SECRETS_ENV = "UQS_ALLOW_PURE_PYTHON_SECRETS"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def pure_python_secrets_allowed() -> bool:
    """Return True when the test-only pure-Python override is switched on."""
    value = os.environ.get(PURE_PYTHON_SECRETS_ENV, "")
    return value.strip().lower() in _TRUE_VALUES


def require_native_backend(available: bool, operation: str, packages: str) -> None:
    """Fail closed before a secret reaches pure-Python arithmetic.

    Args:
        available: Whether the native backend for ``operation`` is importable
            and about to be used.
        operation: Human-readable name of the secret operation.
        packages: The native package or packages that provide it.

    Raises:
        RuntimeError: When no native backend is available and the test-only
            override is not set.
    """
    if available or pure_python_secrets_allowed():
        return
    raise RuntimeError(
        f"{operation} requires {packages}; pure-Python secret arithmetic is "
        f"disabled because it is not constant time. Install the native "
        f"package, or set {PURE_PYTHON_SECRETS_ENV}=1 only for deterministic "
        "test vectors and development."
    )


def require_pure_python_allowed(operation: str, packages: str) -> None:
    """Gate a code path that is known to be the pure-Python reference."""
    require_native_backend(False, operation, packages)
