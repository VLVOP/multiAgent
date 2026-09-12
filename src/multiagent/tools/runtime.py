from __future__ import annotations

import os

from multiagent.tools.backends import OnlineNCBIBackend
from multiagent.tools.interfaces import LBDRetrievalBackend


_backend: LBDRetrievalBackend | None = None


def get_backend() -> LBDRetrievalBackend:
    """Return the active retrieval backend.

    The default is the online NCBI adapter. Tests and the future frozen MEDLINE backend can
    inject another implementation with set_backend without changing any Agent code.
    """
    global _backend
    if _backend is not None:
        return _backend

    backend_name = os.getenv("LBD_BACKEND", "online").strip().lower()
    if backend_name in {"online", "ncbi", "online_ncbi"}:
        _backend = OnlineNCBIBackend()
        return _backend

    raise ValueError(
        f"Unsupported LBD_BACKEND={backend_name!r}. "
        "Only the online backend is implemented in Tool Layer v1."
    )


def set_backend(backend: LBDRetrievalBackend) -> None:
    """Inject a backend implementation, primarily for tests and future frozen corpora."""
    global _backend
    _backend = backend


def reset_backend() -> None:
    global _backend
    _backend = None
