"""Memory-aware worker-count helpers for PT2 gradient computations."""

from __future__ import annotations

import os


_GIB = 1024**3


def _parse_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _available_memory_bytes() -> int | None:
    """Best-effort available-memory probe without adding a dependency."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            return None
        if isinstance(pages, int) and isinstance(page_size, int):
            return pages * page_size
    return None


def estimate_pt2_fd_worker_bytes(nmo: int, *, use_cpp_transform: bool) -> int:
    """Estimate peak bytes for one PT2 finite-difference worker.

    The C++ four-index transform path allocates several full ``nmo**4``
    work arrays. The Python fallback also materializes dense transformed
    tensors. This deliberately overestimates; it is a placement guard, not
    a profiler.
    """
    tensor_bytes = max(1, int(nmo)) ** 4 * 8
    multiplier = 10 if use_cpp_transform else 7
    matrix_bytes = max(1, int(nmo)) ** 2 * 8
    return max(256 * 1024**2, multiplier * tensor_bytes + 12 * matrix_bytes)


def choose_pt2_fd_n_jobs(
    *,
    npr: int,
    nmo: int,
    use_cpp_transform: bool,
) -> int:
    """Return a memory-safe process count for PT2 FD orbital gradients.

    Defaults are intentionally conservative for release hardening. Users
    can raise the cap with ``VIBEQC_PT2_GRADIENT_MAX_JOBS`` after they have
    sized a host, or force an exact count with
    ``VIBEQC_PT2_GRADIENT_N_JOBS``.
    """
    if npr <= 4:
        return 1

    cpu_limit = max(1, min(npr, os.cpu_count() or 1))
    forced = _parse_positive_int_env("VIBEQC_PT2_GRADIENT_N_JOBS")
    if forced is not None:
        return max(1, min(npr, forced))

    hard_cap = _parse_positive_int_env("VIBEQC_PT2_GRADIENT_MAX_JOBS") or 4
    worker_bytes = estimate_pt2_fd_worker_bytes(
        nmo, use_cpp_transform=use_cpp_transform
    )
    available = _available_memory_bytes()
    if available is None:
        memory_limit = 1
    else:
        reserve = max(2 * _GIB, available // 4)
        usable = max(0, available - reserve)
        memory_limit = max(1, usable // worker_bytes)

    return max(1, min(cpu_limit, hard_cap, int(memory_limit)))
