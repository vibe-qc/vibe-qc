from __future__ import annotations

from vibeqc.gradient import _parallel_helpers as ph


def test_pt2_helpers_do_not_expose_legacy_copying_mapper():
    assert not hasattr(ph, "parallel_dE2dk")
    assert not hasattr(ph, "_compute_one_pair")


def test_pt2_fd_jobs_small_work_is_serial(monkeypatch):
    monkeypatch.delenv("VIBEQC_PT2_GRADIENT_N_JOBS", raising=False)
    monkeypatch.delenv("VIBEQC_PT2_GRADIENT_MAX_JOBS", raising=False)

    assert ph.choose_pt2_fd_n_jobs(npr=4, nmo=40, use_cpp_transform=True) == 1


def test_pt2_fd_jobs_default_hard_cap(monkeypatch):
    monkeypatch.delenv("VIBEQC_PT2_GRADIENT_N_JOBS", raising=False)
    monkeypatch.delenv("VIBEQC_PT2_GRADIENT_MAX_JOBS", raising=False)
    monkeypatch.setattr(ph.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(ph, "_available_memory_bytes", lambda: 512 * ph._GIB)

    assert ph.choose_pt2_fd_n_jobs(npr=40, nmo=20, use_cpp_transform=True) == 4


def test_pt2_fd_jobs_memory_cap(monkeypatch):
    monkeypatch.delenv("VIBEQC_PT2_GRADIENT_N_JOBS", raising=False)
    monkeypatch.setenv("VIBEQC_PT2_GRADIENT_MAX_JOBS", "16")
    monkeypatch.setattr(ph.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(ph, "_available_memory_bytes", lambda: 6 * ph._GIB)

    # nmo=80 estimates several GiB per worker, so the memory limiter wins.
    assert ph.choose_pt2_fd_n_jobs(npr=40, nmo=80, use_cpp_transform=True) == 1


def test_pt2_fd_jobs_force_env_is_still_bounded_by_work(monkeypatch):
    monkeypatch.setenv("VIBEQC_PT2_GRADIENT_N_JOBS", "99")
    monkeypatch.setattr(ph.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(ph, "_available_memory_bytes", lambda: 1 * ph._GIB)

    assert ph.choose_pt2_fd_n_jobs(npr=5, nmo=80, use_cpp_transform=True) == 5
