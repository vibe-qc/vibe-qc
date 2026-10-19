"""#411: scripts/_safe_build_env.sh sizes CMAKE_BUILD_PARALLEL_LEVEL from the
memory a build can actually get (MemAvailable and cgroup headroom), not from
MemTotal.

Every case sources the real helper under bash with a fake /proc/meminfo, a
fake /proc/self/cgroup and a fake cgroup tree, so the tests run the same on
macOS and Linux and never touch the host's own memory figures. The nice/ionice
re-exec and the tooling probe are switched off (VIBEQC_BUILD_NICED,
VIBEQC_SKIP_PREFLIGHT); only section (3) is exercised.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts" / "_safe_build_env.sh"
BASH = "/bin/bash" if Path("/bin/bash").is_file() else shutil.which("bash")
GIB = 1024**3

pytestmark = pytest.mark.skipif(BASH is None, reason="the helper is a Bash script")

# The #411 host (2026-09-29), whose GPU driver reserves part of RAM:
# MemTotal ~60 GB, ~36 GB held by the amdgpu driver.
GPU_HOST_TOTAL_KB = 60_000 * 1024
GPU_HOST_AVAIL_KB = 24_000 * 1024


def _meminfo(path: Path, total_kb: int, avail_kb: int | None) -> Path:
    lines = [f"MemTotal:       {total_kb} kB", "MemFree:         1000000 kB"]
    if avail_kb is not None:
        lines.append(f"MemAvailable:   {avail_kb} kB")
    lines.append("Cached:          2000000 kB")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(tmp_path: Path, *, meminfo: Path | None, nproc: int = 16,
         self_cgroup: Path | None = None, cgroup_root: Path | None = None,
         extra_env: dict[str, str] | None = None,
         script: str = "") -> tuple[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "VIBEQC_BUILD_NICED": "1",
        "VIBEQC_SKIP_PREFLIGHT": "1",
        "_VIBEQC_NPROC": str(nproc),
        "_VIBEQC_MEMINFO": str(meminfo if meminfo is not None else tmp_path / "absent"),
        "_VIBEQC_PROC_SELF_CGROUP": str(self_cgroup or tmp_path / "no-self-cgroup"),
        "_VIBEQC_CGROUP_ROOT": str(cgroup_root or tmp_path / "no-cgroup-root"),
    }
    env.update(extra_env or {})
    body = (
        "set -euo pipefail\n"
        f". {HELPER}\n"
        f"{script}\n"
        'echo "LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-unset}"\n'
    )
    done = subprocess.run([BASH, "-c", body], env=env, capture_output=True,
                          text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    levels = re.findall(r"^LEVEL=(\S+)$", done.stdout, flags=re.M)
    return levels[-1], done.stdout


def _v2_tree(tmp_path: Path, limits: dict[str, tuple[int | str, int, int]]) -> tuple[Path, Path]:
    """A cgroup-v2 tree; ``limits`` maps a relative path to (max, current, inactive_file)."""
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cgroup.controllers").write_text("cpu memory\n", encoding="utf-8")
    leaf = "/user.slice/user-1000.slice/user-manager-1000.service/app.slice/vq-admin-update-x.service"
    parts = leaf.strip("/").split("/")
    for depth in range(1, len(parts) + 1):
        rel = "/" + "/".join(parts[:depth])
        d = root / rel.lstrip("/")
        d.mkdir(exist_ok=True)
        mx, cur, inactive = limits.get(rel, ("max", 0, 0))
        (d / "memory.max").write_text(f"{mx}\n", encoding="utf-8")
        (d / "memory.current").write_text(f"{cur}\n", encoding="utf-8")
        (d / "memory.stat").write_text(f"anon 1\ninactive_file {inactive}\n", encoding="utf-8")
    self_cgroup = tmp_path / "self-cgroup"
    self_cgroup.write_text(f"0::{leaf}\n", encoding="utf-8")
    return self_cgroup, root


def test_gpu_reserved_host_sizes_from_memavailable_not_memtotal(tmp_path: Path) -> None:
    """The #411 host: MemTotal would allow 4 workers, MemAvailable does not."""
    meminfo = _meminfo(tmp_path / "meminfo", GPU_HOST_TOTAL_KB, GPU_HOST_AVAIL_KB)
    level, out = _run(tmp_path, meminfo=meminfo, nproc=32,
                      extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "6000"})
    assert level == "4", out  # 24000 // 6000
    level, out = _run(tmp_path, meminfo=meminfo, nproc=32,
                      extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "15000"})
    assert level == "1", out  # 24000 // 15000; main gave 4 from MemTotal
    assert "MemAvailable=24000MB" in out and "MemTotal=60000MB" in out


def test_floor_is_one_worker_and_says_why(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 64_000 * 1024, 3_000 * 1024)
    level, out = _run(tmp_path, meminfo=meminfo, extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "8000"})
    assert level == "1"
    assert "below one worker's budget" in out


def test_nproc_and_hard_cap_still_bound(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000 * 1024, 900_000 * 1024)
    assert _run(tmp_path, meminfo=meminfo, nproc=3)[0] == "3"
    assert _run(tmp_path, meminfo=meminfo, nproc=64)[0] == "8"


def test_old_kernel_without_memavailable_falls_back_to_memtotal(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 64_000 * 1024, None)
    level, out = _run(tmp_path, meminfo=meminfo, extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "16000"})
    assert level == "4"
    assert "MemTotal(no MemAvailable)" in out


def test_cgroup_v2_ancestor_limit_bounds_the_level(tmp_path: Path) -> None:
    """A MemoryMax on user.slice (or any ancestor) caps a host with lots of free RAM."""
    meminfo = _meminfo(tmp_path / "meminfo", 512_000 * 1024, 480_000 * 1024)
    self_cgroup, root = _v2_tree(tmp_path, {
        # 20 GiB limit, 6 GiB in use of which 2 GiB is reclaimable cache → 16 GiB headroom.
        "/user.slice": (20 * GIB, 6 * GIB, 2 * GIB),
    })
    level, out = _run(tmp_path, meminfo=meminfo, nproc=64, self_cgroup=self_cgroup,
                      cgroup_root=root, extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "4096"})
    assert level == "4", out  # 16384 // 4096
    assert "cgroup_headroom=16384MB" in out and "from cgroup" in out


def test_cgroup_v2_tightest_level_wins(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 512_000 * 1024, 480_000 * 1024)
    self_cgroup, root = _v2_tree(tmp_path, {
        "/user.slice": (100 * GIB, 10 * GIB, 0),
        "/user.slice/user-1000.slice/user-manager-1000.service": (12 * GIB, 4 * GIB, 0),
    })
    level, out = _run(tmp_path, meminfo=meminfo, nproc=64, self_cgroup=self_cgroup,
                      cgroup_root=root, extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "2048"})
    assert level == "4", out  # 8 GiB // 2 GiB


def test_cgroup_v2_without_limits_is_ignored(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 64_000 * 1024, 40_000 * 1024)
    self_cgroup, root = _v2_tree(tmp_path, {})
    level, out = _run(tmp_path, meminfo=meminfo, self_cgroup=self_cgroup, cgroup_root=root,
                      extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "10000"})
    assert level == "4", out
    assert "cgroup_headroom=none" in out


def test_cgroup_v1_hierarchical_limit_bounds_the_level(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 256_000 * 1024, 200_000 * 1024)
    root = tmp_path / "cgroup"
    d = root / "memory" / "pbs_jobs" / "123"
    d.mkdir(parents=True)
    (d / "memory.stat").write_text(
        f"cache 0\nhierarchical_memory_limit {32 * GIB}\ntotal_inactive_file {1 * GIB}\n",
        encoding="utf-8")
    (d / "memory.usage_in_bytes").write_text(f"{9 * GIB}\n", encoding="utf-8")
    self_cgroup = tmp_path / "self-cgroup"
    self_cgroup.write_text("11:memory:/pbs_jobs/123\n10:cpu,cpuacct:/pbs_jobs/123\n",
                           encoding="utf-8")
    level, out = _run(tmp_path, meminfo=meminfo, nproc=64, self_cgroup=self_cgroup,
                      cgroup_root=root, extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "6144"})
    assert level == "4", out  # (32 - 9 + 1) GiB = 24 GiB // 6 GiB


def test_cgroup_v1_unlimited_is_ignored(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 64_000 * 1024, 40_000 * 1024)
    root = tmp_path / "cgroup"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "memory.stat").write_text(
        "hierarchical_memory_limit 9223372036854771712\ntotal_inactive_file 0\n", encoding="utf-8")
    (root / "memory" / "memory.usage_in_bytes").write_text("0\n", encoding="utf-8")
    self_cgroup = tmp_path / "self-cgroup"
    self_cgroup.write_text("11:memory:/\n", encoding="utf-8")
    level, _ = _run(tmp_path, meminfo=meminfo, self_cgroup=self_cgroup, cgroup_root=root,
                    extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "10000"})
    assert level == "4"


def test_explicit_level_wins_verbatim(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", GPU_HOST_TOTAL_KB, 1_000 * 1024)
    level, out = _run(tmp_path, meminfo=meminfo, extra_env={"CMAKE_BUILD_PARALLEL_LEVEL": "7"})
    assert level == "7"
    assert "==> CMAKE_BUILD_PARALLEL_LEVEL" not in out


def test_derived_level_is_recomputed_at_each_build_start(tmp_path: Path) -> None:
    """A level the helper derived follows the memory free when the next phase starts."""
    meminfo = _meminfo(tmp_path / "meminfo", 128_000 * 1024, 64_000 * 1024)
    later = _meminfo(tmp_path / "meminfo-later", 128_000 * 1024, 16_000 * 1024)
    level, out = _run(
        tmp_path, meminfo=meminfo, nproc=32,
        extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "8000"},
        script=(
            'echo "LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"\n'
            f"_VIBEQC_MEMINFO={later} vibeqc_refresh_build_parallelism 'next phase'\n"
        ),
    )
    first = re.findall(r"^LEVEL=(\S+)$", out, flags=re.M)[0]
    assert (first, level) == ("8", "2"), out
    assert "for next phase" in out


def test_non_linux_leaves_the_level_unset(tmp_path: Path) -> None:
    level, out = _run(tmp_path, meminfo=None)
    assert level == "unset"
    assert "CMAKE_BUILD_PARALLEL_LEVEL=" not in out.replace("LEVEL=unset", "")


def test_invalid_budget_falls_back_to_default(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 1_000_000 * 1024, 900_000 * 1024)
    level, out = _run(tmp_path, meminfo=meminfo, nproc=4,
                      extra_env={"VIBEQC_BUILD_MEM_PER_JOB_MB": "lots"})
    assert level == "4", out


def _source_from(tmp_path: Path, name: str, meminfo: Path, nproc: int = 32) -> str:
    """Source the helper from a script called ``name``, the way build_*.sh do."""
    caller = tmp_path / name
    caller.write_text(
        "set -euo pipefail\n"
        f". {HELPER}\n"
        'echo "LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-unset}"\n', encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "VIBEQC_BUILD_NICED": "1",
        "VIBEQC_SKIP_PREFLIGHT": "1",
        "_VIBEQC_NPROC": str(nproc),
        "_VIBEQC_MEMINFO": str(meminfo),
        "_VIBEQC_PROC_SELF_CGROUP": str(tmp_path / "no-self-cgroup"),
    }
    done = subprocess.run([BASH, str(caller)], env=env, capture_output=True, text=True,
                          timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.mark.parametrize(("caller", "level"), [
    ("build_libint.sh", "4"),        # 24000 // 5000: libint unity TUs peak <= 4 GiB
    ("setup_native_deps.sh", "1"),   # 24000 // 15000: core-class budget
    ("update.sh", "1"),
])
def test_gpu_reserved_host_phase_budgets(tmp_path: Path, caller: str, level: str) -> None:
    """#411 on that host's figures: the libint phase keeps parallelism, the core phase does not OOM."""
    meminfo = _meminfo(tmp_path / "meminfo", GPU_HOST_TOTAL_KB, GPU_HOST_AVAIL_KB)
    out = _source_from(tmp_path, caller, meminfo)
    assert re.findall(r"^LEVEL=(\S+)$", out, flags=re.M) == [level], out
    assert f"for {caller}" in out


def test_explicit_budget_argument_for_the_core_build(tmp_path: Path) -> None:
    meminfo = _meminfo(tmp_path / "meminfo", 128_000 * 1024, 64_000 * 1024)
    level, out = _run(tmp_path, meminfo=meminfo, nproc=32,
                      script="vibeqc_refresh_build_parallelism 'vibe-qc core' 15000")
    assert level == "4", out  # 64000 // 15000
    assert "for vibe-qc core" in out and "15000MB-per-worker" in out


def test_core_pip_build_is_resized_from_memory_free_at_that_moment(tmp_path: Path) -> None:
    """vibeqc_pip_install_editable re-derives the level just before the core build."""
    early = _meminfo(tmp_path / "meminfo", 128_000 * 1024, 100_000 * 1024)
    late = _meminfo(tmp_path / "meminfo-late", 128_000 * 1024, 31_000 * 1024)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    stub = venv / "bin" / "python"
    stub.write_text('#!/bin/sh\necho "PIP_SAW_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"\n', encoding="utf-8")
    stub.chmod(0o755)
    helpers = REPO_ROOT / "scripts" / "_venv_helpers.sh"
    _, out = _run(
        tmp_path, meminfo=early, nproc=32,
        script=(
            f". {helpers}\n"
            f"_VIBEQC_MEMINFO={late} vibeqc_pip_install_editable {venv} ''\n"
        ),
    )
    # The build-requirements pip step still sees the level derived at start
    # (100000 // 15000 = 6); the core build itself sees 31000 // 15000 = 2.
    assert re.findall(r"^PIP_SAW_LEVEL=(\S+)$", out, flags=re.M) == ["6", "2"], out
    assert "for vibe-qc core" in out
