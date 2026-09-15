"""Per-job system manifest — shape, hostname redaction, run_job wiring.

Pins the contract documented in :mod:`vibeqc.system_info` and in
``docs/user_guide/output_files.md § The .system manifest``:

  1. ``system_info()`` returns a fixed-shape dict — every section + key
     listed in the docs is present, even when a probe falls back to
     ``"unknown"``.
  2. The dict round-trips through stdlib ``tomllib`` after being
     rendered by ``write_system_manifest`` (the on-disk format is
     parseable TOML, not just convention-shaped text).
  3. ``VIBEQC_NO_HOSTNAME=1`` and ``record_hostname=False`` both emit
     ``hostname = "<redacted>"`` (privacy contract — engineering's
     bundled docs runs depend on this).
  4. ``run_job(output=...)`` writes the manifest sibling to the ``.out``
     file with a populated ``[run].wall_seconds``.
  5. Probe failures (a monkeypatched ``platform.processor`` that
     raises) never abort the calculation — the manifest still writes
     with ``cpu.model = "unknown"``.
  6. The manifest records the validation boundary: external QC programs
     are references only, run out-of-process, and are not imported as
     vibe-qc backends.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

from vibeqc import Atom, Molecule, run_job, system_info, write_system_manifest


# ---------------------------------------------------------------------------
# Sections + keys we promise to always emit. If you add a key to
# system_info(), add it here too — and document it in
# docs/user_guide/output_files.md.
# ---------------------------------------------------------------------------

EXPECTED_SHAPE: dict[str, set[str]] = {
    "vibeqc":    {"version", "codename", "git_sha", "git_branch", "is_release"},
    "host":      {"hostname", "os", "os_release", "os_pretty", "arch"},
    "cpu":       {"model", "physical_cores", "logical_cores", "omp_threads_used"},
    "memory":    {"total_gb", "available_gb"},
    "python":    {"version", "implementation", "executable"},
    "libraries": {"libint", "libxc", "spglib", "libecpint", "fftw3"},
    "validation": {
        "external_programs_policy",
        "execution_boundary",
        "native_backend_policy",
    },
}


# ---------------------------------------------------------------------------
# 1. Shape + TOML round-trip
# ---------------------------------------------------------------------------

def test_system_manifest_shape() -> None:
    """Every documented section + key is present in system_info() and
    in the rendered TOML; the rendered TOML round-trips through
    stdlib tomllib."""
    info = system_info()
    for section, keys in EXPECTED_SHAPE.items():
        assert section in info, f"missing section: {section}"
        for key in keys:
            assert key in info[section], f"missing key: {section}.{key}"


def test_system_manifest_prefers_full_immutable_git_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.output.formats.system_info as system_info_module

    full_sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(
        system_info_module,
        "build_info",
        lambda: {
            "sha": full_sha[:7],
            "sha_full": full_sha,
            "branch": "main",
            "is_release": False,
        },
    )
    assert system_info_module.system_info()["vibeqc"]["git_sha"] == full_sha


def test_system_manifest_round_trips_through_tomllib(tmp_path: Path) -> None:
    """The hand-rolled TOML emitter must produce output that stdlib
    tomllib parses back to the same shape — no ad-hoc format that
    only round-trips through itself."""
    manifest_path = write_system_manifest(
        tmp_path / "x.out",
        wall_seconds=1.234,
        basename="x",
    )
    assert manifest_path == tmp_path / "x.system"
    assert manifest_path.is_file()

    with open(manifest_path, "rb") as f:
        parsed = tomllib.load(f)

    for section, keys in EXPECTED_SHAPE.items():
        assert section in parsed
        for key in keys:
            assert key in parsed[section]

    # Per-run section populated by write_system_manifest itself.
    assert "run" in parsed
    assert parsed["run"]["wall_seconds"] == pytest.approx(1.234)
    assert parsed["run"]["basename"] == "x"
    assert isinstance(parsed["run"]["timestamp_iso"], str)


def test_system_manifest_records_external_validation_boundary() -> None:
    info = system_info()
    validation = info["validation"]
    assert "validation references only" in validation["external_programs_policy"]
    assert "out-of-process" in validation["execution_boundary"]
    assert "do not import" in validation["execution_boundary"]
    assert "vibe-qc-owned" in validation["native_backend_policy"]


# ---------------------------------------------------------------------------
# 2. Hostname redaction (privacy contract)
# ---------------------------------------------------------------------------

def test_system_manifest_no_hostname_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VIBEQC_NO_HOSTNAME", "1")
    info = system_info()
    assert info["host"]["hostname"] == "<redacted>"


def test_system_manifest_no_hostname_via_kwarg() -> None:
    info = system_info(record_hostname=False)
    assert info["host"]["hostname"] == "<redacted>"


def test_system_manifest_no_hostname_also_redacts_exe_path() -> None:
    """record_hostname=False must also home-relativise the Python exe
    path — the same opt-out that suppresses the hostname must not leave
    /Users/<user>/... leaking via the python.executable field (§12)."""
    info = system_info(record_hostname=False)
    exe = info["python"]["executable"]
    home = str(Path.home())
    assert not exe.startswith(home), (
        f"python.executable {exe!r} still starts with home {home!r} "
        f"after record_hostname=False"
    )


def test_system_manifest_no_hostname_env_also_redacts_exe_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VIBEQC_NO_HOSTNAME=1 must also home-relativise the Python exe."""
    monkeypatch.setenv("VIBEQC_NO_HOSTNAME", "1")
    info = system_info()
    exe = info["python"]["executable"]
    home = str(Path.home())
    assert not exe.startswith(home), (
        f"python.executable {exe!r} still starts with home {home!r} "
        f"with VIBEQC_NO_HOSTNAME=1"
    )


def test_system_manifest_hostname_present_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no opt-out, the live hostname is recorded — we don't drop
    the field. Test a non-pathological default state by clearing the
    env var explicitly so a developer running with it set in their
    shell doesn't see a surprise pass-through."""
    monkeypatch.delenv("VIBEQC_NO_HOSTNAME", raising=False)
    info = system_info(record_hostname=True)
    assert info["host"]["hostname"] != "<redacted>"
    assert info["host"]["hostname"]  # non-empty


# ---------------------------------------------------------------------------
# 3. run_job integration
# ---------------------------------------------------------------------------

def test_system_manifest_run_job_integration(tmp_path: Path) -> None:
    """run_job writes the manifest sibling to the .out file with a
    populated wall_seconds field — that's the headline UX contract:
    one call writes both files, no extra step from the user."""
    mol = Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])
    stem = tmp_path / "tmp_test"
    run_job(mol, basis="sto-3g", method="rhf", output=stem)

    manifest = stem.with_suffix(".system")
    assert manifest.is_file(), (
        f"run_job(output={stem!r}) should write {manifest.name} "
        f"sibling to {stem.with_suffix('.out').name}"
    )
    with open(manifest, "rb") as f:
        parsed = tomllib.load(f)
    assert parsed["run"]["wall_seconds"] > 0.0
    assert parsed["run"]["basename"] == "tmp_test"
    rows = {row["path"]: row for row in parsed["outputs"]["files"]}
    self_row = rows[str(manifest)]
    assert self_row["written"] is True
    assert self_row["bytes"] == 0
    assert self_row["sha256"] == ""
    assert self_row["checksum_status"] == "self-excluded"


def test_system_manifest_run_job_record_hostname_false(
    tmp_path: Path,
) -> None:
    """run_job(record_hostname=False) propagates to the manifest — no
    need to also set the env var for one-off scripts."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    stem = tmp_path / "h2"
    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=stem,
        record_hostname=False,
    )

    with open(stem.with_suffix(".system"), "rb") as f:
        parsed = tomllib.load(f)
    assert parsed["host"]["hostname"] == "<redacted>"


# ---------------------------------------------------------------------------
# 4. Probe-failure resilience
# ---------------------------------------------------------------------------

def test_system_manifest_invalid_platform_doesnt_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A pathological platform.processor (e.g. broken sysctl on a
    locked-down container) must not propagate up and abort the
    calculation. The manifest should still write with the relevant
    field falling back to ``"unknown"`` or another platform-specific
    fallback."""
    import platform

    def _boom() -> str:
        raise RuntimeError("simulated platform-probe failure")

    monkeypatch.setattr(platform, "processor", _boom)

    # system_info itself must not raise.
    info = system_info()
    assert "model" in info["cpu"]

    # And the on-disk write path must succeed.
    manifest_path = write_system_manifest(
        tmp_path / "robust.out",
        wall_seconds=0.0,
        basename="robust",
    )
    assert manifest_path.is_file()
    with open(manifest_path, "rb") as f:
        parsed = tomllib.load(f)
    assert "model" in parsed["cpu"]


# ---------------------------------------------------------------------------
# 5. TOML formatter — value-quoting contract
# ---------------------------------------------------------------------------

def test_toml_string_with_quotes_round_trips(tmp_path: Path) -> None:
    """Strings carrying double-quotes / backslashes (Windows paths,
    quoted CPU vendor strings) must be escaped so the manifest parses
    back."""
    from vibeqc.system_info import _format_manifest

    info = {
        "host": {"hostname": 'has-"quotes"'},
        "python": {"executable": r"C:\Python\python.exe"},
    }
    text = _format_manifest(info)
    target = tmp_path / "weird.system"
    target.write_text(text, encoding="utf-8")
    with open(target, "rb") as f:
        parsed = tomllib.load(f)
    assert parsed["host"]["hostname"] == 'has-"quotes"'
    assert parsed["python"]["executable"] == r"C:\Python\python.exe"
