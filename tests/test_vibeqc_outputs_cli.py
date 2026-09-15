"""``vibeqc-outputs`` console script — Phase D3.

Pins the contract for the manifest-inspection CLI:

  1. ``vibeqc-outputs <stem>`` reads ``{stem}.system``, prints the
     job header (method/basis/functional, status, wall_seconds)
     and a per-file table (status / role / format / size / checksum
     / path).
  2. Accepts ``<stem>``, ``<stem>.out``, or ``<stem>.system`` —
     all resolve to the same .system file via Path.with_suffix.
  3. Missing manifest → exit 1.
  4. Pre-Phase-O1 manifest (no [plan]) → exit 2.
  5. ``--strict`` exits 3 if any always-on declared file is
     missing on disk.
  6. ``--paths-only`` prints just one path per line (pipeline-
     friendly).
  7. ``--missing-only`` restricts the table to MISS rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeqc.output import OutputPlan, dry_run_manifest
from vibeqc.output.outputs_cli import main as cli_main


def _seed(tmp_path: Path, **overrides) -> Path:
    stem = tmp_path / "job"
    kw = dict(
        output=stem, method="rks", basis="def2-svp",
        functional="PBE",
    )
    kw.update(overrides)
    plan = OutputPlan.from_run_job_kwargs(**kw)
    dry_run_manifest(plan, print_summary=False)
    return stem


# ---------------------------------------------------------------------- #
# Default stdout path
# ---------------------------------------------------------------------- #

def test_prints_job_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed(tmp_path)
    rc = cli_main([str(stem)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RKS" in out
    assert "def2-svp" in out
    assert "PBE" in out
    assert "molecular_scf" in out


def test_prints_per_file_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed(tmp_path)
    cli_main([str(stem)])
    out = capsys.readouterr().out
    # Per-file table headers should be present.
    assert "status" in out and "role" in out and "format" in out
    # And declared files (log/manifest/orbitals/...).
    assert "job.out" in out
    assert "job.system" in out


def test_accepts_dot_out_stem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed(tmp_path)
    rc = cli_main([str(stem.with_suffix(".out"))])
    assert rc == 0
    assert "manifest:" in capsys.readouterr().out


def test_accepts_dot_system_stem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed(tmp_path)
    rc = cli_main([str(stem.with_suffix(".system"))])
    assert rc == 0
    assert "manifest:" in capsys.readouterr().out


# ---------------------------------------------------------------------- #
# Status / written column
# ---------------------------------------------------------------------- #

def test_dry_run_manifest_shows_dry_run_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A dry-run has a real self-manifest but missing calculation files."""
    stem = _seed(tmp_path)
    cli_main([str(stem)])
    out = capsys.readouterr().out
    assert "dry_run" in out
    # Several files declared always=true but not on disk.
    assert "MISS" in out
    assert "self-excluded" in out


def test_legacy_manifest_self_hash_is_never_presented_as_trustworthy(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Inspectors override the impossible stale self-digest in old files."""
    stem = tmp_path / "legacy-self"
    manifest = stem.with_suffix(".system")
    stale = "ad3977f00badcafe"
    manifest.write_text(
        f'''[plan]
job_kind = "molecular_scf"
method = "RHF"
basis = "sto-3g"
functional = ""

[[plan.files]]
role = "manifest"
path = "{manifest}"
format = "toml"
always = true
description = "Runtime manifest."

[outputs]
status = "complete"
finished_at_iso = "2026-07-16T00:00:00+00:00"

[[outputs.files]]
path = "{manifest}"
written = true
bytes = 12223
sha256 = "{stale}"
wall_time_s = 1.0
''',
        encoding="utf-8",
    )

    assert cli_main([str(stem)]) == 0
    out = capsys.readouterr().out
    assert "self-excluded" in out
    assert stale[:8] not in out


# ---------------------------------------------------------------------- #
# --strict
# ---------------------------------------------------------------------- #

def test_strict_with_missing_files_returns_3(
    tmp_path: Path,
) -> None:
    stem = _seed(tmp_path)
    rc = cli_main([str(stem), "--strict"])
    assert rc == 3


def test_strict_with_all_present_returns_0(
    tmp_path: Path,
) -> None:
    """Materialise every declared file on disk; --strict should pass."""
    stem = _seed(tmp_path)
    # Touch each declared file so on_disk = True.
    import tomllib
    body = tomllib.loads(
        stem.with_suffix(".system").read_text(),
    )
    # Mark every always=true file as written in [outputs.files] and
    # actually create the file on disk.
    for row in body["plan"]["files"]:
        if not row["always"]:
            continue
        p = Path(row["path"])
        # The dry-run-seeded paths are absolute; touch as-is.
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    # Also flip the written flag in [outputs.files] — the CLI reads
    # that to decide between OK and MISS.
    for row in body["outputs"]["files"]:
        if row["path"] != str(stem.with_suffix(".system")):
            row["written"] = True
    # Re-emit the manifest with the flips.
    import json
    # Simplest: don't re-serialise TOML; just manually patch the file
    # in place by walking the lines. (The CLI's check uses
    # `written` from [outputs.files] AND on-disk existence; touching
    # the files alone is enough as long as the path exists.)
    # We just need on-disk existence and any plan_file row's
    # written-flag check. Re-write the file as TOML via a quick
    # round-trip via a serialiser. Easier: use a fresh plan that
    # declares only the .system file (which is auto-written).
    # — actually that won't exercise the all-OK case. Let me write
    # a tiny inline TOML emitter for the [outputs.files] flip.
    lines = stem.with_suffix(".system").read_text().splitlines()
    new_lines = []
    in_outputs_file = False
    for ln in lines:
        if ln.startswith("[[outputs.files]]"):
            in_outputs_file = True
        elif ln.startswith("[") and in_outputs_file:
            in_outputs_file = False
        if in_outputs_file and ln.startswith("written"):
            new_lines.append("written       = true")
            continue
        new_lines.append(ln)
    stem.with_suffix(".system").write_text("\n".join(new_lines))

    rc = cli_main([str(stem), "--strict"])
    assert rc == 0


# ---------------------------------------------------------------------- #
# --paths-only
# ---------------------------------------------------------------------- #

def test_paths_only_emits_one_path_per_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed(tmp_path)
    rc = cli_main([str(stem), "--paths-only"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) >= 4  # log + manifest + xyz + bibtex at least
    # No human header noise.
    assert "manifest:" not in out
    assert "status" not in out


# ---------------------------------------------------------------------- #
# --missing-only
# ---------------------------------------------------------------------- #

def test_missing_only_restricts_to_miss_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed(tmp_path)
    rc = cli_main([str(stem), "--missing-only"])
    assert rc == 0
    out = capsys.readouterr().out
    # Every row shown should be MISS.
    body_lines = [ln for ln in out.splitlines()
                   if ln.strip() and not ln.startswith((
                       "  ", "manifest:", "  job:", "  status:",
                       "  finished_at:", "  wall_seconds:",
                   ))]
    # Easier check: no "OK" rows in the table.
    table_lines = [ln for ln in out.splitlines()
                    if ln.startswith("  ") and len(ln.strip()) > 0
                    and not ln.startswith("  status")
                    and not ln.startswith("  ----")]
    # The OK rows would start with "  OK"; none expected.
    assert not any(ln.lstrip().startswith("OK") for ln in table_lines)


# ---------------------------------------------------------------------- #
# Error paths
# ---------------------------------------------------------------------- #

def test_missing_manifest_returns_1(tmp_path: Path) -> None:
    rc = cli_main([str(tmp_path / "nope")])
    assert rc == 1


def test_pre_phase_o1_manifest_returns_2(
    tmp_path: Path,
) -> None:
    stem = tmp_path / "legacy"
    stem.with_suffix(".system").write_text(
        '[vibeqc]\nversion = "0.7.0"\n'
        '[run]\nbasename = "legacy"\n',
        encoding="utf-8",
    )
    rc = cli_main([str(stem)])
    assert rc == 2


def test_malformed_toml_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stem = tmp_path / "broken"
    stem.with_suffix(".system").write_text(
        "this is not [[ valid toml\n",
        encoding="utf-8",
    )
    rc = cli_main([str(stem)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "failed to parse" in err
