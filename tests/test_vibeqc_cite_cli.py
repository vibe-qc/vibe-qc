"""``vibeqc-cite`` console script — re-emit citations for a run.

Pins the contract documented in
``docs/design_output_module.md § Phase O3 — vibeqc-cite CLI``:

  1. ``vibeqc-cite <stem>`` reads ``{stem}.system``, extracts the
     ``[plan]`` section, walks the citation database, and prints
     the plain-text reference list to stdout.
  2. ``--write`` writes ``{stem}.bibtex`` + ``{stem}.references``
     siblings instead of printing.
  3. ``--bibtex-only`` selects only the BibTeX surface (stdout or
     file).
  4. A missing manifest exits non-zero with a clear stderr message.
  5. A pre-Phase-O1 manifest (no ``[plan]`` section) exits non-zero
     and instructs the user to re-run the job.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from vibeqc.output import OutputPlan, dry_run_manifest
from vibeqc.output.citations.cli import main as cite_main


def _seed_manifest(tmp_path: Path, **overrides) -> Path:
    """Write a dry-run manifest at ``{tmp_path}/job.system`` and
    return the stem path. ``overrides`` are passed to
    ``OutputPlan.from_run_job_kwargs``."""
    stem = tmp_path / "job"
    kw = dict(
        output=stem,
        method="rks",
        basis="def2-svp",
        functional="PBE",
    )
    kw.update(overrides)
    plan = OutputPlan.from_run_job_kwargs(**kw)
    dry_run_manifest(plan, print_summary=False)
    return stem


# ---------------------------------------------------------------------- #
# Default stdout path
# ---------------------------------------------------------------------- #

def test_cite_prints_plain_references_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed_manifest(tmp_path)
    rc = cite_main([str(stem)])
    assert rc == 0
    out = capsys.readouterr().out
    # Plain-text formatter starts each entry with [N] and includes
    # the software citation first.
    assert "[1]" in out
    assert "vibe-qc" in out.lower()
    # PBE-on-def2-svp should pull libxc + Perdew + Weigend-Ahlrichs.
    assert "Lehtola" in out
    assert "Perdew" in out
    assert "Weigend" in out


def test_cite_bibtex_only_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed_manifest(tmp_path)
    rc = cite_main([str(stem), "--bibtex-only"])
    assert rc == 0
    out = capsys.readouterr().out
    # BibTeX entries start with @<type>{.
    assert "@" in out
    # PBE-on-def2-svp pulls the canonical bibtex keys we registered.
    assert "perdew_burke_ernzerhof_1996" in out
    assert "lehtola_libxc_2018" in out


# ---------------------------------------------------------------------- #
# --write
# ---------------------------------------------------------------------- #

def test_cite_write_emits_both_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed_manifest(tmp_path)
    rc = cite_main([str(stem), "--write"])
    assert rc == 0
    assert stem.with_suffix(".bibtex").is_file()
    assert stem.with_suffix(".references").is_file()
    # The CLI reports the files it wrote on stdout.
    out = capsys.readouterr().out
    assert ".bibtex" in out
    assert ".references" in out


def test_cite_write_bibtex_only_skips_references(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed_manifest(tmp_path)
    rc = cite_main([str(stem), "--write", "--bibtex-only"])
    assert rc == 0
    assert stem.with_suffix(".bibtex").is_file()
    assert not stem.with_suffix(".references").is_file()


# ---------------------------------------------------------------------- #
# Error paths
# ---------------------------------------------------------------------- #

def test_cite_missing_manifest_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stem = tmp_path / "nope"
    with pytest.raises(SystemExit) as ei:
        cite_main([str(stem)])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "manifest not found" in err


def test_cite_pre_phase_o1_manifest_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare .system without [plan] should fail loud."""
    stem = tmp_path / "legacy"
    stem.with_suffix(".system").write_text(
        "[vibeqc]\nversion = \"0.7.0\"\n"
        "[run]\nbasename = \"legacy\"\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as ei:
        cite_main([str(stem)])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "[plan]" in err


# ---------------------------------------------------------------------- #
# Argument handling — accepting stem with .out / .system suffix
# ---------------------------------------------------------------------- #

def test_cite_accepts_path_with_out_suffix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stem = _seed_manifest(tmp_path)
    # User-friendly: pass the .out path; vibeqc-cite normalises to
    # the .system sibling internally via Path.with_suffix.
    rc = cite_main([str(stem.with_suffix(".out"))])
    assert rc == 0
    assert "[1]" in capsys.readouterr().out
