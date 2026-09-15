"""Run pre-existing CRYSTAL14 .d12 files through the vibe-basis pipeline.

Handles the format produced by the user's basis-set optimization
code::

    title
    CRYSTAL
    0 0 0
    spacegroup
    lattice_param
    natoms
    Z f_x f_y f_z
    OPTGEOM              ← geometry optimization (not single-point)
    ENDOPT
    ENDGEOM
    Z nshells ...        ← inline basis set
    99 0
    ENDBS
    DFT / PW1PW / END[DFT]
    TOLINTEG N N N N N
    SHRINK S S
    END

This differs from :func:`emit_input` which produces single-point
SCF decks with ``BASISSET`` keyword and ``TOLDEE``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from vibe_basis.backends.crystal import (
    CrystalEnergyResult,
    parse_output_file,
)
from vibe_basis.transports.base import Transport
from vibe_basis.workdir import resolve_workdir


@dataclass
class D12RunResult:
    """Result of running a single .d12 file."""

    d12_path: Path
    compound: str
    ok: bool
    energy: Optional[float]
    method: Optional[str]
    last_cycle: Optional[int]
    failure_mode: Optional[str]
    n_lines: int

    @property
    def energy_ha(self) -> Optional[float]:
        return self.energy


@dataclass
class D12BatchReport:
    """Aggregate report from running multiple .d12 files."""

    files_total: int
    files_completed: int
    files_failed: int
    results: list[D12RunResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f".d12 batch — {self.files_completed}/{self.files_total} completed"]
        for r in sorted(self.results, key=lambda r: r.compound):
            if r.ok:
                lines.append(
                    f"  {r.compound:<8} {r.method:<6} "
                    f"{r.energy:>14.6f} Ha  (cycle {r.last_cycle})"
                )
            else:
                lines.append(f"  {r.compound:<8} FAILED — {r.failure_mode}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# .d12 metadata extractor (don't run CRYSTAL, just read the file)
# ---------------------------------------------------------------------------

_TITLE_RE = re.compile(r"^([A-Za-z].*)$")
_GEOMETRY_RE = re.compile(r"OPTGEOM", re.IGNORECASE)
_PW1PW_RE = re.compile(r"PW1PW", re.IGNORECASE)
_FUNCTIONAL_RE = re.compile(r"^\s*(PW1PW|PBE|B3LYP|PBE0|HF|RHF)\s*$", re.IGNORECASE)
_INLINE_BASIS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


def inspect_d12(path: Path) -> dict[str, str]:
    """Extract metadata from a .d12 file without running CRYSTAL.

    Returns a dict with keys: compound, method, has_optgeom,
    has_inline_basis, format.
    """
    text = path.read_text()
    lines = text.splitlines()
    info: dict[str, str] = {
        "compound": path.stem.replace("_seg_PW1PW", ""),
        "method": "unknown",
        "has_optgeom": "no",
        "has_inline_basis": "no",
    }

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if _TITLE_RE.match(stripped) and info["compound"] == path.stem:
            info["compound"] = stripped.split()[0]
        if _GEOMETRY_RE.search(stripped):
            info["has_optgeom"] = "yes"
        if _PW1PW_RE.search(stripped):
            info["method"] = "PW1PW"
        elif "ENDDFT" in stripped.upper():
            pass  # closing DFT block
        elif _FUNCTIONAL_RE.match(stripped) and info["method"] == "unknown":
            info["method"] = stripped.upper()
        if _INLINE_BASIS_RE.match(stripped):
            # "12 12" or "8 13" — Z followed by nshells
            info["has_inline_basis"] = "yes"

    return info


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_d12_file(
    d12_path: Path,
    transport: Transport,
    *,
    workdir_root: str | Path | None = None,
    cpus: int = 4,
    wall_time_s: int = 7200,
    crystal_wrapper: str = "crystal",
    timeout_s: float = 86_400.0,
) -> D12RunResult:
    """Run a single pre-existing .d12 file through CRYSTAL14.

    Parameters
    ----------
    d12_path
        Path to the .d12 input file (on the local machine).
        The basename will be used as the run name.
    transport
        Transport to use (LocalTransport or VqTransport).
    workdir_root
        Directory for per-run staging. When ``None`` (default),
        resolves via :func:`vibe_basis.workdir.resolve_workdir`
        (``$VQ_WORKDIR`` when running under vq, otherwise a new
        ``tempfile.mkdtemp``). This keeps scratch files out of
        the git checkout.
    cpus, wall_time_s
        Passed to transport.
    crystal_wrapper
        Command or script that launches CRYSTAL14.
    timeout_s
        Maximum wait time for transport.wait().

    Returns
    -------
    D12RunResult
    """
    d12_path = Path(d12_path)
    compound = d12_path.stem.replace("_seg_PW1PW", "")

    if workdir_root is None:
        workdir_root = resolve_workdir()
    workdir = Path(workdir_root) / compound
    if workdir.exists():
        import shutil

        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    # Copy the .d12 into the workdir.
    dest = workdir / f"{compound}.d12"
    dest.write_text(d12_path.read_text())

    try:
        job = transport.run(
            workdir,
            command=[crystal_wrapper, f"{compound}.d12"],
            dest=workdir / "fetched",
            cpus=cpus,
            wall_time_s=wall_time_s,
            label=f"d12/{compound}",
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return D12RunResult(
            d12_path=d12_path,
            compound=compound,
            ok=False,
            energy=None,
            method=None,
            last_cycle=None,
            failure_mode=f"transport_error: {exc}",
            n_lines=0,
        )

    out_path = job.output_dir / f"{compound}.out"
    try:
        parsed = parse_output_file(out_path)
    except FileNotFoundError:
        return D12RunResult(
            d12_path=d12_path,
            compound=compound,
            ok=False,
            energy=None,
            method=None,
            last_cycle=None,
            failure_mode="output_file_not_found",
            n_lines=0,
        )

    return D12RunResult(
        d12_path=d12_path,
        compound=compound,
        ok=parsed.ok,
        energy=parsed.energy,
        method=parsed.method,
        last_cycle=parsed.last_cycle,
        failure_mode=None if parsed.ok else parsed.failure_mode,
        n_lines=parsed.n_lines_scanned,
    )


def run_d12_batch(
    d12_paths: Iterable[Path],
    transport: Transport,
    *,
    workdir_root: str | Path | None = None,
    cpus: int = 4,
    wall_time_s: int = 7200,
    crystal_wrapper: str = "crystal",
    timeout_s: float = 86_400.0,
) -> D12BatchReport:
    """Run multiple .d12 files and collect results.

    Parameters match :func:`run_d12_file`; applied to each file.
    """
    paths = list(d12_paths)
    results: list[D12RunResult] = []
    completed = 0
    failed = 0

    if workdir_root is None:
        workdir_root = resolve_workdir()

    for p in paths:
        r = run_d12_file(
            p,
            transport,
            workdir_root=workdir_root,
            cpus=cpus,
            wall_time_s=wall_time_s,
            crystal_wrapper=crystal_wrapper,
            timeout_s=timeout_s,
        )
        results.append(r)
        if r.ok:
            completed += 1
        else:
            failed += 1

    return D12BatchReport(
        files_total=len(paths),
        files_completed=completed,
        files_failed=failed,
        results=results,
    )
