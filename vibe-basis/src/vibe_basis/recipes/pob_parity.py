"""Goal 8 Stage 0 — pob-TZVP HF parity reproduction driver.

Wires the existing building blocks into an end-to-end pipeline
smoke test::

    structures → emit_input → transport.run → parse_output → compare

This driver is the immediate next milestone from the 2026-05-14
handover.  It proves the CRYSTAL14 + vq + output-parser pipeline
works end-to-end *before* touching the basis.

Acceptance gate (per :ref:`go8-stage0`):
  Σᵢ |E_HF(vibe-basis pipeline) − E_HF(PT2013 SI Table 2)| < 0.1 mHa

**Important caveat.**  PT2013 SI Table 2 energies were computed at
the *paper's relaxed geometries*, while the structure database
uses *experimental lattice constants*.  The gate measures pipeline
correctness, not numerical parity.  When geometries differ by
~0.01 Å, the per-compound energy drift can exceed 0.1 mHa for
sensitive systems (hydrides).  The actual Stage-0 threshold should
be relaxed to ~1 mHa per compound if geometry effects dominate.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from vibe_basis.backends.crystal import (
    CrystalEnergyResult,
    emit_input,
    parse_output_file,
)
from vibe_basis.io.references import get_ref
from vibe_basis.io.structures import Structure
from vibe_basis.transports.base import Transport

# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompoundResult:
    """Per-compound result from one pipeline evaluation."""

    compound: str
    formula: str
    ok: bool
    energy: Optional[float]  # Ha per unit cell
    method: Optional[str]
    last_cycle: Optional[int]
    ref_energy: Optional[float]  # from PT2013 SI Table 2
    delta_mha: Optional[float]  # (computed − ref) in mHa
    failure_mode: Optional[str]  # if not ok
    notes: str = ""


@dataclass
class ParityReport:
    """Aggregate report from a complete Stage 0 parity run."""

    basis: str
    method: str
    compounds_total: int
    compounds_emitted: int  # non-AFM cubic that produced a deck
    compounds_converged: int
    compounds_compared: int  # converged AND have ref energy
    results: list[CompoundResult] = field(default_factory=list)
    sum_abs_delta_mha: float = 0.0

    @property
    def all_converged(self) -> bool:
        return self.compounds_converged == self.compounds_emitted

    @property
    def passes_acceptance(self) -> bool:
        """Acceptance gate: Σ|ΔE| < 0.1 mHa.

        **Caveat** — this threshold assumes the paper's relaxed
        geometry.  With experimental lattices the gate should be
        interpreted as pipeline-correctness, not µHa-accurate
        numerical parity.
        """
        return self.sum_abs_delta_mha < 0.1

    def summary(self) -> str:
        """Return a human-readable summary of the parity run."""
        lines: list[str] = []
        lines.append(f"Stage 0 — {self.basis.upper()} {self.method.upper()} parity")
        lines.append(
            f"  compounds: {self.compounds_total} total → "
            f"{self.compounds_emitted} emitted → "
            f"{self.compounds_converged} converged → "
            f"{self.compounds_compared} compared"
        )
        for r in sorted(self.results, key=lambda r: r.compound):
            if r.ok and r.delta_mha is not None:
                lines.append(
                    f"  {r.compound:<8} {r.energy:>14.6f}  Δ = {r.delta_mha:+.3f} mHa"
                )
            elif r.ok:
                lines.append(f"  {r.compound:<8} {r.energy:>14.6f}  (no reference)")
            elif r.notes:
                lines.append(
                    f"  {r.compound:<8} FAILED — {r.failure_mode}  [{r.notes}]"
                )
            else:
                lines.append(f"  {r.compound:<8} FAILED — {r.failure_mode}")
        lines.append(f"  Σ|ΔE| = {self.sum_abs_delta_mha:.3f} mHa")
        lines.append(
            f"  gate = {self.sum_abs_delta_mha:.3f} / 0.1 mHa  "
            f"{'PASS' if self.passes_acceptance else 'FAIL'}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_compound_result(
    s: Structure,
    parsed: CrystalEnergyResult,
) -> CompoundResult:
    """Build a CompoundResult from parsed CRYSTAL14 output."""
    ref_e = get_ref(s.name)
    delta_mha = None
    if parsed.ok and ref_e is not None and parsed.energy is not None:
        delta_mha = (parsed.energy - ref_e) * 1000.0  # Ha → mHa

    notes = ""
    if parsed.ok and ref_e is None:
        notes = "no reference in PT2013 SI Table 2"
    elif parsed.ok and s.name == "NaF":
        notes = (
            "PT2013 SI Table 2 lists a dash for NaF/HF "
            "(SCF didn't converge in the paper)"
        )

    return CompoundResult(
        compound=s.name,
        formula=s.formula,
        ok=parsed.ok,
        energy=parsed.energy,
        method=parsed.method,
        last_cycle=parsed.last_cycle,
        ref_energy=ref_e,
        delta_mha=delta_mha,
        failure_mode=None if parsed.ok else parsed.failure_mode,
        notes=notes,
    )


def _dedup_sort(structures: Iterable[Structure]) -> list[Structure]:
    """Deduplicate + sort the structure list by name."""
    seen: set[str] = set()
    out: list[Structure] = []
    for s in sorted(structures, key=lambda s: (s.crystal_system, s.name)):
        if s.name not in seen:
            seen.add(s.name)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Core driver
# ---------------------------------------------------------------------------


def run_pob_parity(
    structures: Iterable[Structure],
    transport: Transport,
    *,
    basis: str = "pob-tzvp",
    method: str = "rhf",
    workdir_root: str | Path = "pob_parity_runs",
    cpus: int = 14,
    wall_time_s: int = 1800,
    crystal_wrapper: str = "crystal",
    shrink: int = 8,
    toldee: int = 8,
    timeout_s: float = 86_400.0,
) -> ParityReport:
    """Run the Stage 0 pob-TZVP HF parity pipeline.

    For every non-AFM cubic structure in *structures*:

    1. Emit a CRYSTAL14 ``.d12`` deck via :func:`emit_input`.
    2. Submit it through *transport*.
    3. Parse the output via :func:`parse_output_file`.
    4. Compare the converged energy to the PT2013 SI Table 2
       reference, if available.

    Parameters
    ----------
    structures
        Iterable of :class:`vibe_basis.io.structures.Structure`
        entries.  Typically ``in_table("PT2013-T4")`` for the 13
        cubic ionics, or ``all_structures()`` for the full
        database.
    transport
        Concrete :class:`Transport` implementation.  Use
        :class:`LocalTransport` for laptop smoke tests,
        :class:`VqTransport` for compute-host production runs.
    basis
        CRYSTAL basis keyword.  Must match a basis CRYSTAL
        recognises on the target machine.
    method
        ``"rhf"``/``"hf"`` (no DFT block) or a recognised
        CRYSTAL functional keyword (``"pw1pw"``, ``"pbe"``, …).
    workdir_root
        Directory under which per-compound work directories are
        created.
    cpus, wall_time_s
        Passed to ``transport.submit``.
    crystal_wrapper
        Command (or path to a wrapper script) that launches
        CRYSTAL14.  For compute-host this is a ``run-crystal.sh``
        script; for local dev an adjustable wrapper.
    shrink, toldee
        SCF convergence parameters — see
        :func:`vibe_basis.backends.crystal.emit_input`.
    timeout_s
        Maximum wall-time for ``transport.wait()`` per compound.

    Returns
    -------
    ParityReport
        Summary with per-compound deltas and pass/fail status.
    """
    workdir = Path(workdir_root)
    workdir.mkdir(parents=True, exist_ok=True)

    ordered = _dedup_sort(structures)
    n_total = len(ordered)
    n_emitted = 0
    n_converged = 0
    n_compared = 0
    results: list[CompoundResult] = []

    for s in ordered:
        # Emit .d12 deck — AFM / unsupported compounds return None.
        deck = emit_input(s, basis, method, shrink=shrink, toldee=toldee)
        if deck is None:
            continue
        n_emitted += 1

        # Per-compound work directory.
        wd = workdir / s.name
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir()
        (wd / f"{s.name}.d12").write_text(deck)

        # Submit → wait → fetch.
        try:
            job = transport.run(
                wd,
                command=[crystal_wrapper, f"{s.name}.d12"],
                dest=wd / "fetched",
                cpus=cpus,
                wall_time_s=wall_time_s,
                label=f"pob_parity/{s.name}",
                timeout_s=timeout_s,
            )
        except Exception as exc:
            results.append(
                CompoundResult(
                    compound=s.name,
                    formula=s.formula,
                    ok=False,
                    energy=None,
                    method=None,
                    last_cycle=None,
                    ref_energy=get_ref(s.name),
                    delta_mha=None,
                    failure_mode=f"transport_error: {exc}",
                )
            )
            continue

        # Parse CRYSTAL14 output.
        out_path = job.output_dir / f"{s.name}.out"
        try:
            parsed = parse_output_file(out_path)
        except FileNotFoundError:
            parsed = CrystalEnergyResult(
                ok=False,
                energy=None,
                method=None,
                last_cycle=None,
                converged=False,
                truncated=False,
                failure_mode="output_file_not_found",
                n_lines_scanned=0,
            )

        result = _make_compound_result(s, parsed)
        if parsed.ok and result.delta_mha is not None:
            n_compared += 1
        if parsed.ok:
            n_converged += 1
        results.append(result)

    sum_delta = sum(
        abs(r.delta_mha) for r in results if r.ok and r.delta_mha is not None
    )
    return ParityReport(
        basis=basis,
        method=method,
        compounds_total=n_total,
        compounds_emitted=n_emitted,
        compounds_converged=n_converged,
        compounds_compared=n_compared,
        results=results,
        sum_abs_delta_mha=sum_delta,
    )
