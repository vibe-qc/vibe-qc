"""Tolerance-based pass / marginal / fail classification of ΔE."""
from __future__ import annotations

import math
from numbers import Real
from typing import Optional, Tuple

from .case import CodeRow
from .spec import ExpectedRef


def _positive(value: Optional[int]) -> Optional[int]:
    return value if value is not None and value > 0 else None


def _set_normalized_deltas(row: CodeRow) -> None:
    """Populate mHa-scaled and size-normalized delta fields."""
    if row.delta_ha_vs_ref is None:
        row.delta_mha_vs_ref = None
        row.abs_delta_mha_vs_ref = None
        row.delta_mha_per_atom_vs_ref = None
        row.abs_delta_mha_per_atom_vs_ref = None
        row.delta_mha_per_electron_vs_ref = None
        row.abs_delta_mha_per_electron_vs_ref = None
        row.delta_mha_per_basis_function_vs_ref = None
        row.abs_delta_mha_per_basis_function_vs_ref = None
        return

    delta_mha = 1000.0 * row.delta_ha_vs_ref
    row.delta_mha_vs_ref = delta_mha
    row.abs_delta_mha_vs_ref = abs(delta_mha)

    n_atoms = _positive(row.n_atoms)
    row.delta_mha_per_atom_vs_ref = (
        delta_mha / n_atoms if n_atoms is not None else None
    )
    row.abs_delta_mha_per_atom_vs_ref = (
        abs(row.delta_mha_per_atom_vs_ref)
        if row.delta_mha_per_atom_vs_ref is not None else None
    )

    n_electrons = _positive(row.n_electrons)
    row.delta_mha_per_electron_vs_ref = (
        delta_mha / n_electrons if n_electrons is not None else None
    )
    row.abs_delta_mha_per_electron_vs_ref = (
        abs(row.delta_mha_per_electron_vs_ref)
        if row.delta_mha_per_electron_vs_ref is not None else None
    )

    n_bf = _positive(row.n_basis_functions)
    row.delta_mha_per_basis_function_vs_ref = (
        delta_mha / n_bf if n_bf is not None else None
    )
    row.abs_delta_mha_per_basis_function_vs_ref = (
        abs(row.delta_mha_per_basis_function_vs_ref)
        if row.delta_mha_per_basis_function_vs_ref is not None else None
    )


def _normalization_note(row: CodeRow) -> str:
    if row.abs_delta_mha_vs_ref is None:
        return ""
    parts = [f"|Δ|={row.abs_delta_mha_vs_ref:.3f} mHa"]
    if row.abs_delta_mha_per_atom_vs_ref is not None:
        parts.append(f"{row.abs_delta_mha_per_atom_vs_ref:.3f} mHa/atom")
    if row.abs_delta_mha_per_electron_vs_ref is not None:
        parts.append(f"{row.abs_delta_mha_per_electron_vs_ref:.3f} mHa/electron")
    if row.abs_delta_mha_per_basis_function_vs_ref is not None:
        parts.append(
            f"{row.abs_delta_mha_per_basis_function_vs_ref:.3f} mHa/bf"
        )
    return "; normalized " + ", ".join(parts)


def classify(
    delta_ha: Optional[float], tolerance_ha: float, *,
    marginal_factor: float = 3.0,
) -> Tuple[str, str]:
    """Return (status, note) for a ΔE against a tolerance.

    pass     : |Δ| <= tolerance
    marginal : tolerance < |Δ| <= marginal_factor * tolerance
    fail     : |Δ| > marginal_factor * tolerance
    error    : invalid policy or unavailable/nonfinite delta
    """
    if not _finite_number(tolerance_ha) or tolerance_ha < 0:
        return "error", "tolerance must be finite and nonnegative"
    if not _finite_number(marginal_factor) or marginal_factor < 1:
        return "error", "marginal factor must be finite and at least one"
    if not _finite_number(delta_ha):
        return "error", "delta unavailable or nonfinite"
    abs_d = abs(delta_ha)
    if abs_d <= tolerance_ha:
        return "pass", f"|Δ|={abs_d:.3e} Ha <= tol {tolerance_ha:.3e}"
    if abs_d <= marginal_factor * tolerance_ha:
        return "marginal", (
            f"|Δ|={abs_d:.3e} Ha within {marginal_factor:.0f}× tol "
            f"({tolerance_ha:.3e})"
        )
    return "fail", (
        f"|Δ|={abs_d:.3e} Ha exceeds {marginal_factor:.0f}× tol "
        f"({tolerance_ha:.3e})"
    )


def _finite_number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def scf_energy_problem(converged, energy) -> str:
    """An energy alone does not attest a successfully converged calculation."""
    if converged is not True:
        return "SCF convergence is false or unconfirmed"
    if not _finite_number(energy):
        return "energy unavailable or nonfinite"
    return ""


def _case_key(row):
    return row.system_id, row.basis, row.method_id, row.kmesh


def _case_mismatch(row, reference) -> bool:
    if _case_key(row) != _case_key(reference):
        return True
    return any(getattr(row, key) is not None and getattr(reference, key) is not None
               and getattr(row, key) != getattr(reference, key)
               for key in ("n_atoms", "n_electrons", "n_basis_functions"))


def paired_wall_ratio(row: CodeRow, reference: CodeRow) -> Optional[float]:
    """Accuracy-filtered timing; resource equivalence needs separate evidence."""
    if (row.status != "pass" or reference.status != "reference"
            or row.ref_code != reference.code or _case_mismatch(row, reference)
            or scf_energy_problem(row.converged, row.energy_ha)
            or scf_energy_problem(reference.converged, reference.energy_ha)):
        return None
    if not all(_finite_number(r.wall_s) and r.wall_s > 0 for r in (row, reference)):
        return None
    ratio = row.wall_s / reference.wall_s
    return ratio if math.isfinite(ratio) else None


def annotate_against_reference(
    rows: list[CodeRow], ref_code: str, expected: ExpectedRef,
) -> None:
    """Classify only finite, explicitly converged, matching case rows.

    Preserve runner errors/unavailability. Labels and reported counts must
    match, but this is not proof of equal basis primitives, grids, Hamiltonian
    conventions or runtime provenance; the numerical campaign supplies those.
    """
    pending = [r for r in rows if r.status in ("pending", "pass", "marginal", "fail", "reference")]
    for r in pending:
        r.delta_ha_vs_ref = None
        r.ref_code = ""
        _set_normalized_deltas(r)

    def error(row, reason):
        row.status = "error"
        row.note = f"{row.note} | {reason}" if row.note else reason

    candidates = [r for r in rows if r.code == ref_code]
    issue = ""
    ref_row = candidates[0] if len(candidates) == 1 else None
    if ref_row is None:
        issue = f"reference code {ref_code!r} missing or ambiguous"
    elif ref_row.status not in ("pending", "pass", "marginal", "fail", "reference"):
        issue = f"reference code {ref_code!r} has status {ref_row.status}"
    else:
        issue = scf_energy_problem(ref_row.converged, ref_row.energy_ha)
        expected_mesh = "mol" if tuple(expected.kmesh) == (0, 0, 0) else "x".join(map(str, expected.kmesh))
        if _case_key(ref_row) != (expected.system_id, expected.basis, expected.method_id, expected_mesh):
            issue = "reference case does not match the expected comparison"
    if not issue and classify(0., expected.tolerance_ha)[0] == "error":
        issue = "invalid comparison tolerance"
    if issue:
        for r in pending:
            error(r, issue)
        return

    ref_row.delta_ha_vs_ref = 0.0
    ref_row.ref_code = ref_code
    ref_row.status = "reference"
    _set_normalized_deltas(ref_row)
    if not ref_row.note:
        ref_row.note = f"reference (E={ref_row.energy_ha:.6f} Ha)"

    for r in pending:
        if r is ref_row:
            continue
        issue = scf_energy_problem(r.converged, r.energy_ha)
        if _case_mismatch(r, ref_row):
            issue = "candidate case or reported counts differ from reference"
        if issue:
            error(r, issue)
            continue
        delta = r.energy_ha - ref_row.energy_ha
        status, note = classify(delta, expected.tolerance_ha)
        if status == "error":
            error(r, note)
            continue
        r.delta_ha_vs_ref = delta
        r.ref_code = ref_code
        _set_normalized_deltas(r)
        r.status = status
        note += _normalization_note(r)
        r.note = f"{r.note} | {note}" if r.note else note
