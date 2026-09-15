"""Tolerance-based pass / marginal / fail classification of ΔE."""
from __future__ import annotations

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
    error    : Δ unavailable (one side did not converge)
    """
    if delta_ha is None:
        return "error", "delta unavailable"
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


def annotate_against_reference(
    rows: list[CodeRow], ref_code: str, expected: ExpectedRef,
) -> None:
    """Mutate rows in-place: set delta_ha_vs_ref / ref_code / status / note
    for every non-reference row.

    The reference row itself gets status = 'reference' and Δ = 0.
    """
    ref_row = next((r for r in rows if r.code == ref_code), None)
    if ref_row is None or ref_row.energy_ha is None:
        for r in rows:
            if r.status == "pending":
                r.status = "error"
                if not r.note:
                    r.note = f"reference code {ref_code!r} missing or did not converge"
        return

    ref_e = ref_row.energy_ha
    ref_row.delta_ha_vs_ref = 0.0
    ref_row.ref_code = ref_code
    ref_row.status = "reference"
    _set_normalized_deltas(ref_row)
    if not ref_row.note:
        ref_row.note = f"reference (E={ref_e:.6f} Ha)"

    for r in rows:
        if r is ref_row or r.status not in ("pending",):
            continue
        if r.energy_ha is None:
            r.status = "error"
            if not r.note:
                r.note = "did not converge"
            continue
        r.delta_ha_vs_ref = r.energy_ha - ref_e
        r.ref_code = ref_code
        status, note = classify(r.delta_ha_vs_ref, expected.tolerance_ha)
        _set_normalized_deltas(r)
        r.status = status
        note = note + _normalization_note(r)
        if r.note:
            r.note = f"{r.note} | {note}"
        else:
            r.note = note
