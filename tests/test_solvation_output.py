"""The solvation block of the ``.out`` states the model that produced the energy.

The segment-charge kernel and the cavity move a solvated energy by percent, so
a file that printed only the energy would hide the choice that matters most
(#681). Every line is checked against the result object it was read from, not
against a literal, so the block cannot drift from the run.
"""

from __future__ import annotations

import re

import pytest

import vibeqc as vq

_WATER = [
    vq.Atom(8, (0.0, 0.0, 0.2257)),
    vq.Atom(1, (0.0, 1.4375, -0.9027)),
    vq.Atom(1, (0.0, -1.4375, -0.9027)),
]


def _run(tmp_path, solvent, **kw):
    stem = tmp_path / "w"
    result = vq.run_job(
        vq.Molecule(list(_WATER), 0, 1),
        basis="sto-3g",
        method="rhf",
        solvent=solvent,
        output=str(stem),
        write_molden_file=False,
        progress=False,
        **kw,
    )
    return result, stem.with_suffix(".out").read_text()


def _field(text, label):
    m = re.search(rf"^\s*{re.escape(label)}\s*= (.*)$", text, flags=re.M)
    assert m, f"no {label!r} line in the solvation block"
    return m.group(1)


def test_the_default_block_names_the_point_charge_model(tmp_path):
    result, text = _run(tmp_path, "water")
    sol = result.solvent_result
    assert _field(text, "Solvent").startswith("water, epsilon = 78.390")
    assert f"= {sol.screening.f:.5f}" in _field(text, "Screening")
    assert _field(text, "Screening").startswith("CPCM")
    cavity = _field(text, "Cavity")
    assert "switched Lebedev spheres" in cavity and "302 points per sphere" in cavity
    seg = _field(text, "Cavity segments")
    assert seg.startswith(f"{sol.cavity.n_points},")
    area = float(sol.cavity.weights.sum())
    assert f"area {area:.2f} bohr^2" in seg
    assert _field(text, "Segment charges").startswith("point charges")
    assert _field(text, "Reaction-field cycles") == f"{sol.n_macro_iter} (converged)"


def test_the_gaussian_kernel_is_named(tmp_path):
    sm = vq.SolventModel(epsilon=78.39, name="water", charge_representation="gaussian")
    _, text = _run(tmp_path, sm)
    assert _field(text, "Segment charges").startswith("Gaussian")


def test_cosmo_prints_its_own_screening_factor(tmp_path):
    sm = vq.SolventModel(epsilon=78.39, name="water", variant="cosmo")
    result, text = _run(tmp_path, sm)
    screening = _field(text, "Screening")
    assert screening.startswith("COSMO") and "eps + 0.5" in screening
    assert f"{result.solvent_result.screening.f:.5f}" in screening


def test_the_fine_cavity_is_named(tmp_path):
    sm = vq.SolventModel(epsilon=78.39, name="water", cavity="fine")
    result, text = _run(tmp_path, sm)
    assert _field(text, "Cavity").startswith("COSMO FINE cavity")
    assert _field(text, "Segment charges").startswith("Gaussian")
    assert _field(text, "Cavity segments").startswith(f"{result.solvent_result.cavity.n_points},")


def test_a_gas_phase_job_prints_no_solvation_block(tmp_path):
    stem = tmp_path / "gas"
    vq.run_job(
        vq.Molecule(list(_WATER), 0, 1),
        basis="sto-3g",
        method="rhf",
        output=str(stem),
        write_molden_file=False,
        progress=False,
    )
    assert "Implicit solvation" not in stem.with_suffix(".out").read_text()


def test_an_msindo_cosmo_job_does_not_claim_a_lebedev_cavity(tmp_path):
    """MSINDO COSMO uses a GEPOL cavity and has no Gaussian-basis result to read
    the Lebedev description from, so its block carries no model lines."""
    stem = tmp_path / "m"
    vq.run_job(
        vq.Molecule(list(_WATER), 0, 1),
        method="msindo",
        solvent="water",
        output=str(stem),
        verbose=0,
    )
    text = stem.with_suffix(".out").read_text()
    assert "Implicit solvation (COSMO)" in text
    assert "Lebedev" not in text
    assert "Segment charges" not in text
