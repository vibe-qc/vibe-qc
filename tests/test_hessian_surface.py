"""A reported Hessian must name the surface it was built on.

``run_job`` resolves a correlated request down to the mean-field reference
its SCF runs (``_select_method("mp2", ...) -> "rhf"``) and
``compute_hessian_fd`` differentiates *that*. Before this module's fix the
output said so nowhere: ``run_job(method="mp2", hessian=True)`` produced a
frequency block byte-for-byte identical to the ``method="rhf"`` one, and
the thermochemistry rows were labelled ``E(elec)`` while carrying the RHF
electronic energy in a job whose reported energy was MP2. Nothing in the
``.out``, the ``.system`` manifest or the ``.qvf`` told a reader which
surface the numbers described.

These tests pin the three places that now say it, for one correlated
method (MP2) and its mean-field reference:

* the ``.out`` frequency block's ``Surface:`` line and, when the surface
  is not the requested method's, the explicit note underneath it;
* the thermochemistry rows, whose labels name the surface the electronic
  energy came from;
* the ``[hessian]`` section of the ``.system`` manifest and the
  ``surface`` key of the QVF ``vibrations`` metadata, so a machine
  consumer can gate on it without parsing prose.

A method that does not resolve to a mean-field reference at all (CASSCF,
CISD, ...) reports no numbers, so it has no honesty problem -- but it used
to leak ``compute_hessian_fd``'s raw ``ValueError`` into the block after
the whole job had run. That refusal is pinned here too.

Everything is water / STO-3G: each job is a fraction of a second.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from vibeqc import run_job
from vibeqc.molecule import Atom, Molecule
from vibeqc.output import (
    hessian_surface_label,
    hessian_surface_lines,
    hessian_surface_manifest_fields,
    hessian_unsupported_surface_lines,
    thermochemistry_energy_labels,
)

_A2B = 1.8897259886


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.7572 * _A2B, 0.5865 * _A2B]),
            Atom(1, [0.0, -0.7572 * _A2B, 0.5865 * _A2B]),
        ],
        0,
        1,
    )


def _freq_block(out: str) -> str:
    """The frequency + thermochemistry span of a ``.out``."""
    start = out.index("## Vibrational Frequencies")
    end = out.index("Timings (wall clock", start)
    return out[start:end]


@pytest.fixture(scope="module")
def mp2_job(tmp_path_factory) -> tuple[str, str, Path]:
    """One ``method='mp2', hessian=True`` run: (.out, .system, stem)."""
    stem = tmp_path_factory.mktemp("mp2") / "h2o"
    run_job(
        _h2o(),
        basis="sto-3g",
        method="mp2",
        hessian=True,
        output=str(stem),
        verbose=0,
    )
    return (
        stem.with_suffix(".out").read_text(),
        stem.with_suffix(".system").read_text(),
        stem,
    )


@pytest.fixture(scope="module")
def rhf_job(tmp_path_factory) -> tuple[str, str, Path]:
    """The same job on the bare reference, for the side-by-side pin."""
    stem = tmp_path_factory.mktemp("rhf") / "h2o"
    run_job(
        _h2o(),
        basis="sto-3g",
        method="rhf",
        hessian=True,
        output=str(stem),
        verbose=0,
    )
    return (
        stem.with_suffix(".out").read_text(),
        stem.with_suffix(".system").read_text(),
        stem,
    )


# ---------------------------------------------------------------------------
# The regression: a correlated Hessian names its surface
# ---------------------------------------------------------------------------


def test_correlated_hessian_block_names_the_surface(mp2_job) -> None:
    """The frequency block says RHF, and says it is not MP2."""
    out, _, _ = mp2_job
    block = _freq_block(out)
    assert "Surface: RHF/sto-3g" in block
    assert "(NOT MP2)" in block
    assert "no MP2 second derivatives were computed" in block


def test_correlated_thermochemistry_rows_name_the_surface(mp2_job) -> None:
    """``E(elec)`` was ambiguous in a job whose energy is MP2; the rows
    now name the surface the electronic energy actually came from."""
    out, _, _ = mp2_job
    block = _freq_block(out)
    assert "E(RHF) + ZPE" in block
    assert "H = E(RHF) + H_corr" in block
    assert "G = E(RHF) + G_corr" in block
    assert "E(elec)" not in block


def test_correlated_thermochemistry_uses_the_reference_energy(mp2_job) -> None:
    """The labelled number really is the RHF energy, not the MP2 total.

    The label has to match the arithmetic or it only moves the problem:
    ``E(RHF) + ZPE`` minus the ZPE must reproduce the printed RHF
    reference energy, and must not be the MP2 total.
    """
    out, _, _ = mp2_job

    def value_after(label: str) -> float:
        return float(out.split(label)[1].split("=")[1].split()[0])

    e_rhf = value_after("E(RHF reference)")
    e_mp2 = value_after("E(MP2 total)")
    zpe = value_after("Zero-point energy")
    e_plus_zpe = value_after("E(RHF) + ZPE")

    assert e_plus_zpe - zpe == pytest.approx(e_rhf, abs=1e-9)
    assert e_plus_zpe - zpe != pytest.approx(e_mp2, abs=1e-6)


def test_frequency_blocks_differ_only_by_the_surface_note(
    mp2_job, rhf_job
) -> None:
    """The original defect, pinned directly.

    Before the fix these two blocks were byte-for-byte identical, so no
    reader could tell an MP2 job's frequencies from an RHF job's. The
    numbers still agree (the MP2 Hessian *is* the RHF one), but the MP2
    block now carries lines the RHF block does not.
    """
    mp2_out, _, _ = mp2_job
    rhf_out, _, _ = rhf_job
    mp2_block = _freq_block(mp2_out)
    rhf_block = _freq_block(rhf_out)

    assert mp2_block != rhf_block

    # Same physics: every frequency row is identical.
    def freq_rows(block: str) -> list[str]:
        return [
            ln.strip()
            for ln in block.splitlines()
            if ln.strip().startswith(("1 ", "2 ", "3 "))
        ]

    assert freq_rows(mp2_block) == freq_rows(rhf_block)
    assert freq_rows(mp2_block)  # the pin is worthless on an empty table

    # Different provenance: the correlated block carries the note.
    assert "(NOT MP2)" in mp2_block
    assert "(NOT" not in rhf_block
    assert "Surface: RHF/sto-3g" in rhf_block


def test_meanfield_hessian_names_its_surface_without_a_note(rhf_job) -> None:
    """A plain RHF job is not a substitution, so it gets the surface line
    and no warning paragraph."""
    out, _, _ = rhf_job
    block = _freq_block(out)
    assert "Surface: RHF/sto-3g" in block
    assert "second derivatives were computed" not in block
    assert "E(RHF) + ZPE" in block


def test_open_shell_correlated_hessian_names_uhf(tmp_path: Path) -> None:
    """The label follows the resolved reference, it is not hardcoded to
    RHF: an open-shell MP2 request resolves to UHF, so the surface is
    UHF and the thermochemistry rows say E(UHF)."""
    stem = tmp_path / "triplet"
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.7572 * _A2B, 0.5865 * _A2B]),
            Atom(1, [0.0, -0.7572 * _A2B, 0.5865 * _A2B]),
        ],
        0,
        3,
    )
    run_job(
        mol,
        basis="sto-3g",
        method="mp2",
        hessian=True,
        output=str(stem),
        verbose=0,
    )
    block = _freq_block(stem.with_suffix(".out").read_text())
    assert "Surface: UHF/sto-3g  (NOT MP2)" in block
    assert "the analytic UHF gradient" in block
    assert "E(UHF) + ZPE" in block


def test_dft_surface_names_the_functional(tmp_path: Path) -> None:
    """RKS alone does not identify a surface; the functional does."""
    stem = tmp_path / "h2o"
    run_job(
        _h2o(),
        basis="sto-3g",
        method="rks",
        functional="b3lyp",
        hessian=True,
        output=str(stem),
        verbose=0,
    )
    block = _freq_block(stem.with_suffix(".out").read_text())
    assert "Surface: RKS(B3LYP)/sto-3g" in block
    # The row labels stay at the reference so the column of "=" holds.
    assert "E(RKS) + ZPE" in block


def test_semiempirical_keeps_its_own_skip_message(tmp_path: Path) -> None:
    """PM6 has no Gaussian-basis FD Hessian and already said so; the new
    unsupported-surface branch must not take that message over."""
    stem = tmp_path / "h2o"
    run_job(
        _h2o(), method="pm6", basis="", hessian=True,
        output=str(stem), verbose=0,
    )
    out = stem.with_suffix(".out").read_text()
    assert "available only for Gaussian-basis molecular methods" in out
    assert "no Hessian is available on the PM6 surface" not in out
    section = stem.with_suffix(".system").read_text().split("[hessian]")[1]
    assert "available      = false" in section.split("[", 1)[0]


# ---------------------------------------------------------------------------
# Machine-readable provenance: .system manifest and .qvf
# ---------------------------------------------------------------------------


def test_manifest_records_the_correlated_surface(mp2_job) -> None:
    _, system, _ = mp2_job
    assert "[hessian]" in system
    section = system.split("[hessian]")[1].split("[", 1)[0]
    assert 'requested_method = "mp2"' in section
    assert 'surface        = "RHF/sto-3g"' in section
    assert 'surface_method = "rhf"' in section
    assert "surface_is_requested_method = false" in section
    assert "available      = true" in section


def test_manifest_records_a_matching_surface_as_matching(rhf_job) -> None:
    _, system, _ = rhf_job
    section = system.split("[hessian]")[1].split("[", 1)[0]
    assert 'requested_method = "rhf"' in section
    assert "surface_is_requested_method = true" in section


def test_manifest_has_no_hessian_section_without_a_hessian(
    tmp_path: Path,
) -> None:
    """The section is conditional: an energy-only job gains no stub."""
    stem = tmp_path / "h2o"
    run_job(
        _h2o(), basis="sto-3g", method="rhf", output=str(stem), verbose=0
    )
    assert "[hessian]" not in stem.with_suffix(".system").read_text()


def test_qvf_vibrations_metadata_carries_the_surface(mp2_job) -> None:
    """``job_spec.method`` in a QVF already reads "rhf" for an MP2 job, so
    the vibrations section has to say which surface produced it."""
    _, _, stem = mp2_job
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as zf:
        meta = json.loads(zf.read("vibrations/metadata.json"))
    surface = meta["surface"]
    assert surface["surface"] == "RHF/sto-3g"
    assert surface["requested_method"] == "mp2"
    assert surface["surface_is_requested_method"] is False


# ---------------------------------------------------------------------------
# A surface with no finite-difference Hessian at all
# ---------------------------------------------------------------------------


def test_unsupported_surface_is_skipped_not_failed(tmp_path: Path) -> None:
    """CASSCF does not resolve to a mean-field reference, so there is no
    gradient to differentiate. That used to surface as
    ``FAILED: ValueError: FD Hessian: unknown method 'CASSCF'`` after the
    whole job had run."""
    stem = tmp_path / "h2o"
    run_job(
        _h2o(),
        basis="sto-3g",
        method="casscf",
        active_space=(2, 2),
        hessian=True,
        output=str(stem),
        verbose=0,
    )
    out = stem.with_suffix(".out").read_text()
    block = _freq_block(out)
    assert "SKIPPED" in block
    assert "no Hessian is available on the CASSCF surface" in block
    assert "FAILED" not in block
    assert "ValueError" not in block
    # No numbers were reported, so there is nothing to mislabel.
    assert "Freq/cm" not in block
    assert "Thermochemistry" not in block
    # The energy still lands -- skipping the Hessian must not cost the job.
    assert "CASSCF" in out

    section = stem.with_suffix(".system").read_text().split("[hessian]")[1]
    assert "available      = false" in section.split("[", 1)[0]


def test_unsupported_casscf_points_at_the_api_route(tmp_path: Path) -> None:
    """``compute_hessian_casscf`` exists but is not run_job-shaped; the
    refusal names it and says why it is not wired in."""
    stem = tmp_path / "h2o"
    run_job(
        _h2o(),
        basis="sto-3g",
        method="casscf",
        active_space=(2, 2),
        hessian=True,
        output=str(stem),
        verbose=0,
    )
    block = _freq_block(stem.with_suffix(".out").read_text())
    assert "vibeqc.hessian_casscf.compute_hessian_casscf" in block
    assert "unprojected" in block


# ---------------------------------------------------------------------------
# The wording itself -- unit level, no SCF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resolved, functional, basis, expected",
    [
        ("rhf", None, None, "RHF"),
        ("rhf", None, "sto-3g", "RHF/sto-3g"),
        ("uks", "b3lyp", "def2-svp", "UKS(B3LYP)/def2-svp"),
        ("rohf", None, "6-31g", "ROHF/6-31g"),
    ],
)
def test_surface_label(resolved, functional, basis, expected) -> None:
    assert (
        hessian_surface_label(resolved, functional=functional, basis=basis)
        == expected
    )


@pytest.mark.parametrize("requested", ["mp2", "ccsd", "ccsd(t)", "ovgf"])
def test_every_post_scf_route_gets_the_note(requested) -> None:
    """The note keys off requested != resolved, not off a list of
    correlated methods, so a route added later needs no edit here."""
    lines = hessian_surface_lines(
        requested_method=requested, resolved_method="rhf", basis="sto-3g"
    )
    assert f"(NOT {requested.upper()})" in lines
    assert "no " + requested.upper() + " second derivatives" in lines


@pytest.mark.parametrize("requested", ["auto", "rhf", "RHF", ""])
def test_no_note_when_the_surface_is_what_was_asked_for(requested) -> None:
    """``auto`` asks vibe-qc to choose, so its choice is not a
    substitution; neither is a bare RHF request, in any case."""
    lines = hessian_surface_lines(
        requested_method=requested, resolved_method="rhf", basis="sto-3g"
    )
    assert lines.strip() == "Surface: RHF/sto-3g"


def test_thermo_labels_align_with_the_rows_above_them() -> None:
    """The block's other rows pad their label to 25 characters; a surface
    label that overflows would break the column of ``=`` signs."""
    for reference in ("rhf", "uhf", "rohf", "rks", "uks"):
        for label in thermochemistry_energy_labels(reference):
            assert len(label) == 25, (reference, label)
            assert label.rstrip() == label.strip()


def test_manifest_fields_are_flat_scalars() -> None:
    """The ``[hessian]`` TOML emitter takes scalars only."""
    fields = hessian_surface_manifest_fields(
        requested_method="mp2",
        resolved_method="rhf",
        functional=None,
        basis="sto-3g",
    )
    assert all(
        isinstance(v, (str, bool, int, float)) for v in fields.values()
    ), fields
    assert fields["surface_is_requested_method"] is False


def test_unsupported_lines_name_the_available_surfaces() -> None:
    lines = hessian_unsupported_surface_lines(
        requested_method="cisd", resolved_method="cisd"
    )
    assert "no Hessian is available on the CISD surface" in lines
    for reference in ("RHF", "UHF", "ROHF", "RKS", "UKS"):
        assert reference in lines
    # Only CASSCF has an API-only route worth pointing at.
    assert "compute_hessian_casscf" not in lines
