"""A ``.xyz`` written by ``run_job`` must name the method its energy is on.

``run_job`` writes the ``.xyz`` comment line from the **SCF** result's energy,
and ``_select_method`` has already collapsed a correlated request down to its
mean-field reference, so ``method="mp2"`` produced a geometry file carrying the
RHF energy under a bare ``energy=`` tag. ``energy=`` is the conventional
single-frame energy tag and ASE reads it as *the* energy of that structure, so
someone post-processing MP2 optimizations through ASE silently got Hartree-Fock
energies -- and the MP2 job's file was byte-identical to an RHF job's, leaving
nothing to tell them apart (#360).

The number is deliberately unchanged: ``energy=`` is still the SCF energy. What
is new is that the file says so, with ``energy_method=`` naming what the energy
is and ``method=`` naming the job. The sibling artifacts were checked and left
alone, for reasons recorded in the merge request: the ``.opt`` trajectory writer
is public API with no ``run_job`` call site, the ASE ``.traj`` is binary with no
comment line, and the periodic runner has no correlated-to-mean-field collapse
so its extended-XYZ energy is never another method's.

Water / STO-3G: each job is a fraction of a second.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeqc import run_job
from vibeqc.molecule import Atom, Molecule

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


def _comment(stem: Path) -> str:
    return stem.with_suffix(".xyz").read_text().splitlines()[1]


def _tag(comment: str, key: str) -> str | None:
    for token in comment.split():
        if token.startswith(f"{key}="):
            return token.split("=", 1)[1]
    return None


@pytest.fixture(scope="module")
def mp2_stem(tmp_path_factory) -> Path:
    stem = tmp_path_factory.mktemp("mp2") / "h2o"
    run_job(
        _h2o(), basis="sto-3g", method="mp2", output=str(stem), verbose=0
    )
    return stem


@pytest.fixture(scope="module")
def rhf_stem(tmp_path_factory) -> Path:
    stem = tmp_path_factory.mktemp("rhf") / "h2o"
    run_job(
        _h2o(), basis="sto-3g", method="rhf", output=str(stem), verbose=0
    )
    return stem


def test_correlated_xyz_names_the_energy_method(mp2_stem: Path) -> None:
    comment = _comment(mp2_stem)
    assert _tag(comment, "energy_method") == "rhf"
    assert _tag(comment, "method") == "mp2"


def test_the_tagged_energy_really_is_the_reference_energy(
    mp2_stem: Path,
) -> None:
    """The tag has to match the number, or it only moves the problem."""
    out = mp2_stem.with_suffix(".out").read_text()

    def value_after(label: str) -> float:
        return float(out.split(label)[1].split("=")[1].split()[0])

    e_rhf = value_after("E(RHF reference)")
    e_mp2 = value_after("E(MP2 total)")
    energy = float(_tag(_comment(mp2_stem), "energy"))

    assert energy == pytest.approx(e_rhf, abs=1e-9)
    assert energy != pytest.approx(e_mp2, abs=1e-6)


def test_the_energy_number_is_unchanged(mp2_stem: Path, rhf_stem: Path) -> None:
    """This fix labels, it does not move a number: the MP2 job's energy
    tag is still the SCF one, identical to the RHF job's."""
    assert _tag(_comment(mp2_stem), "energy") == _tag(
        _comment(rhf_stem), "energy"
    )


def test_correlated_and_reference_xyz_are_no_longer_identical(
    mp2_stem: Path, rhf_stem: Path
) -> None:
    """The original symptom, pinned directly."""
    mp2_text = mp2_stem.with_suffix(".xyz").read_text()
    rhf_text = rhf_stem.with_suffix(".xyz").read_text()
    assert mp2_text != rhf_text
    # They differ in the comment line only -- same geometry, same energy.
    assert mp2_text.splitlines()[2:] == rhf_text.splitlines()[2:]


def test_meanfield_job_tags_agree(rhf_stem: Path) -> None:
    comment = _comment(rhf_stem)
    assert _tag(comment, "energy_method") == "rhf"
    assert _tag(comment, "method") == "rhf"


def test_auto_method_reports_what_it_resolved_to(tmp_path: Path) -> None:
    """``method="auto"`` is a request to choose; the tag names the choice
    rather than echoing "auto", which would name no surface at all."""
    stem = tmp_path / "h2o"
    run_job(
        _h2o(), basis="sto-3g", method="auto", output=str(stem), verbose=0
    )
    comment = _comment(stem)
    assert _tag(comment, "method") == "rhf"
    assert _tag(comment, "energy_method") == "rhf"


def test_optimized_geometry_carries_the_same_tags(tmp_path: Path) -> None:
    """``optimize=True`` relaxes the reference surface for a correlated
    request too, so the optimized file needs the tags just as much."""
    stem = tmp_path / "h2o"
    run_job(
        _h2o(), basis="sto-3g", method="mp2", optimize=True,
        output=str(stem), verbose=0,
    )
    comment = _comment(stem)
    assert _tag(comment, "energy_method") == "rhf"
    assert _tag(comment, "method") == "mp2"


def test_ase_reads_the_energy_and_the_method(mp2_stem: Path) -> None:
    """ASE is the consumer the ``energy=`` tag exists for. It must still
    parse the energy, and it must surface the new tags -- otherwise the
    label is invisible to exactly the reader that was being misled.
    """
    ase_io = pytest.importorskip("ase.io")
    atoms = ase_io.read(str(mp2_stem.with_suffix(".xyz")))
    assert atoms.info["energy_method"] == "rhf"
    assert atoms.info["method"] == "mp2"
    # The energy still round-trips, unchanged.
    out = mp2_stem.with_suffix(".out").read_text()
    e_rhf = float(out.split("E(RHF reference)")[1].split("=")[1].split()[0])
    assert atoms.get_potential_energy() == pytest.approx(e_rhf, abs=1e-9)


@pytest.mark.parametrize(
    "method",
    ["ccsd(t)", "qcisd(t)", "cepa(0)", "scs-mp2", "dlpno-ccsd(t)", "mp2"],
)
def test_method_spellings_survive_the_ase_round_trip(
    method: str, tmp_path: Path
) -> None:
    """Method names carry parentheses and hyphens, which is exactly the
    kind of thing a ``key=value`` comment-line convention breaks on. Pin
    that ASE still recovers both the tag and the energy for every
    spelling that can reach the writer.
    """
    ase_io = pytest.importorskip("ase.io")
    from vibeqc.output.formats.xyz import format_xyz

    path = tmp_path / "t.xyz"
    path.write_text(
        format_xyz(
            _h2o(), energy_ha=-76.1, energy_method="rhf", method=method
        )
    )
    atoms = ase_io.read(str(path))
    assert atoms.info["method"] == method
    assert atoms.info["energy_method"] == "rhf"
    assert atoms.get_potential_energy() == pytest.approx(-76.1)
