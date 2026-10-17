"""An optimised basis goes into the user basis store and comes back exact.

``vibeqc.basis_optimization.export`` turns the optimiser's
``{symbol: CrystalAtomBasis}`` into a ``vibe-basis`` QVF-Basis record and
imports it into the user store (#410). Pinned here:

* the record holds the optimiser's doubles exactly, says what produced it
  (kind, origin, run, objective) and cites the optimiser plus only those
  starting-basis routes that cover its elements;
* the candidate files the optimiser scores (``basis_crystal.emit_g94``) carry
  every double exactly, so the optimiser scores the bundled basis it starts
  from, not a ten-decimal copy;
* a basis saved from a real optimisation and loaded by name in a fresh
  process reproduces the optimiser's final energy;
* a calculation on the stored basis cites the optimiser and the starting basis;
* two store-import gaps the hand-off exposed: a record's own provenance kind
  is honoured, and a vibe-basis sidecar's objective lands in the notes.

Needs a built vibe-qc. Every store and cache is a temporary directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

vq = pytest.importorskip("vibeqc")

from vibeqc import basis_store  # noqa: E402
from vibeqc.basis_crystal import emit_g94, parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization import (  # noqa: E402
    BDIIS_CITATION_KEYS,
    BasisParametrisation,
    FreeSpec,
    Transform,
    optimised_basis_record,
    save_optimised_basis,
    write_optimised_basis,
)
from vibeqc.basis_optimization.export import (  # noqa: E402
    OPTIMISER_UPSTREAM,
    starting_basis_citations,
)
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_toolkit import ProvenanceKind, from_g94, library  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
POB = Path(vq.__file__).parent / "basis_library" / "sources" / "pob-TZVP"
AWKWARD = 0.18918312345678912     # a double no ten-decimal format keeps
H2_BOND = 1.4


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A private store and cache; LIBINT_DATA_PATH restored afterwards."""
    monkeypatch.setenv("VIBEQC_BASIS_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("VIBEQC_BSE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("LIBINT_DATA_PATH", os.environ.get("LIBINT_DATA_PATH", ""))
    return tmp_path


def _pob(*pairs):
    return {sym: parse_crystal_atom_basis_file(POB / src) for sym, src in pairs}


def _h2():
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, H2_BOND])], multiplicity=1
    )


def _route_keys(record):
    return [k for route in record.provenance.citations for k in route.keys]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_record_carries_the_exact_doubles_and_says_what_made_it():
    atoms = _pob(("H", "01_H"), ("O", "08_O"))
    atoms["H"].shells[2].exponents[0] = AWKWARD
    objective = {"reference": "RHF", "systems": ["H2O"], "final": -76.0123}
    rec = optimised_basis_record(
        atoms, name="H2O-Opt-Test", starting_basis="pob-TZVP",
        method_citations=BDIIS_CITATION_KEYS, run_id="run-1", objective=objective,
    )
    assert rec.name == "h2o-opt-test"
    assert set(rec.elements) == {"H", "O"}
    assert rec.elements["H"].shells[2].primitives[0].exponent == AWKWARD
    src_o = atoms["O"].shells[0]
    got_o = rec.elements["O"].shells[0].primitives
    assert [p.exponent for p in got_o] == list(src_o.exponents)
    assert [p.coefficient for p in got_o] == list(src_o.coefficients)

    prov = rec.provenance
    assert prov.kind is ProvenanceKind.VIBE_BASIS and prov.origin == "generated"
    assert prov.run_id == "run-1"
    assert json.loads(prov.notes) == objective
    names = [u.name for u in prov.upstream]
    assert names == [OPTIMISER_UPSTREAM, "pob-TZVP"]
    assert prov.upstream[0].version == vq.__version__

    keys = _route_keys(rec)
    for key in BDIIS_CITATION_KEYS + ("peintinger_pob_tzvp_2013",):
        assert key in keys
    # pob-TZVP's fifth-period and ECP routes do not cover H or O; the
    # registry cites every key a store record names, so they must not ride along
    assert "laun_pob_fifth_period_2018" not in keys
    assert "andrae_ecp_1990" not in keys


def test_starting_basis_routes_are_restricted_to_the_elements():
    for_h = starting_basis_citations("pob-TZVP", ["H"])
    assert [r.keys for r in for_h] == [["peintinger_pob_tzvp_2013"]]
    assert for_h[0].elements is None
    for_ag = starting_basis_citations("pob-TZVP", ["Ag", "H"])
    restricted = {tuple(r.keys): r.elements for r in for_ag if r.elements is not None}
    assert restricted[("laun_pob_fifth_period_2018",)] == ["Ag"]
    assert all(r.elements in (None, ["Ag"]) for r in for_ag)


def test_unknown_starting_basis_and_a_scale_factor_are_refused():
    atoms = _pob(("H", "01_H"))
    with pytest.raises(ValueError, match="not a bundled record"):
        optimised_basis_record(atoms, name="x", starting_basis="no-such-basis")
    atoms["H"].shells[1].scale_factor = 1.2
    with pytest.raises(ValueError, match="scale factor"):
        optimised_basis_record(atoms, name="x")


# ---------------------------------------------------------------------------
# The candidate files the optimiser scores
# ---------------------------------------------------------------------------


def test_emit_g94_round_trips_every_double():
    atoms = _pob(("H", "01_H"), ("C", "06_C"))
    atoms["H"].shells[2].exponents[0] = AWKWARD
    atoms["C"].shells[0].coefficients[0] = 1.2345678901234567e-05
    parsed = from_g94(emit_g94(list(atoms.values())), name="rt")
    for sym, atom in atoms.items():
        for shell, back in zip(atom.shells, parsed.elements[sym].shells):
            assert [p.exponent for p in back.primitives] == list(shell.exponents)
            assert [p.coefficient for p in back.primitives] == list(shell.coefficients)


def test_the_optimiser_scores_the_bundled_basis_it_starts_from():
    """pob-TZVP's Be file carries digits past the tenth decimal (every pob
    element file from Li on does; H does not). The candidate file must keep
    them, or the optimiser's x0 is a rounded copy of the bundled basis."""
    atoms = _pob(("Be", "04_Be"))
    lost = [x for sh in atoms["Be"].shells for x in (*sh.exponents, *sh.coefficients)
            if float(f"{x:.10f}") != x]
    assert lost, "the probe needs a number that ten decimals do not hold"

    def be():
        return vq.Molecule([vq.Atom(4, [0.0, 0.0, 0.0])], multiplicity=1)

    bundled = library.load_record("pob-tzvp").elements["Be"]
    mol = be()
    e_bundled = vq.run_rhf(mol, vq.BasisSet(mol, "pob-tzvp")).energy
    with TempBasisLibrary() as lib:
        name = lib.write_g94(atoms, basis_name="x0-copy")
        written = from_g94((Path(lib.path) / "basis" / f"{name}.g94").read_text(),
                           name="x0-copy").elements["Be"]
        mol = be()
        e_candidate = vq.run_rhf(mol, vq.BasisSet(mol, name)).energy
    # The deterministic pin: the candidate file holds the bundled doubles.
    # (Ten decimals moved this energy by only 2e-13 Ha, so an energy test
    # alone cannot tell the formats apart.)
    for mine, ref in zip(written.shells, bundled.shells):
        assert [(p.exponent, p.coefficient) for p in mine.primitives] == [
            (p.exponent, p.coefficient) for p in ref.primitives
        ]
    assert e_candidate == pytest.approx(e_bundled, abs=1e-12)


# ---------------------------------------------------------------------------
# Save, resolve, reproduce
# ---------------------------------------------------------------------------


def _optimise_h2():
    from vibeqc.basis_optimization.recipes.molecular import optimize_molecular_basis

    par = BasisParametrisation(
        atoms=_pob(("H", "01_H")),
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None)),
              FreeSpec("H", 3, 0, "exponent", transform=Transform.LOG, bounds=(0.05, None))],
    )
    return optimize_molecular_basis(par, _h2, max_iter=60)


_CHILD = """
import json, sys
import vibeqc as vq
mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, {bond}])], multiplicity=1)
r = vq.run_rhf(mol, vq.BasisSet(mol, {name!r}))
print(json.dumps({{"energy": r.energy, "converged": bool(r.converged)}}))
"""


def test_saved_optimum_reproduces_the_optimised_energy_in_a_fresh_process(roots):
    res = _optimise_h2()
    assert res.converged, res.message
    stored = res.save_to_store("h2-pob-tzvp-opt", starting_basis="pob-TZVP",
                               run_id="h2-test")
    assert stored.kind == "vibe-basis" and stored.name == "h2-pob-tzvp-opt"

    # in this process, after activation
    mol = _h2()
    e_here = vq.run_rhf(mol, vq.BasisSet(mol, "h2-pob-tzvp-opt")).energy
    assert e_here == pytest.approx(res.optimal_objective, abs=1e-10)

    # in a new process, as a new shell would start it
    env = dict(os.environ)
    env.pop("LIBINT_DATA_PATH", None)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD.format(bond=H2_BOND, name="h2-pob-tzvp-opt")],
        env=env, capture_output=True, text=True, check=True, cwd=roots,
    )
    child = json.loads(out.stdout.strip().splitlines()[-1])
    assert child["converged"]
    assert child["energy"] == pytest.approx(res.optimal_objective, abs=1e-10)

    record = library.load_record(stored.path)
    notes = json.loads(record.provenance.notes)
    assert notes["recipe"] == "optimize_molecular_basis" and notes["reference"] == "RHF"
    assert notes["optimal_objective_hartree"] == pytest.approx(res.optimal_objective)


def test_a_run_on_the_stored_basis_cites_the_optimiser_and_the_start(roots, tmp_path):
    atoms = _pob(("H", "01_H"))
    atoms["H"].shells[2].exponents[0] = AWKWARD
    save_optimised_basis(atoms, name="h-cite-test", starting_basis="pob-TZVP",
                         method_citations=BDIIS_CITATION_KEYS, run_id="cite-1")
    from vibeqc.output.citations.registry import _store_record_citations

    keys, note = _store_record_citations("h-cite-test")
    assert "daga_optbasis_2020" in keys and "peintinger_pob_tzvp_2013" in keys
    assert "laun_pob_fifth_period_2018" not in keys
    assert "vibe-basis record" in note


def test_saving_inside_a_candidate_library_leaves_it_alone(roots):
    atoms = _pob(("H", "01_H"))
    with TempBasisLibrary() as lib:
        before = os.environ["LIBINT_DATA_PATH"]
        save_optimised_basis(atoms, name="h-inside-test", starting_basis="pob-TZVP")
        assert os.environ["LIBINT_DATA_PATH"] == before == str(lib.path)
    assert any(r.name == "h-inside-test" for r in basis_store.list_records())


# ---------------------------------------------------------------------------
# The two store-import gaps
# ---------------------------------------------------------------------------


def test_a_written_record_imports_with_no_flags(roots, tmp_path):
    atoms = _pob(("H", "01_H"))
    path = write_optimised_basis(atoms, tmp_path / "share", name="H-Shared",
                                 starting_basis="pob-TZVP", run_id="share-1",
                                 objective={"reference": "RHF"})
    assert path.name == "h-shared.qvf.json"
    stored = basis_store.import_record(path)          # kind from the record itself
    assert stored.kind == "vibe-basis"
    again = basis_store.import_record(path, name="h-shared-2",
                                      kind=ProvenanceKind.VIBE_BASIS)   # enum accepted
    assert again.kind == "vibe-basis"
    rec = library.load_record(stored.path)
    assert json.loads(rec.provenance.notes) == {"reference": "RHF"}
    assert rec.provenance.run_id == "share-1"


def test_a_vibe_basis_sidecar_objective_becomes_the_notes(roots, tmp_path):
    sys.path.insert(0, str(REPO / "vibe-basis" / "src"))
    from vibe_basis.io.export import write_basis_for_import
    from vibe_basis.topology import BasisTopology, Shell

    topology = BasisTopology(atoms={"H": (Shell("S", (3.42525091, 0.62391373, 0.1688554),
                                                 (0.15432897, 0.53532814, 0.44463454)),)})
    g94, _ = write_basis_for_import(topology, tmp_path / "run-9", name="H-Side",
                                    run_id="run-9", objective={"mad": 0.1, "systems": 3})
    stored = basis_store.import_record(g94)
    rec = library.load_record(stored.path)
    assert json.loads(rec.provenance.notes) == {"mad": 0.1, "systems": 3}
    explicit = basis_store.import_record(g94, name="h-side-2", notes="hand note")
    assert library.load_record(explicit.path).provenance.notes == "hand note"


# ---------------------------------------------------------------------------
# The Stage-2 result carries its basis
# ---------------------------------------------------------------------------


def test_one_compound_result_carries_and_saves_its_basis(roots):
    from types import SimpleNamespace

    from vibeqc.basis_optimization.recipes.one_compound import run_one_compound

    par = BasisParametrisation(
        atoms=_pob(("H", "01_H")),
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG, bounds=(0.05, 1.0))],
    )

    def runner(system, basis, **kw):
        d = np.log(basis.h) - np.log(0.15)
        return SimpleNamespace(energy=-1.0 + 0.5 * d * d, converged=True)

    res = run_one_compound(
        par, lambda: None, compound="mock", driver="scipy", max_evaluations=40,
        scf_runner=runner,
        basis_factory=lambda s, n, atoms: SimpleNamespace(h=atoms["H"].shells[2].exponents[0]),
    )
    assert res.optimized_atoms["H"].shells[2].exponents[0] == pytest.approx(res.optimal_value)
    stored = res.save_to_store("h-mock-crystal", run_id="mock-1")
    notes = json.loads(library.load_record(stored.path).provenance.notes)
    assert notes["recipe"] == "run_one_compound" and notes["compound"] == "mock"
