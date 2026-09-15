"""Atomization energies for mean-field methods (vibeqc.atomization + run_job).

Pins the free-atom reference machinery (ground-state multiplicities + cached
atomic SCFs) and the run_job atomization block.  HF underbinds (it misses
correlation, which is larger in the molecule than in the separated atoms), so
the HF atomization is well below experiment — the right direction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeqc import run_job
from vibeqc.atomization import (atomic_ground_state_energy, atomization_energy,
                                supported_elements)
from vibeqc.molecule import Atom, Molecule

_A2B = 1.8897259886


def _h2o():
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.7572 * _A2B, 0.5865 * _A2B]),
         Atom(1, [0.0, -0.7572 * _A2B, 0.5865 * _A2B])], 0, 1)


def test_free_atom_ground_state_energies():
    """Open-shell (O triplet) + closed-shell (He) free-atom SCFs converge to
    the known HF/6-31G energies; results are cached (same object on re-call)."""
    e_h = atomic_ground_state_energy(1, "rhf", "6-31g")     # H doublet (UHF)
    e_o = atomic_ground_state_energy(8, "rhf", "6-31g")     # O triplet (UHF)
    assert e_h == pytest.approx(-0.49823, abs=1e-4)
    assert e_o == pytest.approx(-74.78031, abs=1e-4)
    # Cache: identical value, no recompute drift.
    assert atomic_ground_state_energy(8, "rhf", "6-31g") == e_o


def test_atomization_water_hf():
    """H₂O atomization at HF/6-31G: Σ E(atom) − E(mol), positive (bound), and
    well below experiment (HF underbinds).  Pinned as a regression anchor."""
    from vibeqc._vibeqc_core import (BasisSet, Molecule as CMol, Atom as CAtom,
                                     RHFOptions, run_rhf)
    mol = CMol([CAtom(8, [0, 0, 0]),
                CAtom(1, [0, 0.7572 * _A2B, 0.5865 * _A2B]),
                CAtom(1, [0, -0.7572 * _A2B, 0.5865 * _A2B])], 0, 1)
    hf = run_rhf(mol, BasisSet(mol, "6-31g"), RHFOptions())
    r = atomization_energy(mol, hf.energy, "rhf", "6-31g")
    assert r.atomization > 0.0                          # bound
    assert r.e_atoms_sum == pytest.approx(
        r.atomic_energies[8] + 2 * r.atomic_energies[1])
    assert r.atomization == pytest.approx(0.207199, abs=1e-4)
    assert r.atomization_kcal == pytest.approx(130.0, abs=0.5)
    assert r.n_atoms == 3
    assert r.atomization_kcal < 232.0                  # HF < experimental De


def test_atomization_rejects_transition_metals():
    """3d transition metals are excluded (no black-box atomic UHF ground state)."""
    fe = Molecule([Atom(26, [0, 0, 0]), Atom(8, [0, 0, 3.0])], 0, 1)
    with pytest.raises(NotImplementedError, match="transition metals|free-atom"):
        atomization_energy(fe, -1300.0, "rhf", "6-31g")


def test_supported_elements_main_group():
    s = supported_elements()
    assert {1, 6, 7, 8, 9, 17}.issubset(s)   # H, C, N, O, F, Cl
    assert 26 not in s                        # Fe (3d TM) excluded


def test_run_job_atomization_block(tmp_path: Path):
    """run_job(atomization=True) writes an atomization block + structured event;
    off by default."""
    r = run_job(_h2o(), basis="6-31g", method="rhf", atomization=True,
                output=str(tmp_path / "h2o"), verbose=0)
    out = (tmp_path / "h2o.out").read_text()
    assert "## Atomization energy" in out
    assert "Atomization energy" in out
    assert "kcal/mol" in out

    # Off by default.
    run_job(_h2o(), basis="6-31g", method="rhf",
            output=str(tmp_path / "h2o_off"), verbose=0)
    assert "## Atomization energy" not in (tmp_path / "h2o_off.out").read_text()
