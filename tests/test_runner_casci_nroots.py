"""run_job wiring for multi-root CASCI (casci_options.nroots).

Engine-level multi-root validation lives in tests/test_casci_direct.py and
the solver suite; this file pins the public pass-through (2026-06 docs
audit: ``SolverResult.root_energies`` documented multi-root CASCI but the
runner never plumbed a root count for ``method="casci"``): the
``casci_options=CASCIOptions(nroots=N)`` kwarg, root surfacing on the
result, ground-root invariance of the headline energy, the .out root
table, and parity with the standalone ``vibeqc.solvers.casci`` solver.
"""

from __future__ import annotations

import numpy as np
from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import BasisSet
from vibeqc.runner import run_job
from vibeqc.solvers import (
    CASCIOptions,
    build_hamiltonian_mo,
    casci,
    get_hf_orbital_provider,
)

# H2 at R = 1.4 bohr: CAS(2,2)/STO-3G is the full CI space (4 determinants
# in the M_s=0 sector), so nroots=3 exercises ground + triplet + excited
# singlet without any truncation ambiguity.
H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])


def _run(tmp_path, name, **kw):
    return run_job(
        H2,
        basis="sto-3g",
        method="casci",
        active_space=(2, 2),
        output=tmp_path / name,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        **kw,
    )


def test_run_job_casci_nroots_surfaces_roots(tmp_path):
    """nroots=3 lands three ascending roots; energy stays the ground root."""
    res = _run(tmp_path, "h2-3roots", casci_options=CASCIOptions(nroots=3))
    assert res.converged
    assert res.root_energies is not None and len(res.root_energies) == 3
    assert res.root_energies == sorted(res.root_energies)
    assert res.energy == res.root_energies[0]
    # M_s=0 determinant sector, no S^2 filter: the triplet (S(S+1)=2)
    # interleaves between the two singlets — the documented composition.
    assert res.root_s2 is not None and len(res.root_s2) == 3
    assert np.allclose(res.root_s2, [0.0, 2.0, 0.0], atol=1e-8)
    out_text = (tmp_path / "h2-3roots.out").read_text()
    assert "Root energies:" in out_text
    assert out_text.count("root ") >= 3


def test_run_job_casci_nroots_matches_direct_solver(tmp_path):
    """run_job roots == standalone vibeqc.solvers.casci(..., nroots=3)."""
    res = _run(tmp_path, "h2-parity", casci_options=CASCIOptions(nroots=3))

    b = BasisSet(H2, "sto-3g")
    C = get_hf_orbital_provider(H2, b, method="rhf")
    H = build_hamiltonian_mo(H2, b, C)
    ref = casci(
        H.h1e,
        H.h2e,
        n_active_elec=2,
        n_active_orb=2,
        n_core=0,
        nuclear_repulsion=H.nuclear_repulsion,
        ms2=0,
        nroots=3,
    )
    assert np.allclose(res.root_energies, ref.e_totals, atol=1e-10)


def test_run_job_casci_default_stays_single_root(tmp_path):
    """No casci_options: single root, no root table — and the ground-root
    energy is unchanged by requesting more roots."""
    res = _run(tmp_path, "h2-default")
    assert res.root_energies is None
    multi = _run(tmp_path, "h2-multi", casci_options=CASCIOptions(nroots=2))
    assert abs(res.energy - multi.energy) < 1e-12


def test_casci_options_ignored_by_other_methods(tmp_path):
    """casci_options is method="casci"-only: a single-state CASPT2 keeps its
    single-root CASCI reference (root surfacing unchanged) when the kwarg
    is passed by mistake."""
    res = run_job(
        H2,
        basis="sto-3g",
        method="caspt2",
        active_space=(2, 2),
        casci_options=CASCIOptions(nroots=3),
        output=tmp_path / "h2-caspt2",
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert res.converged
    assert res.root_energies is None
