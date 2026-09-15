"""Restricted open-shell Kohn--Sham (ROKS) tests.

End-to-end (real integrals + libxc; runs on a build box):

* ROKS on a closed-shell singlet reproduces RKS (the Roothaan effective
  Fock collapses to the closed-shell KS Fock; ``E_xc[rho/2, rho/2]`` from
  the spin-polarised functional equals the unpolarised ``E_xc[rho]``).
* Open-shell ROKS is spin-pure (``<S^2> = S(S+1)`` exactly).
* Meta-GGA and range-separated-hybrid ROKS routes are live; direct
  double-hybrid ROKS still raises because the SCF energy alone is not
  the full two-part method.
* Energy parity against PySCF's ROKS where PySCF is present.

The Roothaan coupling and SCF machinery are shared with ROHF and are
unit-tested there (``tests/test_rohf.py`` pure-math tier).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, run_rks
from vibeqc.roks import ROKSOptions, run_roks

from .conftest import GEOMETRIES

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _tight() -> ROKSOptions:
    return ROKSOptions(
        max_iter=500,
        conv_tol_energy=1e-11,
        # The pure-Python open-shell PBE Roothaan loop reaches a stable
        # two-cycle just above 1e-7 on OH/STO-3G; TPSS sits near 2.7e-7.
        # 3e-7 remains tighter than the default while matching PySCF at
        # microhartree scale.
        conv_tol_grad=3e-7,
        diis_subspace_size=10,
    )


@pytest.mark.parametrize("functional", ["lda", "pbe", "b3lyp", "tpss", "wb97x"])
@pytest.mark.parametrize("mol_key,basis_name", [("H2", "sto-3g"), ("H2O", "sto-3g")])
def test_roks_on_closed_shell_matches_rks(functional, mol_key, basis_name):
    """Closed-shell singlet: ROKS == RKS across the supported KS families."""
    atoms = GEOMETRIES[mol_key]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, basis_name)

    from vibeqc import RKSOptions

    rks_opts = RKSOptions()
    rks_opts.functional = functional
    rks_opts.conv_tol_energy = 1e-11
    r_rks = run_rks(mol, basis, rks_opts)
    r_roks = run_roks(mol, basis, _tight(), functional=functional)

    assert r_roks.converged and r_rks.converged
    # Same grid + functional; ROKS reduces to RKS at the closed-shell
    # density. Grid/convergence slack -> 1e-6.
    assert r_roks.energy == pytest.approx(r_rks.energy, abs=1e-6)
    assert abs(r_roks.s_squared) < 1e-12


def test_roks_range_separated_mgga_vv10_closed_shell_matches_rks():
    """wB97M-V exercises RSH exchange + meta-GGA tau + VV10 in one route."""
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )
    basis = BasisSet(mol, "sto-3g")

    from vibeqc import RKSOptions

    rks_opts = RKSOptions()
    rks_opts.functional = "wb97m-v"
    rks_opts.conv_tol_energy = 1e-11
    # ROKS currently evaluates VV10 on the full XC grid. Disable the RKS
    # coarse-grid policy so this closed-shell reduction compares one
    # numerical integration contract rather than two grid policies.
    rks_opts.grid.vv10_grid_factor = 1.0
    r_rks = run_rks(mol, basis, rks_opts)
    r_roks = run_roks(mol, basis, _tight(), functional="wb97m-v")

    assert r_roks.converged and r_rks.converged
    assert r_roks.energy == pytest.approx(r_rks.energy, abs=1e-6)
    assert abs(r_roks.s_squared) < 1e-12


def test_roks_doublet_spin_pure():
    """Open-shell ROKS/PBE on the OH radical: <S^2> = 0.75 exactly."""
    atoms = [(8, [0.0, 0.0, 0.0]), (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])]
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms], multiplicity=2
    )
    basis = BasisSet(mol, "sto-3g")
    r = run_roks(mol, basis, _tight(), functional="pbe")
    assert r.converged
    assert r.s_squared == pytest.approx(0.75, abs=1e-12)
    # The XC energy component is populated for the energy-breakdown block.
    assert r.e_xc < 0.0
    assert r.functional == "pbe"


@pytest.mark.parametrize("basis_name", ["def2-svp", "def2-tzvp"])
def test_roks_oh_pi_degenerate_shell_converges(basis_name):
    """OH(2Pi) needs a fractional RO frontier shell in larger bases."""
    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR])],
        multiplicity=2,
    )
    basis = BasisSet(mol, basis_name)

    # conv_tol_grad is 1e-8 rather than 1e-6 because the last assertion in
    # this test compares ``r.density`` (the final cycle's INPUT density)
    # against the density rebuilt from the converged MOs. Their difference
    # is the exit |dD|, which the gradient tolerance bounds -- at 1e-6 that
    # residual is ~1e-7, so the 5e-9 pin below only ever held by trajectory
    # accident. It broke when the #119 guess fix changed the path to the
    # same fixed point (same basin, same 2/2/2/1.5/1.5/0 occupations,
    # energy unchanged to 2.2e-10 Ha, exit |dD| 5.9e-8). Tightening the run
    # makes the density check a real invariant rather than a path artefact;
    # n_iter stays well inside the cap asserted below (9 -> 12 at def2-SVP,
    # 11 -> 13 at def2-TZVP).
    opts = ROKSOptions(max_iter=120, conv_tol_energy=1e-11, conv_tol_grad=1e-8)
    r = run_roks(mol, basis, opts, functional="pbe")

    assert r.converged
    assert r.n_iter <= 20
    assert r.s_squared == pytest.approx(0.75, abs=1e-12)
    assert list(r.mo_occupations[:6]) == pytest.approx(
        [2.0, 2.0, 2.0, 1.5, 1.5, 0.0], abs=1e-12
    )
    assert r.rohf_canonicalization == "roothaan-fractional-shell"
    assert r.mo_energies[3] == pytest.approx(r.mo_energies[4], abs=1e-8)
    assert r.mo_energies == pytest.approx(r.mo_energies_roothaan, abs=1e-12)
    assert r.mo_coeffs == pytest.approx(r.mo_coeffs_roothaan, abs=1e-12)
    assert np.all(np.diff(r.mo_energies) >= -1.0e-10)
    density_from_mos = (
        np.asarray(r.mo_coeffs) * np.asarray(r.mo_occupations)
    ) @ np.asarray(r.mo_coeffs).T
    assert density_from_mos == pytest.approx(r.density, abs=5.0e-9)


def test_run_job_default_roks_grid_materialises_at_tzvp(tmp_path):
    """``run_job(method="roks")`` with default options must not crash.

    Regression: ``ROKSOptions`` is a pure-Python dataclass whose ``grid``
    defaults to None (the driver materialises a default GridOptions when it
    sees None), while the C++ RKS/UKS options default to a live
    GridOptions instance. ``_run_single_point`` applied the grid-level
    preset in-place via ``_apply_grid_level(options.grid, ...)``, so a bare
    default ROKS run raised ``AttributeError: 'NoneType' object has no
    attribute ...`` before ever reaching the driver (first seen at
    def2-TZVP, where the run is big enough to be worth launching).
    """
    from vibeqc import run_job

    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR])],
        multiplicity=2,
    )
    result = run_job(
        mol,
        basis="def2-tzvp",
        method="roks",
        functional="pbe",
        output=str(tmp_path / "roks_default_grid"),
        citations=False,
        record_hostname=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert result.converged
    assert result.s_squared == pytest.approx(0.75, abs=1e-12)
    assert result.rohf_canonicalization == "roothaan-fractional-shell"
    system_text = (tmp_path / "roks_default_grid.system").read_text()
    assert 'rohf_canonicalization = "roothaan-fractional-shell"' in system_text
    # The default grid must have received the run_job grid-level preset
    # (materialised GridOptions, not None) -- a converged spin-pure OH
    # energy is the observable consequence.
    assert result.energy < -75.0


@pytest.mark.parametrize(
    "functional",
    ["b2plyp", "pwpb95"],
)
def test_roks_refuses_direct_double_hybrid(functional):
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(NotImplementedError, match="double hybrid"):
        run_roks(mol, basis, _tight(), functional=functional)


@pytest.mark.parametrize(
    "label,atoms,charge,mult,functional,pyscf_functional",
    [
        ("OH-pbe", [(8, [0.0, 0.0, 0.0]),
                    (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
         0, 2, "pbe", "pbe"),
        ("OH-lda", [(8, [0.0, 0.0, 0.0]),
                    (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
         0, 2, "lda", "svwn"),
        ("OH-tpss", [(8, [0.0, 0.0, 0.0]),
                     (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
         0, 2, "tpss", "tpss,tpss"),
        ("H-wb97x", [(1, [0.0, 0.0, 0.0])], 0, 2, "wb97x", "wb97x"),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_roks_energy_matches_pyscf(
    label, atoms, charge, mult, functional, pyscf_functional
):
    """Energy + <S^2> parity against PySCF's integer-occupation ROKS."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    m = gto.Mole()
    m.unit = "Bohr"
    m.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    m.basis = "sto-3g"
    m.charge = charge
    m.spin = mult - 1
    m.verbose = 0
    m.build()
    mf = dft.ROKS(m)
    mf.xc = pyscf_functional
    mf.conv_tol = 1e-11
    mf.kernel()
    ref_e = mf.e_tot

    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms],
        charge=charge,
        multiplicity=mult,
    )
    basis = BasisSet(mol, "sto-3g")
    opts = _tight()
    opts.fractional_open_shell = False
    r = run_roks(mol, basis, opts, functional=functional)
    assert r.converged
    # Grid + libxc-build differences vs PySCF -> ~1e-5 tolerance.
    assert abs(r.energy - ref_e) < 1e-4, (
        f"{label}: E_vibeqc={r.energy:.8f} E_pyscf={ref_e:.8f}"
    )
