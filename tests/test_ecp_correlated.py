"""Post-HF, restricted open-shell and determinant routes on ECP references.

Until 2026-09 every route beyond mean-field refused an ECP reference
because the kernels partitioned their occupied space from
``Molecule.n_electrons()``.  They now use the reference's valence count and
an ECP-aware frozen core.  The parity pins compare against PySCF, run in
this process only as a reference generator (CLAUDE.md § 10: no PySCF
import under ``python/vibeqc``); every system is H2S-sized so the file
stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.correlation_conventions import (
    effective_electron_count,
    published_frozen_core_orbital_count,
    resolve_frozen_core_count,
)

pyscf = pytest.importorskip("pyscf")

_H2S_XYZ = [(16, [0.0, 0.0, 0.0]), (1, [0.0, 1.815, 1.425]), (1, [0.0, -1.815, 1.425])]
_SH_XYZ = [(16, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 2.55])]


def _molecule(atoms, charge=0, multiplicity=1):
    return vq.Molecule([vq.Atom(z, xyz) for z, xyz in atoms], charge, multiplicity)


def _pyscf_mol(atoms, charge=0, multiplicity=1):
    from pyscf import gto

    symbols = {16: "S", 1: "H"}
    atom = "; ".join(f"{symbols[z]} {x} {y} {w}" for z, (x, y, w) in atoms)
    return gto.M(
        atom=atom, unit="Bohr", basis="lanl2dz", ecp="lanl2dz",
        charge=charge, spin=multiplicity - 1, verbose=0,
    )


def _tight(opts):
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return opts


@pytest.fixture(scope="module")
def h2s_rhf():
    mol = _molecule(_H2S_XYZ)
    basis = vq.BasisSet(mol, "lanl2dz")
    result = vq.run_rhf(mol, basis, _tight(vq.RHFOptions()))
    assert result.converged and result.ecp_total_ncore == 10
    return mol, basis, result


@pytest.fixture(scope="module")
def h2s_pyscf():
    from pyscf import scf

    mf = scf.RHF(_pyscf_mol(_H2S_XYZ))
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


# ---------------------------------------------------------------------------
# Reference and counts
# ---------------------------------------------------------------------------


def test_rhf_reference_matches_pyscf(h2s_rhf, h2s_pyscf):
    _, _, result = h2s_rhf
    assert result.energy == pytest.approx(h2s_pyscf.e_tot, abs=2e-6)


def test_effective_electron_count_is_the_valence_count(h2s_rhf):
    mol, _, result = h2s_rhf
    assert mol.n_electrons() == 18
    assert effective_electron_count(mol, result) == 8


def test_published_frozen_core_subtracts_the_ecp_core(h2s_rhf):
    """ORCA's table freezes ten electrons on sulfur; LANL2DZ already removed
    exactly those ten, so nothing is frozen on the ECP reference."""
    mol, _, result = h2s_rhf
    assert published_frozen_core_orbital_count(mol) == 5
    assert published_frozen_core_orbital_count(mol, result) == 0
    assert resolve_frozen_core_count(mol, None, reference=result) == 0
    assert resolve_frozen_core_count(mol, "published", reference=result) == 0
    assert resolve_frozen_core_count(mol, 2, reference=result) == 2


def test_small_core_ecp_leaves_the_semicore_frozen():
    """Iodine in dhf-TZVP: the published core is 36 electrons, the
    Dirac-Fock ECP removes 28, so 4s4p (8 electrons, 4 orbitals) stay
    frozen per atom while nothing is frozen twice. dhf-TZVP starts at Rb,
    so I2 is the light-partner-free test system."""
    mol = vq.Molecule([vq.Atom(53, [0, 0, 0]), vq.Atom(53, [0, 0, 5.0])], 0, 1)
    basis = vq.BasisSet(mol, "dhf-tzvp")
    opts = vq.RHFOptions()
    opts.max_iter = 1
    result = vq.run_rhf(mol, basis, opts)
    assert result.ecp_total_ncore == 2 * 28
    assert published_frozen_core_orbital_count(mol) == 2 * 18
    assert published_frozen_core_orbital_count(mol, result) == 2 * 4


# ---------------------------------------------------------------------------
# Closed-shell post-HF against PySCF
# ---------------------------------------------------------------------------


def test_mp2_on_ecp_reference_matches_pyscf(h2s_rhf, h2s_pyscf):
    from pyscf import mp

    mol, basis, result = h2s_rhf
    opts = vq.MP2Options()
    opts.n_frozen_core = 0
    opts.density_fit = False
    ours = vq.run_mp2(mol, basis, result, opts)
    ref = mp.MP2(h2s_pyscf).run(frozen=0)
    assert ours.e_correlation == pytest.approx(ref.e_corr, abs=2e-6)


def test_mp2_default_frozen_core_resolves_through_the_wrapper(h2s_rhf):
    """MP2Options() leaves n_frozen_core=-1; the native kernel refuses to
    apply the element table to an ECP reference, and the public wrapper
    resolves the ECP-aware count (zero here) instead."""
    mol, basis, result = h2s_rhf
    opts = vq.MP2Options()
    opts.density_fit = False
    with pytest.raises(Exception, match="cannot be applied natively to an ECP reference"):
        vq.run_mp2(mol, basis, result, opts)
    scs = vq.run_scs_mp2(mol, basis, result, density_fit=False)
    assert scs.e_correlation < 0.0


def test_ccsd_on_ecp_reference_matches_pyscf(h2s_rhf, h2s_pyscf):
    from pyscf import cc

    mol, basis, result = h2s_rhf
    opts = vq.CCSDOptions()
    opts.n_frozen_core = 0
    opts.density_fit = False
    opts.compute_triples = False
    ours = vq.run_ccsd(mol, basis, result, opts)
    ref = cc.CCSD(h2s_pyscf, frozen=0).run()
    assert ours.converged
    assert ours.e_ccsd_correlation == pytest.approx(ref.e_corr, abs=5e-6)


def test_casci_on_ecp_reference_matches_pyscf(h2s_pyscf, tmp_path):
    """#740: CASCI on an ECP reference against PySCF's own CAS with the same
    LANL2DZ ECP. This is the load-bearing check of the whole CAS/MR-PT half:
    the family is handed the reference-built Hamiltonian, so if the ECP
    operator or the valence count were wrong the CAS energy would move.
    Measured 2026-09-07: agreement to 5.4e-08 Ha."""
    mcscf = pytest.importorskip("pyscf.mcscf")
    ref = mcscf.CASCI(h2s_pyscf, 4, 4).run()
    ours = vq.run_job(
        _molecule(_H2S_XYZ), basis="lanl2dz", method="casci",
        active_space=(4, 4), output=tmp_path / "casci-ecp", progress=False,
    )
    assert float(ours.energy) == pytest.approx(float(ref.e_tot), abs=1e-6)


def test_dlpno_mp2_on_ecp_reference_tracks_canonical(h2s_rhf):
    from vibeqc.dlpno.mp2 import DLPNOMP2Options, run_dlpno_mp2

    mol, basis, result = h2s_rhf
    canonical = vq.MP2Options()
    canonical.n_frozen_core = 0
    canonical.density_fit = False
    e_canonical = vq.run_mp2(mol, basis, result, canonical).e_correlation
    aux = vq.BasisSet(mol, "def2-svp-rifit", require_all_atoms=False)
    df = vq.DensityFitting(basis, aux, aux_basis_name="def2-svp-rifit", molecule=mol)
    opts = DLPNOMP2Options()
    opts.n_frozen = 0
    dlpno = run_dlpno_mp2(mol, basis, result, df, opts)
    assert dlpno.e_corr == pytest.approx(e_canonical, abs=5e-4)


# ---------------------------------------------------------------------------
# Open-shell routes
# ---------------------------------------------------------------------------


def test_ump2_on_ecp_reference_matches_pyscf():
    from pyscf import mp, scf

    mol = _molecule(_SH_XYZ, 0, 2)
    basis = vq.BasisSet(mol, "lanl2dz")
    uhf = vq.run_uhf(mol, basis, _tight(vq.UHFOptions()))
    assert uhf.converged and uhf.ecp_total_ncore == 10
    mf = scf.UHF(_pyscf_mol(_SH_XYZ, 0, 2))
    mf.conv_tol = 1e-12
    mf.kernel()
    assert uhf.energy == pytest.approx(mf.e_tot, abs=2e-6)
    opts = vq.UMP2Options()
    opts.n_frozen_core = 0
    opts.density_fit = False
    ours = vq.run_ump2(mol, basis, uhf, opts)
    ref = mp.UMP2(mf).run(frozen=0)
    assert ours.e_correlation == pytest.approx(ref.e_corr, abs=2e-6)


def test_rohf_on_ecp_reference_matches_pyscf():
    from pyscf import scf

    mol = _molecule(_SH_XYZ, 0, 2)
    basis = vq.BasisSet(mol, "lanl2dz")
    opts = vq.rohf.ROHFOptions(max_iter=200, conv_tol_energy=1e-12, conv_tol_grad=1e-10)
    rohf = vq.run_rohf(mol, basis, opts)
    assert rohf.converged
    assert rohf.ecp_operator_applied and rohf.ecp_total_ncore == 10
    assert rohf.n_alpha + rohf.n_beta == 7
    assert list(rohf.ecp_effective_charges) == pytest.approx([6.0, 1.0])
    mf = scf.ROHF(_pyscf_mol(_SH_XYZ, 0, 2))
    mf.conv_tol = 1e-12
    mf.kernel()
    assert rohf.energy == pytest.approx(mf.e_tot, abs=2e-6)


def test_roks_on_ecp_reference_runs_with_the_valence_count():
    mol = _molecule(_SH_XYZ, 0, 2)
    basis = vq.BasisSet(mol, "lanl2dz")
    opts = vq.roks.ROKSOptions(max_iter=200)
    roks = vq.run_roks(mol, basis, opts, functional="pbe")
    assert roks.converged
    assert roks.ecp_operator_applied and roks.ecp_total_ncore == 10
    assert roks.n_alpha + roks.n_beta == 7


# ---------------------------------------------------------------------------
# The determinant family consumes the reference Hamiltonian
# ---------------------------------------------------------------------------


def test_cisd_on_ecp_reference_matches_pyscf(h2s_pyscf, tmp_path):
    from pyscf import ci

    mol = _molecule(_H2S_XYZ)
    result = vq.run_job(mol, basis="lanl2dz", method="cisd", output=tmp_path / "cisd")
    ref = ci.CISD(h2s_pyscf, frozen=0).run()
    assert result.energy == pytest.approx(ref.e_tot, abs=5e-6)


# ---------------------------------------------------------------------------
# run_job routes that used to refuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mp2", "ccsd", "rohf"])
def test_run_job_accepts_ecp_references_on_lifted_routes(method, tmp_path):
    """run_job used to refuse these outright on an ECP basis. LANL2DZ has no
    fitting aux, so the conventional (non-DF) route is the one to exercise."""
    mol = _molecule(_H2S_XYZ) if method != "rohf" else _molecule(_SH_XYZ, 0, 2)
    kwargs = {}
    if method in ("mp2", "ccsd"):
        opts = vq.MP2Options() if method == "mp2" else vq.CCSDOptions()
        opts.density_fit = False
        kwargs["mp2_options" if method == "mp2" else "ccsd_options"] = opts
    result = vq.run_job(
        mol, basis="lanl2dz", method=method, output=tmp_path / method, **kwargs
    )
    assert result.converged
    out = (tmp_path / f"{method}.out").read_text()
    assert "libecpint" in out.lower() or "ecp" in out.lower()


def test_run_uccsd_from_mos_needs_an_explicit_frozen_core_on_ecp_arrays():
    mol = _molecule(_SH_XYZ, 0, 2)
    basis = vq.BasisSet(mol, "lanl2dz")
    uhf = vq.run_uhf(mol, basis, _tight(vq.UHFOptions()))
    with pytest.raises(ValueError, match="explicit n_frozen_core"):
        vq.cc.run_uccsd_from_mos(
            mol, basis, np.asarray(uhf.mo_coeffs_alpha), np.asarray(uhf.mo_coeffs_beta),
            np.asarray(uhf.fock_alpha), np.asarray(uhf.fock_beta), uhf.energy,
            ecp_total_ncore=uhf.ecp_total_ncore,
        )


# ---------------------------------------------------------------------------
# Orbital-basis coverage guard
# ---------------------------------------------------------------------------


def test_basisset_refuses_an_element_the_file_omits():
    """cc-pVDZ stops at Kr: silver used to enter with zero shells (measured
    on def2-TZVP before its Rb-Rn blocks were bundled)."""
    mol = vq.Molecule([vq.Atom(47, [0, 0, 0]), vq.Atom(1, [0, 0, 3.1])], 0, 1)
    with pytest.raises(RuntimeError, match=r"no functions for Z=47"):
        vq.BasisSet(mol, "cc-pvdz")
    partial = vq.BasisSet(mol, "cc-pvdz", require_all_atoms=False)
    assert all(s.atom_index == 1 for s in partial.shells())


def test_def2_beyond_kr_matches_pyscf_with_the_def2_ecp():
    """AgH in def2-TZVP: the appended BSE block and its def2-ECP sidecar
    reproduce PySCF's built-in def2-TZVP + def2-ECP to the microhartree."""
    from pyscf import gto, scf

    mol = vq.Molecule([vq.Atom(47, [0, 0, 0]), vq.Atom(1, [0, 0, 3.1])], 0, 1)
    basis = vq.BasisSet(mol, "def2-tzvp")
    ours = vq.run_rhf(mol, basis, _tight(vq.RHFOptions()))
    assert ours.converged and ours.ecp_total_ncore == 28
    ref = scf.RHF(
        gto.M(atom="Ag 0 0 0; H 0 0 3.1", unit="Bohr", basis="def2-tzvp",
              ecp="def2-tzvp", verbose=0)
    )
    ref.conv_tol = 1e-12
    ref.kernel()
    assert ours.energy == pytest.approx(ref.e_tot, abs=2e-6)


def test_basisset_ghost_centre_is_exempt_from_the_coverage_guard():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(0, [0, 0, 1.4])], 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    assert basis.nbasis >= 1
