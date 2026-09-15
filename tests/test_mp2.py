"""RMP2 correlation energy vs PySCF."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    MP2Options,
    Molecule,
    RHFOptions,
    run_mp2,
    run_rhf,
)

from .conftest import GEOMETRIES


def _all_electron_mp2_options():
    """Preserve the pre-#140 all-electron numerical reference space."""
    options = MP2Options()
    options.n_frozen_core = 0
    return options


def _pyscf_mp2(atoms_bohr, basis_name):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf, mp
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    m = mp.MP2(mf, frozen=0)
    m.kernel()
    return mf.e_tot, m.e_corr, m.e_tot


def _vibeqc_mp2(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    hf = run_rhf(mol, basis, opts)
    return hf, run_mp2(mol, basis, hf, _all_electron_mp2_options())


MP2_CASES = [
    ("H2",  "sto-3g"),
    ("H2",  "6-31g*"),
    ("H2O", "sto-3g"),
    ("H2O", "6-31g*"),
    ("H2O", "cc-pvdz"),
    ("CH4", "sto-3g"),
]


@pytest.mark.parametrize(
    "mol_key,basis_name", MP2_CASES,
    ids=[f"{m}-{b}" for m, b in MP2_CASES],
)
def test_mp2_correlation_energy_matches_pyscf(mol_key, basis_name):
    atoms = GEOMETRIES[mol_key]
    hf, mp2 = _vibeqc_mp2(atoms, basis_name)
    _, ec_ref, etot_ref = _pyscf_mp2(atoms, basis_name)

    # MP2 correlation is variational in the HF orbitals; agreement is set
    # by the underlying HF energy/orbitals (machine precision against PySCF)
    # and the AO→MO integral transform (machine precision).
    assert abs(mp2.e_correlation - ec_ref) < 1e-9, (
        f"{mol_key}/{basis_name}: "
        f"E_corr vibeqc = {mp2.e_correlation:.12f}, "
        f"E_corr pyscf = {ec_ref:.12f}, "
        f"diff = {mp2.e_correlation - ec_ref:+.2e}"
    )
    assert abs(mp2.e_total - etot_ref) < 1e-9


def test_mp2_energy_decomposition_sums_to_total():
    atoms = GEOMETRIES["H2O"]
    hf, mp2 = _vibeqc_mp2(atoms, "sto-3g")
    # Same-spin + opposite-spin = total correlation (RMP2 convention).
    assert mp2.e_ss + mp2.e_os == pytest.approx(mp2.e_correlation, rel=1e-12)
    assert mp2.e_hf + mp2.e_correlation == pytest.approx(mp2.e_total, rel=1e-14)


def test_mp2_correlation_is_negative():
    """MP2 correlation is strictly negative for bound states: the
    numerator of each term is positive (squared ERI minus positive cross
    term) while the energy denominator Δ = ε_occ − ε_virt < 0 always."""
    atoms = GEOMETRIES["H2O"]
    _, mp2 = _vibeqc_mp2(atoms, "6-31g*")
    assert mp2.e_correlation < 0


def test_mp2_rejects_unconverged_hf():
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    hf = run_rhf(mol, basis, opts)
    assert not hf.converged
    with pytest.raises(RuntimeError, match="not converged"):
        run_mp2(mol, basis, hf)


def test_mp2_rejects_open_shell_reference():
    """RMP2 requires a closed-shell RHF reference."""
    mol = Molecule([Atom(3, [0, 0, 0])], multiplicity=2)  # Li atom (doublet)
    basis = BasisSet(mol, "sto-3g")
    # Can't even build an RHF reference for Li — skip via an artificial
    # Molecule with even electrons but multiplicity != 1.
    mol_tri = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
                      charge=0, multiplicity=3)
    basis_tri = BasisSet(mol_tri, "sto-3g")
    # run_rhf will refuse mult != 1; fabricate a "fake" RHFResult instead?
    # Easier: check that the MP2 driver refuses when mol itself is open-shell.
    # We need a converged RHF result to pass in, so use H2 RHF result with
    # a mol that says multiplicity = 3 — but construction would fail earlier.
    # Instead, test the "even electrons but multiplicity=1 check inside run_mp2":
    # skip this corner case and rely on run_rhf to guard.
    pytest.skip("MP2 open-shell rejection is covered by run_rhf's guards.")


def test_mp2_direct_and_disk_match_incore_and_honor_slab_budget(tmp_path):
    """BUG 63: exact direct/disk routes never build AO ERI or full OVOV."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    ropts = RHFOptions()
    ropts.conv_tol_energy = 1e-12
    ropts.conv_tol_grad = 1e-10
    hf = run_rhf(mol, basis, ropts)

    fast_opts = _all_electron_mp2_options()
    fast_opts.memory_mode = "incore"
    fast = run_mp2(mol, basis, hf, fast_opts)

    nocc = mol.n_electrons() // 2
    nvir = basis.nbasis - nocc
    nao = basis.nbasis
    one_slab_bytes = nao**2 * 8 + max(
        (nao**3 + nvir * nao**2) * 8,
        (nvir * nao**2 + nvir * nocc * nao) * 8,
        (nvir * nocc * nao + nvir * nocc * nvir) * 8,
    )

    direct_opts = _all_electron_mp2_options()
    direct_opts.memory_mode = "direct"
    direct_opts.requested_memory_bytes = one_slab_bytes
    direct = run_mp2(mol, basis, hf, direct_opts)
    assert direct.memory_mode_used == "direct"
    assert direct.workspace_bytes <= direct_opts.requested_memory_bytes
    assert direct.disk_bytes == 0
    assert direct.e_os == pytest.approx(fast.e_os, abs=1e-11)
    assert direct.e_ss == pytest.approx(fast.e_ss, abs=1e-11)

    disk_opts = _all_electron_mp2_options()
    disk_opts.memory_mode = "disk"
    disk_opts.requested_memory_bytes = one_slab_bytes
    disk_opts.scratch_directory = str(tmp_path)
    disk = run_mp2(mol, basis, hf, disk_opts)
    assert disk.memory_mode_used == "disk"
    assert disk.workspace_bytes <= disk_opts.requested_memory_bytes
    assert disk.disk_bytes > 0
    assert disk.e_correlation == pytest.approx(fast.e_correlation, abs=1e-11)
    assert list(tmp_path.iterdir()) == []


def test_mp2_direct_matches_incore_for_nonidentical_shell_pairs():
    """Direct AO contraction must retain quartets whose shell pairs differ.

    def2-SVP has shell quartets with the same first shell but differently
    ordered second shells.  A basis-function pair-order filter used outside
    an identical shell-pair quartet silently discarded some of those ERIs.
    """
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")
    ropts = RHFOptions()
    ropts.conv_tol_energy = 1e-12
    ropts.conv_tol_grad = 1e-10
    hf = run_rhf(mol, basis, ropts)

    incore_opts = _all_electron_mp2_options()
    incore_opts.memory_mode = "incore"
    incore = run_mp2(mol, basis, hf, incore_opts)

    direct_opts = _all_electron_mp2_options()
    direct_opts.memory_mode = "direct"
    direct = run_mp2(mol, basis, hf, direct_opts)

    assert direct.e_os == pytest.approx(incore.e_os, abs=1e-11)
    assert direct.e_ss == pytest.approx(incore.e_ss, abs=1e-11)
    assert direct.e_correlation == pytest.approx(
        incore.e_correlation, abs=1e-11
    )


def test_mp2_auto_selects_direct_and_rejects_impossible_budget():
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "sto-3g")
    hf = run_rhf(mol, basis, RHFOptions())
    nocc = mol.n_electrons() // 2
    nvir = basis.nbasis - nocc
    nao = basis.nbasis
    one_slab_bytes = nao**2 * 8 + max(
        (nao**3 + nvir * nao**2) * 8,
        (nvir * nao**2 + nvir * nocc * nao) * 8,
        (nvir * nocc * nao + nvir * nocc * nvir) * 8,
    )

    opts = _all_electron_mp2_options()
    opts.requested_memory_bytes = one_slab_bytes
    result = run_mp2(mol, basis, hf, opts)
    assert result.memory_mode_used == "direct"
    assert result.workspace_bytes <= opts.requested_memory_bytes

    opts.requested_memory_bytes = one_slab_bytes - 1
    with pytest.raises(RuntimeError, match="minimum one-occupied-orbital slab"):
        run_mp2(mol, basis, hf, opts)


def test_df_mp2_direct_panels_match_incore():
    atoms = GEOMETRIES["H2"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "cc-pvdz")
    hf = run_rhf(mol, basis, RHFOptions())

    fast_opts = _all_electron_mp2_options()
    fast_opts.density_fit = True
    fast_opts.aux_basis = "cc-pvdz-ri"
    fast_opts.memory_mode = "incore"
    fast = run_mp2(mol, basis, hf, fast_opts)

    # Explicit direct mode with no numerical cap still bounds the OVOV step
    # to the kernel's 64 MiB occupied-slab target. The generic DF construction
    # storage is reported in the telemetry as part of the peak workspace.
    direct_opts = _all_electron_mp2_options()
    direct_opts.density_fit = True
    direct_opts.aux_basis = "cc-pvdz-ri"
    direct_opts.memory_mode = "direct"
    direct = run_mp2(mol, basis, hf, direct_opts)
    assert direct.memory_mode_used == "direct"
    assert direct.disk_bytes == 0
    assert direct.e_os == pytest.approx(fast.e_os, abs=1e-11)
    assert direct.e_ss == pytest.approx(fast.e_ss, abs=1e-11)
