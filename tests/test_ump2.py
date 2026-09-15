"""UMP2 correlation energy on a UHF reference vs PySCF."""

from __future__ import annotations

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    UHFOptions,
    UMP2Options,
    run_uhf,
    run_ump2,
)

from .conftest import ANGSTROM_TO_BOHR


def _all_electron_ump2_options():
    """Preserve the pre-#140 all-electron numerical reference space."""
    options = UMP2Options()
    options.n_frozen_core = 0
    return options


# Open-shell systems. Geometries in Angstrom; converted to bohr for vibe-qc,
# PySCF takes Angstrom directly.
OPEN_SHELL_CASES = [
    # name,        atoms_A,                              charge, mult, basis
    ("H-doublet",  [("H", 0.0, 0.0, 0.0)],                    0, 2, "sto-3g"),
    ("H-doublet-pvdz", [("H", 0.0, 0.0, 0.0)],                0, 2, "cc-pvdz"),
    ("N-quartet",  [("N", 0.0, 0.0, 0.0)],                    0, 4, "sto-3g"),
    ("OH-doublet", [("O", 0.0, 0.0, 0.0),
                    ("H", 0.0, 0.0, 0.969329)],               0, 2, "sto-3g"),
    ("OH-doublet-631gs", [("O", 0.0, 0.0, 0.0),
                          ("H", 0.0, 0.0, 0.969329)],         0, 2, "6-31g*"),
    ("O2-triplet", [("O", 0.0, 0.0, 0.0),
                    ("O", 0.0, 0.0, 1.208)],                  0, 3, "sto-3g"),
    ("O2-triplet-631gs", [("O", 0.0, 0.0, 0.0),
                          ("O", 0.0, 0.0, 1.208)],            0, 3, "6-31g*"),
]


_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6,
      "N": 7, "O": 8, "F": 9}


def _vibeqc_ump2(atoms_A, charge, mult, basis_name):
    atoms = [Atom(_Z[s], [x * ANGSTROM_TO_BOHR,
                          y * ANGSTROM_TO_BOHR,
                          z * ANGSTROM_TO_BOHR])
             for s, x, y, z in atoms_A]
    mol = Molecule(atoms, charge, mult)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.max_iter = 400
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    # Kernel-parity tests compare against the reference program AT THE
    # SAME SCF SOLUTION. The internal-stability escape (bug
    # UHF-ABOVE-ROHF-VARIATIONAL-INVERSION fix) moves O2 triplet/STO-3G
    # to a lower symmetry-broken UHF solution that PySCF's default UHF
    # does not sit on, which would turn this into a basin comparison
    # rather than an MP2-kernel comparison. Pin the legacy solution.
    opts.stability_check = False
    uhf = run_uhf(mol, basis, opts)
    assert uhf.converged, f"vibeqc UHF did not converge on {basis_name}"
    return uhf, run_ump2(mol, basis, uhf, _all_electron_ump2_options())


def _pyscf_ump2(atoms_A, charge, mult, basis_name):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf, mp
    atom_str = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms_A)
    mol = gto.M(atom=atom_str, basis=basis_name,
                spin=mult - 1, charge=charge,
                unit="Angstrom", verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    m = mp.UMP2(mf, frozen=0)
    m.kernel()
    return mf.e_tot, m.e_corr, m.e_tot


@pytest.mark.parametrize(
    "name,atoms_A,charge,mult,basis_name", OPEN_SHELL_CASES,
    ids=[c[0] for c in OPEN_SHELL_CASES],
)
def test_ump2_matches_pyscf(name, atoms_A, charge, mult, basis_name):
    _, ump2 = _vibeqc_ump2(atoms_A, charge, mult, basis_name)
    _, ec_ref, etot_ref = _pyscf_ump2(atoms_A, charge, mult, basis_name)

    # Agreement set by UHF orbital agreement (~1e-11) and the AO→MO transform
    # (machine precision). 1e-8 is a comfortable bar across all cases tested.
    assert abs(ump2.e_correlation - ec_ref) < 1e-8, (
        f"{name}: E_corr vibeqc = {ump2.e_correlation:.12f}, "
        f"pyscf = {ec_ref:.12f}, diff = {ump2.e_correlation - ec_ref:+.2e}"
    )
    assert abs(ump2.e_total - etot_ref) < 1e-8


def test_ump2_channels_sum_to_total():
    """αα + ββ + αβ == total correlation energy."""
    _, ump2 = _vibeqc_ump2(
        [("O", 0, 0, 0), ("H", 0, 0, 0.969329)],
        charge=0, mult=2, basis_name="6-31g*",
    )
    assert ump2.e_aa + ump2.e_bb + ump2.e_ab == pytest.approx(
        ump2.e_correlation, rel=1e-12, abs=1e-14,
    )
    assert ump2.e_hf + ump2.e_correlation == pytest.approx(
        ump2.e_total, rel=1e-14,
    )


def test_ump2_reduces_to_rmp2_for_closed_shell():
    """On H2 the α and β densities coincide; UMP2 should reproduce RMP2,
    with e_aa == e_bb (pure same-spin) and e_ab == 2 * opposite-spin."""
    from vibeqc import RHFOptions, run_mp2, run_rhf

    atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    mol = Molecule(atoms)
    basis = BasisSet(mol, "6-31g*")

    ropts = RHFOptions()
    ropts.conv_tol_energy = 1e-12
    ropts.conv_tol_grad = 1e-10
    rhf = run_rhf(mol, basis, ropts)
    rmp2 = run_mp2(mol, basis, rhf)

    uopts = UHFOptions()
    uopts.conv_tol_energy = 1e-12
    uopts.conv_tol_grad = 1e-10
    uopts.max_iter = 200
    # Closed-shell forced via multiplicity=1 and even electron count.
    umol = Molecule(atoms, 0, 1)
    uhf = run_uhf(umol, basis, uopts)
    ump2 = run_ump2(umol, basis, uhf)

    assert ump2.e_correlation == pytest.approx(rmp2.e_correlation, abs=1e-10)


def test_ump2_rejects_unconverged_uhf():
    mol = Molecule([Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.832])],
                   charge=0, multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    uhf = run_uhf(mol, basis, opts)
    assert not uhf.converged
    with pytest.raises(RuntimeError, match="not converged"):
        run_ump2(mol, basis, uhf)


def test_ump2_correlation_is_negative():
    """UMP2 correlation is strictly negative for non-degenerate UHF ground
    states: each numerator is a squared integral and the energy denominator
    ε_occ − ε_virt < 0."""
    _, ump2 = _vibeqc_ump2(
        [("O", 0, 0, 0), ("O", 0, 0, 1.208)],
        charge=0, mult=3, basis_name="sto-3g",
    )
    assert ump2.e_correlation < 0
    assert ump2.e_aa <= 0
    assert ump2.e_bb <= 0
    assert ump2.e_ab <= 0


def test_ump2_allows_frozen_core_to_empty_only_the_beta_space():
    """A high-spin alpha-alpha channel survives when active beta is empty."""

    mol = Molecule([Atom(7, [0.0, 0.0, 0.0])], charge=0, multiplicity=4)
    basis = BasisSet(mol, "cc-pvdz")
    uhf_opts = UHFOptions()
    uhf_opts.stability_check = False
    uhf = run_uhf(mol, basis, uhf_opts)
    assert uhf.converged

    options = UMP2Options()
    options.n_frozen_core = 2  # n_beta_total for quartet N
    result = run_ump2(mol, basis, uhf, options)

    assert result.n_frozen_core == 2
    assert result.e_aa < 0.0
    assert result.e_bb == 0.0
    assert result.e_ab == 0.0
    assert result.e_correlation == pytest.approx(result.e_aa)


def test_ump2_direct_and_disk_match_incore_and_clean_scratch(tmp_path):
    atoms_A = [("O", 0, 0, 0), ("H", 0, 0, 0.969329)]
    atoms = [Atom(_Z[s], [x * ANGSTROM_TO_BOHR,
                          y * ANGSTROM_TO_BOHR,
                          z * ANGSTROM_TO_BOHR])
             for s, x, y, z in atoms_A]
    mol = Molecule(atoms, charge=0, multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    uhf_opts = UHFOptions()
    uhf_opts.max_iter = 400
    uhf_opts.stability_check = False
    uhf = run_uhf(mol, basis, uhf_opts)
    assert uhf.converged

    fast_opts = _all_electron_ump2_options()
    fast_opts.memory_mode = "incore"
    fast = run_ump2(mol, basis, uhf, fast_opts)

    nalpha = (mol.n_electrons() + mol.multiplicity - 1) // 2
    nbeta = (mol.n_electrons() - mol.multiplicity + 1) // 2
    nva = basis.nbasis - nalpha
    nvb = basis.nbasis - nbeta
    nao = basis.nbasis

    def one_slab_bytes(nv_bra, no_ket, nv_ket):
        return max(
            (nao**3 + nv_bra * nao**2) * 8,
            (nv_bra * nao**2 + nv_bra * no_ket * nao) * 8,
            (nv_bra * no_ket * nao + nv_bra * no_ket * nv_ket) * 8,
        )

    channel_minima = []
    if nalpha >= 2 and nva >= 2:
        channel_minima.append(one_slab_bytes(nva, nalpha, nva))
    if nbeta >= 2 and nvb >= 2:
        channel_minima.append(one_slab_bytes(nvb, nbeta, nvb))
    if nalpha and nbeta and nva and nvb:
        channel_minima.append(one_slab_bytes(nva, nbeta, nvb))
    minimum_budget = 2 * nao**2 * 8 + max(channel_minima)

    direct_opts = _all_electron_ump2_options()
    direct_opts.memory_mode = "direct"
    direct_opts.requested_memory_bytes = minimum_budget
    direct = run_ump2(mol, basis, uhf, direct_opts)
    assert direct.memory_mode_used == "direct"
    assert direct.workspace_bytes <= direct_opts.requested_memory_bytes
    assert direct.disk_bytes == 0
    assert direct.e_aa == pytest.approx(fast.e_aa, abs=1e-11)
    assert direct.e_bb == pytest.approx(fast.e_bb, abs=1e-11)
    assert direct.e_ab == pytest.approx(fast.e_ab, abs=1e-11)

    disk_opts = _all_electron_ump2_options()
    disk_opts.memory_mode = "disk"
    disk_opts.requested_memory_bytes = minimum_budget
    disk_opts.scratch_directory = str(tmp_path)
    disk = run_ump2(mol, basis, uhf, disk_opts)
    assert disk.memory_mode_used == "disk"
    assert disk.workspace_bytes <= disk_opts.requested_memory_bytes
    assert disk.disk_bytes > 0
    assert disk.e_correlation == pytest.approx(fast.e_correlation, abs=1e-11)
    assert list(tmp_path.iterdir()) == []


def test_df_ump2_direct_panels_match_incore():
    # Li doublet has populated alpha-alpha and alpha-beta channels while
    # remaining a tiny, reliably convergent atomic test.
    atoms = [Atom(3, [0.0, 0.0, 0.0])]
    mol = Molecule(atoms, charge=0, multiplicity=2)
    basis = BasisSet(mol, "cc-pvdz")
    uhf_opts = UHFOptions()
    uhf_opts.stability_check = False
    uhf = run_uhf(mol, basis, uhf_opts)
    assert uhf.converged

    fast_opts = _all_electron_ump2_options()
    fast_opts.density_fit = True
    fast_opts.aux_basis = "cc-pvdz-ri"
    fast_opts.memory_mode = "incore"
    fast = run_ump2(mol, basis, uhf, fast_opts)

    direct_opts = _all_electron_ump2_options()
    direct_opts.density_fit = True
    direct_opts.aux_basis = "cc-pvdz-ri"
    direct_opts.memory_mode = "direct"
    direct = run_ump2(mol, basis, uhf, direct_opts)
    assert direct.memory_mode_used == "direct"
    assert direct.e_aa == pytest.approx(fast.e_aa, abs=1e-11)
    assert direct.e_bb == pytest.approx(fast.e_bb, abs=1e-11)
    assert direct.e_ab == pytest.approx(fast.e_ab, abs=1e-11)
