"""Phase 18 + 19: population analysis + dipole moment.

Validated against PySCF where possible — Mulliken charges and dipole
moments are directly comparable (same sign convention, same origin
convention). Mayer bond orders don't have a single canonical PySCF
output we can round-trip, so we validate them against analytical
expectations and element-wise symmetry.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq


def _h2o() -> vq.Molecule:
    # Experimental-ish geometry (bohr). O at origin; H at ±y offset.
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])


def _h2() -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])


def _rp209_formaldehyde() -> vq.Molecule:
    angstrom_to_bohr = 1.8897259886
    return vq.Molecule(
        [
            vq.Atom(6, [0.0, 0.0, 0.0]),
            vq.Atom(8, [0.0, 0.0, 1.208 * angstrom_to_bohr]),
            vq.Atom(
                1,
                [0.0, 0.943 * angstrom_to_bohr, -0.587 * angstrom_to_bohr],
            ),
            vq.Atom(
                1,
                [0.0, -0.943 * angstrom_to_bohr, -0.587 * angstrom_to_bohr],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Mulliken + Löwdin charges
# ---------------------------------------------------------------------------

def test_mulliken_charges_sum_to_total_charge_neutral():
    """Sum of Mulliken charges = total molecular charge (0 for neutral)."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    charges = vq.mulliken_charges(res, basis, mol)
    assert charges.shape == (3,)
    assert abs(np.sum(charges)) < 1e-10


def test_mulliken_charges_sum_to_total_charge_anion():
    """For OH⁻ (closed-shell, 10 electrons) the Mulliken sum equals −1."""
    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.84]),
    ], charge=-1, multiplicity=1)
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    charges = vq.mulliken_charges(res, basis, mol)
    assert abs(np.sum(charges) - (-1.0)) < 1e-10


def test_loewdin_charges_sum_to_total_charge():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    charges = vq.loewdin_charges(res, basis, mol)
    assert charges.shape == (3,)
    assert abs(np.sum(charges)) < 1e-10


def test_loewdin_def2_svp_h2o2_uses_global_symmetric_orthogonalization():
    """Flexible bases still use the standard global Löwdin definition."""
    from vibeqc._vibeqc_core import compute_overlap

    ang2bohr = 1.0 / 0.529177210903
    mol = vq.Molecule(
        [
            vq.Atom(8, [-0.70600000 * ang2bohr, -0.40100000 * ang2bohr, 0.04800000 * ang2bohr]),
            vq.Atom(8, [0.70600000 * ang2bohr, -0.40100000 * ang2bohr, 0.04800000 * ang2bohr]),
            vq.Atom(1, [-1.18900000 * ang2bohr, 0.29000000 * ang2bohr, 0.28700000 * ang2bohr]),
            vq.Atom(1, [1.18900000 * ang2bohr, 0.29000000 * ang2bohr, 0.28700000 * ang2bohr]),
        ],
        multiplicity=1,
    )
    basis = vq.BasisSet(mol, "def2-svp")
    opts = vq.RKSOptions()
    opts.functional = "b3lyp"
    opts.max_iter = 120
    opts.conv_tol_energy = 1.0e-8
    opts.damping = 0.0
    res = vq.run_rks(mol, basis, opts)

    q = vq.loewdin_charges(res, basis, mol)

    S = np.asarray(compute_overlap(basis))
    eigvals, eigvecs = np.linalg.eigh(S)
    S_half = (eigvecs * np.sqrt(eigvals).reshape(1, -1)) @ eigvecs.T
    lowdin_density = S_half @ np.asarray(res.density) @ S_half
    ao_to_atom = []
    for shell in basis.shells():
        ao_to_atom.extend([int(shell.atom_index)] * (2 * int(shell.l) + 1))
    populations = np.bincount(
        ao_to_atom,
        weights=np.diag(lowdin_density),
        minlength=len(mol.atoms),
    )
    expected = np.array([atom.Z for atom in mol.atoms]) - populations

    np.testing.assert_allclose(q, expected, atol=1.0e-12)
    assert q[0] == pytest.approx(q[1], abs=1.0e-8)
    assert q[2] == pytest.approx(q[3], abs=1.0e-8)
    assert abs(float(np.sum(q))) < 1.0e-8


def test_rp209_formaldehyde_population_properties_match_orca_611():
    """RHF/def2-TZVP population definitions match the RP209 reference."""
    from vibeqc.bond_analysis import wiberg_bond_orders

    mol = _rp209_formaldehyde()
    basis = vq.BasisSet(mol, "def2-tzvp")
    result = vq.run_rhf(mol, basis)

    # Out-of-process ORCA 6.1.1 RP209 reference, generated with RHF,
    # def2-TZVP, TightSCF, and the same geometry.
    assert result.energy == pytest.approx(-113.915782285454, abs=3.0e-8)
    np.testing.assert_allclose(
        vq.mulliken_charges(result, basis, mol),
        [0.184294, -0.381624, 0.098665, 0.098665],
        atol=5.0e-6,
    )
    loewdin = vq.loewdin_charges(result, basis, mol)
    np.testing.assert_allclose(
        loewdin,
        [-0.219249, 0.038056, 0.090597, 0.090597],
        atol=2.0e-6,
    )
    assert not np.array_equal(
        loewdin,
        vq.mulliken_charges(result, basis, mol),
    )

    mayer = vq.mayer_bond_orders(result, basis, mol)
    assert mayer[0, 1] == pytest.approx(1.9938, abs=5.0e-5)
    assert mayer[0, 2] == pytest.approx(0.9510, abs=5.0e-5)
    assert mayer[0, 3] == pytest.approx(0.9510, abs=5.0e-5)

    # Wiberg is a separate Löwdin-basis squared-density metric, not a
    # condition-number fallback or alternate label for Mayer.
    wiberg = wiberg_bond_orders(result, basis, mol)
    assert wiberg[0, 1] == pytest.approx(2.8560697759, abs=1.0e-9)
    assert wiberg[0, 2] == pytest.approx(0.9474497815, abs=1.0e-9)


def test_mulliken_symmetric_h2o():
    """The two H atoms in H₂O should have equal Mulliken charges by
    symmetry (within 1e-8)."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    q = vq.mulliken_charges(res, basis, mol)
    assert abs(q[1] - q[2]) < 1e-8


def test_mulliken_matches_pyscf_h2o():
    """Cross-check vibe-qc's Mulliken against PySCF's on H₂O/STO-3G."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol_py = gto.M(
        atom=[["O", (0.0, 0.0, 0.0)],
              ["H", (0.0, 1.43, -0.98)],
              ["H", (0.0, -1.43, -0.98)]],
        unit="Bohr", basis="sto3g", verbose=0,
    )
    mf = scf.RHF(mol_py).run(conv_tol=1e-10)
    pyscf_pop, pyscf_charges = mf.mulliken_pop(verbose=0)

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    vibeqc_q = vq.mulliken_charges(res, basis, mol)

    for i in range(3):
        assert abs(vibeqc_q[i] - pyscf_charges[i]) < 1e-6, (
            f"atom {i}: vibeqc = {vibeqc_q[i]:.6f}, "
            f"pyscf = {pyscf_charges[i]:.6f}"
        )


# ---------------------------------------------------------------------------
# Mayer bond orders
# ---------------------------------------------------------------------------

def test_mayer_bond_orders_matrix_symmetric():
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    B = vq.mayer_bond_orders(res, basis, mol)
    assert B.shape == (3, 3)
    assert np.allclose(B, B.T, atol=1e-12)
    # Diagonal entries hold per-atom "free valence" — all zero in our
    # implementation (we set off-diagonal bonds and leave the self term
    # zero).
    assert np.allclose(np.diag(B), 0.0)


def test_mayer_h2_bond_order_near_one():
    """Plain H–H single bond in H₂ at equilibrium: Mayer ≈ 1.0.

    Tolerance is generous because small basis sets (STO-3G) don't give
    exactly 1; 6-31G* is closer."""
    mol = _h2()
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    B = vq.mayer_bond_orders(res, basis, mol)
    assert 0.9 < B[0, 1] < 1.1, f"H-H Mayer bond order = {B[0, 1]:.4f}"


def test_prominent_bonds_degenerate_pairs_order_by_atom_index():
    """Row order of noise-degenerate bonds must be machine-stable.

    Symmetry-equivalent bonds (the two O-H bonds of C2v water) carry the
    same bond order up to floating-point noise. A raw descending float sort
    leaks that noise into the display order the .out snapshot goldens
    freeze; ties at (sub-)display precision must break by atom indices."""
    mol = _h2o()
    B = np.zeros((3, 3))
    # (0, 1) infinitesimally below (0, 2): a raw float sort puts (0, 2)
    # first, the quantized tie-break must restore index order.
    B[0, 1] = B[1, 0] = 0.9612345 - 1e-12
    B[0, 2] = B[2, 0] = 0.9612345
    bonds = vq.prominent_bonds(B, mol)
    assert [(i, j) for i, j, _ in bonds] == [(0, 1), (0, 2)]
    # A genuinely larger bond order still floats to the top, regardless of
    # index order.
    B[0, 1] = B[1, 0] = 0.5
    bonds = vq.prominent_bonds(B, mol)
    assert [(i, j) for i, j, _ in bonds] == [(0, 2), (0, 1)]


def test_mayer_h2o_oh_bonds_are_equal_and_hh_small():
    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    B = vq.mayer_bond_orders(res, basis, mol)
    # O-H bond order symmetric between the two hydrogens
    assert abs(B[0, 1] - B[0, 2]) < 1e-6
    # H-H "bond order" should be near zero (the Hs are not bonded)
    assert abs(B[1, 2]) < 0.05
    # O-H bond order is around 0.75 for H2O / 6-31G*
    assert 0.6 < B[0, 1] < 0.85


def test_mayer_ill_conditioned_overlap_keeps_mayer_definition(monkeypatch):
    """A condition-number heuristic must not substitute a Wiberg index."""
    S = np.array(
        [
            [1.0, 0.47665592, -0.85927033, 0.85261941, 0.79932703, 0.9661772],
            [0.47665592, 1.0, -0.04035792, 0.82356327, -0.09613658, 0.54474595],
            [-0.85927033, -0.04035792, 1.0, -0.57981205, -0.97765344, -0.77276365],
            [0.85261941, 0.82356327, -0.57981205, 1.0, 0.46951008, 0.82876683],
            [0.79932703, -0.09613658, -0.97765344, 0.46951008, 1.0, 0.68776102],
            [0.9661772, 0.54474595, -0.77276365, 0.82876683, 0.68776102, 1.0],
        ]
    )
    P = np.array(
        [
            [20.72355392, 16.24888052, -13.9135067, -25.21251264, 0.63875773, -17.71060337],
            [16.24888052, 18.28355159, -13.85513317, -21.18229312, 2.64024255, -17.10842729],
            [-13.9135067, -13.85513317, 10.90688403, 17.67863865, -1.56581989, 13.60291921],
            [-25.21251264, -21.18229312, 17.67863865, 31.03438379, -1.32275072, 22.36863946],
            [0.63875773, 2.64024255, -1.56581989, -1.32275072, 0.84540111, -1.78940911],
            [-17.71060337, -17.10842729, 13.60291921, 22.36863946, -1.78940911, 17.00843148],
        ]
    )

    class _Basis:
        def shells(self):
            return [
                SimpleNamespace(l=0, atom_index=0),
                SimpleNamespace(l=0, atom_index=0),
                SimpleNamespace(l=0, atom_index=0),
                SimpleNamespace(l=0, atom_index=1),
                SimpleNamespace(l=0, atom_index=1),
                SimpleNamespace(l=0, atom_index=1),
            ]

    molecule = SimpleNamespace(atoms=[SimpleNamespace(Z=1), SimpleNamespace(Z=1)])
    result = SimpleNamespace(density=P)
    monkeypatch.setattr("vibeqc.properties.compute_overlap", lambda _basis: S)

    B = vq.mayer_bond_orders(result, _Basis(), molecule)

    PS = P @ S
    expected = np.sum(PS[:3, 3:] * PS[3:, :3].T)
    assert B[0, 1] == pytest.approx(B[1, 0], abs=1.0e-12)
    assert B[0, 1] == pytest.approx(expected, abs=1.0e-12)


# ---------------------------------------------------------------------------
# Dipole moment
# ---------------------------------------------------------------------------

def test_dipole_h2_zero_for_symmetric_diatomic():
    """H₂ is non-polar — dipole components within roundoff of zero."""
    mol = _h2()
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    dip = vq.dipole_moment(res, basis, mol)
    assert abs(dip.total) < 1e-8


def test_dipole_h2o_magnitude_reasonable():
    """H₂O dipole with HF/6-31G* should be ≈ 2.1 Debye (experimental
    1.85; HF/6-31G* slightly overshoots)."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    dip = vq.dipole_moment(res, basis, mol)
    assert 1.8 < dip.total_debye < 2.5


def test_dipole_origin_independence_neutral_molecule():
    """For a neutral molecule the dipole is origin-independent."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    dip_com = vq.dipole_moment(res, basis, mol)
    dip_origin = vq.dipole_moment(res, basis, mol, origin=(0.0, 0.0, 0.0))
    dip_far = vq.dipole_moment(res, basis, mol, origin=(10.0, -3.0, 2.0))
    # Magnitudes match; directions match (the Debye ratio is ~1).
    assert abs(dip_com.total - dip_origin.total) < 1e-10
    assert abs(dip_com.total - dip_far.total) < 1e-10


def test_dipole_points_along_expected_axis_h2o():
    """H₂O with Hs at ±y and in −z direction: dipole should lie along z
    and be negative (O partial charge pulls electrons toward +z relative
    to the H-plane)."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    dip = vq.dipole_moment(res, basis, mol)
    # x/y components are zero by symmetry
    assert abs(dip.x) < 1e-8
    assert abs(dip.y) < 1e-8
    # z is dominant
    assert abs(dip.z) > 0.5


def test_dipole_uhf_result_accepted():
    """Dipole computation works on UHF results (uses density_alpha +
    density_beta)."""
    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.832]),
    ], charge=0, multiplicity=2)
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_uhf(mol, basis)
    dip = vq.dipole_moment(res, basis, mol)
    assert np.isfinite(dip.total)


def test_dipole_matches_pyscf_h2o():
    """Quantitative cross-check against PySCF."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol_py = gto.M(
        atom=[["O", (0.0, 0.0, 0.0)],
              ["H", (0.0, 1.43, -0.98)],
              ["H", (0.0, -1.43, -0.98)]],
        unit="Bohr", basis="631g*", verbose=0,
    )
    mf = scf.RHF(mol_py).run(conv_tol=1e-10)
    # PySCF's dipole_moment returns Debye by default.
    pyscf_dip = mf.dip_moment(verbose=0)

    mol = _h2o()
    basis = vq.BasisSet(mol, "6-31g*")
    res = vq.run_rhf(mol, basis)
    dip = vq.dipole_moment(res, basis, mol)

    # Convert PySCF output to e·bohr — PySCF's dip_moment returns Debye.
    pyscf_debye = np.asarray(pyscf_dip)
    vibeqc_debye = np.array(dip.components_debye())
    # Tolerance: both codes solve the same equations; remaining
    # difference comes from SCF convergence tolerance. 1e-5 is comfortable.
    assert np.max(np.abs(vibeqc_debye - pyscf_debye)) < 1e-5, (
        f"vibeqc = {vibeqc_debye}, pyscf = {pyscf_debye}"
    )


# ---------------------------------------------------------------------------
# run_job integration
# ---------------------------------------------------------------------------

def test_run_job_output_contains_properties_block():
    mol = _h2o()
    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        vq.run_job(mol, basis="6-31g*", method="rhf", output=out_stem,
                   write_molden_file=False)
        text = (out_stem.with_suffix(".out")).read_text()
    assert "Atomic charges" in text
    assert "Mulliken" in text
    assert "Löwdin" in text
    # Hirshfeld column added when the grid build + SAD promolecule
    # succeed (always the case for 6-31G* H₂O).
    assert "Hirshfeld" in text
    assert "Bond orders (Mayer)" in text
    assert "Dipole moment" in text
    assert "Debye" in text


def test_run_job_uhf_properties_block():
    """UHF jobs also get a properties block, using the α+β total density."""
    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.832]),
    ], charge=0, multiplicity=2)
    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        vq.run_job(mol, basis="6-31g*", method="uhf", output=out_stem,
                   write_molden_file=False)
        text = (out_stem.with_suffix(".out")).read_text()
    assert "Atomic charges" in text
    assert "Dipole moment" in text


# ---------------------------------------------------------------------------
# Center of mass
# ---------------------------------------------------------------------------

def test_center_of_mass_h2o_on_z_axis():
    mol = _h2o()
    com = vq.center_of_mass(mol)
    # By symmetry the COM is on the z-axis.
    assert abs(com[0]) < 1e-10
    assert abs(com[1]) < 1e-10
    # COM is displaced from O by roughly (H_mass × H_z × 2) / total_mass.
    assert com[2] != 0.0


# ---------------------------------------------------------------------------
# Complex (periodic, Hermitian) densities — real-observable handling
#
# Periodic SCF results carry a complex-typed, Hermitian density with a
# phase-dependent imaginary part. The property code forms real observables
# (tr(P·O), per-atom (P·S), density on a real grid), so it reduces the
# density to its real part via properties._real_if_hermitian instead of
# letting a silent float() cast discard the imaginary part (which spammed
# ComplexWarning on every periodic run — properties.py:120/473/573 et al.).
# ---------------------------------------------------------------------------
class _FakeResult:
    """Minimal duck-typed RHF result carrying an arbitrary density."""

    def __init__(self, density):
        self.density = density


def _hermitian_complex(P_real):
    """A complex-Hermitian matrix = real symmetric P + i·(tiny antisymmetric).
    (i·B)^H = −i·B^T = i·B for real antisymmetric B, so this stays Hermitian
    while carrying a machine-noise imaginary part like a periodic density."""
    n = P_real.shape[0]
    B = np.triu(np.ones((n, n)), 1)
    B = B - B.T  # real antisymmetric
    return P_real.astype(np.complex128) + 1e-15j * B


def test_real_if_hermitian_real_passthrough():
    from vibeqc.properties import _real_if_hermitian

    P = np.array([[1.0, 0.3], [0.3, 1.0]])
    out = _real_if_hermitian(P)
    assert not np.iscomplexobj(out)
    np.testing.assert_array_equal(out, P)


def test_real_if_hermitian_complex_hermitian_noise_is_silent():
    import warnings

    from vibeqc.properties import _real_if_hermitian

    P = _hermitian_complex(np.array([[1.0, 0.3], [0.3, 1.0]]))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        out = _real_if_hermitian(P)
    assert not np.iscomplexobj(out)
    assert not any("imaginary" in str(w.message) for w in wl)


def test_complex_hermitian_density_with_large_imaginary_part_is_quiet():
    """Bloch densities may be exactly Hermitian while carrying sizable
    imaginary off-diagonal entries; only a Hermiticity residual is a bug."""
    import warnings

    from vibeqc.properties import _real_if_hermitian

    P = np.array([[1.0 + 0.0j, 0.2 + 0.5j],
                  [0.2 - 0.5j, 1.0 + 0.0j]])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        out = _real_if_hermitian(P)
    assert not np.iscomplexobj(out)
    assert not wl
    np.testing.assert_allclose(out, P.real, atol=0.0)


def test_real_if_hermitian_non_hermitian_warns():
    import warnings

    from vibeqc.properties import _real_if_hermitian

    # Asymmetric imaginary part → genuinely non-Hermitian (a density bug).
    P = np.array([[1.0 + 0j, 0.2 + 0.5j], [0.2 + 0.0j, 1.0 + 0j]])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        out = _real_if_hermitian(P)
    assert not np.iscomplexobj(out)
    assert any("non-Hermitian" in str(w.message) for w in wl)


def _hermitian_k_stack(nk, n=6, seed=7):
    """``(nk, n, n)`` stack of complex Hermitian blocks with O(1) imaginary
    off-diagonals -- a per-k periodic density (``List[np.ndarray]``) after
    ``np.asarray``."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nk, n, n)) + 1j * rng.normal(size=(nk, n, n))
    return A + A.conj().swapaxes(-1, -2)


@pytest.mark.parametrize("nk", [1, 2, 6])
def test_real_if_hermitian_per_k_stack_is_quiet(nk):
    """Each k-block is Hermitian, so the check must adjoint the AO axes only.
    ``P.conj().T`` has one branch per case against these n = 6 blocks:
    nk = 1 broadcast (1, n, n) against (n, n, 1) and warned on an exactly
    Hermitian density; nk = 2 (nk != n) raised ValueError; nk = 6 (nk == n)
    broadcast, compared unrelated entries and warned."""
    import warnings

    from vibeqc.properties import _real_if_hermitian

    P = _hermitian_k_stack(nk)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        out = _real_if_hermitian(list(P))
    assert not wl
    assert out.shape == (nk, 6, 6)
    np.testing.assert_array_equal(out, P.real)


def test_real_if_hermitian_per_k_stack_warns_on_one_bad_block():
    import warnings

    from vibeqc.properties import _real_if_hermitian

    P = _hermitian_k_stack(2)
    P[1, 0, 1] += 0.5j  # breaks Hermiticity in the second k-block only
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        _real_if_hermitian(list(P))
    assert any("non-Hermitian" in str(w.message) for w in wl)


def test_periodic_complex_density_properties_no_warning_and_match_real():
    """dipole / Mulliken / Mayer on a complex-but-Hermitian density emit no
    ComplexWarning and equal the real-density values (regression guard for
    the periodic-property ComplexWarning spam)."""
    import warnings

    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    r_real = vq.run_rhf(mol, basis)
    P_real = np.asarray(r_real.density)
    r_complex = _FakeResult(_hermitian_complex(P_real))

    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        d_c = vq.dipole_moment(r_complex, basis, mol)
        q_c = vq.mulliken_charges(r_complex, basis, mol)
        b_c = vq.mayer_bond_orders(r_complex, basis, mol)
    assert not any(
        "imaginary" in str(w.message)
        or "ComplexWarning" in type(w.message).__name__
        for w in wl
    ), "periodic complex-Hermitian density must not emit ComplexWarning"

    # Real part recovers the real-density observables exactly.
    assert d_c.total == pytest.approx(vq.dipole_moment(r_real, basis, mol).total, abs=1e-12)
    np.testing.assert_allclose(q_c, vq.mulliken_charges(r_real, basis, mol), atol=1e-12)
    np.testing.assert_allclose(b_c, vq.mayer_bond_orders(r_real, basis, mol), atol=1e-12)


# ---------------------------------------------------------------------------
# Public property path, end to end
#
# Regression guard for COMPUTE-DIPOLE-SIGNATURE-MISMATCH (rp213 mega-matrix
# "D-properties" sub-wave, 2026-08-14): 20/20 cases died on
#
#     dip = vq.compute_dipole(molecule, basis_obj, scf.density)
#     TypeError: compute_dipole(): incompatible function arguments.
#
# That three-argument form has never existed. ``compute_dipole`` is the
# low-level *integral* builder -- ``(basis, origin) -> DipoleIntegrals``,
# bound that way in d60a910c0 and unchanged since. The dipole *moment* is
# ``vq.dipole_moment(result, basis, molecule) -> DipoleMoment``. The tests
# below make the three things the wave carrier got wrong mechanical.
# ---------------------------------------------------------------------------

_MU_ROW = re.compile(r"\|mu\|\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def test_compute_dipole_binding_takes_basis_and_origin():
    """``compute_dipole`` is the integral builder: ``(basis[, origin])``."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")

    default_origin = vq.compute_dipole(basis)
    explicit_origin = vq.compute_dipole(basis, [0.0, 0.0, 0.0])

    for dip in (default_origin, explicit_origin):
        for component in (dip.x, dip.y, dip.z):
            mat = np.asarray(component)
            assert mat.shape == (basis.nbasis, basis.nbasis)
            # Dipole integrals are symmetric in the AO indices.
            np.testing.assert_allclose(mat, mat.T, atol=1e-12)

    np.testing.assert_allclose(np.asarray(default_origin.z),
                               np.asarray(explicit_origin.z), atol=1e-14)


def test_compute_dipole_rejects_the_rp213_three_argument_call():
    """The drifted ``(molecule, basis, density)`` form must stay a hard
    ``TypeError`` rather than silently acquiring a meaning."""
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis)

    with pytest.raises(TypeError, match="incompatible function arguments"):
        vq.compute_dipole(mol, basis, result.density)


def test_run_job_dipole_row_matches_dipole_moment_api():
    """The public runner already writes the dipole to the ``.out``; a
    properties payload never has to recompute it to get it into an
    artifact, and when it does the two must agree."""
    mol = _h2o()
    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        vq.run_job(mol, basis="sto-3g", method="rhf", output=out_stem,
                   write_molden_file=False)
        text = (out_stem.with_suffix(".out")).read_text()

    assert "Dipole moment (origin: center of mass)" in text
    match = _MU_ROW.search(text)
    assert match is not None, f"no |mu| row in .out:\n{text}"

    basis = vq.BasisSet(mol, "sto-3g")
    mu = vq.dipole_moment(vq.run_rhf(mol, basis), basis, mol)
    # The .out row carries 8 and 4 decimals respectively.
    assert float(match.group(1)) == pytest.approx(mu.total, abs=1e-7)
    assert float(match.group(2)) == pytest.approx(mu.total_debye, abs=1e-3)


def test_property_sidecar_dipole_and_polarizability():
    """The supported form of the rp213 carrier block: ``run_job`` for the
    standard artifact set, then ``dipole_moment`` + CPHF
    ``dipole_polarizability_rhf`` for a properties sidecar.

    ``run_job`` has no polarizability option, so a sidecar is the route
    for it -- but the dipole comes from ``dipole_moment``, never from
    ``compute_dipole``.
    """
    mol = _h2o()
    options = vq.RHFOptions()
    options.conv_tol_energy = 1.0e-8

    with tempfile.TemporaryDirectory() as d:
        out_stem = Path(d) / "run"
        vq.run_job(mol, basis="sto-3g", method="rhf", rhf_options=options,
                   output=out_stem, write_molden_file=False)
        out_text = (out_stem.with_suffix(".out")).read_text()

        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis=basis, options=options)
        mu = vq.dipole_moment(result, basis, mol)
        alpha = np.asarray(
            vq.dipole_polarizability_rhf(result, basis, mol), dtype=float)

        assert alpha.shape == (3, 3)
        assert np.all(np.isfinite(alpha))
        np.testing.assert_allclose(alpha, alpha.T, atol=1e-8)
        iso = float(np.trace(alpha) / 3.0)
        # H₂O/STO-3G CPHF: small but strictly positive isotropic response.
        assert 0.0 < iso < 20.0

        sidecar = Path(str(out_stem) + ".properties.json")
        sidecar.write_text(json.dumps({
            "polarizability_au": alpha.tolist(),
            "polarizability_isotropic_au": iso,
            "dipole_au": [mu.x, mu.y, mu.z],
            "dipole_total_au": mu.total,
            "dipole_total_debye": mu.total_debye,
            "dipole_origin_bohr": list(mu.origin),
        }, indent=1))
        payload = json.loads(sidecar.read_text())

    # Both artifacts exist and agree on the dipole.
    assert payload["dipole_total_au"] == pytest.approx(mu.total, abs=1e-10)
    match = _MU_ROW.search(out_text)
    assert match is not None
    assert float(match.group(1)) == pytest.approx(
        payload["dipole_total_au"], abs=1e-7)



# ---------------------------------------------------------------------------
# GitLab #642: nuclear_charges keyword
# ---------------------------------------------------------------------------


def test_nuclear_charges_keyword_defaults_to_bare_z_and_validates_shape():
    """``nuclear_charges=None`` is bare Z (byte-identical to the pre-#642
    behaviour for all-electron runs); an explicit bare-Z vector reproduces
    it; a wrong-length vector is refused rather than silently broadcast."""
    import numpy as np
    import pytest
    import vibeqc as vq
    from vibeqc.properties import dipole_moment, loewdin_charges, mulliken_charges

    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 1.43, 1.108]), vq.Atom(1, [0.0, -1.43, 1.108])],
        0, 1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis, vq.RHFOptions())
    bare = np.array([8.0, 1.0, 1.0])
    d_default = dipole_moment(res, basis, mol)
    d_bare = dipole_moment(res, basis, mol, nuclear_charges=bare)
    assert (d_default.x, d_default.y, d_default.z) == (d_bare.x, d_bare.y, d_bare.z)
    np.testing.assert_array_equal(
        mulliken_charges(res, basis, mol), mulliken_charges(res, basis, mol, nuclear_charges=bare)
    )
    np.testing.assert_array_equal(
        loewdin_charges(res, basis, mol), loewdin_charges(res, basis, mol, nuclear_charges=bare)
    )
    with pytest.raises(ValueError, match="one entry per atom"):
        dipole_moment(res, basis, mol, nuclear_charges=[8.0, 1.0])
    with pytest.raises(ValueError, match="one entry per atom"):
        mulliken_charges(res, basis, mol, nuclear_charges=[8.0])


# ---------------------------------------------------------------------------
# Canonical AO-to-atom map (GitLab #205 follow-up)
# ---------------------------------------------------------------------------


def _oh_cc_pvdz_pure_and_cartesian():
    """cc-pVDZ on OH, as bundled (pure) and rebuilt Cartesian.

    The d shell is the only one whose count differs: 5 pure spherical
    harmonics vs 6 Cartesian products, so nbasis goes 19 -> 20. Every
    other shell is s or p, where the two conventions agree.
    """
    mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])], 0, 2)
    pure = vq.BasisSet(mol, "cc-pVDZ")
    return mol, pure, _as_cartesian(mol, pure)


def _as_cartesian(molecule, pure: "vq.BasisSet") -> "vq.BasisSet":
    """Rebuild ``pure`` shell for shell with the Cartesian convention."""
    return vq.BasisSet(
        molecule,
        [
            vq.ShellInfo(
                shell.atom_index,
                shell.l,
                False,
                shell.exponents,
                shell.coefficients,
                shell.origin,
            )
            for shell in pure.shells()
        ],
        "cc-pVDZ-cartesian",
        True,
    )


def _ao_to_atom_helpers():
    """Every basis-only AO-to-atom map in the tree.

    All of them now delegate to ``properties._shell_to_atom``; this list
    is what keeps a future copy from drifting back out of sync. Anything
    that maps AOs to atoms from ``basis.shells()`` alone belongs here.
    ``localization._ao_metadata`` derives the same map but also needs a
    periodic system and translations, so it is covered separately by
    :func:`test_ao_metadata_atom_map_is_the_canonical_one`.
    """
    from vibeqc.bands import _shell_to_atom as bands_map
    from vibeqc.bond_analysis import _shell_to_atom as bond_map
    from vibeqc.nbo import _shell_to_atom as nbo_map
    from vibeqc.output.formats.population import _basis_ao_to_atom as population_map
    from vibeqc.periodic.chi.localization import (
        _reference_atom_indices as chi_reference_map,
    )
    from vibeqc.periodic.chi.properties import _ao_atom_indices as chi_properties_map
    from vibeqc.properties import _shell_to_atom as properties_map

    return {
        "properties": properties_map,
        "bands": bands_map,
        "bond_analysis": bond_map,
        "nbo": nbo_map,
        "output.formats.population": population_map,
        "periodic.chi.localization": chi_reference_map,
        "periodic.chi.properties": chi_properties_map,
    }


def test_ao_to_atom_map_length_matches_nbasis_pure_and_cartesian():
    """Every AO-to-atom map spans the full basis in both conventions.

    A map derived as ``2l+1`` per shell is silently one entry short for
    each d shell of a Cartesian basis, which misaligns (rather than
    raises on) the per-atom sums built from it. Asserting the length
    against ``nbasis`` catches exactly that.
    """
    _mol, pure, cartesian = _oh_cc_pvdz_pure_and_cartesian()

    # Guard the fixture itself: if these ever coincide the test is vacuous.
    assert pure.nbasis == 19
    assert cartesian.nbasis == 20
    assert any(int(shell.l) == 2 for shell in pure.shells())

    for name, helper in _ao_to_atom_helpers().items():
        for label, basis in (("pure", pure), ("cartesian", cartesian)):
            ao_to_atom = helper(basis)
            assert len(ao_to_atom) == basis.nbasis, (
                f"{name} AO-to-atom map has {len(ao_to_atom)} entries for the "
                f"{label} basis, expected nbasis={basis.nbasis}"
            )
            # Both atoms must actually own AOs, and no index may dangle.
            assert set(int(a) for a in ao_to_atom) == {0, 1}


def test_ao_to_atom_maps_agree_across_modules():
    """The four helpers are one derivation, not four that happen to agree."""
    _mol, pure, cartesian = _oh_cc_pvdz_pure_and_cartesian()
    helpers = _ao_to_atom_helpers()
    reference = helpers["properties"]

    for basis in (pure, cartesian):
        expected = np.asarray(reference(basis), dtype=np.int64)
        for name, helper in helpers.items():
            np.testing.assert_array_equal(
                np.asarray(helper(basis), dtype=np.int64),
                expected,
                err_msg=f"{name} disagrees with properties._shell_to_atom",
            )


def test_nbo_overlap_fallback_is_sized_by_the_canonical_map(monkeypatch):
    """The ImportError fallback in ``nbo.compute_overlap_fallback`` returns an
    identity of ``nbasis`` on a Cartesian basis too (#223).

    It used to size that identity as ``sum(2l+1)``, the pure count, so the
    fallback overlap was one row short per Cartesian d shell. Making the
    native import fail is the only way to reach the fallback, so the core
    module is blanked in ``sys.modules`` for the duration of the call.
    """
    import sys

    from vibeqc.nbo import compute_overlap_fallback

    _mol, pure, cartesian = _oh_cc_pvdz_pure_and_cartesian()
    assert cartesian.nbasis > pure.nbasis  # the d shells are Cartesian
    monkeypatch.setitem(sys.modules, "vibeqc._vibeqc_core", None)
    for basis in (pure, cartesian):
        fallback = np.asarray(compute_overlap_fallback(basis))
        assert fallback.shape == (basis.nbasis, basis.nbasis)
        np.testing.assert_array_equal(fallback, np.eye(basis.nbasis))


def test_bond_analysis_shell_to_atom_returns_plain_int_list():
    """bond_analysis indexes the map in a scalar loop; keep it a list."""
    from vibeqc.bond_analysis import _shell_to_atom

    _mol, pure, _cartesian = _oh_cc_pvdz_pure_and_cartesian()
    ao_to_atom = _shell_to_atom(pure)
    assert isinstance(ao_to_atom, list)
    assert all(type(a) is int for a in ao_to_atom)


def test_wiberg_bond_orders_on_cartesian_basis():
    """End-to-end: a Cartesian SCF must survive per-atom partitioning.

    This is the failure the consolidation fixes -- bond_analysis used to
    build a 19-entry map for a 20-function basis and walk off the end of
    it. A custom-named basis has no SAD data, hence the HCORE guess.
    """
    from vibeqc.bond_analysis import wiberg_bond_orders
    from vibeqc.properties import mulliken_charges

    mol, _pure, cartesian = _oh_cc_pvdz_pure_and_cartesian()
    options = vq.UHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    result = vq.run_uhf(mol, cartesian, options)

    bond_orders = wiberg_bond_orders(result, cartesian, mol)
    assert bond_orders.shape == (2, 2)
    np.testing.assert_allclose(bond_orders, bond_orders.T, atol=1e-10)
    # O-H is a single bond; allow a wide band since this is a shape check.
    assert 0.3 < bond_orders[0, 1] < 1.6

    # Sharper than any tolerance band: the partial charges of a neutral
    # molecule sum to zero only if every AO was assigned to some atom.
    # A map one entry short silently drops that AO's population instead.
    charges = mulliken_charges(result, cartesian, mol)
    assert charges.shape == (2,)
    np.testing.assert_allclose(charges.sum(), 0.0, atol=1e-10)


def test_ao_metadata_atom_map_is_the_canonical_one():
    """``localization._ao_metadata`` partitions AOs the same way.

    It is the one AO-to-atom derivation that cannot join
    :func:`_ao_to_atom_helpers`, since it also takes a lattice and the
    translation list. It lays shell origins into arrays sized at
    ``nbasis``, so its own ``2l+1`` count left a tail of rows unwritten
    on a Cartesian basis: those AOs read back as sitting at the cell
    origin on atom 0, which is where the first atom happens to be.
    """
    from vibeqc.periodic.chi.localization import _ao_metadata
    from vibeqc.properties import _shell_to_atom

    lattice = np.diag([9.0, 9.5, 10.0])
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.8])],
    )
    molecule = system.unit_cell_molecule()
    pure = vq.BasisSet(molecule, "cc-pVDZ")
    cartesian = _as_cartesian(molecule, pure)
    assert pure.nbasis == 19
    assert cartesian.nbasis == 20

    # Fractional position of each atom, to index with the atom map.
    atom_fractional = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8 / 10.0]])

    for label, basis in (("pure", pure), ("cartesian", cartesian)):
        fractional, atoms = _ao_metadata(
            basis, system, np.asarray([[0, 0, 0]], dtype=int)
        )
        expected = _shell_to_atom(basis)
        np.testing.assert_array_equal(
            atoms,
            expected,
            err_msg=f"{label} _ao_metadata disagrees with properties._shell_to_atom",
        )
        # Every origin row must be written through to the last AO, not
        # left at the zeros the array was allocated with.
        assert fractional.shape == (basis.nbasis, 3), label
        np.testing.assert_allclose(
            fractional, atom_fractional[expected], atol=1.0e-12
        )
