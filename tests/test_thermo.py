"""Phase 17a-3 — molecular thermochemistry post-processor.

Contracts pinned here:

1. **API surface** — :func:`vibeqc.compute_thermochemistry`,
   :class:`vibeqc.ThermoOptions`, :class:`vibeqc.ThermoResult`
   are public.

2. **PySCF parity on H₂O / STO-3G** — every per-component thermal
   contribution (ZPE, E_trans/E_rot/E_vib, S_trans/S_rot/S_vib/S_elec,
   total entropy, U/H/G corrections) matches
   :func:`pyscf.hessian.thermo.thermo` to within FD-frequency
   precision.

3. **Linear vs nonlinear rotor handling** — H₂ (linear) gets
   ``rotor_type='linear'`` and 2-DOF rotational energy ``= kT``;
   H₂O (nonlinear) gets ``rotor_type='nonlinear'`` and 3-DOF
   rotational energy ``= 3/2 kT``.

4. **Atomic gas** — a single atom has rotor_type='atom', no
   vibration, no rotation, only translation + electronic.

5. **Symmetry number affects rotational entropy by R·ln(σ)** —
   doubling σ changes S_rot by exactly k_B · ln(σ_new/σ_old).

6. **Imaginary modes are excluded with a count** — a non-stationary
   geometry produces ``n_imaginary_modes_excluded > 0`` and the
   thermo numbers are computed only over the real modes.

7. **Validation** of inputs — temperature/pressure/sigma/
   electronic_degeneracy must be positive.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------

def _h2o_mol():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)


def _h2_mol(R_bohr: float = 1.346):
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, R_bohr])],
        charge=0, multiplicity=1,
    )


def _he_atom():
    """Single He atom — atomic gas case."""
    return vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])],
                        charge=0, multiplicity=1)


def _h2o_hessian():
    mol = _h2o_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    return mol, vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                       scf_options=opts)


def _h2_hessian():
    mol = _h2_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    return mol, vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                       scf_options=opts)


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_thermo_api_exposed():
    assert hasattr(vq, "compute_thermochemistry")
    assert hasattr(vq, "ThermoOptions")
    assert hasattr(vq, "ThermoResult")


def test_thermo_default_options():
    o = vq.ThermoOptions()
    assert o.temperature == 298.15
    assert o.pressure == 101325.0
    assert o.symmetry_number == 1
    assert o.electronic_degeneracy is None


# ---------------------------------------------------------------------------
# 2. PySCF parity on H₂O / STO-3G
# ---------------------------------------------------------------------------

def _pyscf_thermo_h2o(temperature=298.15, pressure=101325.0):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.hessian import thermo

    mol = gto.Mole()
    mol.atom = [
        ["O", (0.0, 0.0, 0.0)],
        ["H", (0.0,  0.7572, 0.5868)],
        ["H", (0.0, -0.7572, 0.5868)],
    ]
    mol.basis = "sto-3g"
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    h = mf.Hessian().kernel()
    freq_info = thermo.harmonic_analysis(mol, h)
    return thermo.thermo(mf, freq_info["freq_au"],
                         temperature=temperature, pressure=pressure)


def test_h2o_thermo_matches_pyscf_298():
    """Per-component agreement to within FD precision on the
    standard 298.15 K / 1 atm condition."""
    mol, hess = _h2o_hessian()
    t = vq.compute_thermochemistry(
        mol, hess,
        options=vq.ThermoOptions(temperature=298.15, pressure=101325.0,
                                  symmetry_number=2),
    )
    ps = _pyscf_thermo_h2o(298.15, 101325.0)

    # Energies (Hartree). FD-vs-analytic Hessian → ~1e-7 ZPE drift.
    # E_trans and E_rot are pure k_B·T expressions; tolerance is set by
    # the relative precision of the Boltzmann constant in the two
    # codes (PySCF uses CODATA 2014; we use CODATA 2018).
    assert t.zpe == pytest.approx(ps["ZPE"][0], abs=2e-7)
    assert t.e_trans == pytest.approx(ps["E_trans"][0], rel=1e-6)
    assert t.e_rot == pytest.approx(ps["E_rot"][0], rel=1e-6)
    # PySCF's "E_vib" includes ZPE; ours separates ZPE and thermal vib.
    pyscf_thermal_vib = ps["E_vib"][0] - ps["ZPE"][0]
    assert t.e_vib == pytest.approx(pyscf_thermal_vib, abs=1e-9)

    # Entropies (Hartree/K) — perfectly defined by partition functions
    assert t.s_trans == pytest.approx(ps["S_trans"][0], abs=1e-9)
    assert t.s_rot == pytest.approx(ps["S_rot"][0], abs=1e-9)
    assert t.s_vib == pytest.approx(ps["S_vib"][0], abs=1e-10)
    assert t.s_elec == pytest.approx(ps["S_elec"][0], abs=1e-12)
    assert t.s_total == pytest.approx(ps["S_tot"][0], abs=1e-9)

    # Cumulative thermal corrections.
    pyscf_g_thermal = ps["G_tot"][0] - ps["E0"][0]
    pyscf_h_thermal = ps["H_tot"][0] - ps["E0"][0]
    pyscf_u_thermal = ps["E_tot"][0] - ps["E0"][0]
    assert t.u_thermal == pytest.approx(pyscf_u_thermal, abs=2e-7)
    assert t.h_thermal == pytest.approx(pyscf_h_thermal, abs=2e-7)
    assert t.g_thermal == pytest.approx(pyscf_g_thermal, abs=2e-7)


def test_h2o_thermo_matches_pyscf_500K():
    """Cross-check at a higher temperature, where the vibrational
    population is non-trivial (the e^(-x) terms aren't tiny)."""
    mol, hess = _h2o_hessian()
    T = 500.0
    t = vq.compute_thermochemistry(
        mol, hess,
        options=vq.ThermoOptions(temperature=T, pressure=101325.0,
                                  symmetry_number=2),
    )
    ps = _pyscf_thermo_h2o(T, 101325.0)
    # Heat capacities should also match (relative tolerance limited by
    # the CODATA-2014 vs CODATA-2018 Boltzmann constant difference).
    assert t.cv_trans == pytest.approx(ps["Cv_trans"][0], rel=1e-6)
    assert t.cv_rot == pytest.approx(ps["Cv_rot"][0], rel=1e-6)
    # Cv_vib at 500K is more sensitive to the frequency values (FD vs
    # PySCF analytic Hessian); 1e-3 absorbs both Boltzmann-constant and
    # FD-truncation noise.
    assert t.cv_vib == pytest.approx(ps["Cv_vib"][0], rel=1e-3)
    # Entropies still match.
    assert t.s_total == pytest.approx(ps["S_tot"][0], rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Linear vs nonlinear rotor handling
# ---------------------------------------------------------------------------

def test_h2_rotor_type_linear():
    mol, hess = _h2_hessian()
    t = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(symmetry_number=2),
    )
    assert t.rotor_type == "linear"
    # E_rot for linear molecule = kT (2 rotational DOFs)
    expected_e_rot = 3.166811563e-6 * 298.15  # k_B[Ha/K] × T[K] = kT
    assert t.e_rot == pytest.approx(expected_e_rot, rel=1e-9)
    # Cv_rot for linear = R (= k_B per molecule, same numeric value
    # in our atomic-unit-of-entropy convention)
    assert t.cv_rot == pytest.approx(3.166811563e-6, rel=1e-9)


def test_h2o_rotor_type_nonlinear():
    mol, hess = _h2o_hessian()
    t = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(symmetry_number=2),
    )
    assert t.rotor_type == "nonlinear"
    expected_e_rot = 1.5 * 3.166811563e-6 * 298.15
    assert t.e_rot == pytest.approx(expected_e_rot, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Atomic gas
# ---------------------------------------------------------------------------

def test_atom_thermo_he():
    """Single He atom: only translational + electronic contributions.
    No ZPE, no rotational anything."""
    mol = _he_atom()
    # Build a fake HessianResult — atom has no Hessian, but our thermo
    # function should still handle n_atoms == 1. We synthesise a result
    # consistent with what compute_hessian_fd would have returned.
    fake = vq.HessianResult(
        hessian=np.zeros((3, 3)),
        hessian_mw=np.zeros((3, 3)),
        frequencies_cm1=np.zeros(3),
        normal_modes=np.eye(3),
        imaginary_count=0,
        n_displacements=0,
        is_linear=False,
        masses_amu=np.array([4.003]),
    )
    t = vq.compute_thermochemistry(
        mol, fake, options=vq.ThermoOptions(symmetry_number=1),
    )
    assert t.rotor_type == "atom"
    assert t.zpe == 0.0
    assert t.e_rot == 0.0
    assert t.e_vib == 0.0
    assert t.s_rot == 0.0
    assert t.s_vib == 0.0
    assert t.cv_rot == 0.0
    assert t.cv_vib == 0.0
    # Translational entropy of He at STP is well-known: ~126 J/(K·mol).
    s_trans_expected_si = 126.151  # J/(K·mol) tabulated
    s_trans_au = (s_trans_expected_si
                  / (4.3597447222071e-18 * 6.02214076e23))
    assert t.s_trans == pytest.approx(s_trans_au, rel=2e-3)


# ---------------------------------------------------------------------------
# 5. Symmetry number affects rotational entropy
# ---------------------------------------------------------------------------

def test_symmetry_number_changes_s_rot_by_kln_sigma():
    """Rotational partition function scales as 1/σ, so S_rot picks up
    a -k_B ln(σ) contribution. Doubling σ from 1 to 2 must reduce
    S_rot by exactly k_B · ln(2)."""
    mol, hess = _h2o_hessian()
    t1 = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(symmetry_number=1),
    )
    t2 = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(symmetry_number=2),
    )
    expected_diff = 3.166811563e-6 * np.log(2.0)  # k_B ln(2)
    assert t1.s_rot - t2.s_rot == pytest.approx(expected_diff, rel=1e-9)


def test_electronic_degeneracy_default_from_multiplicity():
    """Default S_elec uses the molecule's multiplicity; explicit
    override takes precedence."""
    # Triplet O2 molecule (multiplicity 3 → S_elec = k_B ln 3)
    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(8, [0.0, 0.0, 1.21 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=3)
    opts = vq.UHFOptions()
    opts.conv_tol_grad = 1e-9
    hess = vq.compute_hessian_fd(mol, "sto-3g", method="UHF",
                                  scf_options=opts)
    # Default should pick up multiplicity=3
    t = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(symmetry_number=2),
    )
    expected_s_elec = 3.166811563e-6 * np.log(3.0)
    assert t.s_elec == pytest.approx(expected_s_elec, rel=1e-9)

    # Override forces a different value
    t2 = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(symmetry_number=2,
                                              electronic_degeneracy=1),
    )
    assert t2.s_elec == 0.0


# ---------------------------------------------------------------------------
# 6. Imaginary modes are excluded with a count
# ---------------------------------------------------------------------------

def test_imaginary_modes_are_excluded_from_partition_function():
    """A geometry slightly off-equilibrium produces small imaginary
    modes (or near-zero numerical artefacts). They must not enter
    the harmonic partition function. We check by handing the
    function a result with manually injected imaginary frequencies
    and verify the thermo numbers ignore them."""
    mol, hess = _h2o_hessian()
    # Inject a fake imaginary mode at the lowest frequency slot
    freqs = hess.frequencies_cm1.copy()
    # Replace the smallest-positive frequency with a negative one
    smallest_real_idx = int(np.argmin(np.abs(freqs - 2000.0)))
    freqs_with_imag = freqs.copy()
    freqs_with_imag[smallest_real_idx] = -100.0
    fake = vq.HessianResult(
        hessian=hess.hessian, hessian_mw=hess.hessian_mw,
        frequencies_cm1=freqs_with_imag,
        normal_modes=hess.normal_modes,
        imaginary_count=1,
        n_displacements=hess.n_displacements,
        is_linear=hess.is_linear,
        masses_amu=hess.masses_amu,
    )
    t_real = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(symmetry_number=2),
    )
    t_imag = vq.compute_thermochemistry(
        mol, fake, options=vq.ThermoOptions(symmetry_number=2),
    )
    # The imaginary mode contributes nothing — its ZPE/thermal
    # entries are dropped — so the result should differ by only
    # the contribution of the original real mode that we replaced.
    assert t_imag.n_imaginary_modes_excluded == 1
    # ZPE drops by half the original frequency in Hartree.
    dropped_zpe = 0.5 * freqs[smallest_real_idx] / 219474.6313632
    assert (t_real.zpe - t_imag.zpe) == pytest.approx(dropped_zpe, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Input validation
# ---------------------------------------------------------------------------

def test_invalid_temperature_raises():
    mol, hess = _h2o_hessian()
    with pytest.raises(ValueError, match="temperature"):
        vq.compute_thermochemistry(
            mol, hess, options=vq.ThermoOptions(temperature=-10.0),
        )


def test_invalid_pressure_raises():
    mol, hess = _h2o_hessian()
    with pytest.raises(ValueError, match="pressure"):
        vq.compute_thermochemistry(
            mol, hess, options=vq.ThermoOptions(pressure=0.0),
        )


def test_invalid_symmetry_number_raises():
    mol, hess = _h2o_hessian()
    with pytest.raises(ValueError, match="symmetry_number"):
        vq.compute_thermochemistry(
            mol, hess, options=vq.ThermoOptions(symmetry_number=0),
        )


def test_invalid_electronic_degeneracy_raises():
    mol, hess = _h2o_hessian()
    with pytest.raises(ValueError, match="electronic_degeneracy"):
        vq.compute_thermochemistry(
            mol, hess, options=vq.ThermoOptions(electronic_degeneracy=0),
        )
