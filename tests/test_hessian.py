"""Phase 17a-1 — finite-difference Hessian + harmonic frequencies.

Contracts pinned here:

1. **API surface** — :func:`vibeqc.compute_hessian_fd`,
   :class:`vibeqc.HessianFDOptions`, :class:`vibeqc.HessianResult`
   are public.

2. **PySCF parity on H2O / STO-3G (RHF)** — the Hessian elements
   match PySCF's analytic Hessian to ~1e-5 Ha/bohr² (FD step
   truncation) and the harmonic frequencies match to <1 cm⁻¹.

3. **Trans/rot projection** — ``project_trans_rot=True`` (default)
   produces exactly 6 zeros for nonlinear molecules and 5 zeros
   for linear molecules (auto-detected via the inertia tensor
   rank). The vibrational subset is then ``frequencies_cm1[6:]``
   (or ``[5:]`` for linear).

4. **Symmetry** — the returned ``hessian`` is symmetric to FD
   noise level, regardless of whether ``sym_strict=True``.

5. **Method dispatch** — RHF, UHF, RKS, UKS all run end-to-end
   on small molecules and produce non-imaginary frequencies at a
   stationary geometry.

6. **Isotope override** — substituting D₂ for H₂ via
   ``atomic_masses_amu`` shifts the stretching frequency by
   ~1/√2 (the harmonic-oscillator mass-scaling law).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213

# ---------------------------------------------------------------------------
# Fixtures: geometries near the HF/STO-3G stationary points
# ---------------------------------------------------------------------------

def _h2_mol(R_bohr: float = 1.346) -> vq.Molecule:
    """H2 near HF/STO-3G optimum (R ≈ 1.346 bohr)."""
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, R_bohr])],
        charge=0, multiplicity=1,
    )


def _h2o_mol() -> vq.Molecule:
    """H2O at the canonical experimental geometry (close to HF/STO-3G
    optimum: r(O-H) ≈ 0.96 Å, ∠HOH ≈ 104.5°)."""
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)


def _o2_triplet_mol() -> vq.Molecule:
    """O2 in its triplet ground state. Distance ~ 1.21 Å for HF/STO-3G."""
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(8, [0.0, 0.0, 1.21 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=3)


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_hessian_api_exposed():
    assert hasattr(vq, "compute_hessian_fd")
    assert hasattr(vq, "HessianFDOptions")
    assert hasattr(vq, "HessianResult")


# ---------------------------------------------------------------------------
# 2. PySCF parity on H2O / STO-3G
# ---------------------------------------------------------------------------

def _pyscf_rhf_hessian_h2o(basis: str = "sto-3g"):
    """Returns (H_3N3N, frequencies_cm1) from PySCF for our H2O geometry."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.hessian import thermo

    mol = gto.Mole()
    mol.atom = [
        ["O", (0.0, 0.0, 0.0)],
        ["H", (0.0,  0.7572, 0.5868)],
        ["H", (0.0, -0.7572, 0.5868)],
    ]
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    h = mf.Hessian().kernel()
    n = mol.natm
    H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            H[3*i:3*i+3, 3*j:3*j+3] = h[i, j]
    info = thermo.harmonic_analysis(mol, h)
    freqs = np.real(info["freq_wavenumber"])
    return H, np.sort(freqs)


def test_hessian_matches_pyscf_h2o_rhf():
    """H2O / STO-3G FD Hessian elements within 5e-5 Ha/bohr² of PySCF
    analytic Hessian (FD truncation at step 0.005 bohr)."""
    mol = _h2o_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9

    H_ps, freqs_ps = _pyscf_rhf_hessian_h2o("sto-3g")
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts)

    # Hessian elements: FD truncation error scales as O(δ²).
    np.testing.assert_allclose(result.hessian, H_ps, atol=5e-5, rtol=0,
                               err_msg="H2O/STO-3G Hessian disagrees with PySCF")

    # Top three vibrational frequencies (bend + 2 stretches): tight match.
    freqs_vq = np.sort(result.frequencies_cm1)[-3:]
    np.testing.assert_allclose(freqs_vq, freqs_ps, atol=1.0,
                               err_msg="H2O/STO-3G frequencies disagree with PySCF")


# ---------------------------------------------------------------------------
# 3. Trans / rot projection
# ---------------------------------------------------------------------------

def test_linear_molecule_has_five_zero_modes():
    """H2 is linear → 5 trans/rot zero modes (3 trans, 2 rot perpendicular
    to the bond axis), 1 vibration."""
    mol = _h2_mol()
    opts = vq.RHFOptions(); opts.conv_tol_grad = 1e-9
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts)
    assert result.is_linear
    n_zero = int(np.sum(np.abs(result.frequencies_cm1) < 1e-3))
    assert n_zero == 5
    n_finite = result.frequencies_cm1.size - n_zero
    assert n_finite == 1


def test_nonlinear_molecule_has_six_zero_modes():
    """H2O is nonlinear → 6 trans/rot zero modes, 3 vibrations."""
    mol = _h2o_mol()
    opts = vq.RHFOptions(); opts.conv_tol_grad = 1e-9
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts)
    assert not result.is_linear
    n_zero = int(np.sum(np.abs(result.frequencies_cm1) < 1e-3))
    assert n_zero == 6


def test_no_imaginary_modes_at_minimum():
    """At a near-stationary geometry, no imaginary modes after projection."""
    mol = _h2o_mol()
    opts = vq.RHFOptions(); opts.conv_tol_grad = 1e-9
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts)
    assert result.imaginary_count == 0


def test_hessian_is_symmetric():
    """The FD Hessian is built one column at a time but symmetrized at the
    end (sym_strict=True). With strict-sym off the residual asymmetry must
    still be at the FD-noise level."""
    mol = _h2o_mol()
    opts = vq.RHFOptions(); opts.conv_tol_grad = 1e-9

    h_opts = vq.HessianFDOptions(sym_strict=True)
    r_sym = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                   scf_options=opts, hessian_options=h_opts)
    np.testing.assert_allclose(r_sym.hessian, r_sym.hessian.T, atol=1e-12)

    h_opts = vq.HessianFDOptions(sym_strict=False)
    r_raw = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                   scf_options=opts, hessian_options=h_opts)
    asym = r_raw.hessian - r_raw.hessian.T
    assert np.max(np.abs(asym)) < 1e-3   # FD noise floor


# ---------------------------------------------------------------------------
# 4. Method dispatch — runs end-to-end on each method
# ---------------------------------------------------------------------------

def test_hessian_runs_on_rks_lda_h2():
    mol = _h2_mol()
    opts = vq.RKSOptions()
    opts.functional = "LDA"
    opts.conv_tol_grad = 1e-9
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RKS",
                                    scf_options=opts)
    assert result.imaginary_count == 0
    # H2 LDA stretch is in the 4500-5500 cm^-1 range (different from HF
    # but well above zero). Just sanity-check the magnitude.
    nonzero = result.frequencies_cm1[np.abs(result.frequencies_cm1) > 1.0]
    assert nonzero.size == 1
    assert 3000.0 < nonzero[0] < 6000.0


def test_hessian_runs_on_uhf_triplet_o2():
    """Triplet O2 / STO-3G gradient + Hessian path. The geometry is not
    exactly stationary so we accept up to 1 imaginary mode; main goal is
    that the per-spin gradient code feeds correctly into the FD driver."""
    mol = _o2_triplet_mol()
    opts = vq.UHFOptions()
    opts.conv_tol_grad = 1e-9
    result = vq.compute_hessian_fd(mol, "sto-3g", method="UHF",
                                    scf_options=opts)
    # 6 atoms × DOF = 6, of which 5 are trans/rot zeros (linear), 1 is the
    # O2 stretch. Stretch should be a few thousand cm^-1.
    assert result.is_linear
    nonzero = np.abs(result.frequencies_cm1)[np.abs(result.frequencies_cm1) > 1.0]
    assert nonzero.size == 1


def test_hessian_runs_on_uks_lda_triplet_o2():
    mol = _o2_triplet_mol()
    opts = vq.UKSOptions()
    opts.functional = "LDA"
    opts.conv_tol_grad = 1e-9
    result = vq.compute_hessian_fd(mol, "sto-3g", method="UKS",
                                    scf_options=opts)
    assert result.is_linear
    nonzero = np.abs(result.frequencies_cm1)[np.abs(result.frequencies_cm1) > 1.0]
    assert nonzero.size == 1


# ---------------------------------------------------------------------------
# 5. Isotope substitution
# ---------------------------------------------------------------------------

def test_isotope_substitution_d2_vs_h2():
    """D₂ stretch ≈ H₂ stretch / √2 by the harmonic mass-scaling law:
    ω = √(k/μ); μ_D2 / μ_H2 = 2, so ω_D2 / ω_H2 = 1/√2.

    We use the same geometry (the equilibrium R is essentially mass-
    independent in the Born-Oppenheimer picture) and only swap the
    masses in HessianFDOptions, so the same SCF/gradient evaluations
    are reused and the only thing that changes is the mass-weighting.
    """
    mol = _h2_mol()
    opts = vq.RHFOptions(); opts.conv_tol_grad = 1e-9

    # H2: default masses (1.008 amu each)
    r_h2 = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                  scf_options=opts)

    # D2: 2.014 amu each
    h_opts = vq.HessianFDOptions(atomic_masses_amu=[2.014, 2.014])
    r_d2 = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                  scf_options=opts,
                                  hessian_options=h_opts)

    f_h2 = r_h2.frequencies_cm1[-1]
    f_d2 = r_d2.frequencies_cm1[-1]
    ratio = f_d2 / f_h2
    expected = np.sqrt(1.008 / 2.014)
    # The mass scaling is exact in the harmonic picture; tiny deviation
    # is from the 1.008 vs exact 1.00794 amu used in the table.
    assert abs(ratio - expected) < 5e-3


# ---------------------------------------------------------------------------
# 6. Validation / error paths
# ---------------------------------------------------------------------------

def test_unknown_method_raises():
    mol = _h2_mol()
    with pytest.raises(ValueError, match="unknown method"):
        vq.compute_hessian_fd(mol, "sto-3g", method="MP2")


def test_invalid_step_raises():
    mol = _h2_mol()
    h_opts = vq.HessianFDOptions(step_bohr=-0.001)
    with pytest.raises(ValueError, match="step_bohr"):
        vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                               hessian_options=h_opts)


def test_isotope_override_wrong_length_raises():
    mol = _h2_mol()
    h_opts = vq.HessianFDOptions(atomic_masses_amu=[1.0])  # length 1, need 2
    with pytest.raises(ValueError, match="atomic_masses_amu"):
        vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                               hessian_options=h_opts)


def test_default_hessian_options():
    """Defaults are documented as 0.005 bohr step, sym_strict=True,
    project_trans_rot=True, no isotope override, no dipole derivs."""
    o = vq.HessianFDOptions()
    assert o.step_bohr == 0.005
    assert o.sym_strict is True
    assert o.project_trans_rot is True
    assert o.atomic_masses_amu is None
    assert o.include_dipole_derivatives is False


# ---------------------------------------------------------------------------
# 7. Phase 17a-2 — IR intensities
# ---------------------------------------------------------------------------

def test_ir_intensities_api_exposed():
    assert hasattr(vq, "ir_intensities")


def test_ir_intensities_default_off():
    """Without ``include_dipole_derivatives=True`` the dipole-derivative
    fields stay None and ``ir_intensities`` raises a clean ValueError
    (rather than failing with an attribute error)."""
    mol = _h2_mol()
    opts = vq.RHFOptions(); opts.conv_tol_grad = 1e-9
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts)
    assert result.dipole_derivatives is None
    assert result.dipole_origin is None
    with pytest.raises(ValueError, match="dipole derivatives missing"):
        vq.ir_intensities(result)


def test_ir_intensity_h2_homonuclear_is_zero():
    """H₂ has no permanent dipole and no dipole derivative (homonuclear
    diatomic). The single stretching mode must be IR-inactive within
    floating-point noise."""
    mol = _h2_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    h_opts = vq.HessianFDOptions(include_dipole_derivatives=True)
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts, hessian_options=h_opts)
    ir = vq.ir_intensities(result)
    # Each entry must be zero — homonuclear vibration + masked zero
    # modes leave nothing.
    assert np.max(np.abs(ir)) < 1e-10


def test_ir_intensity_zero_modes_are_masked():
    """For H₂O the trans/rot zero modes must have IR intensity exactly
    zero. (Without the mask, the eigh decomposition of the projected
    Hessian returns an arbitrary basis of the kernel that mixes
    translations with rotations, and the latter have non-zero
    ∂μ/∂Q — see the ir_intensities docstring.)"""
    mol = _h2o_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    h_opts = vq.HessianFDOptions(include_dipole_derivatives=True)
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts, hessian_options=h_opts)
    ir = vq.ir_intensities(result)
    # Six zero-frequency modes (nonlinear molecule).
    zero_idx = np.where(np.abs(result.frequencies_cm1) < 1e-3)[0]
    assert zero_idx.size == 6
    np.testing.assert_array_equal(ir[zero_idx], 0.0)


def test_ir_intensity_hf_heteronuclear_nonzero():
    """HF (heteronuclear diatomic) has a strongly IR-active stretch.
    Order of magnitude check: HF/STO-3G IR intensity for the stretch
    is in the 10–80 km/mol range across typical HF references."""
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(9, [0.0, 0.0, 1.733]),  # ~ HF/STO-3G equilibrium
    ], charge=0, multiplicity=1)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    h_opts = vq.HessianFDOptions(include_dipole_derivatives=True)
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts, hessian_options=h_opts)
    ir = vq.ir_intensities(result)
    # Five trans/rot zeros (linear); one vibration — the H-F stretch.
    nonzero = ir[np.abs(result.frequencies_cm1) > 1.0]
    assert nonzero.size == 1
    assert 5.0 < nonzero[0] < 200.0


def test_ir_h2o_dipole_derivatives_match_pyscf():
    """Cross-check the Cartesian dipole-derivative tensor against
    PySCF's own FD-on-dipole at the same step. Factors out the IR
    formula and pins just the dipole derivative computation. Agreement
    to ~1e-6 e (FD truncation level)."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = _h2o_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    h_opts = vq.HessianFDOptions(include_dipole_derivatives=True)
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts, hessian_options=h_opts)

    # ---- PySCF FD reference ---------------------------------------------
    origin = list(result.dipole_origin)
    positions = np.array([list(a.xyz) for a in mol.atoms])

    def pyscf_dipole(pos_bohr):
        m = gto.Mole()
        m.unit = "Bohr"
        m.atom = [
            ["O", pos_bohr[0]],
            ["H", pos_bohr[1]],
            ["H", pos_bohr[2]],
        ]
        m.basis = "sto-3g"
        m.verbose = 0
        m.build()
        mf = scf.RHF(m)
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        with m.with_common_origin(origin):
            ao_dip = m.intor_symmetric("int1e_r", comp=3)
        el = -np.einsum("xij,ji->x", ao_dip, dm)
        nuc = np.zeros(3)
        for k in range(m.natm):
            nuc += m.atom_charge(k) * (m.atom_coord(k) - np.array(origin))
        return el + nuc

    delta = 0.005
    B_ps = np.zeros((9, 3))
    for i in range(3):
        for cart in range(3):
            pp = positions.copy(); pp[i, cart] += delta
            pm = positions.copy(); pm[i, cart] -= delta
            B_ps[3 * i + cart, :] = (pyscf_dipole(pp) - pyscf_dipole(pm)) / (2 * delta)

    np.testing.assert_allclose(result.dipole_derivatives, B_ps,
                               atol=5e-6, rtol=0,
                               err_msg="dipole-derivative tensor disagrees with PySCF FD")


def test_ir_intensity_h2o_matches_published_values():
    """RHF/STO-3G H₂O has published IR intensities from FD-of-dipole
    treatments (Koput 1992 and subsequent works): bend ≈ 11 km/mol,
    sym stretch ≈ 36–37, antisym stretch ≈ 20–22. The relative
    ordering and rough magnitudes are pinned here as a regression
    guard against a unit-conversion drift."""
    mol = _h2o_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    h_opts = vq.HessianFDOptions(include_dipole_derivatives=True)
    result = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                    scf_options=opts, hessian_options=h_opts)
    ir = vq.ir_intensities(result)

    # Three vibrational modes after sorting by frequency: bend,
    # sym stretch, antisym stretch.
    nonzero_idx = np.where(np.abs(result.frequencies_cm1) > 1.0)[0]
    assert nonzero_idx.size == 3
    bend, sym_str, antisym_str = ir[nonzero_idx]

    # Magnitudes — 50% tolerance, pinning the order of magnitude.
    assert 5.0 < bend < 30.0
    assert 20.0 < sym_str < 80.0    # widest band, sym stretch is the strongest
    assert 5.0 < antisym_str < 60.0
    # Sym stretch dominates (largest derivative through the bond axis).
    assert sym_str > bend


def test_ir_intensities_translational_invariance():
    """A rigid translation of the molecule cannot change physical IR
    intensities (the SCF + dipole are translation-invariant). Pinning
    the contract guards against origin-handling bugs."""
    base = _h2o_mol()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    h_opts = vq.HessianFDOptions(include_dipole_derivatives=True)

    r0 = vq.ir_intensities(vq.compute_hessian_fd(
        base, "sto-3g", method="RHF",
        scf_options=opts, hessian_options=h_opts))

    # Shift everyone by (0.2, 0.3, 0.5) bohr.
    shift = np.array([0.2, 0.3, 0.5])
    shifted = vq.Molecule([
        vq.Atom(int(a.Z),
                [a.xyz[0] + shift[0], a.xyz[1] + shift[1], a.xyz[2] + shift[2]])
        for a in base.atoms
    ], charge=base.charge, multiplicity=base.multiplicity)
    r1 = vq.ir_intensities(vq.compute_hessian_fd(
        shifted, "sto-3g", method="RHF",
        scf_options=opts, hessian_options=h_opts))

    # Pair up by ascending frequency in each result and check
    # element-wise equality (the freq ordering itself is invariant).
    np.testing.assert_allclose(np.sort(r0), np.sort(r1), atol=1e-6)


def test_ir_intensities_isotope_substitution():
    """For a heteronuclear diatomic, the harmonic-oscillator IR
    intensity scales as ``|∂μ/∂Q|² ∝ 1/μ`` where ``μ`` is the reduced
    mass (the normal-mode amplitude L is ``∝ 1/√μ``). Substituting
    H → D in HF changes the reduced mass from
    ``μ_HF = m_H m_F / (m_H + m_F) ≈ 0.957 amu``
    to ``μ_DF ≈ 1.821 amu``, so the IR-intensity ratio is

        I_DF / I_HF = μ_HF / μ_DF ≈ 0.526.

    Same dipole-derivative tensor (geometry-only), different mass
    weighting via ``HessianFDOptions.atomic_masses_amu``."""
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(9, [0.0, 0.0, 1.733]),
    ], charge=0, multiplicity=1)
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    h_opts_h = vq.HessianFDOptions(include_dipole_derivatives=True)
    r_h = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                 scf_options=opts, hessian_options=h_opts_h)

    h_opts_d = vq.HessianFDOptions(include_dipole_derivatives=True,
                                    atomic_masses_amu=[2.014, 19.00])
    r_d = vq.compute_hessian_fd(mol, "sto-3g", method="RHF",
                                 scf_options=opts, hessian_options=h_opts_d)

    ir_h = vq.ir_intensities(r_h)
    ir_d = vq.ir_intensities(r_d)
    band_h = ir_h[np.abs(r_h.frequencies_cm1) > 1.0][0]
    band_d = ir_d[np.abs(r_d.frequencies_cm1) > 1.0][0]

    mu_HF = 1.008 * 19.00 / (1.008 + 19.00)
    mu_DF = 2.014 * 19.00 / (2.014 + 19.00)
    expected_ratio = mu_HF / mu_DF       # ≈ 0.526
    ratio = band_d / band_h
    assert abs(ratio - expected_ratio) < 0.01


@pytest.mark.parametrize("displacement", [-0.1, 0.1])
def test_hessian_displacement_projects_preloaded_read_from_reference_basis(displacement):
    from vibeqc.hessian import _scf_and_gradient_at
    from vibeqc.guess_read import _to_current_basis

    source = _h2_mol()
    source_basis = vq.BasisSet(source, "sto-3g")
    prior = vq.run_rhf(source, source_basis)
    options = vq.RHFOptions()
    options.initial_guess = vq.InitialGuess.READ
    options.read_density = prior.density
    displaced = _h2_mol(1.346 + displacement)
    basis, result, gradient = _scf_and_gradient_at(
        displaced, "sto-3g", "RHF", options, vq.GridOptions(),
        reference_mol=source,
    )
    np.testing.assert_array_equal(options.read_density, prior.density)
    read_options = vq.RHFOptions()
    read_options.initial_guess = vq.InitialGuess.READ
    expected = vq.run_rhf(displaced, basis, read_options, read_from=prior)
    assert result.converged
    assert result.energy == pytest.approx(expected.energy, abs=1e-11)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(_to_current_basis(prior.density, source_basis, basis) - prior.density) > 1e-3
    np.testing.assert_array_equal(options.read_density, prior.density)
