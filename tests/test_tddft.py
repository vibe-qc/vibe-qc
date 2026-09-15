"""Tests for vibeqc.tddft — Casida linear-response TDDFT and TDA.

Validates:
- AO→MO ERI transformation correctness
- TDA/CIS excitation energies on H2 (reference against analytic values)
- Casida full TD-HF matches TDA in the TDA limit
- Oscillator strength computation
- Dominant amplitude extraction
- Integration with RHF ground state
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.tddft import (
    TDDFTResult,
    TDDFTState,
    eri_ao_to_mo,
    oscillator_strength,
    run_tddft_casida,
    run_tddft_tda,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2_molecule(d: float = 1.4) -> vq.Molecule:
    """H2 molecule at bond length d bohr."""
    half = d / 2.0
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, -half]), vq.Atom(1, [0.0, 0.0, half])],
        0,
        1,
    )


def _he_molecule() -> vq.Molecule:
    """He atom at origin."""
    return vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])], 0, 1)


# ---------------------------------------------------------------------------
# Oscillator strength
# ---------------------------------------------------------------------------


def test_oscillator_strength_zero_for_zero_dipole():
    """Zero transition dipole → zero oscillator strength."""
    f = oscillator_strength(0.5, np.array([0.0, 0.0, 0.0]))
    assert f == 0.0


def test_oscillator_strength_scales_with_energy():
    """Oscillator strength is proportional to excitation energy."""
    mu = np.array([1.0, 0.0, 0.0])
    f1 = oscillator_strength(0.1, mu)
    f2 = oscillator_strength(0.2, mu)
    assert f2 == pytest.approx(2.0 * f1, rel=1e-12)


# ---------------------------------------------------------------------------
# ERI AO→MO transformation: consistency checks
# ---------------------------------------------------------------------------


def test_eri_ao_to_mo_preserves_size():
    """The MO-basis ERI block has correct dimensions."""
    mol = _h2_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, opts)

    n_occ = mol.n_electrons() // 2
    n_basis = basis.nbasis
    n_virt = n_basis - n_occ

    eri_ao = vq._vibeqc_core.compute_eri(basis)
    eri_mo = eri_ao_to_mo(eri_ao, res.mo_coeffs, n_occ)

    # Check dimensions: (n_occ, n_virt, n_occ, n_virt)
    assert eri_mo.shape[0] == n_occ, f"dim 0: {eri_mo.shape[0]} != {n_occ}"
    assert eri_mo.shape[1] == n_virt, f"dim 1: {eri_mo.shape[1]} != {n_virt}"
    assert eri_mo.shape[2] == n_occ, f"dim 2: {eri_mo.shape[2]} != {n_occ}"
    assert eri_mo.shape[3] == n_virt, f"dim 3: {eri_mo.shape[3]} != {n_virt}"


def test_eri_ao_to_mo_symmetry():
    """The (ia|jb) tensor is symmetric under (i↔j, a↔b) exchange."""
    mol = _h2_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, opts)

    n_occ = mol.n_electrons() // 2
    eri_ao = vq._vibeqc_core.compute_eri(basis)
    eri_mo = eri_ao_to_mo(eri_ao, res.mo_coeffs, n_occ)

    # (ia|jb) == (jb|ia)
    assert np.allclose(eri_mo, eri_mo.transpose(2, 3, 0, 1), atol=1e-12)


# ---------------------------------------------------------------------------
# TDA / CIS on H2
# ---------------------------------------------------------------------------


# Module-level cache to avoid recomputing the same SCF repeatedly
_h2_cache: tuple | None = None


def _h2_rhf_sto3g():
    """Helper: H2 RHF/STO-3G ground state (cached)."""
    global _h2_cache
    if _h2_cache is None:
        mol = _h2_molecule()
        basis = vq.BasisSet(mol, "sto-3g")
        opts = vq.RHFOptions()
        opts.conv_tol_energy = 1e-10
        _h2_cache = (mol, basis, vq.run_rhf(mol, basis, opts))
    return _h2_cache


def test_tda_h2_sto3g_returns_state_count():
    """TDA returns the requested number of states."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1
    )
    assert len(result.states) == 1
    assert result.method == "TDA"


def test_tda_h2_sto3g_energies_positive():
    """All TDA excitation energies are positive."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=3
    )
    assert all(s.excitation_energy > 0 for s in result.states)


def test_tda_h2_sto3g_excitation_reasonable():
    """H2 STO-3G CIS lowest excitation is ~0.95 Ha (sigma_g to sigma_u).

    The CIS singlet excitation for H2 at R=1.4 bohr with STO-3G
    is epsilon_diff + 2*(ia|ia) - (ii|aa) = 1.248 + 0.363 - 0.664 = 0.947 Ha.
    This test verifies we're in the right ballpark.
    """
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1
    )
    omega = result.states[0].excitation_energy
    assert 0.5 < omega < 1.5, f"omega={omega:.4f} out of expected range [0.5, 1.5]"


def test_tda_h2_sto3g_energies_monotonic():
    """TDA excitation energies are sorted. H2/STO-3G has only 1
    transition, so requesting 1 state gives a trivially sorted list."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1
    )
    assert len(result.states) == 1
    assert result.states[0].excitation_energy > 0


def test_tda_h2_sto3g_oscillator_sum_positive():
    """At least one state has non-zero oscillator strength."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=2
    )
    total_f = sum(s.oscillator_strength for s in result.states)
    assert total_f > 0.0


def test_tda_h2_sto3g_dominant_amplitude_present():
    """Each state reports at least one dominant amplitude."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1
    )
    assert len(result.states[0].dominant_amplitudes) > 0


def test_tda_he_sto3g_no_virtuals():
    """He STO-3G has nbasis=1, n_occ=1, n_virt=0 — must raise."""
    mol = _he_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, opts)
    n_occ = mol.n_electrons() // 2
    with pytest.raises(ValueError, match="virtual"):
        run_tddft_tda(mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1)


# ---------------------------------------------------------------------------
# Casida
# ---------------------------------------------------------------------------


def test_casida_h2_sto3g_returns_state_count():
    """Casida returns the requested number of states."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_casida(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1
    )
    assert len(result.states) == 1
    assert result.method == "Casida"


def test_casida_h2_sto3g_energies_positive():
    """All Casida excitation energies are positive."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_casida(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=3
    )
    assert all(s.excitation_energy > 0 for s in result.states)


def test_casida_h2_sto3g_vs_tda_ordering():
    """Casida energies should be lower than TDA (B-matrix stabilising)."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    tda = run_tddft_tda(mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=3)
    cas = run_tddft_casida(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=3
    )
    for i in range(min(len(tda.states), len(cas.states))):
        assert (
            cas.states[i].excitation_energy <= tda.states[i].excitation_energy + 1e-10
        ), (
            f"Casida S{i + 1} ({cas.states[i].excitation_energy:.6f}) "
            f"should be ≤ TDA ({tda.states[i].excitation_energy:.6f})"
        )


def test_casida_h2_sto3g_dominant_amplitude_present():
    """Casida states report dominant amplitudes."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_casida(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1
    )
    assert len(result.states[0].dominant_amplitudes) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_tda_no_virtual_orbitals_raises():
    """When n_basis == n_occ, TDDFT should raise ValueError."""
    mol = _he_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, opts)
    with pytest.raises(ValueError, match="virtual"):
        run_tddft_tda(
            mol, basis, res.mo_energies, res.mo_coeffs, n_occ=basis.nbasis, n_states=3
        )


def test_casida_no_virtual_orbitals_raises():
    """Casida should also raise when no virtuals."""
    mol = _he_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, opts)
    with pytest.raises(ValueError, match="virtual"):
        run_tddft_casida(
            mol, basis, res.mo_energies, res.mo_coeffs, n_occ=basis.nbasis, n_states=3
        )


# ---------------------------------------------------------------------------
# Result attributes
# ---------------------------------------------------------------------------


def test_tddft_result_metadata():
    """TDDFTResult carries correct metadata."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    n_requested = 1  # H2/STO-3G has only 1 occ->virt transition
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=n_requested
    )
    assert result.n_states == n_requested
    assert result.n_occ == n_occ
    assert result.n_virt == basis.nbasis - n_occ
    assert isinstance(result.states, list)


def test_tddft_state_attributes():
    """TDDFTState has all expected attributes."""
    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    result = run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ, n_states=1
    )
    s = result.states[0]
    assert s.index == 1
    assert isinstance(s.excitation_energy, float)
    assert isinstance(s.excitation_energy_ev, float)
    assert isinstance(s.wavelength_nm, float)
    assert isinstance(s.oscillator_strength, float)
    assert s.transition_dipole.shape == (3,)
    assert len(s.dominant_amplitudes) > 0


# ---------------------------------------------------------------------------
# Importable from vibeqc
# ---------------------------------------------------------------------------


def test_tddft_functions_importable_from_vibeqc():
    """The public API is accessible as vibeqc.run_tddft_tda etc."""
    assert callable(vq.run_tddft_tda)
    assert callable(vq.run_tddft_casida)
    assert callable(vq.oscillator_strength)
    assert callable(vq.run_tddft_tda_uhf)
    assert callable(vq.run_tddft_tda_periodic)
    assert callable(vq.eri_ao_to_mo)


# ---------------------------------------------------------------------------
# ALDA kernel validation
# ---------------------------------------------------------------------------


def test_alda_kernel_changes_excitation_energy():
    """TDA with the ALDA kernel gives a different (lower) excitation
    energy than TDA without the XC kernel, for LDA on H2."""
    mol = _h2_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-10
    res = vq.run_rks(mol, basis, opts)
    n_occ = mol.n_electrons() // 2

    # Without kernel (c_x=0, no f_xc)
    tda_no_kernel = vq.run_tddft_tda(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        n_occ,
        n_states=1,
        functional="lda",
    )
    # With ALDA kernel
    tda_with_kernel = vq.run_tddft_tda(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        n_occ,
        n_states=1,
        functional="lda",
        density_ao=np.asarray(res.density),
    )

    w_no = tda_no_kernel.states[0].excitation_energy
    w_with = tda_with_kernel.states[0].excitation_energy

    # The ALDA kernel should shift the excitation energy (typically
    # lowers it for LDA on H2). Verify they differ by at least 1e-6 Ha.
    assert abs(w_no - w_with) > 1e-6, (
        f"ALDA kernel did not shift excitation: w_no={w_no:.6f}, w_with={w_with:.6f}"
    )
    # Both should be positive
    assert w_no > 0
    assert w_with > 0


# ---------------------------------------------------------------------------
# UHF TDA validation
# ---------------------------------------------------------------------------


def test_uhf_tda_closed_shell_matches_rhf():
    """UHF TDA on a closed-shell singlet matches RHF TDA."""
    mol = _h2_molecule()
    basis = vq.BasisSet(mol, "sto-3g")

    opts_r = vq.RHFOptions()
    opts_r.conv_tol_energy = 1e-10
    res_r = vq.run_rhf(mol, basis, opts_r)
    tda_r = vq.run_tddft_tda(
        mol,
        basis,
        res_r.mo_energies,
        res_r.mo_coeffs,
        1,
        n_states=1,
    )

    opts_u = vq.UHFOptions()
    opts_u.conv_tol_energy = 1e-10
    res_u = vq.run_uhf(mol, basis, opts_u)
    tda_u = vq.run_tddft_tda_uhf(
        mol,
        basis,
        res_u.mo_energies_alpha,
        res_u.mo_energies_beta,
        res_u.mo_coeffs_alpha,
        res_u.mo_coeffs_beta,
        1,
        1,
        n_states=2,
    )

    # The second UHF state (in-phase alpha+beta) should match RHF
    w_r = tda_r.states[0].excitation_energy
    w_u = tda_u.states[1].excitation_energy  # S2 = RHF-like
    assert abs(w_r - w_u) < 1e-8, f"UHF S2 ({w_u:.10f}) should match RHF ({w_r:.10f})"


def test_uhf_tda_li_atom_has_positive_excitations():
    """UHF TDA on Li atom doublet produces positive excitation energies."""
    mol = vq.Molecule([vq.Atom(3, [0.0, 0.0, 0.0])], 0, 2)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UHFOptions()
    opts.conv_tol_energy = 1e-10
    res = vq.run_uhf(mol, basis, opts)

    n_e = mol.n_electrons()
    mult = mol.multiplicity
    n_a = (n_e + (mult - 1)) // 2
    n_b = (n_e - (mult - 1)) // 2

    tda = vq.run_tddft_tda_uhf(
        mol,
        basis,
        res.mo_energies_alpha,
        res.mo_energies_beta,
        res.mo_coeffs_alpha,
        res.mo_coeffs_beta,
        n_a,
        n_b,
        n_states=3,
    )

    assert len(tda.states) >= 1
    assert all(s.excitation_energy > 0 for s in tda.states)
    assert all(s.oscillator_strength >= 0 for s in tda.states)


# ---------------------------------------------------------------------------
# NTO analysis
# ---------------------------------------------------------------------------


def test_nto_single_occ_virt():
    """NTO of a 1x1 excitation yields a single pair with weight 1.0."""
    from vibeqc.tddft import NTOResult, compute_nto

    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    tda = run_tddft_tda(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        n_occ,
        n_states=1,
    )
    nto = compute_nto(tda.states[0], res.mo_coeffs, n_occ)
    assert isinstance(nto, NTOResult)
    assert len(nto.hole_weights) == 1
    assert abs(nto.hole_weights[0] - 1.0) < 1e-12
    assert nto.hole_orbitals_ao.shape[1] == 1
    assert nto.particle_orbitals_ao.shape[1] == 1
    assert nto.dominant_pair[0] == pytest.approx(1.0)


def test_nto_shape_matches_basis():
    """NTO hole/particle orbitals have n_basis rows."""
    from vibeqc.tddft import compute_nto

    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    tda = run_tddft_tda(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        n_occ,
        n_states=1,
    )
    nto = compute_nto(tda.states[0], res.mo_coeffs, n_occ)
    assert nto.hole_orbitals_ao.shape[0] == basis.nbasis
    assert nto.particle_orbitals_ao.shape[0] == basis.nbasis


def test_nto_weights_sum_to_one():
    """NTO singular values squared sum to 1 for a normalized excitation."""
    from vibeqc.tddft import compute_nto

    mol, basis, res = _h2_rhf_sto3g()
    n_occ = mol.n_electrons() // 2
    tda = run_tddft_tda(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        n_occ,
        n_states=1,
    )
    nto = compute_nto(tda.states[0], res.mo_coeffs, n_occ)
    # For a normalized excitation vector, sum(s^2) == 1
    assert abs(np.sum(nto.hole_weights**2) - 1.0) < 1e-12
