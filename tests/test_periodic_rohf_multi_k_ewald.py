"""Multi-k periodic ROHF (EWALD_3D) tests — run on a build box.

The Roothaan coupling (incl. the complex-Hermitian non-Γ case), occupation
and the per-k SCF wiring are unit-checked; these pin the *multi-k periodic*
integration:

* closed-shell and one-electron limits agree with the independently wired
  BIPOLE drivers using the same corrected Ewald-exchange convention,
* exact ⟨S²⟩ = S(S+1) for an open shell.

The non-special-mesh LiH+ anchor is validated out of process against
PySCF 2.13.1 KROHF/GDF with the same BvK spin counts and ``exxdiv='ewald'``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc as vq


def _options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.damping = 0.3
    o.max_iter = 60
    return o


def _h2_box(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _h_atom_box(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 2
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _assert_rohf_result_state_is_consistent(result, kmesh):
    """Every returned observable must describe the returned RO orbitals."""
    from vibeqc._vibeqc_core import real_space_density_from_kpoints_fractional

    occ_alpha = [
        (np.asarray(occ) > 0.0).astype(float)
        for occ in result.mo_occupations
    ]
    occ_beta = [
        (np.asarray(occ) == 2.0).astype(float)
        for occ in result.mo_occupations
    ]
    cells = list(result.density_alpha.cells)
    density_alpha = real_space_density_from_kpoints_fractional(
        result.mo_coeffs, occ_alpha, kmesh, cells
    )
    density_beta = real_space_density_from_kpoints_fractional(
        result.mo_coeffs, occ_beta, kmesh, cells
    )
    for rebuilt, returned in zip(
        density_alpha.blocks, result.density_alpha.blocks, strict=True
    ):
        assert np.asarray(returned) == pytest.approx(
            np.asarray(rebuilt), abs=1.0e-12
        )
    for rebuilt, returned in zip(
        density_beta.blocks, result.density_beta.blocks, strict=True
    ):
        assert np.asarray(returned) == pytest.approx(
            np.asarray(rebuilt), abs=1.0e-12
        )
    for alpha, beta, total in zip(
        result.density_alpha.blocks,
        result.density_beta.blocks,
        result.density.blocks,
        strict=True,
    ):
        assert np.asarray(total) == pytest.approx(
            np.asarray(alpha) + np.asarray(beta), abs=1.0e-12
        )
    for energies, coeffs, fock in zip(
        result.mo_energies, result.mo_coeffs, result.fock, strict=True
    ):
        diagonal = np.real(np.diag(coeffs.conj().T @ fock @ coeffs))
        assert np.asarray(energies) == pytest.approx(diagonal, abs=1.0e-12)
    assert result.energy == pytest.approx(result.scf_trace[-1].energy, abs=1.0e-12)


def test_multik_rohf_complex_diis_on_non_special_mesh():
    """A non-special mesh exercises complex Pulay residuals without casts."""
    box = 12.0
    centre = box / 2.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [
            vq.Atom(3, [centre - 0.6, centre, centre]),
            vq.Atom(1, [centre + 1.0, centre, centre]),
        ],
        charge=1,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [3, 1, 1])
    options = _options()
    options.lattice_opts.cutoff_bohr = 12.0
    options.lattice_opts.nuclear_cutoff_bohr = 15.0
    options.max_iter = 100
    options.conv_tol_energy = 1.0e-8
    options.conv_tol_grad = 1.0e-6
    complex_warning = getattr(getattr(np, "exceptions", np), "ComplexWarning")

    with warnings.catch_warnings():
        warnings.simplefilter("error", complex_warning)
        result = vq.run_rohf_periodic_multi_k_ewald3d(
            system,
            basis,
            kmesh,
            options,
            omega=0.0,
            spacing_bohr=0.5,
        )

    assert result.converged
    # Pulay DIIS convergence is defined by the vanishing Fock-density
    # commutator, not by a fixed number of extrapolation cycles (Pulay,
    # J. Comput. Chem. 3, 556 (1982), DOI 10.1002/jcc.540030413).  The
    # exact cycle count is sensitive to harmless floating-point changes in
    # the Bloch transforms and DIIS linear solve, so pin the physical and
    # algorithmic state instead.
    assert 1 < result.n_iter < options.max_iter
    assert len(result.scf_trace) == result.n_iter
    assert result.scf_trace[-1].iter == result.n_iter
    assert abs(result.scf_trace[-1].delta_e) < options.conv_tol_energy
    assert result.scf_trace[-1].grad_norm < options.conv_tol_grad
    assert max(step.diis_subspace for step in result.scf_trace) >= 2
    # Recompute the Pulay error from the returned orbitals and Fock matrices,
    # independently of the driver's convergence flag and stored trace.  This
    # catches a stale final state while remaining insensitive to how many
    # extrapolation cycles produced it.
    returned_grad = 0.0
    for fock, coeff, occ, overlap, weight in zip(
        result.fock,
        result.mo_coeffs,
        result.mo_occupations,
        result.overlap,
        result.kpoint_weights,
        strict=True,
    ):
        dm_total = (coeff * np.asarray(occ)[None, :]) @ coeff.conj().T
        s_eval, s_vec = np.linalg.eigh(overlap)
        orth = (s_vec / np.sqrt(s_eval)[None, :]) @ s_vec.conj().T
        fds = fock @ dm_total @ overlap
        error = orth.conj().T @ (fds - fds.conj().T) @ orth
        returned_grad += float(weight) * float(np.linalg.norm(error))
    assert returned_grad < options.conv_tol_grad
    # PySCF 2.13.1 KROHF/GDF, LiH+/STO-3G, 12-bohr cube, (3,1,1),
    # BvK nelec=(6,3), exxdiv='ewald': -7.519325226764678 Ha.
    # The 0.0622 mHa residual is the padded-SR/reciprocal-envelope floor.
    # GitLab #651: the default Ewald alpha on this 12-bohr box is the
    # cutoff bound 0.438 (CRYSTAL's 0.233 leaves erfc(alpha r)/r at 7.5e-5 on
    # the 12-bohr image shell). Measured on the fixer's laptop: CRYSTAL alpha
    # at cutoff 12 gave the old pin -7.5192630688; the bound gives
    # -7.5192632201; both alphas at cutoff 30 give -7.51926711 (within 5e-9
    # of each other). The residual 3.9e-6 is the one-electron / density
    # lattice truncation of a cutoff equal to the box edge, not the split.
    assert result.energy == pytest.approx(-7.5192632201, abs=1.0e-8)
    assert abs(result.energy - (-7.519325226764678)) < 1.0e-4
    assert result.s_squared == pytest.approx(0.75, abs=1.0e-12)
    assert len(result.mo_coeffs) == 3
    assert all(np.allclose(occ, [2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
               for occ in result.mo_occupations)
    matrices = [*result.mo_coeffs, *result.fock, *result.overlap, *result.hcore]
    assert max(float(np.max(np.abs(np.asarray(matrix).imag)))
               for matrix in matrices) > 0.1
    _assert_rohf_result_state_is_consistent(result, kmesh)


def test_multik_rohf_closed_shell_matches_corrected_bipole_rhf():
    """Closed-shell H2 agrees with the corrected Ewald-exchange RHF path."""
    sysp, basis = _h2_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_rhf = vq.run_pbc_bipole_rhf(
        sysp,
        basis,
        km,
        _options(),
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        progress=False,
    )
    r_rohf = vq.run_rohf_periodic_multi_k_ewald3d(
        sysp, basis, km, _options(), omega=0.0, spacing_bohr=0.3
    )
    assert r_rhf.converged and r_rohf.converged
    # 1e-7: both routes bound their default alpha by the 12-bohr cutoff and
    # resolve their reciprocal envelopes for it (#651 / #674); measured
    # -1.1170858289 (BIPOLE) vs -1.1170858285 (EWALD_3D). Pre-#674 BIPOLE
    # read -1.1170669756 and this window was 5e-5.
    assert r_rohf.energy == pytest.approx(r_rhf.energy, abs=1e-7)
    assert abs(r_rohf.s_squared) < 1e-12


def test_multik_rohf_gamma_mesh_matches_corrected_bipole_uhf_for_one_electron():
    """One-electron Γ ROHF agrees with corrected Ewald-exchange UHF."""
    sysp, basis = _h_atom_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_multi = vq.run_rohf_periodic_multi_k_ewald3d(
        sysp, basis, km, _options(), omega=0.0, spacing_bohr=0.3
    )
    r_gamma = vq.run_pbc_bipole_uhf(
        sysp,
        basis,
        km,
        _options(),
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        progress=False,
    )
    assert r_multi.converged and r_gamma.converged
    # 1e-7 (was 1e-5 with a 1e-6 margin): measured -0.4667330009 (BIPOLE
    # UHF) vs -0.4667330007 (EWALD_3D ROHF); pre-#674 BIPOLE read
    # -0.4667239569.
    assert r_multi.energy == pytest.approx(r_gamma.energy, abs=1e-7)
    assert r_multi.s_squared == pytest.approx(0.75, abs=1e-12)


def test_multik_rohf_spin_pure_doublet():
    """Open-shell H-atom box: ⟨S²⟩ = 0.75 exactly on a [1,1,1] mesh."""
    sysp, basis = _h_atom_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rohf_periodic_multi_k_ewald3d(
        sysp, basis, km, _options(), omega=0.5, spacing_bohr=0.3
    )
    assert r.converged
    assert r.s_squared == pytest.approx(0.75, abs=1e-12)
    _assert_rohf_result_state_is_consistent(r, km)


def test_multik_rohf_iteration_cap_returns_evaluated_orbital_state():
    """A one-cycle density guess must not leak an unevaluated MO payload."""
    sysp, basis = _h_atom_box(10.0)
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    options = _options()
    options.max_iter = 1
    options.initial_guess = vq.InitialGuess.SAD
    options.damping = 0.0
    options.lattice_opts.cutoff_bohr = 6.0
    options.lattice_opts.nuclear_cutoff_bohr = 8.0

    result = vq.run_rohf_periodic_multi_k_ewald3d(
        sysp,
        basis,
        km,
        options,
        auto_optimize_truncation=False,
        spacing_bohr=0.6,
        sr_image_precision=None,
        progress=False,
    )

    assert not result.converged
    assert result.n_iter == 1
    assert len(result.scf_trace) == 1
    _assert_rohf_result_state_is_consistent(result, km)


# ---------------------------------------------------------------------------
# GitLab #651: the Ewald split parameter and the real-space image cutoff are
# one truncation contract, and the long-range exchange envelope has to follow
# the alpha actually used.
# ---------------------------------------------------------------------------

from vibeqc.pbc_bipole_common import ewald_alpha_lower_bound  # noqa: E402


class _OptionsWithEwaldOmega(vq.PeriodicRHFOptions):
    """The pybind options carry no ``ewald_omega`` slot and no ``__dict__``;
    the driver reads the override through ``getattr``, so a Python subclass
    is how a test pins the splitting parameter."""


def _options_with_omega(omega: float):
    o = _OptionsWithEwaldOmega()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.damping = 0.3
    o.max_iter = 60
    o.ewald_omega = omega
    return o


# Measured on the fixer's laptop (core rebuilt at 39680600d), H2/STO-3G in
# the 30-bohr box at (1,1,1), lattice cutoff 12 bohr:
#   omega 0.0933 (CRYSTAL 2.8/30), cutoff 12         -1.1170669756 Ha  erfc tail cut
#   omega 0.0933, cutoff 30                          -1.1170858190 Ha
#   omega 0.438 (bound), K_max from the volume only  -1.0789613510 Ha  K_LR cut
# and with the reciprocal envelope resolved for the alpha in use, omega 0.2
# and 0.438 at cutoff 12 and 0.438 at cutoff 30 all give -1.1170858285 Ha:
# the split-invariant total. The corrected BIPOLE RHF of the cross-route
# test above read -1.1170669756 until GitLab #674 bounded its alpha the
# same way (its K_SR erfc arm is cut at the Fock output cells) and scaled
# its reciprocal envelope with the alpha; it now reads -1.1170858289 and
# the two cross-route windows are 1e-7.
_H2_30_CONVERGED = -1.1170858285


def test_multik_rohf_total_is_invariant_to_the_ewald_split_at_fixed_cutoff():
    sysp, basis = _h2_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    bound = ewald_alpha_lower_bound(12.0, 1.0e-12)
    energies = {}
    for omega in (0.2, bound):
        r = vq.run_rohf_periodic_multi_k_ewald3d(
            sysp, basis, km, _options_with_omega(omega), omega=0.0, spacing_bohr=0.3
        )
        assert r.converged
        assert r.omega == pytest.approx(omega)
        energies[omega] = float(r.energy)
    assert max(energies.values()) - min(energies.values()) < 1.0e-8, energies
    for e in energies.values():
        assert e == pytest.approx(_H2_30_CONVERGED, abs=2.0e-8)
    # With nothing requested the driver resolves the bound and lands on the
    # same total; pre-fix it resolved CRYSTAL's 0.0933 and was 1.9e-5 short.
    r = vq.run_rohf_periodic_multi_k_ewald3d(
        sysp, basis, km, _options(), omega=0.0, spacing_bohr=0.3
    )
    assert r.converged
    assert r.omega == pytest.approx(bound, rel=1e-9)
    assert r.energy == pytest.approx(_H2_30_CONVERGED, abs=2.0e-8)


def test_multik_rohf_unbounded_crystal_alpha_is_short_at_the_fixed_cutoff():
    """The pre-fix configuration, requested explicitly, must still be short by
    the erfc tail beyond 12 bohr; otherwise the fixture no longer discriminates
    the defect."""
    sysp, basis = _h2_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rohf_periodic_multi_k_ewald3d(
        sysp, basis, km, _options_with_omega(2.8 / 30.0), omega=0.0, spacing_bohr=0.3
    )
    assert r.converged
    assert r.omega == pytest.approx(2.8 / 30.0)
    assert r.energy - _H2_30_CONVERGED > 1.0e-5, r.energy
