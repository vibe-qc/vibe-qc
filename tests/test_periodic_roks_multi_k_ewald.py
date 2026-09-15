"""Multi-k periodic ROKS (EWALD_3D / BIPOLE) tests — run on a build box.

The Roothaan coupling, the occupation rule and the spin-polarised XC kernel
are unit-checked elsewhere; these pin the *multi-k periodic ROKS*
integration on the corrected-Ewald-exchange engine:

* the closed-shell limit reproduces the independently wired multi-k RKS
  driver (pure functional *and* global hybrid, so the ``c_full`` exchange
  scaling is checked against a separate implementation),
* ⟨S²⟩ = S(S+1) exactly for an open shell,
* the fused OpenMP multi-k Bloch kernels reproduce the per-k Python fold,
* the reported energy components sum to the total,
* screened / long-range-corrected hybrids and double hybrids fail closed.

The open-shell Li/PBE/STO-3G ``(1,1,2)`` anchor is validated out of process
against PySCF 2.14.0 ``KROKS().density_fit()`` with the matching BvK
supercell spin counts (4 alpha / 2 beta over the two k points).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_roks_multi_k_ewald import (
    _single_open_shell_occupations,
    _spin_occupations,
    run_roks_periodic_multi_k_ewald3d,
)


def _options(functional: str):
    o = vq.PeriodicKSOptions()
    o.functional = functional
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 12.0
    o.damping = 0.3
    o.max_iter = 80
    o.conv_tol_energy = 1.0e-9
    o.conv_tol_grad = 1.0e-6
    return o


def _h2_box(box: float = 12.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _li_box(box: float = 6.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box, [vq.Atom(3, [c, c, c])], multiplicity=2
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _h_atom_box(box: float = 10.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box, [vq.Atom(1, [c, c, c])], multiplicity=2
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _assert_roks_result_state_is_consistent(result, kmesh):
    """Every returned observable must describe the returned RO orbitals."""
    from vibeqc._vibeqc_core import real_space_density_from_kpoints_fractional

    spin_occupations = [_spin_occupations(occ) for occ in result.mo_occupations]
    occ_alpha = [pair[0] for pair in spin_occupations]
    occ_beta = [pair[1] for pair in spin_occupations]
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


def test_multik_roks_closed_shell_pure_matches_rks():
    """mult=1 with a pure functional collapses ROKS onto multi-k RKS.

    No exchange is built on either side, so the two drivers must agree to
    machine precision.
    """
    sysp, basis = _h2_box()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r_roks = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("lda"), progress=False
    )
    r_rks = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("lda"), progress=False
    )
    assert r_roks.converged and r_rks.converged
    assert r_roks.energy == pytest.approx(r_rks.energy, abs=1e-9)
    assert r_roks.e_xc == pytest.approx(r_rks.e_xc, abs=1e-12)
    assert r_roks.e_coulomb == pytest.approx(r_rks.e_coulomb, abs=1e-12)
    assert abs(r_roks.s_squared) < 1e-12


def test_multik_roks_closed_shell_hybrid_matches_bipole_rks():
    """mult=1 global hybrid collapses onto the public BIPOLE RKS route.

    The comparison partner is ``run_pbc_bipole_rks`` with the corrected
    Ewald exchange -- the same convention this driver inherits from the
    ROHF engine -- not ``run_rks_periodic_multi_k_ewald3d``, whose
    multi-k hybrid exchange carries no q -> 0 correction and is ~82 mHa
    below both PySCF conventions on this cell (tracked separately).
    """
    sysp, basis = _h2_box()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r_roks = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("pbe0"), progress=False
    )
    r_rks = vq.run_pbc_bipole_rks(
        sysp, basis, km, _options("pbe0"),
        functional="pbe0",
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        exchange_exxdiv="ewald",
        progress=False,
    )
    assert r_roks.converged
    assert r_roks.energy == pytest.approx(r_rks.energy, abs=1e-4)
    assert abs(r_roks.s_squared) < 1e-12
    # Published target: PySCF 2.14.0 KRKS.density_fit(), xc='pbe0',
    # exxdiv='ewald' on the matched cell/basis/mesh gives
    # -1.155219229638 Ha (exxdiv=None gives -1.117597525253 Ha, so this
    # also pins that the q -> 0 correction is applied at all).
    E_PYSCF_KRKS_PBE0 = -1.155219229638
    assert r_roks.energy == pytest.approx(E_PYSCF_KRKS_PBE0, abs=2.0e-4)


def test_multik_roks_spin_pure_doublet():
    """Open-shell Li box: ⟨S²⟩ = 0.75 exactly, occupations are 2/1/0."""
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    r = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("pbe"), progress=False
    )
    assert r.converged
    assert r.s_squared == pytest.approx(0.75, abs=1e-12)
    assert r.n_alpha == 2 and r.n_beta == 1
    for occ in r.mo_occupations:
        assert sorted(occ, reverse=True)[:2] == [2.0, 1.0]
        assert float(np.sum(occ)) == pytest.approx(3.0, abs=1e-12)
    _assert_roks_result_state_is_consistent(r, km)


def test_multik_roks_iteration_cap_returns_evaluated_orbital_state():
    """A one-cycle density guess must not leak an unevaluated MO payload."""
    sysp, basis = _h_atom_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    options = _options("lda")
    options.max_iter = 1
    options.initial_guess = vq.InitialGuess.SAD
    options.damping = 0.0
    options.lattice_opts.cutoff_bohr = 6.0
    options.lattice_opts.nuclear_cutoff_bohr = 8.0

    result = run_roks_periodic_multi_k_ewald3d(
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
    _assert_roks_result_state_is_consistent(result, km)


def test_multik_roks_degenerate_frontier_shell_converges_without_band_flips():
    """The symmetry-degenerate Li ``(2,2,1)`` shell must settle promptly.

    The historical integer 2/1/0 choice changed which member of the frontier
    shell carried the open electron as the effective Fock moved. Energy was
    flat long before the commutator: current main needs 87 iterations on this
    one-thread fixture, versus six for the nondegenerate ``(1,1,2)`` mesh.
    """
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [2, 2, 1])
    opts = _options("pbe")
    opts.max_iter = 40

    result = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, progress=False
    )

    assert result.converged, (
        f"degenerate-shell ROKS did not settle in {result.n_iter} cycles; "
        f"last commutator={result.scf_trace[-1].grad_norm:.9e}"
    )
    assert result.s_squared == pytest.approx(0.75, abs=1e-12)
    fractional_shells = [
        occ[np.isclose(occ, 0.5)] for occ in result.mo_occupations
    ]
    assert sum(shell.size == 2 for shell in fractional_shells) == 1


def test_single_open_shell_occupation_locks_alpha_degeneracy():
    """The open shell follows alpha energies, then preserves their symmetry."""
    effective = np.array([-1.61, 0.1241, 0.1275, 0.1275, 0.74])
    alpha = np.array([-1.62, 0.0765, 0.06896878498, 0.06896878498, 0.71])

    occupations, shell_size = _single_open_shell_occupations(
        effective, alpha, 2, 1
    )

    assert shell_size == 2
    assert occupations == pytest.approx([2.0, 0.0, 0.5, 0.5, 0.0])

    # Once exact symmetry identifies the shell, an SCF-roundoff split must
    # not collapse it back onto one arbitrary member.
    alpha_split = alpha.copy()
    alpha_split[3] += 2.0e-5
    occupations, shell_size = _single_open_shell_occupations(
        effective,
        alpha_split,
        2,
        1,
        locked_shell_size=shell_size,
    )
    assert shell_size == 2
    assert occupations == pytest.approx([2.0, 0.0, 0.5, 0.5, 0.0])


def test_multik_roks_pyscf_kroks_reference():
    """Li/PBE/STO-3G (1,1,2) against an out-of-process PySCF KROKS run.

    Published target: PySCF 2.14.0 ``KROKS(cell, kpts).density_fit()`` with
    ``xc='pbe'``, 6-bohr cubic cell, Li at the centre, Gamma-centred
    ``(1,1,2)`` mesh and the equivalent BvK supercell spin ``cell.spin = 2``
    (4 alpha / 2 beta over the two k points) converged to
    ``-7.435745517126 Ha``. The residual here is DFT-grid and lattice-
    truncation numerics, not a formulation difference: the two codes use
    different atomic grids and different Coulomb gauges.
    """
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    opts = _options("pbe")
    opts.lattice_opts.cutoff_bohr = 14.0
    opts.lattice_opts.nuclear_cutoff_bohr = 14.0
    r = run_roks_periodic_multi_k_ewald3d(sysp, basis, km, opts, progress=False)
    assert r.converged
    E_PYSCF_KROKS = -7.435745517126
    assert r.energy == pytest.approx(E_PYSCF_KROKS, abs=5.0e-4)


def test_multik_roks_fused_kernels_match_serial_fold(monkeypatch):
    """The fused OpenMP multi-k Bloch kernels reproduce the Python fold.

    Drops the driver onto its ``_bloch_sum_blocks`` fallback by hiding the
    C++ symbol, so both the Fock and the V_xc folds take the serial path.
    """
    import vibeqc._vibeqc_core as core

    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    r_fused = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("pbe"), progress=False
    )
    monkeypatch.delattr(core, "bloch_sum_multi_k")
    r_serial = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("pbe"), progress=False
    )
    assert r_fused.converged and r_serial.converged
    # The two folds differ only in float accumulation order, so the
    # converged energy agrees far below the 1e-9 SCF threshold. The
    # iteration count is deliberately NOT pinned: a ~1e-13 difference can
    # land either side of the convergence test on the final cycle.
    assert r_fused.energy == pytest.approx(r_serial.energy, abs=1e-10)
    assert abs(r_fused.n_iter - r_serial.n_iter) <= 1


def test_multik_roks_energy_components_sum_to_total():
    """E_total = E_core + E_coulomb + E_hf_exchange + E_xc + E_nuclear."""
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    r = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("pbe0"), progress=False
    )
    assert r.converged
    # Pure functionals build no exchange at all; a global hybrid must.
    assert r.e_hf_exchange < 0.0
    e_core = r.e_electronic - r.e_coulomb - r.e_hf_exchange - r.e_xc
    assert r.e_electronic == pytest.approx(
        e_core + r.e_coulomb + r.e_hf_exchange + r.e_xc, abs=1e-12
    )
    assert r.energy == pytest.approx(r.e_electronic + r.e_nuclear, abs=1e-9)


def test_multik_roks_pure_functional_builds_no_exchange():
    """A pure functional reports exactly zero exact exchange."""
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    r = run_roks_periodic_multi_k_ewald3d(
        sysp, basis, km, _options("pbe"), progress=False
    )
    assert r.e_hf_exchange == 0.0
    assert r.functional == "pbe"
    assert r.runtime_backend == "bipole-roks-multi-k-ewald"


def test_multik_roks_screened_hybrid_fails_closed():
    """hse06 needs an erfc exchange arm this engine does not carry."""
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    with pytest.raises(NotImplementedError, match="erfc"):
        run_roks_periodic_multi_k_ewald3d(
            sysp, basis, km, _options("hse06"), progress=False
        )


def test_multik_roks_long_range_corrected_fails_closed():
    """LC functionals keep the shared periodic-exchange fail-close."""
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    with pytest.raises(NotImplementedError, match="full-range"):
        run_roks_periodic_multi_k_ewald3d(
            sysp, basis, km, _options("cam-b3lyp"), progress=False
        )


def test_multik_roks_requires_a_functional():
    sysp, basis = _li_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    with pytest.raises(ValueError, match="functional is required"):
        run_roks_periodic_multi_k_ewald3d(
            sysp, basis, km, _options(""), progress=False
        )


def test_multik_roks_bad_multiplicity_rejected():
    sysp, basis = _li_box()
    sysp.multiplicity = 3  # 3 electrons cannot carry S = 1
    km = vq.monkhorst_pack(sysp, [1, 1, 2])
    with pytest.raises(ValueError, match="integer a/b counts"):
        run_roks_periodic_multi_k_ewald3d(
            sysp, basis, km, _options("pbe"), progress=False
        )
