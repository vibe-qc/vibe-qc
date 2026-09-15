"""Phase C1a tests: Saunders-Hillier level shift on periodic SCF.

Contracts exercised:

1. **Inertness at convergence** — for both the Γ-Ewald and multi-k
   Ewald drivers, ``level_shift = 0.3`` reproduces the ``level_shift =
   0`` energy to ~µHa. The shift is by construction inert at the SCF
   fixed point: it shifts virtual eigenvalues during iteration but
   leaves the converged density unchanged.

2. **MO eigenvalues are physical at convergence** — the reported
   ``mo_energies`` are the un-shifted physical orbital energies (the
   final self-consistency pass omits the shift), so a converged
   calculation produces the same orbital spectrum regardless of the
   shift value used during iteration.

3. **Default behavior preserved** — the default ``level_shift = 0``
   means a calculation that doesn't set the field reproduces the
   pre-C1a SCF dynamics exactly.

4. **Field is exposed on all three option structs** — PeriodicRHFOptions,
   PeriodicSCFOptions, PeriodicKSOptions all have ``level_shift`` as
   a settable double (default 0.0).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2_chain(a: float = 8.0):
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * a,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _gamma_ewald_options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.damping = 0.5
    o.max_iter = 80
    o.use_diis = True
    return o


def _multi_k_options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.damping = 0.5
    o.max_iter = 60
    o.use_diis = True
    o.conv_tol_grad = 1e-5    # multi-k orbital-degeneracy plateau
    return o


# ---------------------------------------------------------------------------
# 1. Field exposure on all three option structs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    vq.PeriodicRHFOptions,
    vq.PeriodicSCFOptions,
    vq.PeriodicKSOptions,
])
def test_level_shift_field_default_is_zero(cls):
    o = cls()
    assert hasattr(o, "level_shift")
    assert o.level_shift == 0.0
    o.level_shift = 0.5
    assert o.level_shift == 0.5


@pytest.mark.parametrize("cls", [
    vq.PeriodicRHFOptions,
    vq.PeriodicSCFOptions,
    vq.PeriodicKSOptions,
])
def test_level_shift_warmup_and_schedule_fields(cls):
    """The full unified option set (warm-up + explicit schedule) is
    exposed on every periodic options struct, matching the molecular
    structs field-for-field."""
    o = cls()
    assert o.level_shift_warmup_cycles == -1
    o.level_shift_warmup_cycles = 3
    assert o.level_shift_warmup_cycles == 3
    assert list(o.level_shift_schedule) == []
    o.level_shift_schedule = [0.5, 0.4, 0.0]
    assert list(o.level_shift_schedule) == [0.5, 0.4, 0.0]


def test_gamma_ewald_schedule_inertness_at_convergence():
    """An explicit decaying schedule on the Γ-Ewald driver reaches the
    same total energy as no shift; the schedule damps the path only.
    Exercises the unified opts.level_shift_schedule field on the periodic
    side (empty ⇒ legacy behaviour; non-empty ⇒ shared C++ helper)."""
    from vibeqc.level_shift_schedule import LevelShiftSchedule
    sysp, basis = _h2_chain(a=8.0)

    opts = _gamma_ewald_options()
    r0 = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )

    opts_s = _gamma_ewald_options()
    LevelShiftSchedule([0.4, 0.3, 0.2, 0.1, 0.0]).apply_to(opts_s)
    r1 = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts_s, omega=0.5, spacing_bohr=0.3,
    )
    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Γ-Ewald: same energy with and without level shift
# ---------------------------------------------------------------------------

def test_gamma_ewald_level_shift_inertness_at_convergence():
    """level_shift = 0.3 reaches the same total energy as level_shift = 0
    on a tight H2 chain. The shift is "inert" at the SCF fixed point."""
    sysp, basis = _h2_chain(a=8.0)
    opts = _gamma_ewald_options()

    opts.level_shift = 0.0
    r0 = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.3
    r1 = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)


def test_gamma_ewald_mo_energies_physical_at_convergence():
    """Reported MO eigenvalues are the *un-shifted* physical energies.
    Two SCFs run with different ``level_shift`` values during iteration
    must produce the same physical orbital spectrum at convergence."""
    sysp, basis = _h2_chain(a=8.0)
    opts = _gamma_ewald_options()

    opts.level_shift = 0.0
    r0 = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.3
    r1 = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    eps0 = np.sort(np.real(r0.mo_energies))
    eps1 = np.sort(np.real(r1.mo_energies))
    np.testing.assert_allclose(eps0, eps1, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. Multi-k: same energy with and without level shift
# ---------------------------------------------------------------------------

def test_multi_k_level_shift_inertness_at_convergence():
    sysp, basis = _h2_chain(a=10.0)
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _multi_k_options()

    opts.level_shift = 0.0
    r0 = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.3
    r1 = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-6)


def test_multi_k_mo_eigenvalues_unshifted_at_convergence():
    """At convergence each k-point's MO eigenvalues must match between
    shifted and un-shifted runs (within numerical noise from slightly
    different convergence paths)."""
    sysp, basis = _h2_chain(a=10.0)
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    opts = _multi_k_options()

    opts.level_shift = 0.0
    r0 = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.3
    r1 = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    for idx in range(len(km.kpoints)):
        eps0 = np.sort(np.real(r0.mo_energies[idx]))
        eps1 = np.sort(np.real(r1.mo_energies[idx]))
        np.testing.assert_allclose(eps0, eps1, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Default level_shift=0 reproduces pre-C1a behavior exactly
# ---------------------------------------------------------------------------

def test_default_level_shift_zero_reproduces_baseline():
    """An options object that doesn't touch level_shift must produce
    identical SCF dynamics to one that explicitly sets level_shift=0."""
    sysp, basis = _h2_chain(a=10.0)
    opts_implicit = _gamma_ewald_options()   # never touches level_shift
    opts_explicit = _gamma_ewald_options()
    opts_explicit.level_shift = 0.0

    r_imp = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts_implicit, omega=0.5, spacing_bohr=0.3,
    )
    r_exp = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts_explicit, omega=0.5, spacing_bohr=0.3,
    )
    assert r_imp.n_iter == r_exp.n_iter
    assert r_imp.energy == pytest.approx(r_exp.energy, abs=1e-12)


# ---------------------------------------------------------------------------
# Regression: dispatcher must forward level_shift to the backend
# ---------------------------------------------------------------------------
#
# The dispatchers ``run_rhf_periodic_scf`` /
# ``run_rhf_periodic_gamma_scf`` translate ``PeriodicSCFOptions ↔
# PeriodicRHFOptions`` field-by-field. A regression in the
# translation helpers (commit 714b3a3) silently dropped
# ``level_shift``: the bare backend respected it but the recommended
# dispatcher entry points produced byte-identical SCF behavior for
# any ``level_shift`` value. The original level-shift tests below
# only exercised the bare backends, so the regression slipped
# through. These tests close that gap by going through the
# dispatcher and verifying the iteration-count signature of an
# active level shift.


def _gamma_ewald_options_for_dispatcher():
    """Same options as ``_gamma_ewald_options`` but with EWALD_3D
    coulomb_method set so the dispatcher routes to the Ewald backend.
    Tighter cell (a = 4.5 bohr) so the iteration count differs visibly
    between level-shift values — on the loose 8-bohr cell DIIS
    converges in 2 iters regardless."""
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.damping = 0.5
    o.max_iter = 60
    o.use_diis = True
    return o


def test_dispatcher_gamma_scf_forwards_level_shift():
    """The Γ-only dispatcher must converge to the same energy with /
    without level shift, and the iteration count must reflect the
    shift's presence (no-op forwarding would give iter-count
    invariance — the symptom of the original regression)."""
    sysp, basis = _h2_chain(a=8.0)
    opts = _gamma_ewald_options_for_dispatcher()

    opts.level_shift = 0.0
    r0 = vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.5
    rL = vq.run_rhf_periodic_gamma_scf(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r0.converged and rL.converged
    # Same converged energy.
    assert rL.energy == pytest.approx(r0.energy, abs=1e-9)


def test_dispatcher_multi_k_scf_forwards_level_shift():
    """Same regression check on the multi-k dispatcher."""
    sysp, basis = _h2_chain(a=10.0)
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _gamma_ewald_options_for_dispatcher()
    opts.conv_tol_grad = 1e-5    # multi-k orbital-degeneracy plateau

    opts.level_shift = 0.0
    r0 = vq.run_rhf_periodic_scf(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.5
    rL = vq.run_rhf_periodic_scf(
        sysp, basis, km, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r0.converged and rL.converged
    assert rL.energy == pytest.approx(r0.energy, abs=1e-7)


def test_dispatcher_translation_preserves_level_shift_field():
    """Direct test of the field-by-field option translation: the
    dispatcher's internal ``_copy_options_to_*`` helpers must copy
    convergence-control fields. Imported via the private
    helper for a tight regression guard."""
    from vibeqc.periodic_rhf_dispatch import (
        _copy_options_to_rhf, _copy_options_to_scf,
    )
    src = vq.PeriodicSCFOptions()
    src.level_shift = 0.42
    src.fock_mixing = 0.30
    src.smearing_temperature = 0.013
    src.quadratic_fallback_iter = 17
    src.quadratic_fallback_shift = 0.25
    src.quadratic_fallback_max_step = 0.07
    rhf_copy = _copy_options_to_rhf(src)
    scf_copy = _copy_options_to_scf(src)
    assert rhf_copy.level_shift == 0.42
    assert scf_copy.level_shift == 0.42
    assert rhf_copy.fock_mixing == 0.30
    assert scf_copy.fock_mixing == 0.30
    assert rhf_copy.smearing_temperature == pytest.approx(0.013)
    assert scf_copy.smearing_temperature == pytest.approx(0.013)
    assert rhf_copy.quadratic_fallback_iter == 17
    assert scf_copy.quadratic_fallback_iter == 17
    assert rhf_copy.quadratic_fallback_shift == pytest.approx(0.25)
    assert scf_copy.quadratic_fallback_shift == pytest.approx(0.25)
    assert rhf_copy.quadratic_fallback_max_step == pytest.approx(0.07)
    assert scf_copy.quadratic_fallback_max_step == pytest.approx(0.07)


def test_dispatcher_translation_preserves_accelerator_and_dynamic_damping():
    """Regression guard for the option-drop bug fixed alongside the
    level-shift forwarder: ``_copy_options_to_rhf`` / ``_copy_options_to_scf``
    must also forward the SCF-accelerator + dynamic-damping fields the
    C++ direct kernels consume. A silent drop here meant a caller's
    ``scf_accelerator = EDIIS_DIIS`` request reached the kernel as the
    default DIIS, with no warning."""
    from vibeqc.periodic_rhf_dispatch import (
        _copy_options_to_rhf, _copy_options_to_scf,
    )
    src = vq.PeriodicSCFOptions()
    src.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS
    src.ediis_diis_switch_threshold = 0.073
    src.dynamic_damping = True
    src.dynamic_damping_min = 0.15
    src.dynamic_damping_max = 0.88
    rhf_copy = _copy_options_to_rhf(src)
    scf_copy = _copy_options_to_scf(src)
    assert rhf_copy.scf_accelerator == vq.SCFAccelerator.EDIIS_DIIS
    assert scf_copy.scf_accelerator == vq.SCFAccelerator.EDIIS_DIIS
    assert rhf_copy.ediis_diis_switch_threshold == pytest.approx(0.073)
    assert scf_copy.ediis_diis_switch_threshold == pytest.approx(0.073)
    assert rhf_copy.dynamic_damping is True
    assert scf_copy.dynamic_damping is True
    assert rhf_copy.dynamic_damping_min == pytest.approx(0.15)
    assert scf_copy.dynamic_damping_min == pytest.approx(0.15)
    assert rhf_copy.dynamic_damping_max == pytest.approx(0.88)
    assert scf_copy.dynamic_damping_max == pytest.approx(0.88)


def test_ks_dispatcher_translation_preserves_accelerator_and_dynamic_damping():
    """Same regression for ``_copy_ks_options`` on the KS side."""
    from vibeqc.periodic_ks_dispatch import _copy_ks_options
    src = vq.PeriodicKSOptions()
    src.scf_accelerator = vq.SCFAccelerator.EDIIS
    src.ediis_diis_switch_threshold = 0.055
    src.dynamic_damping = True
    src.dynamic_damping_min = 0.20
    src.dynamic_damping_max = 0.80
    out = _copy_ks_options(src)
    assert out.scf_accelerator == vq.SCFAccelerator.EDIIS
    assert out.ediis_diis_switch_threshold == pytest.approx(0.055)
    assert out.dynamic_damping is True
    assert out.dynamic_damping_min == pytest.approx(0.20)
    assert out.dynamic_damping_max == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# Periodic UHF: dead-code regression guard for ``run_uhf_periodic_gamma_ewald3d``
# ---------------------------------------------------------------------------
#
# Until this commit the Γ-Ewald UHF kernel read ``opts.level_shift`` but
# never applied it: the variable was bound and then ignored on the path
# from F to ``diagonalize``. The bare-backend tests above only cover RHF
# / RKS / multi-k UHF (which already wired the shift); this section
# closes the gap on single-k UHF with the same contract that
# ``test_molecular_level_shift.py`` pins for molecular UHF.


def _h_atom_doublet_periodic(box: float = 30.0):
    """One H atom in a vacuum supercell — simplest open-shell periodic
    case. Matches the fixture used by ``test_periodic_uhf_ewald.py``."""
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h4_chain_triplet(a: float = 8.0):
    """H4 chain in a cubic supercell, multiplicity 3 (2 unpaired α
    electrons). 4 electrons in 4 STO-3G basis functions is heavy enough
    that DIIS-off SCF takes O(50) iterations, so a non-zero level shift
    visibly perturbs ``n_iter``. The trivial open-shell systems (H atom,
    triplet O / STO-3G) converge in 2 iters regardless and provide no
    signal."""
    atoms = [vq.Atom(1, [0, 0, z]) for z in (0.0, 1.4, 2.8, 4.2)]
    sysp = vq.PeriodicSystem(3, np.eye(3) * a, atoms)
    sysp.multiplicity = 3
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _gamma_ewald_uhf_options():
    o = vq.PeriodicRHFOptions()    # UHF reuses PeriodicRHFOptions
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.damping = 0.3
    o.max_iter = 80
    o.use_diis = True
    return o


def test_uhf_gamma_ewald_level_shift_inertness_at_convergence():
    """level_shift = 0.3 reaches the same UHF total energy and ⟨S²⟩
    as level_shift = 0 on a doublet H atom in a vacuum supercell."""
    sysp, basis = _h_atom_doublet_periodic()
    opts = _gamma_ewald_uhf_options()

    opts.level_shift = 0.0
    r0 = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.3
    r1 = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)
    assert r0.s_squared == pytest.approx(r1.s_squared, abs=1e-7)


def test_uhf_gamma_ewald_mo_energies_physical_at_convergence():
    """Reported per-spin MO eigenvalues are the un-shifted physical
    orbital energies (the final consistency pass diagonalises the
    un-shifted F)."""
    sysp, basis = _h_atom_doublet_periodic()
    opts = _gamma_ewald_uhf_options()

    opts.level_shift = 0.0
    r0 = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    opts.level_shift = 0.3
    r1 = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    eps_a0 = np.sort(np.real(r0.mo_energies_alpha))
    eps_a1 = np.sort(np.real(r1.mo_energies_alpha))
    eps_b0 = np.sort(np.real(r0.mo_energies_beta))
    eps_b1 = np.sort(np.real(r1.mo_energies_beta))
    np.testing.assert_allclose(eps_a0, eps_a1, atol=1e-7)
    np.testing.assert_allclose(eps_b0, eps_b1, atol=1e-7)


def test_uhf_gamma_ewald_default_level_shift_zero_reproduces_baseline():
    """Implicit-default UHF run equals an explicit ``level_shift = 0``
    run byte-for-byte on energy and iteration count."""
    sysp, basis = _h_atom_doublet_periodic()
    opts_implicit = _gamma_ewald_uhf_options()      # never sets level_shift
    opts_explicit = _gamma_ewald_uhf_options()
    opts_explicit.level_shift = 0.0

    r_imp = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts_implicit, omega=0.5, spacing_bohr=0.3,
    )
    r_exp = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts_explicit, omega=0.5, spacing_bohr=0.3,
    )
    assert r_imp.n_iter == r_exp.n_iter
    assert r_imp.energy == pytest.approx(r_exp.energy, abs=1e-12)


def test_uhf_gamma_ewald_level_shift_perturbs_iteration_dynamics():
    """A non-zero level shift must actually change the UHF SCF trajectory
    (different ``n_iter``) even though it converges to the same fixed
    point. This is the explicit guard against the dead-code regression
    where ``opts.level_shift`` was read but never applied to F before
    diagonalisation. DIIS is disabled to isolate the level-shift effect.

    Mirrors ``test_level_shift_perturbs_iteration_dynamics`` in
    ``tests/test_molecular_level_shift.py``. Uses an H4 chain triplet
    (mult=3) — the H-atom and triplet-O / STO-3G fixtures converge in
    2 iters regardless and provide no trajectory signal."""
    sysp, basis = _h4_chain_triplet()

    opts0 = _gamma_ewald_uhf_options()
    opts0.use_diis = False
    opts0.damping = 0.5    # bare-bones damping is needed without DIIS
    opts0.max_iter = 500
    r0 = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, opts0, omega=0.5, spacing_bohr=0.3,
    )

    optsL = _gamma_ewald_uhf_options()
    optsL.use_diis = False
    optsL.damping = 0.5
    # The shifted no-DIIS trajectory needs 527 cycles to satisfy the physical
    # terminal commutator check. The older 500-cycle cap only looked converged
    # before the returned-density fixed-point verification was added.
    optsL.max_iter = 600
    optsL.level_shift = 0.5
    rL = vq.run_uhf_periodic_gamma_ewald3d(
        sysp, basis, optsL, omega=0.5, spacing_bohr=0.3,
    )

    # Same fixed point.
    assert r0.converged and rL.converged
    assert r0.energy == pytest.approx(rL.energy, abs=1e-8)
    # Different trajectory — without DIIS to compensate, the shift
    # changes how fast we get there. If this assertion fires it means
    # ``level_shift`` is being read but not applied (the original bug).
    assert r0.n_iter != rL.n_iter


# ---------------------------------------------------------------------------
# C++ direct-truncated periodic drivers (``run_rhf_periodic_gamma`` Γ,
# ``run_rhf_periodic`` multi-k, ``run_rks_periodic`` multi-k KS).
#
# Regression guard. Until v0.15.x these three drivers exposed
# ``level_shift`` / ``level_shift_warmup_cycles`` / ``level_shift_schedule``
# on their option structs but never read them: the shift was a silent no-op.
# Every C++ periodic driver must satisfy both halves of the contract:
#   * HONORED - the shift changes the SCF path;
#   * INERT   - it never moves the converged fixed point.
# ---------------------------------------------------------------------------

def _h2_631g_cell(a: float = 12.0):
    """H2 in a wide cubic cell, 6-31G.

    The basis matters. In STO-3G an H2 unit cell carries one basis function
    per atom, so both MOs (σ_g, σ_u) are fixed by symmetry alone: a level
    shift moves their eigenvalues but cannot rotate the eigenvectors, and the
    SCF path is provably identical with and without a shift. Such a system
    cannot distinguish "shift applied" from "shift ignored" and is useless as
    a regression probe. 6-31G gives each symmetry block two functions, so the
    occupied-virtual mixing angle is a genuine degree of freedom that the
    Saunders-Hillier shift acts on.
    """
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * a, [vq.Atom(1, [0.0, 0.0, 0.0]),
                           vq.Atom(1, [0.0, 0.0, 1.4])])
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "6-31g")


def _cxx_lattice(o):
    o.lattice_opts.cutoff_bohr = 10.0
    o.lattice_opts.nuclear_cutoff_bohr = 14.0
    return o


def _cxx_path_opts(o):
    """Fixed-length unconverged trace: isolates the shift's effect on the path.

    Comparing converged iteration counts is a weak probe (two different paths
    can land on the same count). Comparing a fixed-length trace of energies is
    exact: any difference at all means the shift reached the Fock matrix.
    DIIS and dynamic damping are off so nothing can compensate for it.
    """
    _cxx_lattice(o)
    o.use_diis = False
    o.dynamic_damping = False
    o.damping = 0.5
    o.conv_tol_energy = 1e-14   # unreachable: run the full max_iter
    o.conv_tol_grad = 1e-12
    o.max_iter = 5
    return o


def _cxx_converged_opts(o):
    _cxx_lattice(o)
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    o.max_iter = 200
    return o


def _trace(result):
    return [it.energy for it in result.scf_trace]


def _assert_path_changed(unshifted, shifted):
    # Iteration 1's energy is built from the guess density, before any shifted
    # diagonalization has fed back into it, so it must agree. The divergence
    # begins at iteration 2.
    assert unshifted[0] == pytest.approx(shifted[0], abs=1e-10)
    assert any(abs(a - b) > 1e-8 for a, b in zip(unshifted[1:], shifted[1:])), (
        "level_shift did not change the SCF path: it is being accepted on the "
        "options struct but never applied to the Fock matrix"
    )


def test_cxx_gamma_level_shift_changes_scf_path():
    sysp, basis = _h2_631g_cell()
    r0 = vq.run_rhf_periodic_gamma(
        sysp, basis, _cxx_path_opts(vq.PeriodicRHFOptions()))
    oL = _cxx_path_opts(vq.PeriodicRHFOptions())
    oL.level_shift = 0.5
    oL.level_shift_warmup_cycles = 0    # persistent
    rL = vq.run_rhf_periodic_gamma(sysp, basis, oL)
    _assert_path_changed(_trace(r0), _trace(rL))


def test_cxx_gamma_level_shift_schedule_changes_scf_path():
    """An explicit decay schedule must also reach the C++ Γ driver."""
    sysp, basis = _h2_631g_cell()
    r0 = vq.run_rhf_periodic_gamma(
        sysp, basis, _cxx_path_opts(vq.PeriodicRHFOptions()))
    oS = _cxx_path_opts(vq.PeriodicRHFOptions())
    oS.level_shift_schedule = [0.6, 0.4, 0.2, 0.0]
    rS = vq.run_rhf_periodic_gamma(sysp, basis, oS)
    _assert_path_changed(_trace(r0), _trace(rS))


def test_cxx_multi_k_rhf_level_shift_changes_scf_path():
    sysp, basis = _h2_631g_cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r0 = vq.run_rhf_periodic(
        sysp, basis, km, _cxx_path_opts(vq.PeriodicSCFOptions()))
    oL = _cxx_path_opts(vq.PeriodicSCFOptions())
    oL.level_shift = 0.5
    oL.level_shift_warmup_cycles = 0
    rL = vq.run_rhf_periodic(sysp, basis, km, oL)
    _assert_path_changed(_trace(r0), _trace(rL))


@pytest.mark.parametrize("shift,schedule", [(0.3, []), (0.0, [0.6, 0.4, 0.2])])
def test_cxx_gamma_level_shift_inert_at_convergence(shift, schedule):
    sysp, basis = _h2_631g_cell()
    r0 = vq.run_rhf_periodic_gamma(
        sysp, basis, _cxx_converged_opts(vq.PeriodicRHFOptions()))
    oL = _cxx_converged_opts(vq.PeriodicRHFOptions())
    oL.level_shift = shift
    oL.level_shift_warmup_cycles = 0
    if schedule:
        oL.level_shift_schedule = schedule
    rL = vq.run_rhf_periodic_gamma(sysp, basis, oL)

    assert r0.converged and rL.converged
    assert r0.energy == pytest.approx(rL.energy, abs=1e-8)


def test_cxx_multi_k_rhf_level_shift_inert_at_convergence():
    sysp, basis = _h2_631g_cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    r0 = vq.run_rhf_periodic(
        sysp, basis, km, _cxx_converged_opts(vq.PeriodicSCFOptions()))
    oL = _cxx_converged_opts(vq.PeriodicSCFOptions())
    oL.level_shift = 0.3
    oL.level_shift_warmup_cycles = 0
    rL = vq.run_rhf_periodic(sysp, basis, km, oL)

    assert r0.converged and rL.converged
    assert r0.energy == pytest.approx(rL.energy, abs=1e-8)


def test_cxx_multi_k_rks_level_shift_honored_and_inert():
    sysp, basis = _h2_631g_cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    o0 = _cxx_converged_opts(vq.PeriodicKSOptions())
    o0.functional = "pbe"
    o0.conv_tol_grad = 1e-6
    r0 = vq.run_rks_periodic(sysp, basis, km, o0)
    oL = _cxx_converged_opts(vq.PeriodicKSOptions())
    oL.functional = "pbe"
    oL.conv_tol_grad = 1e-6
    oL.level_shift = 0.3
    oL.level_shift_warmup_cycles = 0
    rL = vq.run_rks_periodic(sysp, basis, km, oL)

    assert r0.converged and rL.converged
    assert r0.energy == pytest.approx(rL.energy, abs=1e-8)   # inert
    # A persistent shift slows the tail of this KS run; if the shift were
    # ignored the two counts would agree exactly.
    assert r0.n_iter != rL.n_iter                            # honored


# ---------------------------------------------------------------------------
# Periodic UKS Γ-Ewald: level-shift coverage + the operator convention
#
# ``run_uks_periodic_gamma_ewald3d`` had no level-shift test of any kind,
# which is how it shipped with the wrong Saunders-Hillier coefficient. It
# applied ``F_s + b*S - (b/2)*S D_s S`` to *spin* densities. D_alpha and
# D_beta are idempotent in the S metric (D_s S D_s = D_s), so S D_s S is the
# projector onto the occupied manifold and the coefficient must be 1:
# occupied eigenvalues stay put, virtuals rise by b. With b/2 the occupied
# block rose by b/2 as well, so the gap opened by b/2 instead of b. The SCF
# still converged to the same fixed point, which is exactly why no energy
# assertion anywhere could have caught it. Only the closed-shell drivers,
# which carry the total density D = 2P, legitimately use b/2.
# ---------------------------------------------------------------------------

def _uks_gamma_ewald_options(shift: float = 0.0):
    o = vq.PeriodicKSOptions()
    o.functional = "lda"
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.use_diis = False          # isolate the shift
    o.dynamic_damping = False
    o.damping = 0.3
    o.max_iter = 120
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-6
    o.level_shift = shift
    o.level_shift_warmup_cycles = 0    # persistent
    return o


def test_uks_gamma_ewald_level_shift_honored_and_inert():
    sysp, basis = _h4_chain_triplet()
    r0 = vq.run_uks_periodic_gamma_ewald3d(
        sysp, basis, _uks_gamma_ewald_options(0.0))
    rL = vq.run_uks_periodic_gamma_ewald3d(
        sysp, basis, _uks_gamma_ewald_options(0.5))

    assert r0.converged and rL.converged
    assert r0.energy == pytest.approx(rL.energy, abs=1e-7)   # inert
    assert r0.n_iter != rL.n_iter                            # honored


def test_level_shift_operator_convention():
    """The coefficient on ``S D S`` is fixed by the density convention.

    Spin density D_s (idempotent, tr(D_s S) = n_occ): coefficient 1.
    Total density D = 2P (tr(D S) = n_elec):          coefficient 1/2.

    Both leave the occupied eigenvalues untouched and raise every virtual
    by exactly ``b``. Any other coefficient lifts the occupied block too,
    shrinking the gap the shift was asked to open. This pins the rule that
    ``periodic_uks_ewald.py`` violated; it is pure linear algebra and needs
    no SCF driver.
    """
    scipy_linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(0)
    n, n_occ, b = 6, 2, 1.0
    A = rng.normal(size=(n, n))
    S = A @ A.T + n * np.eye(n)          # non-orthogonal AO overlap
    B = rng.normal(size=(n, n))
    F = B + B.T
    eps, C = scipy_linalg.eigh(F, S)     # F C = S C eps, C^T S C = I

    D_spin = C[:, :n_occ] @ C[:, :n_occ].T
    D_total = 2.0 * D_spin
    assert np.allclose(D_spin @ S @ D_spin, D_spin, atol=1e-10)

    for D, coeff in ((D_spin, 1.0), (D_total, 0.5)):
        F_shifted = F + b * S - coeff * b * (S @ D @ S)
        eps_shifted, _ = scipy_linalg.eigh(F_shifted, S)
        np.testing.assert_allclose(
            eps_shifted[:n_occ], eps[:n_occ], atol=1e-9,
            err_msg="occupied eigenvalues must not move under the shift")
        np.testing.assert_allclose(
            eps_shifted[n_occ:], eps[n_occ:] + b, atol=1e-9,
            err_msg="every virtual must rise by exactly b")


def test_no_driver_open_codes_the_shift_operator():
    """Every SCF driver must route through ``apply_level_shift``.

    The Γ-Ewald UKS coefficient bug existed because eleven call sites each
    open-coded ``F + b*S - w*(S @ D @ S)``, so one of them could carry the
    wrong ``w`` indefinitely. A driver-free convention test cannot catch that;
    only binding the drivers to the shared operator can. This guards the
    binding: no SCF source file may spell the operator out by hand again.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    # ``F + b*S - w*(S @ D @ S)`` in numpy, or ``S * D * S`` in Eigen, on a
    # line that also mentions a level-shift magnitude.
    open_coded = re.compile(
        r"(S\s*@\s*\w*[Dd]\w*\s*@\s*S)|(S\s*\*\s*\w*[DdPp]\w*\s*\*\s*S)")
    shift_name = re.compile(r"level_shift|b_shift|\bb\b")

    # Skipped: the shared operator itself; ``bindings.cpp``, whose pybind
    # docstrings quote the formula as prose; and the GAPW / GPW plane-wave
    # family, documented as outside the unified accelerator surface.
    skip = {"level_shift.hpp", "level_shift_schedule.py", "bindings.cpp"}

    offenders = []
    sources = list((root / "python" / "vibeqc").glob("*.py"))
    sources += list((root / "cpp" / "src").glob("*.cpp"))
    for path in sources:
        if path.name in skip or path.name.startswith("periodic_gapw"):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            # Comments and bare string literals are documentation, not code.
            if stripped.startswith(("#", "//", "*", '"', "'")):
                continue
            if open_coded.search(line) and shift_name.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")

    assert not offenders, (
        "these lines open-code the Saunders-Hillier shift instead of calling "
        "apply_level_shift / apply_level_shift_k, which is exactly how the "
        "wrong weight drifted into run_uks_periodic_gamma_ewald3d:\n  "
        + "\n  ".join(offenders)
    )
