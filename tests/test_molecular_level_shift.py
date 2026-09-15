"""Phase C1a-2 tests: Saunders-Hillier level shift on the molecular SCF
drivers (run_rhf, run_uhf, run_rks, run_uks).

Mirrors the contract pinned by ``test_periodic_level_shift.py`` for the
periodic side, restricted to the bits that make sense for molecules:

1. **Field exposure on all four option structs** — RHFOptions,
   UHFOptions, RKSOptions, UKSOptions all carry ``level_shift`` as a
   settable double (default 0.0).

2. **Inertness at convergence** — ``level_shift = 0.3`` reaches the
   same total energy as ``level_shift = 0`` on a tight closed-shell
   (H2O / STO-3G) and on triplet O / STO-3G. The shift is by
   construction inert at the SCF fixed point.

3. **MO eigenvalues are physical at convergence** — the returned
   ``mo_energies`` come from the final self-consistency pass (which
   doesn't include the shift), so the orbital spectrum at convergence
   is independent of the shift value used during iteration.

4. **Default behavior preserved** — ``level_shift = 0`` (default)
   matches pre-C1a-2 behavior byte-identically. Toggling it from 0
   to 0 must not perturb the trace.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _h2o_sto3g():
    """Closed-shell H2O / STO-3G."""
    from .conftest import GEOMETRIES, make_molecule
    mol = make_molecule(GEOMETRIES["H2O"])
    basis = vq.BasisSet(mol, "sto-3g")
    return mol, basis


@pytest.fixture
def _o_triplet_sto3g():
    """Triplet O atom / STO-3G — simplest open-shell case."""
    mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0])],
                      charge=0, multiplicity=3)
    basis = vq.BasisSet(mol, "sto-3g")
    return mol, basis


# ---------------------------------------------------------------------------
# 1. Field exposure on all four option structs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    vq.RHFOptions,
    vq.UHFOptions,
    vq.RKSOptions,
    vq.UKSOptions,
])
def test_molecular_level_shift_field_default_is_zero(cls):
    o = cls()
    assert hasattr(o, "level_shift"), cls.__name__
    assert o.level_shift == 0.0, cls.__name__
    o.level_shift = 0.5
    assert o.level_shift == 0.5, cls.__name__


# ---------------------------------------------------------------------------
# 2. Inertness at convergence — same total energy with and without shift
# ---------------------------------------------------------------------------

def test_rhf_level_shift_inertness_at_convergence(_h2o_sto3g):
    """level_shift = 0.3 reaches the same energy as level_shift = 0
    on H2O / STO-3G."""
    mol, basis = _h2o_sto3g

    o = vq.RHFOptions()
    r0 = vq.run_rhf(mol, basis, o)

    o = vq.RHFOptions()
    o.level_shift = 0.3
    r1 = vq.run_rhf(mol, basis, o)

    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)


def test_rks_level_shift_inertness_at_convergence(_h2o_sto3g):
    """level_shift = 0.3 reaches the same KS energy as level_shift = 0
    on RKS LDA / H2O / STO-3G."""
    mol, basis = _h2o_sto3g

    o = vq.RKSOptions()
    o.functional = "LDA"
    r0 = vq.run_rks(mol, basis, o)

    o = vq.RKSOptions()
    o.functional = "LDA"
    o.level_shift = 0.3
    r1 = vq.run_rks(mol, basis, o)

    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)


def test_uhf_level_shift_inertness_at_convergence(_o_triplet_sto3g):
    """Per-spin level shift on triplet O reaches the same UHF energy
    as the unshifted path."""
    mol, basis = _o_triplet_sto3g

    o = vq.UHFOptions()
    r0 = vq.run_uhf(mol, basis, o)

    o = vq.UHFOptions()
    o.level_shift = 0.3
    r1 = vq.run_uhf(mol, basis, o)

    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)
    # ⟨S²⟩ should be unchanged too
    assert r0.s_squared == pytest.approx(r1.s_squared, abs=1e-7)


def test_uks_level_shift_inertness_at_convergence(_o_triplet_sto3g):
    """Per-spin level shift on triplet O reaches the same UKS LDA
    energy as the unshifted path."""
    mol, basis = _o_triplet_sto3g

    o = vq.UKSOptions()
    o.functional = "LDA"
    r0 = vq.run_uks(mol, basis, o)

    o = vq.UKSOptions()
    o.functional = "LDA"
    o.level_shift = 0.3
    r1 = vq.run_uks(mol, basis, o)

    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. MO eigenvalues are physical at convergence
# ---------------------------------------------------------------------------

def test_rhf_mo_energies_physical_at_convergence(_h2o_sto3g):
    """``mo_energies`` are the un-shifted physical orbital energies.
    Two RHF runs with different ``level_shift`` values during iteration
    must produce the same orbital spectrum at convergence (the final
    self-consistency pass diagonalises the un-shifted F)."""
    mol, basis = _h2o_sto3g

    o = vq.RHFOptions()
    r0 = vq.run_rhf(mol, basis, o)

    o = vq.RHFOptions()
    o.level_shift = 0.3
    r1 = vq.run_rhf(mol, basis, o)

    eps0 = np.sort(r0.mo_energies)
    eps1 = np.sort(r1.mo_energies)
    np.testing.assert_allclose(eps0, eps1, atol=1e-7)


def test_uhf_mo_energies_physical_at_convergence(_o_triplet_sto3g):
    """Same un-shifted-MO contract per-spin for UHF on triplet O."""
    mol, basis = _o_triplet_sto3g

    o = vq.UHFOptions()
    r0 = vq.run_uhf(mol, basis, o)

    o = vq.UHFOptions()
    o.level_shift = 0.3
    r1 = vq.run_uhf(mol, basis, o)

    eps_a0 = np.sort(r0.mo_energies_alpha)
    eps_a1 = np.sort(r1.mo_energies_alpha)
    eps_b0 = np.sort(r0.mo_energies_beta)
    eps_b1 = np.sort(r1.mo_energies_beta)
    np.testing.assert_allclose(eps_a0, eps_a1, atol=1e-7)
    np.testing.assert_allclose(eps_b0, eps_b1, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Default reproduces baseline
# ---------------------------------------------------------------------------

def test_rhf_default_level_shift_zero_reproduces_baseline(_h2o_sto3g):
    """Setting level_shift = 0 (default) produces a byte-identical
    energy and trace to the implicit-default run — no perturbation
    from the C1a-2 wiring on disabled-by-default."""
    mol, basis = _h2o_sto3g

    o1 = vq.RHFOptions()
    r1 = vq.run_rhf(mol, basis, o1)

    o2 = vq.RHFOptions()
    o2.level_shift = 0.0   # explicit
    r2 = vq.run_rhf(mol, basis, o2)

    assert r1.converged and r2.converged
    assert r1.energy == pytest.approx(r2.energy, abs=1e-12)
    assert r1.n_iter == r2.n_iter


def test_level_shift_perturbs_iteration_dynamics(_h2o_sto3g):
    """A non-zero level shift must actually change the SCF trajectory
    (different per-iteration energies) even though it converges to the
    same fixed point. Otherwise the field is doing nothing — guards
    against the periodic-UHF latent regression where ``level_shift``
    was read but never applied."""
    mol, basis = _h2o_sto3g

    o = vq.RHFOptions()
    o.use_diis = False    # isolate level-shift effect from DIIS
    o.damping = 0.0
    o.max_iter = 200
    r0 = vq.run_rhf(mol, basis, o)

    o = vq.RHFOptions()
    o.use_diis = False
    o.damping = 0.0
    o.max_iter = 200
    o.level_shift = 0.5
    r_shift = vq.run_rhf(mol, basis, o)

    # Same fixed point.
    assert r0.converged and r_shift.converged
    assert r0.energy == pytest.approx(r_shift.energy, abs=1e-8)
    # Different trajectory (level shift is a damping mechanism — without
    # DIIS to compensate it slows convergence on a well-behaved system,
    # which is exactly the kind of behavior we expect).
    assert r0.n_iter != r_shift.n_iter


# ---------------------------------------------------------------------------
# 5. Unified level_shift_at_iter helper (single source of truth)
# ---------------------------------------------------------------------------
#
# The C++ helper `level_shift_at_iter` resolves the per-iteration
# Saunders-Hillier shift for both the molecular C++ drivers and the
# periodic Python drivers. These pin its contract directly.


def test_level_shift_at_iter_schedule_takes_precedence():
    """A non-empty schedule supersedes level_shift + warmup: entry i is
    the shift at iteration i+1, and the last entry is reused past the
    schedule length (0.0 releases)."""
    sched = [0.5, 0.4, 0.0]
    # warmup and base are ignored when a schedule is present.
    assert vq.level_shift_at_iter(0.3, -1, sched, 100, 1) == 0.5
    assert vq.level_shift_at_iter(0.3, 0, sched, 100, 2) == 0.4
    assert vq.level_shift_at_iter(0.3, 5, sched, 100, 3) == 0.0
    assert vq.level_shift_at_iter(0.3, 5, sched, 100, 50) == 0.0  # clamp last


def test_level_shift_at_iter_auto_warmup_then_release():
    """warmup = -1 (auto): shifted for up to five startup cycles, then
    released, always leaving ≥1 unshifted tail cycle."""
    for it in range(1, 6):
        assert vq.level_shift_at_iter(0.5, -1, [], 100, it) == 0.5
    assert vq.level_shift_at_iter(0.5, -1, [], 100, 6) == 0.0
    assert vq.level_shift_at_iter(0.5, -1, [], 100, 99) == 0.0


def test_level_shift_at_iter_auto_caps_to_tail_cycle():
    """Auto warm-up is capped to max_iter - 1 so a converging unshifted
    tail cycle always survives even for tiny iteration budgets."""
    # max_iter = 3 → warmup = min(5, 2) = 2.
    assert vq.level_shift_at_iter(0.5, -1, [], 3, 1) == 0.5
    assert vq.level_shift_at_iter(0.5, -1, [], 3, 2) == 0.5
    assert vq.level_shift_at_iter(0.5, -1, [], 3, 3) == 0.0


def test_level_shift_at_iter_persistent():
    """warmup = 0: the shift is held at every iteration (legacy)."""
    assert vq.level_shift_at_iter(0.5, 0, [], 100, 1) == 0.5
    assert vq.level_shift_at_iter(0.5, 0, [], 100, 80) == 0.5


def test_level_shift_at_iter_explicit_length():
    """warmup = N > 0: shifted for iters 1..N, released thereafter."""
    assert vq.level_shift_at_iter(0.5, 3, [], 100, 3) == 0.5
    assert vq.level_shift_at_iter(0.5, 3, [], 100, 4) == 0.0


def test_level_shift_at_iter_zero_base_is_inert():
    """No base shift ⇒ zero regardless of warmup / max_iter."""
    assert vq.level_shift_at_iter(0.0, -1, [], 100, 1) == 0.0
    assert vq.level_shift_at_iter(0.0, 0, [], 100, 1) == 0.0
    # max_iter <= 1 ⇒ no room to warm up ⇒ zero.
    assert vq.level_shift_at_iter(0.5, -1, [], 1, 1) == 0.0


# ---------------------------------------------------------------------------
# 6. New unified fields on every molecular option struct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [
    vq.RHFOptions,
    vq.UHFOptions,
    vq.RKSOptions,
    vq.UKSOptions,
])
def test_molecular_level_shift_warmup_and_schedule_fields(cls):
    o = cls()
    # Warm-up defaults to auto (-1), matching the periodic drivers.
    assert o.level_shift_warmup_cycles == -1, cls.__name__
    o.level_shift_warmup_cycles = 3
    assert o.level_shift_warmup_cycles == 3, cls.__name__
    # Schedule defaults empty and accepts a list.
    assert list(o.level_shift_schedule) == [], cls.__name__
    o.level_shift_schedule = [0.5, 0.4, 0.0]
    assert list(o.level_shift_schedule) == [0.5, 0.4, 0.0], cls.__name__


def test_level_shift_schedule_apply_to_lowers_to_vector():
    """LevelShiftSchedule.apply_to sets the C++ vector field."""
    from vibeqc.level_shift_schedule import LevelShiftSchedule
    o = vq.RHFOptions()
    out = LevelShiftSchedule.crystal_default().apply_to(o)
    assert out is o
    assert list(o.level_shift_schedule) == [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]


# ---------------------------------------------------------------------------
# 7. Schedule / warm-up on the molecular drivers: inert at convergence
# ---------------------------------------------------------------------------

def test_rhf_schedule_inertness_at_convergence(_h2o_sto3g):
    """A decaying CRYSTAL-style schedule reaches the same converged
    energy as no shift on H2O / STO-3G; the schedule only damps the
    path, never the fixed point."""
    from vibeqc.level_shift_schedule import LevelShiftSchedule
    mol, basis = _h2o_sto3g

    r0 = vq.run_rhf(mol, basis, vq.RHFOptions())

    o = vq.RHFOptions()
    LevelShiftSchedule.crystal_default().apply_to(o)
    r1 = vq.run_rhf(mol, basis, o)

    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-9)


def test_rhf_schedule_perturbs_dynamics(_h2o_sto3g):
    """A schedule must actually change the SCF trajectory (guards against
    a read-but-never-applied regression), while landing the same energy."""
    from vibeqc.level_shift_schedule import LevelShiftSchedule
    mol, basis = _h2o_sto3g

    o0 = vq.RHFOptions()
    o0.use_diis = False
    o0.damping = 0.0
    o0.max_iter = 300
    r0 = vq.run_rhf(mol, basis, o0)

    o1 = vq.RHFOptions()
    o1.use_diis = False
    o1.damping = 0.0
    o1.max_iter = 300
    LevelShiftSchedule([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]).apply_to(o1)
    r1 = vq.run_rhf(mol, basis, o1)

    assert r0.converged and r1.converged
    assert r0.energy == pytest.approx(r1.energy, abs=1e-8)
    assert r0.n_iter != r1.n_iter


def test_auto_reducing_shift_beats_persistent(_h2o_sto3g):
    """The auto-reducing shift (warm-up then release) converges to the
    same energy as a persistent shift and never takes more iterations;
    releasing the shift lets the tail converge unhindered. DIIS is off so
    the level shift is the only convergence control at play."""
    mol, basis = _h2o_sto3g

    o_persist = vq.RHFOptions()
    o_persist.use_diis = False
    o_persist.damping = 0.0
    o_persist.max_iter = 400
    o_persist.level_shift = 0.5
    o_persist.level_shift_warmup_cycles = 0        # persistent (legacy)
    r_persist = vq.run_rhf(mol, basis, o_persist)

    o_auto = vq.RHFOptions()
    o_auto.use_diis = False
    o_auto.damping = 0.0
    o_auto.max_iter = 400
    o_auto.level_shift = 0.5
    o_auto.level_shift_warmup_cycles = -1          # auto: warm up then release
    r_auto = vq.run_rhf(mol, basis, o_auto)

    assert r_persist.converged and r_auto.converged
    assert r_persist.energy == pytest.approx(r_auto.energy, abs=1e-8)
    assert r_auto.n_iter <= r_persist.n_iter
