"""Regression guards for the RETIRED BIPOLE multipole far-field branch.

G1 was retired as a production / default-on candidate on 2026-06-20
(see ``docs/bipole_status.md`` G1 + ``python/vibeqc/bipole_fock_multipole``
module docstring). The branch is a gated research artifact that is broken
in general — Ha-scale errors on *both* dipolar cells (composition) and,
via a J_SR double-count, even neutral dipole-free cells. (The historical
"dipole-free cells are accurate" claim reflected an older code state; on
current ``main`` H₂ is ~0.5 Ha off too.)

These tests pin the *diagnosis* and the *gating* so the finding stays
durable (the reproduction itself lives in the gitignored
``references/g1_multipole_probe*.py`` / ``references/g1_diag*.py``):

1. The bare interaction tensor is NOT the bug — it is unit-test-correct
   vs exact Coulomb (covered by ``tests/test_bipole_multipole.py``; we
   re-assert the dipole–dipole channel here as the anchor).
2. The branch NEVER auto-enables — ``user_enable=None`` resolves to
   disabled even where the old heuristic would have fired. Only an
   explicit ``user_enable=True`` activates it, and that warns loudly.
3. Root cause A (charge-incomplete Γ moments): the cell-moment
   contraction on the P0-localized Γ density yields a spurious net
   charge, while the Γ-fold contraction is neutral.
4. An SCF-level request fails before setup, so neither artifact can be
   reached accidentally.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, PeriodicRHFOptions, monkhorst_pack


# --- shared dipolar fixture: LiH with H off-centre -> real cell dipole ---
A_BOHR = 6.0
CUTOFF = 14.0  # the radius the old heuristic auto-enabled at
L_MAX = 2


def _lih_dipolar():
    lattice = A_BOHR * np.eye(3)
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [1.6, 0.0, 0.0])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _opts_one_iter():
    o = PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = CUTOFF
    o.lattice_opts.nuclear_cutoff_bohr = CUTOFF
    o.max_iter = 1
    o.initial_guess = InitialGuess.SAD
    o.use_diis = False
    return o


@pytest.fixture(scope="module")
def lih_one_iter():
    """One-iteration legacy-gauge SCF on the dipolar LiH cell.

    Returns ``(system, basis, result, M_lat)``. The result's density is
    the P0-localized Γ density used everywhere downstream.

    The gauge-independent fold guard (hoisted 2026-08-06) would rightly
    refuse this deliberately-small LiH cutoff; these tests pin the
    retired G1 far-field's *composition* errors by comparing two builds
    on the SAME truncated support, so fold reliability is not their
    subject — bypass the measurement rather than paying the LiH-class
    fold-converged cutoff (~24 bohr) in a retirement sentinel.
    """
    import vibeqc.pbc_bipole_common as common
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf
    from vibeqc._vibeqc_core import compute_multipole_moments_lattice

    system, basis = _lih_dipolar()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = _opts_one_iter()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            common, "s_fold_truncation_drift", lambda *a, **k: 1.0e-9
        )
        res = run_pbc_bipole_rhf(
            system, basis, kmesh, opts,
            use_ewald_j_split=True, use_exchange_ewald_split=False,
            use_multipole_far_field=False, multipole_l_max=L_MAX,
            ewald_precision=1e-8, progress=False,
        )
    M_lat = compute_multipole_moments_lattice(
        basis, system, opts.lattice_opts, L_MAX, (0.0, 0.0, 0.0)
    )
    return system, basis, res, M_lat


# ---------------------------------------------------------------------
# 1. The bare interaction tensor is correct (anchor — not the bug).
# ---------------------------------------------------------------------
def test_interaction_tensor_dipole_dipole_is_correct():
    """Anchor: the L=1 channel matches the analytic dipole-dipole law, so
    the Ha-scale far-field error is NOT a tensor-normalization bug."""
    from vibeqc.bipole_multipole import multipole_pair_energy, lm_index, n_components

    # Two z-aligned point dipoles, head-to-tail, well separated along z:
    #   E = -2 mu_A mu_B / R^3   (Stone, Intermolecular Forces, ch. 3).
    a, R = 0.05, 12.0
    def moments(O):
        M = np.zeros(n_components(L_MAX))
        for q, c in ((+1.0, O + np.array([0, 0, a])), (-1.0, O + np.array([0, 0, -a]))):
            r = c - O
            # Z_{1,0} = z (Stone convention)
            M[lm_index(1, 0)] += q * r[2]
        return M
    O_A, O_B = np.zeros(3), np.array([0.0, 0.0, R])
    E = multipole_pair_energy(moments(O_A), moments(O_B), O_B - O_A,
                              L_max_A=L_MAX, L_max_B=L_MAX)
    mu = 2 * a
    assert math.isclose(E, -2.0 * mu * mu / R**3, rel_tol=1e-3)


# ---------------------------------------------------------------------
# 2. Gating: never silently auto-enables.
# ---------------------------------------------------------------------
def test_never_auto_enables_on_dipolar_cell():
    """``user_enable=None`` must resolve to DISABLED even on the dipolar
    cell at the cutoff where the retired auto-enable heuristic fired."""
    from vibeqc.bipole_fock_multipole import resolve_multipole_config

    system, basis = _lih_dipolar()
    opts = _opts_one_iter()
    cfg = resolve_multipole_config(
        system, basis, opts.lattice_opts, user_enable=None, multipole_l_max=L_MAX
    )
    assert cfg.enabled is False
    # Explicit opt-out is also off.
    cfg_off = resolve_multipole_config(
        system, basis, opts.lattice_opts, user_enable=False, multipole_l_max=L_MAX
    )
    assert cfg_off.enabled is False


def test_explicit_enable_warns_and_attributes_root_cause():
    """An explicit ``user_enable=True`` activates the gated artifact and
    warns; the warning must NOT blame the (correct) interaction tensor."""
    from vibeqc.bipole_fock_multipole import resolve_multipole_config

    system, basis = _lih_dipolar()
    opts = _opts_one_iter()
    with pytest.warns(UserWarning, match="RETIRED"):
        cfg = resolve_multipole_config(
            system, basis, opts.lattice_opts, user_enable=True, multipole_l_max=L_MAX
        )
    assert cfg.enabled is True


# ---------------------------------------------------------------------
# 3. Root cause A: charge-incomplete Γ moments.
# ---------------------------------------------------------------------
def test_cell_moment_contraction_spurious_charge_at_gamma(lih_one_iter):
    """On the P0-localized Γ density the shipped contraction
    ``Σ_g P(g)·M(g)`` collapses to ``Tr[P(0)·M(0)] ≠ N_e`` → spurious net
    charge, whereas the Γ-fold ``Tr[P(0)·Σ_g M(g)]`` is neutral."""
    from vibeqc.bipole_cell_moments import compute_cell_multipole_moments

    system, basis, res, M_lat = lih_one_iter

    # Shipped contraction: monopole is NOT neutral (≈ −0.63 e on LiH).
    cm = compute_cell_multipole_moments(res.density, M_lat, system=system)
    q_buggy = float(np.asarray(cm.moments)[0])
    assert abs(q_buggy) > 0.1, (
        f"expected a spurious net charge from the P0-localized Γ density; "
        f"got monopole {q_buggy:.3e}"
    )

    # Γ-fold contraction: Tr[P(0)·Σ_g S(g)] − Σ Z  is neutral.
    P0 = None
    for c in range(len(res.density.cells)):
        if (np.asarray(res.density.cells[c].index) == 0).all():
            P0 = np.asarray(res.density.blocks[c], dtype=float)
            break
    S_fold = sum(np.asarray(M_lat.blocks[c][0], dtype=float)
                 for c in range(len(M_lat.cells)))
    n_e = float(np.einsum("ij,ij->", P0, S_fold))
    z_sum = sum(float(getattr(a, "Z", 0)) for a in system.unit_cell)
    q_gamma_fold = n_e - z_sum
    assert abs(q_gamma_fold) < 1e-6, (
        f"Γ-fold cell charge should be neutral; got {q_gamma_fold:.3e}"
    )


# ---------------------------------------------------------------------
# 4. Driver requests fail closed.
# ---------------------------------------------------------------------
def test_driver_request_fails_closed_before_setup():
    """No SCF driver may expose the retired low-level artifact."""
    from vibeqc.pbc_bipole import run_pbc_bipole_rhf

    with pytest.raises(NotImplementedError, match="three-translation"):
        run_pbc_bipole_rhf(None, None, None, use_multipole_far_field=True)
