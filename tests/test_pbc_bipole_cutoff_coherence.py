"""Driver-parity guard for the BIPOLE neutral-cell nuclear-cutoff clamp.

``9ea2651d`` fixed a mHa-scale molecular-limit overbinding (nuclear
point-charge tails on cells whose compensating electronic charge is
truncated away) by clamping ``nuclear_cutoff_bohr`` down to
``cutoff_bohr`` under the 3-D Ewald-J split — but only in the RKS
driver body and the ``run_periodic_job`` runner. RHF/UHF/UKS *direct*
callers (the FD-gradient path, validation scripts) kept the unclamped
library defaults (electronic 15.0 vs nuclear 25.0 bohr) and silently
disagreed with the runner route and with RKS on identical physics.

The clamp is now centralised
(``pbc_bipole_common.clamp_bipole_nuclear_cutoff``) and applied by all
four drivers. This test pins the invariant that catches the whole
defect class: the Ewald nuclear repulsion is density- and
method-independent, so all four drivers called directly with identical
DEFAULT lattice options on the same 3-D cell must report the *same*
``e_nuclear_repulsion``. Pre-centralisation this fails: RKS clamps
(E_nn from the 15-bohr Ewald real set) while RHF/UHF/UKS do not
(25-bohr set).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions


def _h2_cell():
    # Small 3-D cell so the default cutoffs (electronic 15 / nuclear 25
    # bohr) genuinely resolve to different Ewald real-space cell sets.
    box = 8.0
    c = box / 2
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _gamma(system):
    return vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)


def _hf_opts():
    # DEFAULT lattice cutoffs on purpose — the defect only appears when
    # nuclear_cutoff_bohr (25) exceeds cutoff_bohr (15).
    opts = PeriodicRHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    return opts


def _ks_opts():
    opts = PeriodicKSOptions()
    opts.functional = "svwn"
    opts.max_iter = 1
    opts.use_diis = False
    return opts


def test_all_four_drivers_share_one_nuclear_cutoff():
    """Every direct driver resolves the same clamped nuclear cutoff.

    Pre-centralisation this failed for RHF, UHF, and UKS: they left the
    unclamped 25-bohr nuclear Ewald set while RKS clamped to the 15-bohr
    electronic set. The clamp used to mutate the caller's
    ``opts.lattice_opts`` in place, which is what this test observed;
    since 19cda20b (#316) every driver resolves its domain on a detached
    copy and records it, so the observable is the result's
    ``resolved_domain`` and the caller's options must stay untouched
    (#503).

    (Do not assert single-iteration energy parity across drivers here:
    non-converged trajectories legitimately differ per driver — guess
    seeding and accelerator sequencing — and on this tiny cell the
    clamp itself moves the energy by only ~2e-11, far below those
    sequencing differences. The discriminating observable is the
    resolved cutoff; the converged closed-shell reduction below is the
    physics-level companion.)
    """
    system, basis = _h2_cell()
    kmesh = _gamma(system)

    opts = {"rhf": _hf_opts(), "uhf": _hf_opts(), "rks": _ks_opts(), "uks": _ks_opts()}
    runners = {
        "rhf": vq.run_pbc_bipole_rhf,
        "rks": vq.run_pbc_bipole_rks,
        "uhf": vq.run_pbc_bipole_uhf,
        "uks": vq.run_pbc_bipole_uks,
    }
    resolved = {}
    for name, o in opts.items():
        result = runners[name](system, basis, kmesh, o, progress=False)
        domain = result.resolved_domain
        one_electron = dict(domain.lattice_1e.values)
        two_electron = dict(domain.lattice_2e.values)
        resolved[name] = (
            one_electron["nuclear_cutoff_bohr"],
            two_electron["cutoff_bohr"],
            domain.ewald_real_cutoff_bohr,
        )
        # The driver works on a detached copy: the caller's options keep
        # the unclamped 25-bohr nuclear default.
        assert o.lattice_opts.nuclear_cutoff_bohr == pytest.approx(25.0)
        assert o.lattice_opts.cutoff_bohr == pytest.approx(15.0)

    for name, (nuclear, electronic, ewald_real) in resolved.items():
        assert nuclear == pytest.approx(electronic), (
            f"{name}: the recorded nuclear_cutoff_bohr was not clamped to "
            f"the electronic cutoff ({nuclear} vs {electronic}); the driver "
            "is resolving its own nuclear/Ewald cutoff instead of the "
            "shared clamp_bipole_nuclear_cutoff"
        )
        assert ewald_real == pytest.approx(electronic)
    assert len({tuple(v) for v in resolved.values()}) == 1, resolved


def test_converged_closed_shell_uks_reduces_to_rks_at_default_cutoffs():
    """Converged closed-shell UKS == RKS with DEFAULT lattice cutoffs.

    The physics-level companion to the clamp-coherence pin: both
    drivers converge to the same stationary point (measured residual
    ~9e-8 on this cell) only when they resolve identical lattice/Ewald
    cutoff sets from the same defaults.
    """
    system, basis = _h2_cell()
    kmesh = _gamma(system)

    def _conv_opts():
        opts = PeriodicKSOptions()
        opts.functional = "svwn"
        opts.max_iter = 60
        opts.use_diis = True
        return opts

    r_rks = vq.run_pbc_bipole_rks(system, basis, kmesh, _conv_opts(), progress=False)
    r_uks = vq.run_pbc_bipole_uks(system, basis, kmesh, _conv_opts(), progress=False)
    assert r_rks.converged and r_uks.converged
    assert float(r_uks.energy) == pytest.approx(float(r_rks.energy), abs=1e-6)


def test_clamp_helper_is_gated_and_mutating():
    """The shared helper clamps only under the 3-D Ewald-J split."""
    from vibeqc._vibeqc_core import LatticeSumOptions
    from vibeqc.pbc_bipole_common import clamp_bipole_nuclear_cutoff
    from vibeqc.progress import resolve_progress

    system, _ = _h2_cell()
    plog = resolve_progress(False, verbose=False)

    lat = LatticeSumOptions()
    assert lat.nuclear_cutoff_bohr > lat.cutoff_bohr  # library defaults
    clamp_bipole_nuclear_cutoff(system, lat, True, plog)
    assert lat.nuclear_cutoff_bohr == lat.cutoff_bohr

    lat2 = LatticeSumOptions()
    before = lat2.nuclear_cutoff_bohr
    clamp_bipole_nuclear_cutoff(system, lat2, False, plog)  # no J split
    assert lat2.nuclear_cutoff_bohr == before
