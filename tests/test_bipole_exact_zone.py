"""BIPOLE-EXACT-ZONE increment 1: restricted exact bielectronic zone.

The exact erfc J_SR/K_SR traversal's OUTPUT zone can be bounded below
the operator/fold cutoff (``exact_zone_bohr``): far operator cells keep
the reciprocal J_LR channel + neutralising background only. This is the
CRYSTAL bielectronic/monoelectronic zone partition — Dovesi, Pisani,
Roetti & Saunders, Phys. Rev. B 28, 5781 (1983), Eq. (24a); Pisani,
Dovesi & Roetti, Lecture Notes in Chemistry 48 (1988), Sec. II.4b/II.4d
— expressed in the Ewald-split gauge, where the reciprocal channel is
the far-field model. It decouples the cheap one-electron
fold-convergence range from the expensive exact-ERI range
(BIPOLE-EXACT-ZONE-UNBOUNDED, agentic-loop/bug-claims.md).

Measured basis for the tolerances (fixed Γ-periodic density, pad 20
bohr, 2026-08-06; probe script + logs archived at
``~/.claude-fixer-runs/bipole-exact-zone/`` on the dev machine):
the omitted SR tail decays much faster than the S(k) fold drift at the
same radius — see test_zone_scf_energy_tail below.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_common import resolve_bipole_exact_zone
from vibeqc.pbc_bipole_fock import _sr_image_padded_jk
from vibeqc._vibeqc_core import monkhorst_pack

ANG2BOHR = 1.0 / 0.529177210903


def _mgo_primitive():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])],
    )


def _lih_primitive():
    a = 4.084 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [a / 2, a / 2, a / 2])],
    )


def _h2_compact_box(L: float = 6.0, sep: float = 1.4):
    """Compact H2 box: fold-converged at small cutoffs (H/STO-3G most
    diffuse exponent 0.168), with real neighbor shells at 6.0 and
    8.49 bohr inside an 8.5-bohr cutoff — a seconds-scale SCF fixture
    that still exercises a genuine zone restriction."""
    half = 0.5 * sep
    return vq.PeriodicSystem(
        3,
        np.diag([L, L, L]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )


def _fixed_density(system, basis, cutoff: float):
    """Γ-periodic fixed density: D(g) = D0 on every cell (exact at n_k=1)."""
    nbf = basis.nbasis
    D0 = np.eye(nbf) * (system.n_electrons() / nbf)
    cells = list(direct_lattice_cells(system, cutoff))
    blocks = [np.asarray(D0, dtype=float) for _ in cells]
    return D0, make_lattice_matrix_set(nbf, cells, blocks)


# ---------------------------------------------------------------------------
# Unit level: the padded builder's zone restriction
# ---------------------------------------------------------------------------


def test_padded_jk_zone_blocks_match_full_build_and_far_blocks_zero():
    system = _mgo_primitive()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    omega = crystal_default_ewald_alpha(
        abs(np.linalg.det(np.asarray(system.lattice, dtype=float)))
    )
    cutoff, pad, zone = 8.0, 12.0, 6.0
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    _, D_lat = _fixed_density(system, basis, cutoff)

    full = _sr_image_padded_jk(basis, system, lo, D_lat, omega, pad)
    zoned = _sr_image_padded_jk(
        basis, system, lo, D_lat, omega, pad, exact_zone_bohr=zone
    )

    assert [tuple(c.index) for c in zoned.J.cells] == [
        tuple(c.index) for c in full.J.cells
    ]
    n_zone = n_far = 0
    for i, c in enumerate(full.J.cells):
        r = float(np.linalg.norm(np.asarray(c.r_cart)))
        jz = np.asarray(zoned.J.blocks[i])
        kz = np.asarray(zoned.K.blocks[i])
        if r <= zone:
            # Inside the zone: bitwise identical to the unrestricted build
            # (same traversal, same screening, same internal ball).
            np.testing.assert_array_equal(jz, np.asarray(full.J.blocks[i]))
            np.testing.assert_array_equal(kz, np.asarray(full.K.blocks[i]))
            n_zone += 1
        else:
            assert np.all(jz == 0.0)
            assert np.all(kz == 0.0)
            n_far += 1
    assert n_zone > 1  # home + at least the NN shell
    assert n_far > 0  # the restriction actually dropped something


def test_padded_jk_zone_rejects_zone_at_or_above_cutoff():
    system = _mgo_primitive()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    omega = crystal_default_ewald_alpha(
        abs(np.linalg.det(np.asarray(system.lattice, dtype=float)))
    )
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 8.0
    _, D_lat = _fixed_density(system, basis, 8.0)
    with pytest.raises(ValueError, match="strictly below"):
        _sr_image_padded_jk(
            basis, system, lo, D_lat, omega, 12.0, exact_zone_bohr=8.0
        )


# ---------------------------------------------------------------------------
# Resolver validation
# ---------------------------------------------------------------------------


def _lat_opts(cutoff: float) -> LatticeSumOptions:
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    return lo


def test_resolver_passthrough_and_validation():
    system = _mgo_primitive()
    lo = _lat_opts(10.0)
    common = dict(
        exchange_split_active=True,
        sr_image_extent=14.0,
        pair_mode=False,
        rep_cell_indices=None,
    )
    assert resolve_bipole_exact_zone(system, lo, None, **common) is None
    assert resolve_bipole_exact_zone(system, lo, 7.0, **common) == 7.0

    with pytest.raises(ValueError, match="corrected Ewald exchange split"):
        resolve_bipole_exact_zone(
            system, lo, 7.0, **{**common, "exchange_split_active": False}
        )
    with pytest.raises(ValueError, match="padded SR image path"):
        resolve_bipole_exact_zone(
            system, lo, 7.0, **{**common, "sr_image_extent": None}
        )
    with pytest.raises(NotImplementedError, match="pair-resolved"):
        resolve_bipole_exact_zone(
            system, lo, 7.0, **{**common, "pair_mode": True}
        )
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        resolve_bipole_exact_zone(
            system, lo, 7.0, **{**common, "rep_cell_indices": [0]}
        )
    with pytest.raises(ValueError, match="positive"):
        resolve_bipole_exact_zone(system, lo, -1.0, **common)
    with pytest.raises(ValueError, match="strictly below"):
        resolve_bipole_exact_zone(system, lo, 10.0, **common)


# ---------------------------------------------------------------------------
# SCF level: energy tail, provenance, gradient fail-closed
# ---------------------------------------------------------------------------


def _rhf_opts(cutoff: float):
    from vibeqc._vibeqc_core import InitialGuess, PeriodicRHFOptions

    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = 60
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    return opts


def test_zone_scf_energy_tail_and_provenance():
    """SCF-level plumbing check on a seconds-scale fixture: compact H2
    box, cutoff 8.5 (fold-converged for the compact H/STO-3G basis),
    zone 7 keeps home + the 6.0-bohr shell and drops the 8.49-bohr
    shell. The zone SCF converges, records provenance, and sits within
    a small envelope of the full build (the dropped shell's SR content
    on a compact neutral cell). The PHYSICS-scale tails are pinned at
    fixed density by the MgO bitwise unit test above plus the session
    probe anchors (MgO 2.7e-3 Ha @ zone 10 / 3.2e-4 @ 12; LiH-class
    diffuse bases keep mHa tails to ~14 bohr) — kept out of SCF tests
    deliberately: an MgO c12 padded SCF pair costs ~1 h under load,
    which is the EXACT-ZONE cost disease itself."""
    system = _h2_compact_box()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    kw = dict(
        ewald_precision=1e-8,
        sr_image_extent_bohr=12.0,
        progress=False,
        use_fock_symmetry=False,
    )
    r_full = run_pbc_bipole_rhf(
        system, basis, kmesh, _rhf_opts(8.5), **kw
    )
    r_zone = run_pbc_bipole_rhf(
        system, basis, kmesh, _rhf_opts(8.5), exact_zone_bohr=7.0, **kw
    )
    assert r_full.converged and r_zone.converged
    assert r_full.exact_zone_bohr is None
    assert r_zone.exact_zone_bohr == 7.0
    assert abs(r_zone.energy - r_full.energy) < 1e-3


def test_zone_analytic_gradient_fails_closed():
    from vibeqc.bipole_gradient import compute_bipole_gradient_rhf

    system = _h2_compact_box()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    r_zone = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _rhf_opts(8.5),
        ewald_precision=1e-8,
        sr_image_extent_bohr=12.0,
        exact_zone_bohr=7.0,
        progress=False,
        use_fock_symmetry=False,
    )
    with pytest.raises(NotImplementedError, match="exact bielectronic zone"):
        compute_bipole_gradient_rhf(system, basis, r_zone, kmesh=kmesh)


# ---------------------------------------------------------------------------
# Runner surface (bipole_exact_zone_bohr)
# ---------------------------------------------------------------------------


def test_runner_zone_keyword_rejections(tmp_path):
    from vibeqc.periodic_runner import run_periodic_job

    system = _mgo_primitive()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = str(tmp_path / "zone")

    with pytest.raises(NotImplementedError, match="jk_method='bipole'"):
        run_periodic_job(
            system, basis, method="RKS", functional="PBE",
            jk_method="gdf", bipole_exact_zone_bohr=8.0, output=stem,
        )
    with pytest.raises(NotImplementedError, match="ROHF"):
        run_periodic_job(
            system, basis, method="ROKS", functional="PBE",
            jk_method="bipole", bipole_exact_zone_bohr=8.0, output=stem,
        )
    with pytest.raises(ValueError, match="positive"):
        run_periodic_job(
            system, basis, method="RHF", jk_method="bipole",
            bipole_exact_zone_bohr=-2.0, output=stem,
        )


# ---------------------------------------------------------------------------
# Open-shell drivers (increment 1b)
# ---------------------------------------------------------------------------


def _h2_triplet_box():
    system = _h2_compact_box(sep=2.0)
    system.multiplicity = 3
    return system


def test_uhf_zone_scf_and_provenance():
    from vibeqc._vibeqc_core import InitialGuess, PeriodicRHFOptions
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    system = _h2_triplet_box()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 8.5
    opts.lattice_opts.nuclear_cutoff_bohr = 8.5
    opts.max_iter = 60
    opts.initial_guess = InitialGuess.SAD
    kw = dict(
        ewald_precision=1e-8,
        sr_image_extent_bohr=12.0,
        progress=False,
        use_fock_symmetry=False,
    )
    r_full = run_pbc_bipole_uhf(system, basis, kmesh, opts, **kw)
    r_zone = run_pbc_bipole_uhf(
        system, basis, kmesh, opts, exact_zone_bohr=7.0, **kw
    )
    assert r_full.converged and r_zone.converged
    assert r_full.exact_zone_bohr is None
    assert r_zone.exact_zone_bohr == 7.0
    assert abs(r_zone.energy - r_full.energy) < 1e-3


def test_uks_zone_scf_and_provenance():
    from vibeqc._vibeqc_core import InitialGuess, PeriodicKSOptions
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    system = _h2_triplet_box()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicKSOptions()
    opts.functional = "svwn"
    opts.lattice_opts.cutoff_bohr = 8.5
    opts.lattice_opts.nuclear_cutoff_bohr = 8.5
    opts.max_iter = 80
    opts.initial_guess = InitialGuess.SAD
    kw = dict(
        ewald_precision=1e-8,
        sr_image_extent_bohr=12.0,
        progress=False,
        use_fock_symmetry=False,
    )
    r_full = run_pbc_bipole_uks(system, basis, kmesh, opts, **kw)
    r_zone = run_pbc_bipole_uks(
        system, basis, kmesh, opts, exact_zone_bohr=7.0, **kw
    )
    assert r_full.converged and r_zone.converged
    assert r_full.exact_zone_bohr is None
    assert r_zone.exact_zone_bohr == 7.0
    assert abs(r_zone.energy - r_full.energy) < 1e-3
