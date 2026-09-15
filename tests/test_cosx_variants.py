"""COSX variant + grid-level + auto-selection parameter surface (M1).

This pins the RIJCOSX upgrade's user-facing controls:

* ``cosx_variant`` — ``CosxVariant.AUTO`` (default) / ``STANDARD`` /
  ``FITTED``. STANDARD is Neese 2009 seminumerical K with no overlap
  fit; FITTED is the overlap-fitted build (global Q-junction, Neese
  2009 §2.4; precursor to the per-grid-point fit of Izsák-Neese 2011).
* ``cosx_grid_level`` — 0 keeps the legacy grid; 1..4 select the GridX
  tiers; -1 auto-selects by basis cardinality.
* ``thresh_cosx`` — drives AUTO variant selection.

The pure resolvers (``resolve_cosx_variant``, ``resolve_cosx_grid_level``,
``cosx_basis_cardinality_from_name``, ``cosx_grid_options_for_level``) are
unit-tested directly; the variant + grid-level energy paths are tested
against direct SCF within the COSX fit band.

References:
  [1] Neese, Wennmohs, Hansen, Becker, Chem. Phys. 356, 98 (2009)
      doi:10.1016/j.chemphys.2008.10.036   [standard COSX]
  [2] Izsák, Neese, J. Chem. Phys. 135, 144105 (2011)
      doi:10.1063/1.3646921                [overlap-fitted COSX]
  [3] Helmich-Paris, de Souza, Neese, Izsák, J. Chem. Phys. 155,
      104109 (2021) doi:10.1063/5.0058766  [improved grids, contraction]
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom, BasisSet, Molecule, RHFOptions, RKSOptions,
    CosxVariant,
    cosx_basis_cardinality_from_name,
    cosx_grid_options_for_level,
    cosx_grid_stages_for_level,
    resolve_cosx_grid_level,
    resolve_cosx_variant,
    run_job, run_rhf, run_rks,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_A = ANGSTROM_TO_BOHR
_AUX = "def2-universal-jkfit"

H2O = [(8, [0.0, 0.0, 0.0]),
       (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
       (1, [0.0, -0.793353 * _A, -0.613510 * _A])]


def _mol(atoms, mult=1):
    return Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms],
                    multiplicity=mult)


# --- pure auto-selection logic (no SCF) ----------------------------------

@pytest.mark.parametrize("name,expected", [
    ("def2-svp", 2), ("def2-SVP", 2), ("def2-sv(p)", 2), ("def2-svpd", 2),
    ("def2-tzvp", 3), ("def2-tzvpp", 3), ("pob-tzvp-rev2", 3),
    ("def2-qzvp", 4), ("def2-qzvpp", 4),
    ("cc-pvdz", 2), ("aug-cc-pvdz", 2), ("cc-pvtz", 3), ("cc-pvqz", 4),
    ("cc-pv5z", 5), ("aug-cc-pv5z", 5),
    ("6-31g", 2), ("6-31g*", 2), ("6-311g", 3), ("6-311+g**", 3),
    ("sto-3g", 0), ("minix", 0),
])
def test_basis_cardinality_from_name(name, expected):
    assert cosx_basis_cardinality_from_name(name) == expected


def test_resolve_variant_auto_legacy_grid_is_fitted():
    """On the legacy grid (grid_level == 0), AUTO always resolves to
    FITTED — the standard build is only accurate on a GridX tier."""
    A, F = CosxVariant.AUTO, CosxVariant.FITTED
    assert resolve_cosx_variant(A, 1e-6, 2, 0) == F   # DZ
    assert resolve_cosx_variant(A, 1e-6, 3, 0) == F   # TZ
    assert resolve_cosx_variant(A, 1e-6, 4, 0) == F   # QZ
    assert resolve_cosx_variant(A, 1e-8, 2, 0) == F   # tight


def test_resolve_variant_auto_always_fitted():
    """AUTO always resolves to FITTED, on any grid/basis/threshold. The
    standard build is not robust enough to auto-select (it destabilises
    open-shell RIJCOSX SCFs); it is an explicit opt-in only."""
    A, F = CosxVariant.AUTO, CosxVariant.FITTED
    for gl in (0, 1, 2, 3, 4, -1):
        for card in (0, 2, 3, 4, 5):
            for thr in (1e-6, 1e-8):
                assert resolve_cosx_variant(A, thr, card, gl) == F


def test_resolve_variant_explicit_passthrough():
    S, F = CosxVariant.STANDARD, CosxVariant.FITTED
    # A non-AUTO variant is returned unchanged regardless of context.
    assert resolve_cosx_variant(S, 1e-12, 5, 0) == S
    assert resolve_cosx_variant(F, 1.0, 0, 3) == F


def test_resolve_grid_level():
    # Explicit override clamps to [1, 4].
    assert resolve_cosx_grid_level(1, 0) == 1
    assert resolve_cosx_grid_level(4, 0) == 4
    assert resolve_cosx_grid_level(7, 0) == 4
    # Auto (<= -1): GridX1 is never auto-selected (too coarse for robust
    # open-shell SCF). DZ/TZ -> GridX2, QZ -> GridX3.
    assert resolve_cosx_grid_level(-1, 4) == 3   # QZ → GridX3
    assert resolve_cosx_grid_level(-1, 3) == 2   # TZ → GridX2
    assert resolve_cosx_grid_level(-1, 2) == 2   # DZ → GridX2
    assert resolve_cosx_grid_level(-1, 0) == 2   # unknown → GridX2


def test_grid_options_for_level_tiers():
    # Tiers are Lebedev-angular with increasing radial + angular order.
    g1 = cosx_grid_options_for_level(1)
    g2 = cosx_grid_options_for_level(2)
    g3 = cosx_grid_options_for_level(3)
    g4 = cosx_grid_options_for_level(4)
    radials = [g1.n_radial, g2.n_radial, g3.n_radial, g4.n_radial]
    orders = [g1.lebedev_order, g2.lebedev_order,
              g3.lebedev_order, g4.lebedev_order]
    assert radials == sorted(radials) and len(set(radials)) == 4
    assert orders == sorted(orders) and len(set(orders)) == 4
    # Clamping: out-of-range maps to the nearest tier.
    assert cosx_grid_options_for_level(0).n_radial == g1.n_radial
    assert cosx_grid_options_for_level(9).n_radial == g4.n_radial


# --- variant routing through the SCF energy ------------------------------

_E_TOL = 1e-3   # COSX fit band vs direct (matches test_rijcosx.py)


def test_explicit_variants_match_direct_and_differ():
    """On a GridX grid, STANDARD and FITTED both land within the COSX fit
    band of direct RHF, and they differ from each other (proving the
    variant flag actually routes the K build)."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")

    o = RHFOptions(); o.conv_tol_energy = 1e-10
    e_direct = run_rhf(mol, basis, o).energy

    def _run(variant):
        o = RHFOptions(); o.conv_tol_energy = 1e-10
        o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
        o.cosx_variant = variant; o.cosx_grid_level = 2   # GridX2
        r = run_rhf(mol, basis, o)
        assert r.converged
        return r.energy

    e_std = _run(CosxVariant.STANDARD)
    e_fit = _run(CosxVariant.FITTED)
    assert abs(e_std - e_direct) < _E_TOL, f"STANDARD Δ={e_std-e_direct:.2e}"
    assert abs(e_fit - e_direct) < _E_TOL, f"FITTED Δ={e_fit-e_direct:.2e}"
    # The two builds are genuinely different code paths.
    assert abs(e_std - e_fit) > 1e-9, (
        "STANDARD and FITTED produced identical energies — the "
        "cosx_variant flag is not routing the K build"
    )


def test_auto_variant_is_fitted_on_any_grid():
    """AUTO resolves to FITTED on both the legacy grid and a GridX grid
    (the robust default; STANDARD is opt-in only). Verified by a
    bit-for-bit energy match to an explicit FITTED run on the same grid."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")

    def _run(variant, gl):
        o = RHFOptions(); o.conv_tol_energy = 1e-10
        o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
        o.cosx_variant = variant; o.cosx_grid_level = gl
        return run_rhf(mol, basis, o).energy

    assert abs(_run(CosxVariant.AUTO, 0)
               - _run(CosxVariant.FITTED, 0)) < 1e-12
    assert abs(_run(CosxVariant.AUTO, 2)
               - _run(CosxVariant.FITTED, 2)) < 1e-12


def test_grid_level_tiers_match_direct():
    """Explicit GridX tiers 1..3 all land within the COSX fit band of
    direct RHF on H2O/def2-svp."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")
    o = RHFOptions(); o.conv_tol_energy = 1e-10
    e_direct = run_rhf(mol, basis, o).energy

    for level in (1, 2, 3):
        o = RHFOptions(); o.conv_tol_energy = 1e-10
        o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
        o.cosx_grid_level = level
        r = run_rhf(mol, basis, o)
        assert r.converged
        assert abs(r.energy - e_direct) < _E_TOL, (
            f"GridX{level} Δ={r.energy - e_direct:.2e} Ha exceeds fit band"
        )


@pytest.mark.slow
def test_tzvp_fitted_matches_direct():
    """H2O/def2-TZVP RHF: FITTED variant on GridX2 within 0.1 kcal/mol
    (1.6e-4 Ha) of direct (prompt acceptance criterion)."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-tzvp")
    o = RHFOptions(); o.conv_tol_energy = 1e-10
    e_direct = run_rhf(mol, basis, o).energy

    o = RHFOptions(); o.conv_tol_energy = 1e-10
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    o.cosx_variant = CosxVariant.FITTED; o.cosx_grid_level = 2
    r = run_rhf(mol, basis, o)
    assert r.converged
    assert abs(r.energy - e_direct) < 1.6e-4, (
        f"TZVP FITTED/GridX2 Δ={r.energy - e_direct:.2e} Ha > 0.1 kcal/mol"
    )


# --- P2: multi-stage 2021 grid progression (DefGrid stages) --------------

def test_grid_stages_progression():
    """cosx_grid_stages_for_level returns the coarse→fine DefGrid stage
    triple: 2-3 grids, the last == cosx_grid_options_for_level(level), each
    a 5-region AngularGrid, with non-decreasing peak angular order."""
    for level in (1, 2, 3, 4):
        stages = cosx_grid_stages_for_level(level)
        assert 2 <= len(stages) <= 3
        fine = cosx_grid_options_for_level(level)
        # The fine stage is the last element.
        assert (list(stages[-1].orca_angular_points)
                == list(fine.orca_angular_points))
        peaks = [max(g.orca_angular_points) for g in stages]
        radials = [g.n_radial for g in stages]
        assert peaks == sorted(peaks), f"L{level} peaks not ascending: {peaks}"
        assert radials == sorted(radials)
        for g in stages:
            assert len(g.orca_angular_points) == 5


def test_multistage_gridx_converges_and_matches_direct():
    """A GridX run (now the multi-stage progression) converges and lands in
    the COSX fit band of direct RHF — proving the staged driver path works
    end to end."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")
    o = RHFOptions(); o.conv_tol_energy = 1e-10
    e_direct = run_rhf(mol, basis, o).energy
    for level in (1, 2, 3):
        o = RHFOptions(); o.conv_tol_energy = 1e-10
        o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
        o.cosx_grid_level = level
        r = run_rhf(mol, basis, o)
        assert r.converged, f"multi-stage GridX{level} did not converge"
        assert abs(r.energy - e_direct) < _E_TOL, (
            f"GridX{level} Δ={r.energy - e_direct:.2e}"
        )


def test_multistage_final_grid_is_single_recompute():
    """BUG87-A structural guard: the SCF must NEVER iterate on the final
    (largest) COSX grid. Published scheme (Helmich-Paris, de Souza, Neese
    & Izsák, J. Chem. Phys. 155, 104109 (2021), Sec. IV.A): "It starts
    with a minimal grid, then is switched to a middle grid when close to
    convergence, and after converging, the exchange energy is recomputed
    using the final larger grid."

    Pre-fix, the staged driver ran a full SCF segment to the user's
    tolerance on the final grid; the inter-grid exchange difference
    re-excites the DIIS error to the 1e-2-Frobenius class, so that
    segment re-converged from scratch at the most expensive
    per-iteration price (11 of 12 fleet-glycine iterations were
    fine-grid K builds; BUG87-A's 33x-vs-ORCA defect).

    The structural pin, from the (now truthful) cross-stage trace:
      * the SECOND-TO-LAST row converged to the user's tolerance
        (the converging middle-grid stage), and
      * the LAST row is the single final-grid recompute: its commutator
        is the inter-grid exchange difference (far above tolerance),
        yet the SCF reports converged — exactly one final-grid build.
      * the reported energy is the final-grid recompute energy.
    """
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")
    o = RHFOptions()
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    o.cosx_grid_level = 2
    r = run_rhf(mol, basis, o)
    assert r.converged
    trace = list(r.scf_trace)
    assert len(trace) >= 3
    assert r.n_iter == len(trace)
    # Converging (middle-grid) stage hit the user's tolerance...
    assert trace[-2].grad_norm < o.conv_tol_grad
    # ...and the last row is the non-iterated final-grid recompute:
    # its commutator is grid-difference-dominated, above tolerance.
    assert trace[-1].grad_norm > o.conv_tol_grad
    # The reported energy is the final-grid recompute energy.
    assert r.energy == pytest.approx(trace[-1].energy, abs=1e-9)
    # Cross-stage renumbering is contiguous 1..n_iter.
    assert [s.iter for s in trace] == list(range(1, len(trace) + 1))


def test_multistage_commutator_floor_falls_back_to_final_grid():
    """BUG87-A fallback guard: when the converging (middle) grid cannot
    reach the caller's tolerance — OH·'s ²Π π_x/π_y degeneracy meets the
    middle GridX grid's angular anisotropy at a ~3e-5 commutator floor,
    above conv_tol_grad = 1e-6 — the driver must fall back to a FULL
    converging segment on the final grid (whose floor sits below 1e-6)
    instead of reporting an unconverged SCF after a single recompute."""
    from vibeqc import UHFOptions, run_uhf
    from vibeqc._vibeqc_core import InitialGuess, SCFAccelerator

    mol = _mol([(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * _A])], mult=2)
    basis = BasisSet(mol, "def2-svp")
    o = UHFOptions()
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-6
    # Bounded budget: the middle-grid stall consumes its max_iter, the
    # final-grid fallback then converges in ~15 iterations.
    o.max_iter = 80
    o.scf_accelerator = SCFAccelerator.KDIIS
    o.density_fit = True
    o.aux_basis = _AUX
    o.cosx = True
    o.initial_guess = InitialGuess.SAD
    r = run_uhf(mol, basis, o)
    assert r.converged, (
        "commutator-floor system must converge, via the final-grid "
        "fallback segment or a converging middle grid"
    )
    trace = list(r.scf_trace)
    # Two documented routes end here, and ``converged`` is honest on both
    # (see run_cosx_staged_scf): on the fallback route the last row IS the
    # convergence point and meets the tolerance on the final grid; on the
    # published fast route the middle grid met the tolerance (the row
    # before the single final-grid recompute) and the last row's commutator
    # is the inter-grid exchange difference, above tolerance by design.
    # Which route OH takes depends on whether its degenerate pi pair stays
    # symmetric, which floating-point noise used to decide (#215); either
    # way a run reporting ``converged`` with NEITHER row below tolerance is
    # the defect this witness guards against.
    met = [row.grad_norm < o.conv_tol_grad for row in trace[-2:]]
    assert any(met), (
        "converged reported but neither the final-grid row nor the "
        f"converging-grid row met conv_tol_grad: {[row.grad_norm for row in trace[-2:]]}"
    )


# --- All-methods reach: post-HF inherits RIJCOSX via the reference SCF ----

def test_postscf_inherits_rijcosx_reference(tmp_path):
    """An MP2 job run with a RIJCOSX reference SCF (cosx on rhf_options)
    produces a finite energy close to the same job on an exact reference —
    i.e. COSX reaches post-HF through the mean-field reference, no dead
    cosx fields on MP2/CCSD options needed."""
    mol = _mol(H2O)

    def _mp2(cosx):
        o = RHFOptions(); o.conv_tol_energy = 1e-10
        o.density_fit = True; o.aux_basis = _AUX
        if cosx:
            o.cosx = True
            o.cosx_variant = CosxVariant.FITTED
            o.cosx_grid_level = 2
        res = run_job(mol, basis="def2-svp", method="mp2", rhf_options=o,
                      output=str(tmp_path / ("cosx" if cosx else "ref")))
        return float(res.energy)

    e_cosx = _mp2(True)
    e_ref = _mp2(False)
    assert np.isfinite(e_cosx)
    # The post-HF total shifts only by the COSX fit-band error on the
    # reference (the correlation step is identical).
    assert abs(e_cosx - e_ref) < _E_TOL, (
        f"MP2(cosx ref) Δ={e_cosx - e_ref:.2e} vs MP2(exact ref)"
    )


def test_rohf_rijcosx_matches_direct():
    """ROHF RIJCOSX (opt-in, single grid) converges and lands within the
    COSX fit band of direct ROHF — the open-shell-singlet reference path
    used by ROHF-CCSD / ROHF-MP2."""
    from vibeqc.rohf import ROHFOptions, run_rohf

    oh = _mol([(8, [0.0, 0.0, 0.0]),
               (1, [0.0, 0.0, 0.97 * _A])], mult=2)
    basis = BasisSet(oh, "def2-svp")
    o = ROHFOptions(); o.conv_tol_energy = 1e-9
    e_direct = run_rohf(oh, basis, o).energy

    o = ROHFOptions(); o.conv_tol_energy = 1e-9
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    o.cosx_grid_level = 2
    r = run_rohf(oh, basis, o)
    assert r.converged, "ROHF RIJCOSX GridX2 did not converge"
    assert abs(r.energy - e_direct) < _E_TOL, (
        f"ROHF RIJCOSX Δ={r.energy - e_direct:.2e}"
    )


# --- Default flip: cosx_grid_level defaults to -1 (AUTO GridX) ------------

def test_default_grid_level_is_auto():
    """The SCF option structs now default cosx_grid_level to -1 (AUTO),
    not 0 (legacy) — the flip to a GridX default when cosx=True."""
    for OptCls in (RHFOptions, RKSOptions):
        assert OptCls().cosx_grid_level == -1


def test_default_cosx_uses_gridx_at_normal_tol():
    """With cosx=True and the default grid level + tolerance, the SCF uses a
    GridX tier: its energy matches an explicit GridX2 run and differs from an
    explicit legacy (grid_level=0) run (proving the default flipped)."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")

    def _run(level):
        o = RHFOptions(); o.conv_tol_energy = 1e-10  # default conv_tol_grad
        o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
        if level is not None:
            o.cosx_grid_level = level
        return run_rhf(mol, basis, o).energy

    e_default = _run(None)   # AUTO -> GridX2 for def2-svp (DZ)
    e_gridx2 = _run(2)
    e_legacy = _run(0)
    assert abs(e_default - e_gridx2) < 1e-9, "default should == explicit GridX2"
    assert abs(e_default - e_legacy) > 1e-6, (
        "default should differ from the legacy grid (the flip did not happen)"
    )


def test_default_cosx_falls_back_to_legacy_at_tight_tol():
    """With cosx=True at a tight conv_tol_grad (< 1e-6), the conv-tol-aware
    AUTO gate falls back to the legacy grid: the default-level energy equals
    an explicit grid_level=0 run bit-for-bit (same grid)."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")

    def _run(level):
        o = RHFOptions(); o.conv_tol_energy = 1e-11; o.conv_tol_grad = 1e-8
        o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
        if level is not None:
            o.cosx_grid_level = level
        return run_rhf(mol, basis, o).energy

    e_default_tight = _run(None)   # AUTO + tight -> legacy fallback
    e_legacy = _run(0)
    assert abs(e_default_tight - e_legacy) < 1e-9, (
        "AUTO at tight conv_tol_grad should fall back to the legacy grid"
    )
