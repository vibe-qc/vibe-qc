"""The RSGDF high-|G| tail is the Hamiltonian-identity seam of the two neutral
controls (GitLab IID 307).

``run_ccm_rhf_direct`` and ``run_ccm_rhf_gdf`` are documented as evaluating the
*same* block-circulant neutral Hamiltonian in two representations. They did not:
at ``nrep=(1,1,1)`` the GDF sibling delegates to the Γ fast path
(:func:`vibeqc.pbc_gdf.run_pbc_gdf_rhf`), which auto-sizes an RSGDF high-``|G|``
tail on the tight-core class, while the direct/fold route had no ``tail_ke_cutoff``
plumbing at all and could not request one. Two different Hamiltonians, and no
result field said so.

Measured on MgO fcc (a = 4.21 Å = 7.956 bohr) / STO-3G / ``nrep=(1,1,1)``,
2026-08-27, before the fix::

    direct  E = -271.5489661527 Ha    (untailed)
    gdf     E = -271.0496299758 Ha    (auto-tailed at 3291.6114 Ha)
    direct - gdf = -4.993362e-01 Ha/cell

The **tailed** side is the accurate one: -271.04963 sits 0.3 mHa from the PySCF
MDF reference -271.0499 quoted in
:func:`vibeqc.pbc_gdf._gamma_dense_core_gdf_parity_held`, and the untailed side
carries the documented dense-core P01 offset. After the fix the same pair agrees
to ``+4.09e-11`` Ha/cell.

Pinned here:

* the resolver mirrors the sibling route's decision exactly -- same classifier,
  same inputs, at ``N_c = 1``; ``None`` at ``N_c > 1`` where the multi-k loop
  deliberately does not auto-tail (so this fix moves nothing there);
* an explicit ``tail_ke_cutoff`` overrides the classifier, and a request the
  builder would ignore (``0``, or anything at or below the base mesh)
  normalises to ``None`` so an opt-out is the historical call exactly;
* end-to-end direct-vs-GDF parity on two tight-core cells (fcc Be, diamond),
  each with the untailed control asserted to *fail* the same gate -- a parity
  test whose negative branch is unreachable is not evidence (LEARNINGS
  L95/L99, and the ``theorem1_parity`` gate that reported a pass-shaped result
  over an empty domain is exactly how this defect survived). MgO itself is
  pinned at the classifier level only; see the note above the end-to-end
  section for the measurement that decided that;
* both result types record the tail they actually used.

Reference for the tail itself: Ye & Berkelbach, *J. Chem. Phys.* **154**, 131104
(2021), doi:10.1063/5.0046617 (range-separated periodic Gaussian density
fitting); the periodic GDF factorisation is Sun, Berkelbach, McClain & Chan,
*J. Chem. Phys.* **147**, 164119 (2017), doi:10.1063/1.4998644.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, PeriodicSystem, make_basis
from vibeqc.pbc_gdf import _auto_rsgdf_tail_ke_cutoff
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
from vibeqc.periodic.ccm.neutral import ccm_neutral_tail_ke_cutoff
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_A = 1.0 / 0.529177210903

#: The D1 cross-route gate this issue was filed against.
TOL_HA_PER_CELL = 1e-6


def _fcc(a_bohr):
    return 0.5 * a_bohr * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float).T


def _binary_fcc(a_ang, z1, z2, frac2=(0.5, 0.5, 0.5)):
    lat = _fcc(a_ang * _A)
    atoms = [Atom(z1, [0, 0, 0])]
    if z2 is not None:
        atoms.append(Atom(z2, (lat @ np.asarray(frac2, dtype=float)).tolist()))
    return PeriodicSystem(3, lat, atoms)


def _be(nrep=(1, 1, 1)):
    """fcc Be / STO-3G -- the cheapest cell that trips the tail classifier.

    ``zeta_max = 30.168`` so ``10 x zeta_max = 301.7 > 200``; the auto tail is
    331.85 Ha, a G-ball only 2.1x the base mesh, so the whole gate runs in
    seconds while still moving the energy 160x the tolerance.
    """
    return CCMSystem(_binary_fcc(3.20, 4, None), nrep, "sto-3g")


def _diamond(nrep=(1, 1, 1)):
    return CCMSystem(
        _binary_fcc(3.567, 6, 6, (0.25, 0.25, 0.25)), nrep, "sto-3g")


def _mgo(nrep=(1, 1, 1)):
    """The canonical P01 cell of IID 307 (a = 4.21 Å = 7.956 bohr)."""
    return CCMSystem(_binary_fcc(4.21, 12, 8), nrep, "sto-3g")


def _lih(nrep=(1, 1, 1)):
    lat = _fcc(7.72)
    atoms = [Atom(3, [0, 0, 0]), Atom(1, (lat @ [0.5, 0.5, 0.5]).tolist())]
    return CCMSystem(PeriodicSystem(3, lat, atoms), nrep, "sto-3g")


# --------------------------------------------------------------------------
# The resolver mirrors the sibling route (structural, no SCF)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build", [_be, _diamond, _mgo, _lih])
def test_n1_resolver_matches_the_gamma_fast_path_classifier(build):
    """At N_c = 1 the resolver returns the sibling's own classifier answer.

    Not "a tail that looks similar": the *same* function on the same (unit
    system, AO basis, ke_cutoff) inputs, because ``ke_cutoff`` is the sibling's
    ``rsgdf_ke_cutoff`` and both reach the builder as its ``ke_cutoff``.
    """
    ccm = build((1, 1, 1))
    unit = ccm.unit_system
    basis = make_basis(unit.unit_cell_molecule(), ccm.basis_name)
    expected = _auto_rsgdf_tail_ke_cutoff(
        unit, "rsgdf", basis, None, rsgdf_ke_cutoff=200.0)
    assert ccm_neutral_tail_ke_cutoff(ccm) == expected


def test_lih_sto3g_needs_no_tail_and_diamond_does():
    """The classifier's own split, pinned so a threshold change is visible.

    LiH/STO-3G (``zeta_max = 16.120``) sits under ``10 x zeta_max <= 200`` and
    is the one N=1 point that already passed at -1.55e-12 in the original
    5-point report; diamond and MgO are over it.
    """
    assert ccm_neutral_tail_ke_cutoff(_lih()) is None
    assert ccm_neutral_tail_ke_cutoff(_diamond()) == pytest.approx(787.785207)
    assert ccm_neutral_tail_ke_cutoff(_mgo()) == pytest.approx(3291.6114)


@pytest.mark.parametrize("nrep", [(2, 1, 1), (2, 2, 1), (2, 2, 2)])
def test_multik_meshes_are_not_auto_tailed(nrep):
    """N_c > 1 must resolve to ``None`` -- this fix moves nothing there.

    The multi-k GDF loop deliberately does not auto-tail
    (``_warn_multik_dense_core_gdf_parity_hold`` records the hold and calls
    multi-k auto-tailing an open maintainer call). Auto-tailing only the fold
    would relocate the 0.5 Ha disagreement from N=1 to N>=2 instead of closing
    it, so the two routes stay on the held-but-agreeing side together.
    """
    assert ccm_neutral_tail_ke_cutoff(_mgo(nrep)) is None


def test_explicit_tail_is_honoured_and_wins_over_the_classifier():
    """An explicit request overrides the classifier, in both directions."""
    assert ccm_neutral_tail_ke_cutoff(_mgo(), tail_ke_cutoff=1234.5) == 1234.5
    # ...including at a multi-k mesh, where the classifier itself says None.
    assert ccm_neutral_tail_ke_cutoff(
        _mgo((2, 1, 1)), tail_ke_cutoff=900.0) == 900.0


@pytest.mark.parametrize("requested", [0.0, 100.0, 200.0])
def test_a_tail_the_builder_would_ignore_normalises_to_none(requested):
    """``0`` and any value at or below the base mesh resolve to ``None``.

    The builders sweep the tail only when ``tail_ke_cutoff > ke_cutoff``, so
    such a request applies no tail. Normalising it here keeps an opt-out
    byte-for-byte identical to the historical call -- ``0.0`` and ``None`` are
    *not* interchangeable at every downstream site
    (``_rsgdf_weighted_3c_tensor_gradient_bloch`` rejects a non-``None`` tail
    alongside external kernel weights) -- and stops a result from recording a
    tail that never ran, which is the reporting failure this issue is about.
    """
    assert ccm_neutral_tail_ke_cutoff(
        _mgo(), ke_cutoff=200.0, tail_ke_cutoff=requested) is None


def test_prebuilt_cderi_refuses_a_tail_request():
    """A prebuilt cderi already fixes its tail, so a tail request must fail
    closed rather than be silently dropped -- the same contract ``aux_basis``
    already had, and the failure mode this whole issue is about."""
    ccm = _be()
    # Raises during argument validation, before the cderi is ever contracted,
    # so a placeholder tensor is enough and no SCF has to run.
    with pytest.raises(ValueError, match="tail_ke_cutoff is ignored"):
        run_ccm_rhf_direct(ccm, cderi=np.zeros((1, 1, 1)), tail_ke_cutoff=500.0)


# --------------------------------------------------------------------------
# End-to-end parity, each with its own falsifying control
# --------------------------------------------------------------------------

def _assert_route_parity(ccm, *, min_prefix_gap):
    """direct == gdf post-fix, AND the untailed control still fails the gate.

    The second half is the point: without it this is a gate whose negative
    branch is never exercised, which is precisely the defect shape that let
    ``theorem1_parity`` report a pass over an empty domain.
    """
    gdf = run_ccm_rhf_gdf(ccm)
    direct = run_ccm_rhf_direct(ccm)
    untailed = run_ccm_rhf_direct(ccm, tail_ke_cutoff=0.0)

    assert gdf.converged and direct.converged and untailed.converged

    # Both routes ran the same Hamiltonian, and both say so.
    assert direct.rsgdf_tail_ke_cutoff == pytest.approx(
        gdf.rsgdf_tail_ke_cutoff)
    assert direct.rsgdf_tail_ke_cutoff is not None

    gap = float(direct.energy) - float(gdf.energy)
    assert abs(gap) < TOL_HA_PER_CELL, (
        f"direct - gdf = {gap:+.6e} Ha/cell exceeds {TOL_HA_PER_CELL:g}")

    prefix_gap = float(untailed.energy) - float(gdf.energy)
    assert abs(prefix_gap) > min_prefix_gap, (
        "the untailed control no longer moves the energy, so this gate can no "
        f"longer discriminate (got {prefix_gap:+.6e} Ha/cell)")
    # An opt-out records "no tail applied", not "a tail of 0" -- see
    # test_a_tail_the_builder_would_ignore_normalises_to_none.
    assert untailed.rsgdf_tail_ke_cutoff is None
    return gap, prefix_gap


def test_be_fcc_n1_direct_matches_gdf():
    """Cheapest tight-core gate: pre-fix -1.602e-04 Ha/cell, post-fix -8.5e-12."""
    gap, prefix_gap = _assert_route_parity(_be(), min_prefix_gap=1e-5)
    assert abs(gap) < abs(prefix_gap) / 1e4


def test_diamond_n1_direct_matches_gdf():
    """Pre-fix -1.589e-02 Ha/cell, post-fix -6.8e-11."""
    _assert_route_parity(_diamond(), min_prefix_gap=1e-3)


# --------------------------------------------------------------------------
# Why MgO -- the canonical cell of IID 307 -- is NOT gated end to end here
#
# It was, and the gate was removed after measuring it: the tailed MgO fold
# (3291.6 Ha, i.e. a 66.8x reciprocal ball) did not finish inside 33 minutes,
# running at ~1 core the whole time even though the pair-FT kernel carries an
# `omp parallel for` (2026-08-27). A test nobody will run is not evidence, so
# the end-to-end numerics are gated on fcc Be and diamond instead (15 tests,
# 266 s as a file on the authoring box). The mechanism is basis-driven, not system-specific -- it is the
# same classifier, the same builder kwarg and the same falsifying control on
# all three -- and diamond's pre-fix gap, -1.589e-02 Ha/cell, already sits
# four orders above the 1e-6 tolerance.
#
# What MgO still pins here: its classifier answer (3291.6114 Ha, in
# test_lih_sto3g_needs_no_tail_and_diamond_does) and the N_c > 1 no-op. Its
# end-to-end numbers were reproduced by hand and are recorded in this module's
# docstring, in CHANGELOG.md, and in the commit body.
#
# Before re-adding an MgO SCF gate, fix the cost: the fold's tail is
# unscreened (build_lpq_native_fft, which the Gamma fast path uses, takes
# tail_pair_ft_screen; build_lpq_bloch_native_fft, which the fold uses, has no
# such kwarg), and the single-core behaviour above is unexplained. Both are
# recorded in handovers/HANDOVER_PERIODIC_BUGSWEEP.md.
# --------------------------------------------------------------------------
