"""EWALD_3D exact exchange: which q -> 0 gauge a k-mesh actually uses.

Reproducer and contract for GitLab #560 and #23. ``run_rhf_periodic_multi_k_ewald3d``
serves two exchange q -> 0 conventions:

* the **corrected Ewald split** -- ``erfc`` in real space, a reciprocal channel
  per ``q = k - k'``, plus the probe-charge ``q + G = 0`` term, i.e. PySCF's
  ``exxdiv='ewald'`` (landed in fafeef8b3, because the bare image sum diverges
  with the lattice cutoff once ``D(g)`` is BvK-torus periodic). Since #23 it is
  the default at **every** mesh, a single Gamma point included;
* the **molecular-limit kernel** -- every image density block is zeroed before
  the exchange contraction, so the sum has one term and there is no
  ``q + G = 0`` (Madelung) contribution at all. This is the CRYSTAL/CCM
  real-space convention, available at a single Gamma point as
  ``exchange_exxdiv='none'``. Until #23 it was the single-Gamma default, so the
  route switched gauge between (1,1,1) and any other mesh.

Both are legitimate conventions with the same infinite-k limit, and they are
**not equal at any finite mesh**: Carrier, Rohra & Goerling, Phys. Rev. B 75,
205126 (2007), doi:10.1103/PhysRevB.75.205126, Sec. IV -- the singularity
correction and the uncorrected exchange energy "vary oppositely with increasing
number of k points", the corrected one converging far faster. Peintinger &
Bredow, J. Comput. Chem. 35, 839 (2014), doi:10.1002/jcc.23550, Eqs. (21)-(25)
and Tables 2-3, demonstrate CCM == SCM-Gamma == k-mesh in the *uncorrected*
real-space convention.

So a Gamma-supercell energy may only be band-folded onto a multi-k energy when
both sides are on one gauge. Selecting it is what ``exchange_exxdiv`` does.

Fixture: dilute H2 molecular crystal, H2 (d = 1.4 bohr) on a simple-cubic
a = 8 bohr lattice, STO-3G -- the same cell as
``tests/test_ccm_periodic_3d.py`` and
``examples/regression/parity_ccm_scm_vs_pyscf.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq

A_BOHR = 8.0
D_BOHR = 1.4

# External anchors. PySCF 2.14.0 run OUT OF PROCESS (CLAUDE.md paragraph 10 --
# vibe-qc never imports it): ``scf.RHF(cell).density_fit()`` and
# ``scf.KRHF(cell, cell.make_kpts([2,2,2])).density_fit()`` with
# ``cell.precision = 1e-10`` and ``mf.conv_tol = 1e-10`` on this exact cell.
# Regenerate with ``examples/regression/parity_ccm_scm_vs_pyscf.py`` (PART C).
# PySCF band-folds these two to 8.7e-13 Ha, in either gauge.
PYSCF_GAMMA_UNIT_EWALD = -1.1452192802432282
PYSCF_KRHF_222_EWALD = -1.1191323122707395
# The same two calculations with exxdiv=None, for scale: the gauge is worth
# madelung * n_elec / 2 = 0.35466218 and 0.17733109 Ha per cell respectively.
PYSCF_GAMMA_UNIT_NONE = -0.7905570953083024
PYSCF_KRHF_222_NONE = -0.9418012198032002

# vibe-qc's own Ewald-3D-FFT-vs-GDF Coulomb-method gap on this cell, measured at
# both meshes and in both gauges: 19-58 uHa. 5e-5 Ha is the bar every assertion
# below uses against a PySCF anchor.
PYSCF_METHOD_GAP_HA = 5.0e-5


def _unit_cell() -> "vq.PeriodicSystem":
    return vq.PeriodicSystem(
        3,
        np.diag([A_BOHR, A_BOHR, A_BOHR]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, D_BOHR])],
        charge=0,
        multiplicity=1,
    )


_RESULT_CACHE: dict = {}


def _multi_k_result(mesh, exchange_exxdiv=None):
    """The SCF result; cached, since several tests want the same runs."""
    key = (tuple(mesh), exchange_exxdiv)
    if key not in _RESULT_CACHE:
        unit = _unit_cell()
        basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
        kwargs = {} if exchange_exxdiv is None else {
            "exchange_exxdiv": exchange_exxdiv
        }
        res = vq.run_rhf_periodic_multi_k_ewald3d(
            unit, basis, vq.monkhorst_pack(unit, list(mesh)), **kwargs
        )
        assert res.converged
        _RESULT_CACHE[key] = res
    return _RESULT_CACHE[key]


def _multi_k(mesh, exchange_exxdiv=None):
    """Energy per unit cell."""
    return float(_multi_k_result(mesh, exchange_exxdiv).energy)


def test_route_default_is_one_gauge_at_every_mesh():
    """#23: the EWALD_3D HF route default is the corrected split at every mesh.

    Until #23 a single Gamma point took the molecular-limit kernel by default
    and every other mesh the corrected split, so a k-convergence sequence taken
    with defaults was not a sequence in one gauge: (1,1,1) -> (2,2,2) read
    -1.1235293 -> -1.1191129 Ha, a 4.4 mHa step that mixed real k convergence
    with a 21.67 mHa gauge switch. Both meshes now agree with PySCF
    exxdiv='ewald' to the Coulomb-method gap.
    """
    e_111 = _multi_k((1, 1, 1))
    e_222 = _multi_k((2, 2, 2))

    assert e_111 == pytest.approx(PYSCF_GAMMA_UNIT_EWALD, abs=PYSCF_METHOD_GAP_HA)
    assert e_222 == pytest.approx(PYSCF_KRHF_222_EWALD, abs=PYSCF_METHOD_GAP_HA)
    assert abs(e_111 - PYSCF_GAMMA_UNIT_NONE) > 0.3
    assert abs(e_222 - PYSCF_KRHF_222_NONE) > 0.1

    # The mesh step is now k convergence alone, the same step PySCF takes
    # between the two meshes in this gauge.
    assert (e_111 - e_222) == pytest.approx(
        PYSCF_GAMMA_UNIT_EWALD - PYSCF_KRHF_222_EWALD, abs=2 * PYSCF_METHOD_GAP_HA
    )


def test_single_gamma_mesh_on_the_corrected_gauge_matches_pyscf():
    """#560: ``exchange_exxdiv='ewald'`` puts a Gamma mesh on the split.

    The capability #560 added; since #23 it is also the default, and naming it
    must fail closed where the split is unavailable rather than fall back.
    """
    e_corrected = _multi_k((1, 1, 1), exchange_exxdiv="ewald")
    assert e_corrected == pytest.approx(
        PYSCF_GAMMA_UNIT_EWALD, abs=PYSCF_METHOD_GAP_HA
    )


def test_molecular_limit_gauge_stays_available_at_a_single_gamma_point():
    """``exchange_exxdiv='none'``: the pre-#23 single-Gamma default, on request.

    The molecular-limit kernel zeroes every image density block, so its sum
    has one term and no ``q + G = 0`` contribution: the CRYSTAL/CCM
    real-space convention, 21.7 mHa above exxdiv='ewald' here and 333 mHa
    below exxdiv=None.
    """
    e_none = _multi_k((1, 1, 1), exchange_exxdiv="none")
    # Re-pinned at #23. The #560-era literal -1.1235292398235723 had drifted
    # by 9.60e-8 Ha by v0.17.4 (see the note on GitLab #23).
    assert e_none == pytest.approx(-1.1235293358, abs=1e-8)
    assert abs(e_none - PYSCF_GAMMA_UNIT_EWALD) > 20.0e-3
    assert abs(e_none - PYSCF_GAMMA_UNIT_NONE) > 0.3
    assert abs(e_none - _multi_k((1, 1, 1))) > 20.0e-3


def test_naming_the_default_gauge_changes_nothing():
    """Negative control: same route, same mesh, the default gauge named.

    The default reproduces an explicit ``'ewald'`` at both meshes, so the
    keyword names the default and moves no number.

    The bound is floating-point reproducibility, not physics: two separate SCF
    runs of the same calculation differ by at most 1 ULP under OpenMP
    (measured spread 2.2e-16 Ha over three repeats at OMP_NUM_THREADS=2, and
    0.0 at 1 thread), and the thread count itself moves the last bits by
    ~2e-15. NOISE_HA is four orders above that and ten orders below the
    21.7 mHa gauge step this control exists to catch.
    """
    NOISE_HA = 1.0e-12
    for mesh in ((1, 1, 1), (2, 2, 2)):
        assert abs(
            _multi_k(mesh, exchange_exxdiv="ewald") - _multi_k(mesh)
        ) < NOISE_HA


def test_results_record_the_gauge_that_ran():
    """#23: a result names its exchange gauge, so nothing downstream guesses it.

    The analytic gradient and the runner's ``.system`` ``[run].exchange_q0``
    used to infer the gauge from the mesh, which stopped being possible once a
    single Gamma point could run either one. The Gamma-only driver always runs
    the molecular limit.
    """
    from vibeqc.periodic.exchange_convention import (
        BVK_EWALD,
        MOLECULAR_LIMIT,
        executed_exchange_q0,
    )

    assert _multi_k_result((1, 1, 1)).exchange_q0 == BVK_EWALD
    assert _multi_k_result((1, 1, 1), "ewald").exchange_q0 == BVK_EWALD
    assert _multi_k_result((2, 2, 2)).exchange_q0 == BVK_EWALD
    assert _multi_k_result((1, 1, 1), "none").exchange_q0 == MOLECULAR_LIMIT

    unit = _unit_cell()
    basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
    gamma = vq.run_rhf_periodic_gamma_ewald3d(unit, basis)
    assert gamma.exchange_q0 == MOLECULAR_LIMIT
    # Same gauge; the two drivers' default lattice options differ, which
    # moves the energy by 2e-6 Ha (tests/test_ccm_periodic_3d.py uses the
    # same bound). The 21.7 mHa gauge step is four orders larger.
    assert gamma.energy == pytest.approx(
        _multi_k((1, 1, 1), exchange_exxdiv="none"), abs=1e-5
    )
    # The declared gauge wins over the knob a caller passed.
    assert executed_exchange_q0(gamma, "ewald") == MOLECULAR_LIMIT


def test_uncorrected_gauge_is_refused_on_a_real_mesh():
    """Fail closed: the bare image sum does not converge off a single Gamma.

    fafeef8b3's finding, restated as a guard -- H2/STO-3G in a 12-bohr cube on
    a (2,1,1) mesh moved 236 mHa over a 12-to-20-bohr cutoff sweep. The refusal
    fires before any SCF work.
    """
    unit = _unit_cell()
    basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="single Gamma point"):
        vq.run_rhf_periodic_multi_k_ewald3d(
            unit,
            basis,
            vq.monkhorst_pack(unit, [2, 2, 2]),
            exchange_exxdiv="none",
        )


def test_unknown_exchange_exxdiv_is_refused():
    unit = _unit_cell()
    basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="exchange_exxdiv must be"):
        vq.run_rhf_periodic_multi_k_ewald3d(
            unit,
            basis,
            vq.monkhorst_pack(unit, [1, 1, 1]),
            exchange_exxdiv="madelung",
        )
