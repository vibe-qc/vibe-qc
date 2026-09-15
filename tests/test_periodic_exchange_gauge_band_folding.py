"""EWALD_3D exact exchange: which q -> 0 gauge a k-mesh actually uses.

Reproducer and contract for GitLab #560. ``run_rhf_periodic_multi_k_ewald3d``
serves two different exchange q -> 0 conventions depending on the mesh it is
handed:

* a **single Gamma point** takes the molecular-limit kernel -- every image
  density block is zeroed before the exchange contraction, so the sum has one
  term and there is no ``q + G = 0`` (Madelung) contribution at all. This is
  the CRYSTAL/CCM real-space convention;
* **any other mesh** takes the corrected Ewald split -- ``erfc`` in real space,
  a reciprocal channel per ``q = k - k'``, plus the probe-charge ``q + G = 0``
  term, i.e. PySCF's ``exxdiv='ewald'`` (landed in fafeef8b3, because the bare
  image sum diverges with the lattice cutoff once ``D(g)`` is BvK-torus
  periodic).

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


_ENERGY_CACHE: dict = {}


def _multi_k(mesh, exchange_exxdiv=None):
    """Energy per unit cell; cached, since several tests want the same runs."""
    key = (tuple(mesh), exchange_exxdiv)
    if key not in _ENERGY_CACHE:
        unit = _unit_cell()
        basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
        kwargs = {} if exchange_exxdiv is None else {
            "exchange_exxdiv": exchange_exxdiv
        }
        res = vq.run_rhf_periodic_multi_k_ewald3d(
            unit, basis, vq.monkhorst_pack(unit, list(mesh)), **kwargs
        )
        assert res.converged
        _ENERGY_CACHE[key] = float(res.energy)
    return _ENERGY_CACHE[key]


def test_route_default_changes_exchange_gauge_between_single_gamma_and_a_mesh():
    """#560: the EWALD_3D HF route is not on one gauge across its own meshes.

    Not a wrong number on either side -- each is right in its own convention --
    but a k-convergence sequence taken with the route defaults is not a
    sequence in one gauge, and that is what makes the Gamma-supercell energy
    fail to band-fold onto the multi-k energy.
    """
    e_111 = _multi_k((1, 1, 1))
    e_222 = _multi_k((2, 2, 2))

    # The real mesh is on the corrected gauge and agrees with PySCF
    # exxdiv='ewald' to the Coulomb-method gap.
    assert e_222 == pytest.approx(PYSCF_KRHF_222_EWALD, abs=PYSCF_METHOD_GAP_HA)
    assert abs(e_222 - PYSCF_KRHF_222_NONE) > 0.1

    # The single-Gamma mesh is on neither PySCF gauge: it is the molecular-limit
    # kernel, 21.7 mHa above exxdiv='ewald' and 333 mHa below exxdiv=None.
    assert e_111 == pytest.approx(-1.1235292398235723, abs=1e-8)
    assert abs(e_111 - PYSCF_GAMMA_UNIT_EWALD) > 20.0e-3
    assert abs(e_111 - PYSCF_GAMMA_UNIT_NONE) > 0.3

    # The gauge discontinuity between the route's own two meshes.
    assert abs(e_111 - e_222) > 4.0e-3


def test_single_gamma_mesh_on_the_corrected_gauge_matches_pyscf():
    """#560: ``exchange_exxdiv='ewald'`` makes a Gamma mesh use the split.

    This is the capability the route lacked. Without it there is no way to ask
    for a Gamma-point calculation in the corrected gauge, so the Gamma-supercell
    energy could not be band-folded onto a multi-k energy at all.
    """
    e_corrected = _multi_k((1, 1, 1), exchange_exxdiv="ewald")
    assert e_corrected == pytest.approx(
        PYSCF_GAMMA_UNIT_EWALD, abs=PYSCF_METHOD_GAP_HA
    )
    # It is the corrected gauge, not a relabelled default: 21.7 mHa away from
    # the molecular-limit kernel the same mesh takes by default.
    assert abs(e_corrected - _multi_k((1, 1, 1))) > 20.0e-3


def test_naming_the_gauge_the_route_already_chose_changes_nothing():
    """Negative control: same route, same mesh, gauge named explicitly.

    Both defaults reproduce the explicit request for the gauge the route
    already picks, so the keyword adds a choice and moves no number.

    The bound is floating-point reproducibility, not physics: two separate SCF
    runs of the same calculation differ by at most 1 ULP under OpenMP
    (measured spread 2.2e-16 Ha over three repeats at OMP_NUM_THREADS=2, and
    0.0 at 1 thread), and the thread count itself moves the last bits by
    ~2e-15. NOISE_HA is four orders above that and ten orders below the
    21.7 mHa gauge step this control exists to catch.
    """
    NOISE_HA = 1.0e-12
    assert abs(
        _multi_k((1, 1, 1), exchange_exxdiv="none") - _multi_k((1, 1, 1))
    ) < NOISE_HA
    assert abs(
        _multi_k((2, 2, 2), exchange_exxdiv="ewald") - _multi_k((2, 2, 2))
    ) < NOISE_HA


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
