"""Low-dimensional gauge consistency of the CCM Coulomb channels (CLAUDE.md §7).

For a **neutral** cell the three Coulomb-bearing pieces must share one gauge, so the
divergent conditional constant ``c`` cancels in the total:

    E_nn        ->  +½ c Z²
    Tr[D·V_ne]  ->  -  c Z N_e
    ½Tr[D·J]    ->  +½ c N_e²
    ------------------------------------
    sum         =  ½ c (N_e - Z)²  =  0     (neutral cell)

**Bug (pinned 2026-07-09, fixed 2026-07-10).** For ``dim < 3`` the direct-torus /
neutral-RI route violated this twice over:

1. ``V_ne``/``E_nn`` came from *bare* lattice sums
   (``_gauge_lat_opts_for_v_ne_and_e_nuc`` returned its input unchanged when
   ``system.dim != 3``) while ``J`` came from the **neutral** (``G+q=0``-dropped)
   cderi. The bare 1-D sums are only conditionally convergent, so ``c`` grew
   without bound with the lattice-sum cutoff and the uncancelled ``-½cZ²`` made the
   total **diverge**: polyethylene (dim=1, sto-3g, (4,1,1)) ran
   ``E_total = -1226.78 -> -1368.41 -> -1518.55`` Ha for ``nuclear_cutoff_bohr``
   ``45 -> 90 -> 180``.

2. More seriously, that neutral ``J`` was **not a Coulomb kernel at all**. Its
   G-mesh (``rsgdf_dense_g_mesh``) pins every non-periodic axis at ``G_perp = 0``,
   so the AO-pair density is replaced by its transverse average and ``4π/|G|²``
   degenerates into a uniform sheet term ``∝ 1/V``. Measured on a dim=1 H₂ chain:
   ``|g_neutral|_max · V = 92.46`` **constant** across transverse vacuum
   ``D = 12/18/24`` bohr (electron repulsion vanishes as the vacuum grows), where
   the mixed-boundary wire four-center is ``D``-invariant at ``1.895512``. The two
   differ by 14 % of the wire scale even after removing the best-fit ``c·S⊗S``. At
   SCF level the dim=1 route ran ``-6.917 -> -6.568`` Ha for ``D = 12 -> 30`` bohr.
   The production multi-k ``run_krhf_periodic_gdf`` shares the defect
   (``-3.458 -> -3.324`` Ha on the same sweep).

So merely giving ``V_ne``/``E_nn`` the neutral gauge would have produced a
cutoff-independent *and meaningless* total -- the §7 paper-over.

**Fix: fail closed.** There is no gauge in which the 3-D-torus route's three
channels agree below three dimensions, so it no longer returns a number.
``_gauge_lat_opts_for_v_ne_and_e_nuc`` raises for ``dim < 3`` (so the multi-k GDF,
the Γ-supercell direct-torus and RIJCOSX all inherit it), and both G-mesh builders
refuse a transverse-collapsed mesh unless the caller opts in for a non-Coulomb use.

The gauge-consistent low-D Hamiltonians are reached **by name**, not by silent
substitution: ``run_ccm_rhf_wire`` (dim=1 mixed-boundary; a *different* operator
from the bare-1/r finite torus, and unverified at scale -- see `192bc645`) and the
four-center route (bare 1/r minimum image, all channels in one gauge). ``dim = 2``
has no mixed-boundary kernel at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc._vibeqc_core import LatticeSumOptions
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _h_chain(vacuum=15.0, nrep=(2, 1, 1)):
    """A 1-D H₂ chain, 6-bohr period, square transverse vacuum box (dim=1)."""
    unit = PeriodicSystem(1, np.diag([6.0, vacuum, vacuum]),
                          [Atom(1, [0.0, vacuum / 2, vacuum / 2]),
                           Atom(1, [1.4, vacuum / 2, vacuum / 2])], 0, 1)
    return CCMSystem(unit, nrep, "sto-3g")


def _h_slab(vacuum=30.0):
    """A 2-D H₂ sheet in a vacuum box (dim=2)."""
    unit = PeriodicSystem(2, np.diag([6.0, 6.0, vacuum]),
                          [Atom(1, [0.0, 0.0, vacuum / 2]),
                           Atom(1, [1.4, 0.0, vacuum / 2])], 0, 1)
    return CCMSystem(unit, (2, 2, 1), "sto-3g")


def test_direct_torus_dim1_fails_closed():
    """dim=1: the divergent bare-sum-vs-collapsed-kernel path is unreachable, and
    the error names the two gauge-consistent alternatives."""
    with pytest.raises(NotImplementedError, match="dim == 1") as exc:
        run_ccm_rhf_direct(_h_chain())
    msg = str(exc.value)
    assert "run_ccm_rhf_wire" in msg and "four-center" in msg


def test_direct_torus_dim2_fails_closed():
    """dim=2 has no mixed-boundary kernel at all, so it must refuse loudly rather
    than return the collapsed-kernel number."""
    with pytest.raises(NotImplementedError, match="dim == 2"):
        run_ccm_rhf_direct(_h_slab())


def test_direct_torus_lowd_cutoff_divergence_is_unreachable():
    """The pinned symptom: the total used to move ~0.3 Ha per doubling of the
    nuclear lattice-sum cutoff. No cutoff can now produce a dim<3 energy at all,
    at any cutoff -- which is the only honest way to make it cutoff-independent
    while the kernel underneath is a sheet term."""
    ccm = _h_chain()
    for nuclear_cutoff in (45.0, 90.0, 180.0):
        lo = LatticeSumOptions()
        lo.nuclear_cutoff_bohr = nuclear_cutoff
        with pytest.raises(NotImplementedError):
            run_ccm_rhf_direct(ccm, lat_opts=lo)


def test_gauge_lat_opts_refuses_low_dimensional_cell():
    """The gauge dispatch site itself: for dim<3 there is no gauge to return. Every
    consumer (multi-k GDF, direct-torus, RIJCOSX) inherits the guard from here.
    dim=3 is untouched and still returns the Ewald-3D clone."""
    from vibeqc._vibeqc_core import CoulombMethod
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc

    for ccm in (_h_chain(), _h_slab()):
        with pytest.raises(NotImplementedError, match="no consistent Coulomb gauge"):
            _gauge_lat_opts_for_v_ne_and_e_nuc(LatticeSumOptions(), ccm.unit_system)

    cube = PeriodicSystem(3, np.diag([6.0, 6.0, 6.0]),
                          [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])], 0, 1)
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(LatticeSumOptions(), cube)
    assert gauge.coulomb_method == CoulombMethod.EWALD_3D


def test_coulomb_g_mesh_refuses_transverse_collapse():
    """The root cause, guarded at source. A dim<3 reciprocal mesh spans only the
    periodic axes -- correct support for a Bloch phase, wrong support for the
    4π/|G|² kernel its consumers apply to it. Opting in explicitly is still allowed
    for a non-Coulomb use, and dim=3 is unaffected."""
    from vibeqc.aux_basis import rsgdf_dense_g_mesh, rsgdf_g_mesh

    chain = _h_chain().unit_system
    with pytest.raises(NotImplementedError, match="transverse-collapsed"):
        rsgdf_dense_g_mesh(chain, ke_cutoff=200.0)
    with pytest.raises(NotImplementedError, match="transverse-collapsed"):
        rsgdf_g_mesh(chain, omega=0.8, precision=1e-4)

    collapsed = rsgdf_dense_g_mesh(chain, ke_cutoff=200.0,
                                   allow_transverse_collapse=True)
    np.testing.assert_allclose(collapsed[:, 1:], 0.0, atol=1e-14)

    cube = PeriodicSystem(3, np.diag([6.0, 6.0, 6.0]),
                          [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [1.4, 0.0, 0.0])], 0, 1)
    dense_3d = rsgdf_dense_g_mesh(cube, ke_cutoff=40.0)
    assert np.abs(dense_3d[:, 1:]).max() > 0.0


def test_low_dimensional_neutral_four_center_is_unconstructible():
    """The neutral cderi/four-center cannot be built for a dim<3 cell any more: it
    would ride the transverse-collapsed mesh.

    This is what forced fail-closed rather than a gauge match. The object it used to
    return was a transverse-uniform sheet term ``∝ 1/V``, not a Coulomb kernel --
    measured before the guard landed, ``|g_neutral|_max · V = 92.46`` was *constant*
    across vacuum ``D = 12/18/24`` bohr (so the electron-electron repulsion vanished
    as the vacuum grew), while the wire four-center is ``D``-invariant. No choice of
    gauge for ``V_ne``/``E_nn`` can repair a kernel with that scaling.
    """
    from vibeqc.periodic.ccm.neutral import ccm_eri_neutral

    with pytest.raises(NotImplementedError, match="transverse-collapsed"):
        ccm_eri_neutral(_h_chain(), ke_cutoff=200.0)


def test_wire_four_center_is_vacuum_box_independent():
    """The positive control the collapsed kernel failed: a genuine mixed-boundary
    kernel is open in the transverse plane, so enlarging the vacuum box cannot
    change it. (``∝1/V`` scaling is the fingerprint of the broken kernel.)"""
    from vibeqc.periodic.ccm.lowd_four_center import ccm_eri_wire

    wire = [np.abs(ccm_eri_wire(_h_chain(vacuum=D))).max() for D in (12.0, 18.0)]
    assert wire[0] == pytest.approx(wire[1], rel=1e-6), (
        f"the wire four-center must be vacuum-box independent, got {wire}")
