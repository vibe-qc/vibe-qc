"""Fixed-H periodic supercell/character band-folding controls.

For one already specified periodic block-circulant Hamiltonian, a Γ-point
supercell calculation and the matching primitive-cell character mesh are
Fourier-related:

    E_CCM(N cluster) == E_Γ(N supercell) == E_periodic-HF(unit cell, k-mesh=N).

These tests verify that ordinary periodic-HF representation identity on a
dilute molecular crystal where the Γ-only Ewald-3D driver is reliable (the
molecular-limit density convention holds; a space-filling ionic crystal such as
LiH/STO-3G is correctly refused by the driver and needs the multi-k GDF route).
They do not show that periodic GDF evaluates the union-and-weight Γ-CCM
construction, nor that Γ-CCM and χ-CCM are identical.

**The identity holds within one exact-exchange q → 0 gauge, not across two**
(#560). The corrected Ewald split and the molecular-limit kernel are different
finite-size conventions with the same infinite-k limit (Carrier, Rohra &
Görling, Phys. Rev. B 75, 205126 (2007), doi:10.1103/PhysRevB.75.205126,
Sec. IV), and the EWALD_3D route picks one per mesh unless told otherwise. The
band-folding test therefore names ``exchange_exxdiv='ewald'`` on both sides;
``tests/test_periodic_exchange_gauge_band_folding.py`` pins what each gauge is
worth here (21.7 mHa at a single Γ point, 2.43 mHa on the (2,2,2) supercell).

The external PySCF cross-check (converging to a common TDL within the Ewald-3D
vs GDF Coulomb-method gap) lives, per CLAUDE.md §10, in
``examples/regression/parity_ccm_scm_vs_pyscf.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic.ccm import CCMSystem

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

# Dilute H2 molecular crystal: H2 (d=1.4 bohr) on a simple-cubic lattice.
A_BOHR = 8.0
D_BOHR = 1.4

# Ewald split parameter: both sides take the driver default. Before GitLab
# #651 this file pinned omega = 0.35 on both calls as a workaround, because the
# supercell's volume-scaled default (0.175 bohr^-1 for the 16-bohr cube) had
# not decayed by the 15-bohr real-space cutoff and cost ~6e-5 Ha/cell. The
# default is now bounded below by sqrt(-ln tol) / cutoff_bohr = 0.3504, which
# both the 8-bohr unit cell and the 16-bohr supercell resolve to, so the two
# sides share one alpha without naming it (pbc_bipole_common.default_ewald_alpha).


def _unit_cell():
    return vq.PeriodicSystem(
        3, np.diag([A_BOHR, A_BOHR, A_BOHR]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, D_BOHR])],
        charge=0, multiplicity=1,
    )


def test_ccm_periodic_plumbing_and_overlap_guard():
    """CCM(1,1,1) Γ run is finite/converged and the Fig-6 overlap guard passes.

    The (1,1,1) cyclic cluster is the unit cell with cyclic BC = a Γ-point
    periodic HF; it must equal vibe-qc's multi-k driver at k=(1,1,1) (a trivial
    band-folding identity that also checks the CCMSystem→periodic-driver wiring).
    """
    ccm = CCMSystem(_unit_cell(), (1, 1, 1), "sto-3g")
    emin, _ = ccm.check_overlap_spectrum()      # Fig-6 guard (must be PD)
    assert emin > 1e-5

    res = vq.run_rhf_periodic_gamma_ewald3d(ccm.cluster_system, ccm.basis)
    assert res.converged
    assert np.isfinite(res.energy)

    unit = _unit_cell()
    basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
    mk = vq.run_rhf_periodic_multi_k_ewald3d(unit, basis, vq.monkhorst_pack(unit, [1, 1, 1]))
    assert res.energy == pytest.approx(mk.energy, abs=1e-5)


@pytest.mark.slow
def test_ccm_equals_scm_gamma_band_folding():
    """Γ-supercell/N == k-folded periodic HF for one fixed operator, to µHa.

    A Γ-point HF on the (2,2,2) supercell (the cyclic cluster) divided by 8
    equals the unit cell's periodic HF on a (2,2,2) Monkhorst–Pack mesh. This is
    band folding for the ordinary periodic Hamiltonian; it is not a comparison
    of the Γ-CCM and χ-CCM construction maps.

    Both sides name ``exchange_exxdiv='ewald'`` (#560). A (1,1,1) mesh *is* the
    Γ-point calculation of the supercell; the keyword is what stops the driver
    short-circuiting to the molecular-limit kernel, which is a different gauge
    and does not band-fold onto a (2,2,2) mesh (2.43 mHa apart, pinned in
    ``tests/test_periodic_exchange_gauge_band_folding.py``).

    Externally anchored: PySCF 2.14.0 out of process gives -1.1191323123 Ha per
    cell for both sides of this identity with ``exxdiv='ewald'`` (see
    ``examples/regression/parity_ccm_scm_vs_pyscf.py``); the two vibe-qc numbers
    below sit -3.0 µHa and +19.4 µHa from it.
    """
    nrep = (2, 2, 2)
    ccm = CCMSystem(_unit_cell(), nrep, "sto-3g")
    res = vq.run_rhf_periodic_multi_k_ewald3d(
        ccm.cluster_system,
        ccm.basis,
        vq.monkhorst_pack(ccm.cluster_system, [1, 1, 1]),
        exchange_exxdiv="ewald",
    )
    e_ccm_per_cell = res.energy / (nrep[0] * nrep[1] * nrep[2])

    unit = _unit_cell()
    basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
    mk = vq.run_rhf_periodic_multi_k_ewald3d(
        unit,
        basis,
        vq.monkhorst_pack(unit, list(nrep)),
        exchange_exxdiv="ewald",
    )

    assert res.converged and mk.converged
    # Same Hamiltonian, same gauge, two BZ-sampling representations → equal to
    # µHa. Measured 2.24e-5 Ha; the residual is Coulomb numerics on a 16-bohr
    # versus an 8-bohr cell, not a gauge difference.
    assert e_ccm_per_cell == pytest.approx(mk.energy, abs=5e-5)


@pytest.mark.slow
def test_ccm_periodic_converges_with_cluster_size():
    """The CCM energy/cell *converges* (bounded, smooth) as the cluster grows.

    Γ-point of a larger supercell samples the BZ more densely; the energy/cell
    shifts by only a few mHa (BZ averaging) and stays bounded — it does **not**
    diverge with cluster size. (Contrast the HANDOVER note that Ewald-3D applied
    to a 1-D chain *does* diverge because the vacuum is treated as periodic; this
    is the genuinely-3-D regime where the route is sound. CLAUDE.md §7: a
    diverging periodic energy would be a bug, not a convergence aid.)
    """
    e = {}
    for nrep in [(1, 1, 1), (2, 2, 2)]:
        ccm = CCMSystem(_unit_cell(), nrep, "sto-3g")
        r = vq.run_rhf_periodic_gamma_ewald3d(ccm.cluster_system, ccm.basis)
        assert r.converged
        e[nrep] = r.energy / (nrep[0] * nrep[1] * nrep[2])
    # Bounded few-mHa BZ-averaging shift, not divergence.
    assert abs(e[(2, 2, 2)] - e[(1, 1, 1)]) < 0.02
