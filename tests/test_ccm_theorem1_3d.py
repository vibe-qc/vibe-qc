"""Theorem 1 of the Γ-CCM paper on measured-3-D cells, pinned at 1e-10 Ha/cell.

Ruling R1 (2026-08-21): Γ-CCM is the neutral finite-BvK-torus Hamiltonian,
and its two admissible producers are ``run_ccm_rhf_direct`` (real-Γ
supercell representation) and ``run_ccm_rhf_gdf`` (Bloch representation).
Theorem 1 of ``paper1/aiccm2026a.tex`` is the statement that the two agree
exactly at every finite N_c once they realise the same Hamiltonian, so a
spread above the self-consistency floor localises a defect rather than
measuring an approximation.

Why this file exists (2026-09-16, campaign item A1): at v0.17.0 the pair
agreed to 1e-12 on LiH (1,1,1), (2,1,1) and (2,2,2) but differed by
6.7e-5 Ha/cell on (2,2,1), unmoved by ``lindep_tol`` and by tighter
convergence. The cause was the Bloch producer's LIBRARY default
``symmetry=True``: on a partially replicated mesh (some ``N_i > 1``, not
all) ``_ccm_gdf`` routes the per-k-pair fits through the CCM space-group
star reduction, whose finite Bloch cell-list residual is documented as
quantitative rather than zero (GitLab IID 337 excluded fully-3-D meshes for
exactly this reason). LiH/STO-3G is ultra-diffuse, so the reconstruction
residual reached 6.7e-5 Ha/cell there. The runner arm always passed
``symmetry=False``; the library default now does too, so both producers
evaluate the production multi-k Hamiltonian by default and the identity is
pinned here at 1e-10 on the four meshes the paper campaign names, on LiH
and on MgO (whose (1,1,1) auto-tail exercises the IID 307 seam).

These are the campaign's own gates and are deliberately slow: MgO/STO-3G at
(1,1,1) builds the tailed fold cderi (tight-core class), and (2,2,2) is a
112-AO torus.

Fixed by vibe-qc #378 (this tree): between v0.17.0 and v0.17.1 the multi-k
driver started fitting every 3-D cell with its new SR/LR source while the
direct route's fold kept the classic fit, so the two producers disagreed by
1.42e-4 Ha/cell on LiH (1,1,1) and the Bloch side became 175x slower. The
Bloch producers now pin ``fit_source="classic"`` and pass the fold's tail,
and both results record the executed source and tail, which this file
asserts to match. A Bloch-side dense-core parity hold is allowed only where
both tails are none (N_c > 1 on MgO/STO-3G, zeta_max 299.24: measured held
on (2,1,1) and (2,2,1) on every tree, 2026-09-17). The shipped SR/LR
source stays reachable with
``fit_source="sr"`` and is asserted below to be a different operator.
The bisection record is kept with the private campaign notes, outside this
repository.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf
from vibeqc.periodic.exchange_convention import assert_matched_exchange_q0

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

ANG = 1.0 / 0.529177210903
MESHES = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)]
TOL_HA_PER_CELL = 1e-10


def _rocksalt(a_ang, z_a, z_b):
    """fcc primitive cell of a rocksalt AB crystal (lattice vectors as COLUMNS)."""
    a = a_ang * ANG
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return PeriodicSystem(
        3, lat, [Atom(z_a, [0.0, 0.0, 0.0]),
                 Atom(z_b, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1)


SYSTEMS = {
    # The paper's fixed CRYSTAL23 deck geometries (Computational Details).
    "LiH": (4.0840, 3, 1),
    "MgO": (4.2239, 12, 8),
}


def _identity_residual(name, mesh):
    a_ang, z_a, z_b = SYSTEMS[name]
    ccm = CCMSystem(_rocksalt(a_ang, z_a, z_b), mesh, "sto-3g")
    g = run_ccm_rhf_gdf(ccm)          # library default: pair symmetry OFF
    d = run_ccm_rhf_direct(ccm)       # cderi_build="fold"
    assert g.converged and d.converged
    assert not d.parity_held
    # The Bloch producer's dense-core parity hold describes the untailed
    # multi-k fit at N_c > 1, which both sides share: allowed only when both
    # tails are none, never on a tailed pair.
    if g.parity_held:
        assert g.rsgdf_tail_ke_cutoff is None and d.rsgdf_tail_ke_cutoff is None
    # #378: one implementation of the fit on both sides, read back from the
    # driver's executed fit-source record, not assumed from the request.
    assert g.fit_source == "classic" == d.fit_source, (g.fit_source, d.fit_source)
    assert g.executed_fit_source is not None
    assert g.executed_fit_source.source == "classic"
    assert_matched_exchange_q0([g.exchange_q0, d.exchange_q0],
                               context=f"{name}{mesh}")
    # IID 307: same reciprocal support on both sides, else not one operator.
    assert g.rsgdf_tail_ke_cutoff == d.rsgdf_tail_ke_cutoff
    # The library default must be the production (unreduced) multi-k fit.
    assert g.gdf_pair_reduction_factor == 1.0
    return g.energy - d.energy / ccm.n_cells, g, d


@pytest.mark.slow
@pytest.mark.parametrize("mesh", MESHES, ids=lambda m: "x".join(map(str, m)))
def test_theorem1_lih_sto3g(mesh):
    """LiH rocksalt, a = 4.0840 Å: the (2,2,1) mesh is the one that failed
    at 6.7e-5 Ha/cell with the star reduction on."""
    resid, g, d = _identity_residual("LiH", mesh)
    assert abs(resid) < TOL_HA_PER_CELL, (
        f"LiH {mesh}: Bloch - real-Gamma = {resid:.3e} Ha/cell "
        f"(gdf {g.energy!r}, direct/N_c {d.energy / np.prod(mesh)!r})")


@pytest.mark.slow
@pytest.mark.parametrize("mesh", MESHES, ids=lambda m: "x".join(map(str, m)))
def test_theorem1_mgo_sto3g(mesh):
    """MgO rocksalt, a = 4.2239 Å: the tight-core class whose (1,1,1) mesh
    auto-tails (IID 307); the pair must resolve the same tail and agree."""
    resid, g, d = _identity_residual("MgO", mesh)
    assert abs(resid) < TOL_HA_PER_CELL, (
        f"MgO {mesh}: Bloch - real-Gamma = {resid:.3e} Ha/cell "
        f"(gdf {g.energy!r}, direct/N_c {d.energy / np.prod(mesh)!r})")


def test_star_reduction_is_opt_in_on_the_bloch_producer():
    """The library default of ``run_ccm_rhf_gdf`` is the production
    multi-k fit; the star reduction is reachable only by ``symmetry=True``
    and is then recorded on the result (``gdf_pair_reduction_factor``),
    so a reduced number can never pass for an unreduced one. Cheap cell."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    ccm = CCMSystem(cell, (2, 1, 1), "sto-3g")
    g = run_ccm_rhf_gdf(ccm)
    assert g.gdf_pair_reduction_factor == 1.0
    assert g.fit_source == "classic"
    import inspect

    assert inspect.signature(run_ccm_rhf_gdf).parameters["symmetry"].default is False
    # The reduction cannot ride on the classic pin (#378): the driver refuses
    # a private fit adapter together with an explicit fit source, and the
    # reduction is approximate on 3-D cells anyway.
    with pytest.raises(ValueError, match="symmetry=True"):
        run_ccm_rhf_gdf(ccm, symmetry=True)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_theorem1_lih_ks_211(functional):
    """Closed-shell KS at N_c = 2 (#378, the KS half): both producers' XC
    density-image sums reach the lattice cutoff. With the real-Γ route at the
    grid's 10-bohr partition radius instead, the pair was off by 1.2e-3 (PBE)
    and 9.1e-4 (PBE0) Ha/cell with identical fits, and the real-Γ side was the
    truncated one (PySCF KRKS on the same cell: 3.6 mHa below it at 10 bohr,
    2.5 mHa at the cutoff). At v0.17.0 the pure-functional pair was off by
    1.0e-4 for another reason, the unfitted Hartree for pure DFT, which the
    classic pin no longer takes. The Γ rows of this identity are
    ``tests/test_ccm_rks_direct.py::test_rks_direct_matches_gdf_*_gamma_post_fix``."""
    from vibeqc.periodic.ccm.direct import run_ccm_rks_direct
    from vibeqc.periodic.ccm.ri import run_ccm_rks_gdf

    a_ang, z_a, z_b = SYSTEMS["LiH"]
    ccm = CCMSystem(_rocksalt(a_ang, z_a, z_b), (2, 1, 1), "sto-3g")
    d = run_ccm_rks_direct(ccm, functional)
    g = run_ccm_rks_gdf(ccm, functional)
    assert d.converged and g.converged
    assert g.fit_source == "classic" == d.fit_source
    assert g.rsgdf_tail_ke_cutoff == d.rsgdf_tail_ke_cutoff
    resid = g.energy - d.energy / ccm.n_cells
    assert abs(resid) < TOL_HA_PER_CELL, f"LiH/{functional} (2,1,1): {resid:.3e} Ha/cell"


def _chain_25bohr():
    """The paper's four-atom alternating hydrogen chain in a 25-bohr box,
    declared 3-D (the largest transverse box the measured-dimensionality guard
    admits without an opt-in)."""
    lat = np.diag([4.0 * ANG, 25.0, 25.0])
    atoms = [Atom(1, [x * ANG, 0.0, 0.0]) for x in (0.0, 0.8, 2.0, 2.8)]
    return CCMSystem(PeriodicSystem(3, lat, atoms, 0, 1), (1, 1, 1), "sto-3g")


@pytest.mark.slow
def test_shipped_sr_source_is_a_different_operator_and_says_so():
    """Negative control for #378: the driver's shipped SR/LR source, requested
    deliberately, is recorded as such and does NOT reproduce the real-Γ fold
    (4.4e-5 Ha/cell on this cell at v0.17.1 and HEAD before the fix), while
    the pinned default does."""
    ccm = _chain_25bohr()
    d = run_ccm_rhf_direct(ccm)
    g_classic = run_ccm_rhf_gdf(ccm)
    g_sr = run_ccm_rhf_gdf(ccm, fit_source="sr")
    assert g_classic.fit_source == "classic" and g_sr.fit_source == "sr"
    assert abs(g_classic.energy - d.energy) < 1e-9
    assert abs(g_sr.energy - d.energy) > 1e-6
