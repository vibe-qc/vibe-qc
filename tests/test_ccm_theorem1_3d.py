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

KNOWN FAILURE on this tree (2026-09-16): a second, independent defect entered
between v0.17.0 and v0.17.1 with the periodic GDF refactor. At v0.17.0 the
pair agrees to 1.6e-12 on LiH (1,1,1) and 2.5e-12 on the paper's 25-bohr
hydrogen chain; at v0.17.1 and on this tree both producers moved (LiH by
4.6 mHa/cell) and disagree by 1.42e-4 (LiH (1,1,1)) and 4.4e-5 (chain),
with the one-electron layer identical between routes and the fitted
two-electron operator different, and the Bloch producer 175x slower on
LiH. This file therefore FAILS on (1,1,1) as well until that defect is
repaired; the star-reduction default change above is necessary but not
sufficient. Mechanism (from the tagged sources): v0.17.1 sets
``bulk_sr = gdf_method == "rsgdf" and dim == 3`` in ``periodic_k_gdf.py``,
which sends every 3-D cell's Bloch producer through the new native SR/LR fit
(Gamma fast path bypassed, RSGDF tail declared obsolete, so
``rsgdf_tail_ke_cutoff`` is None on the Bloch side), while the direct
route's ``ccm_neutral_cderi_fold`` still builds the classic FFT fit and
mirrors the fast path's auto-tail (3291.6 Ha on MgO): the two sides fit
different Coulomb operators by construction, and the matched-tail assertion
below fails on MgO for the same reason. The bisection record is kept
with the private campaign notes, outside this repository.
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
    assert not g.parity_held and not d.parity_held
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
    import inspect

    assert inspect.signature(run_ccm_rhf_gdf).parameters["symmetry"].default is False
