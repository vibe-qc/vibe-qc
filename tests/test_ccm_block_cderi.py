"""Block storage of the real-Γ neutral cderi: identity to the dense fold.

2026-09-29 (speed-up prompt 5). The real-Γ producer's fold returned one dense
real array of ``8 n_aux n_mu^2 N_c^3`` bytes, which cannot hold the paper's
MgO and diamond ladder cells at N = 4 (247 to 543 GB against 128 GB of RAM).
``cderi_storage="block"`` keeps the same per-q unit-cell fits unfolded
(``O(N_c^2)``, :class:`~vibeqc.periodic.ccm.block_cderi.CCMBlockCderi`) and
evaluates J and K in the mesh representation on the circulant projection of
the torus density, through the occupied-rank factor per k, the exchange
``q = 0`` seam per k on the same projection. The dense path stays the
reference and the default; every identity here is pinned against it (1e-10
Ha/cell on the energy, rounding on J/K and on the materialised array), and
the Theorem-1 identity against the Bloch producer is asserted on the same
cells (ruling R1).

Tier: T3 / experimental (Γ-CCM research lane). The H2-cube rows run in
seconds; the LiH and MgO STO-3G (2,2,2) rows and the (3,3,3) cube are the
prompt's acceptance cells and are slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.block_cderi import CCMBlockCderi, is_block_cderi
from vibeqc.periodic.ccm.direct import (
    _k_neutral_fast,
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
)
from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
from vibeqc.periodic.ccm.neutral import (
    carried_cderi,
    ccm_neutral_cderi_blocks,
    ccm_neutral_cderi_fold,
    ccm_neutral_tail_ke_cutoff,
)
from vibeqc.periodic.ccm.ri import (
    ccm_ri_j_neutral,
    ccm_ri_k_neutral,
    run_ccm_rhf_gdf,
    run_ccm_rks_gdf,
)

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

ANG = 1.0 / 0.529177210903
ENERGY_TOL_HA_PER_CELL = 1e-10
ARRAY_TOL = 1e-12


def _h2_cube():
    """Compact H2 in a 6-bohr cube, STO-3G: the fold's historical fixture."""
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _rocksalt(a_ang, z_a, z_b):
    a = a_ang * ANG
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return PeriodicSystem(
        3, lat, [Atom(z_a, [0.0, 0.0, 0.0]),
                 Atom(z_b, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1)


SYSTEMS = {"LiH": (4.0840, 3, 1), "MgO": (4.2239, 12, 8)}

_CACHE: dict = {}


def _blocks_and_dense(key, ccm):
    """One block build per cell per session; the dense array is its fold."""
    if key not in _CACHE:
        blk = ccm_neutral_cderi_blocks(ccm)
        _CACHE[key] = (blk, blk.to_dense())
    return _CACHE[key]


def _max_abs(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a - b)))


def _scale(a):
    return max(1.0, float(np.max(np.abs(a))))


def _random_circulant_psd(blk, rng, rank):
    """A translation-invariant PSD torus density: the circulant projection
    of a random Gram matrix (an average of PSD matrices, hence PSD)."""
    x = rng.standard_normal((blk.n_sc, rank))
    return blk.embed(blk.torus_blocks(x @ x.T))


def _random_circulant_symmetric(blk, rng):
    m = rng.standard_normal((blk.n_sc, blk.n_sc))
    return blk.embed(blk.torus_blocks(m + m.T))


# ---------------------------------------------------------------------------
# The storage against the fold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mesh", [(2, 1, 1), (3, 1, 1), (2, 2, 1), (3, 2, 1)],
                         ids=lambda m: "x".join(map(str, m)))
def test_block_to_dense_is_the_fold_h2_cube(mesh):
    """The block object holds every fit the fold consumes: materialising it
    reproduces the fold's array (same code path) to the bit. (3,1,1) and
    (3,2,1) carry genuine ±q pairs and time-reversal partners."""
    ccm = CCMSystem(_h2_cube(), mesh, "sto-3g")
    blk = ccm_neutral_cderi_blocks(ccm)
    assert is_block_cderi(blk)
    assert blk.n_c == ccm.n_cells and blk.n_mu == 2
    dense = ccm_neutral_cderi_fold(ccm)
    assert _max_abs(blk.to_dense(), dense) < 1e-14
    if ccm.n_cells >= 6:
        # Below that the pruned dense array of an all-self-paired mesh is
        # still the smaller object; the block storage wins by N_c^2/(2 N_kept).
        assert blk.nbytes < dense.nbytes
    assert blk.ws_report["n_omitted"] == 0 and blk.ws_cutoff_bohr is None


@pytest.mark.parametrize("mesh", [(2, 1, 1), (3, 1, 1), (2, 2, 1), (3, 2, 1)],
                         ids=lambda m: "x".join(map(str, m)))
def test_block_j_and_k_match_the_dense_contractions_h2_cube(mesh):
    """J, K (both exchange forms) and the seam K + eta S D S on random
    translation-invariant PSD densities, against the dense contractions."""
    ccm = CCMSystem(_h2_cube(), mesh, "sto-3g")
    blk, dense = _blocks_and_dense(("h2", mesh), ccm)
    rng = np.random.default_rng(20260929 + sum(mesh))
    for rank in (1, 3):
        D = _random_circulant_psd(blk, rng, rank)
        S = _random_circulant_symmetric(blk, rng)
        J_ref = ccm_ri_j_neutral(dense, D)
        K_ref = ccm_ri_k_neutral(dense, D)
        assert _max_abs(blk.coulomb(D), J_ref) < ARRAY_TOL * _scale(J_ref)
        assert _max_abs(blk.exchange(D), K_ref) < ARRAY_TOL * _scale(K_ref)
        assert _max_abs(blk.exchange(D, form="pair"), K_ref) < ARRAY_TOL * _scale(K_ref)
        if all(n <= 2 for n in mesh):
            # Every channel self-paired: the mesh sum is real on its own.
            assert blk.imag_residual < 1e-12 * _scale(K_ref)
        eta = 0.37
        Ks_ref = _k_neutral_fast(dense, D, S, eta)
        assert _max_abs(blk.exchange(D, S, eta), Ks_ref) < ARRAY_TOL * _scale(Ks_ref)
    # A non-circulant input is contracted on its circulant projection.
    x = rng.standard_normal((blk.n_sc, 2))
    D_raw = x @ x.T
    D_proj = blk.embed(blk.torus_blocks(D_raw))
    assert _max_abs(blk.coulomb(D_raw), blk.coulomb(D_proj)) < 1e-13
    assert _max_abs(blk.exchange(D_raw), blk.exchange(D_proj)) < 1e-13


def test_torus_mesh_transforms_round_trip_h2_321():
    ccm = CCMSystem(_h2_cube(), (3, 2, 1), "sto-3g")
    blk, _ = _blocks_and_dense(("h2", (3, 2, 1)), ccm)
    rng = np.random.default_rng(7)
    blocks = rng.standard_normal((blk.n_c, blk.n_mu, blk.n_mu))
    m_sc = blk.embed(blocks)
    assert _max_abs(blk.torus_blocks(m_sc), blocks) < 1e-14
    assert _max_abs(blk.unbloch_real(blk.bloch(blocks)), blocks) < 1e-13
    assert blk.imag_residual < 1e-12


# ---------------------------------------------------------------------------
# The SCF identities on the cheap cell
# ---------------------------------------------------------------------------

def test_block_keyword_builds_and_matches_the_dense_scf_h2_221():
    """The keyword path: ``cderi_storage="block"`` builds the storage, the
    result carries it, ``cderi_info`` records it, and the HF energy is the
    dense path's."""
    ccm = CCMSystem(_h2_cube(), (2, 2, 1), "sto-3g")
    d = run_ccm_rhf_direct(ccm)
    b = run_ccm_rhf_direct(ccm, cderi_storage="block")
    assert d.converged and b.converged
    assert d.cderi_info["storage"] == "dense"
    assert b.cderi_info["storage"] == "block"
    assert b.cderi_info["build"] == "fold"
    assert is_block_cderi(b.cderi)
    assert b.cderi_info["nbytes"] == b.cderi.nbytes
    assert b.cderi_info["ws_cutoff_bohr"] is None
    assert b.cderi_info["ws_n_omitted"] == 0
    assert b.cderi_info["ws_omitted_norm"] == 0.0
    assert len(b.cderi_info["ws_shells"]) >= 2
    assert b.fit_source == "classic" and b.exchange_q0 == d.exchange_q0
    assert b.rsgdf_tail_ke_cutoff == d.rsgdf_tail_ke_cutoff
    resid = (b.energy - d.energy) / ccm.n_cells
    assert abs(resid) < ENERGY_TOL_HA_PER_CELL, f"{resid:.3e} Ha/cell"
    # A prebuilt block object is accepted like a prebuilt array.
    b2 = run_ccm_rhf_direct(ccm, cderi=b.cderi)
    assert b2.cderi_info["build"] == "caller"
    assert b2.cderi_info["storage"] == "block"
    assert abs(b2.energy - b.energy) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL
    # The pair form of the exchange (the dense contraction verbatim).
    b3 = run_ccm_rhf_direct(ccm, cderi=b.cderi, exchange_form="pair")
    assert abs(b3.energy - d.energy) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL


@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_block_ks_matches_the_dense_scf_h2_221(functional):
    """Pure and hybrid KS: the loop's a_x scaling and the seam on the
    exact-exchange channel act on the block K exactly as on the dense one."""
    ccm = CCMSystem(_h2_cube(), (2, 2, 1), "sto-3g")
    blk, dense = _blocks_and_dense(("h2", (2, 2, 1)), ccm)
    d = run_ccm_rks_direct(ccm, functional, cderi=dense)
    b = run_ccm_rks_direct(ccm, functional, cderi=blk)
    assert d.converged and b.converged
    assert b.cderi_info["storage"] == "block"
    resid = (b.energy - d.energy) / ccm.n_cells
    assert abs(resid) < ENERGY_TOL_HA_PER_CELL, f"{functional}: {resid:.3e} Ha/cell"
    assert abs(b.e_xc - d.e_xc) / ccm.n_cells < 1e-9
    assert abs(b.e_hf_exchange - d.e_hf_exchange) / ccm.n_cells < 1e-9


def test_strict_zero_mode_seam_off_matches_h2_221():
    ccm = CCMSystem(_h2_cube(), (2, 2, 1), "sto-3g")
    blk, dense = _blocks_and_dense(("h2", (2, 2, 1)), ccm)
    d = run_ccm_rhf_direct(ccm, cderi=dense, exxdiv=None)
    b = run_ccm_rhf_direct(ccm, cderi=blk, exxdiv=None)
    assert d.converged and b.converged
    assert b.exchange_q0 == d.exchange_q0
    assert abs(b.energy - d.energy) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------

def test_mo_factor_and_mp2_ride_the_blocks_h2_321():
    """The MO factor from the blocks gives the dense factor's (ia|jb) to
    rounding, and run_ccm_mp2 on a block reference equals MP2 on the dense
    reference; carried_cderi materialises the array for the other consumers."""
    ccm = CCMSystem(_h2_cube(), (3, 2, 1), "sto-3g")
    blk, dense = _blocks_and_dense(("h2", (3, 2, 1)), ccm)
    d = run_ccm_rhf_direct(ccm, cderi=dense)
    b = run_ccm_rhf_direct(ccm, cderi=blk)
    C = np.asarray(d.mo_coeffs)
    n_occ = ccm.supercell.n_electrons() // 2
    Co, Cv = C[:, :n_occ], C[:, n_occ:]
    B_ref = np.einsum("Pmn,mi,na->Pia", dense, Co, Cv, optimize=True)
    B_blk = blk.mo_factor(Co, Cv)
    ovov_ref = np.einsum("Pia,Pjb->iajb", B_ref, B_ref, optimize=True)
    ovov_blk = np.einsum("Pia,Pjb->iajb", B_blk, B_blk, optimize=True)
    assert _max_abs(ovov_blk, ovov_ref) < 1e-13 * _scale(ovov_ref)
    m_d = run_ccm_mp2(ccm, d)
    m_b = run_ccm_mp2(ccm, b)
    assert abs(m_b.e_correlation - m_d.e_correlation) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL
    assert abs(m_b.e_total - m_d.e_total) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL
    assert _max_abs(carried_cderi(b), dense) < 1e-14
    # Passing the block object explicitly is the same route.
    m_e = run_ccm_mp2(ccm, d, cderi=blk)
    assert abs(m_e.e_correlation - m_d.e_correlation) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL


# ---------------------------------------------------------------------------
# The Wigner-Seitz partition (paper issue O25)
# ---------------------------------------------------------------------------

def test_ws_partition_orders_omits_and_reports_h2_311():
    """(3,1,1) cube: offsets at 0 and 6 bohr (two: +1 and -1 along the
    18-bohr cluster axis). Default omits nothing; a cutoff below 6 bohr omits both far rows,
    records the norm, changes the energy; a cutoff above omits nothing and
    the energy is the exact one; the projection is idempotent."""
    ccm = CCMSystem(_h2_cube(), (3, 1, 1), "sto-3g")
    blk = ccm_neutral_cderi_blocks(ccm)
    shells = blk.ws_shells()
    assert shells[0] == (0.0, 1)
    assert shells[1][0] == pytest.approx(6.0, abs=1e-6) and shells[1][1] == 2
    dist, mult = blk.ws_distances()
    # 18-bohr cluster axis: +-6 have unique nearest images (no boundary tie).
    assert all(int(m) == 1 for m in mult)
    assert sorted(np.round(dist, 6).tolist()) == [0.0, 6.0, 6.0]
    exact = run_ccm_rhf_direct(ccm, cderi=blk)
    far = run_ccm_rhf_direct(ccm, cderi_storage="block", ws_cutoff_bohr=100.0)
    assert far.cderi_info["ws_cutoff_bohr"] == 100.0
    assert far.cderi_info["ws_n_omitted"] == 0
    assert far.cderi_info["ws_omitted_norm"] == 0.0
    assert abs(far.energy - exact.energy) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL
    near = run_ccm_rhf_direct(ccm, cderi_storage="block", ws_cutoff_bohr=3.0)
    info = near.cderi_info
    assert info["ws_n_omitted"] == 2
    assert info["ws_omitted_norm"] > 0.0
    assert 0.0 < info["ws_omitted_norm_rel"] < 1.0
    assert info["ws_shells"] == shells
    assert near.converged
    assert abs(near.energy - exact.energy) / ccm.n_cells > 1e-6
    # Idempotent: re-applying the same cutoff moves nothing.
    fits_before = [ch.fits.copy() for ch in near.cderi.channels]
    report = near.cderi.apply_ws_cutoff(3.0)
    assert report["n_omitted"] == 2 and report["omitted_norm"] < 1e-12
    for ch, before in zip(near.cderi.channels, fits_before):
        assert _max_abs(ch.fits, before) < 1e-13


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_block_storage_refusals_h2_211():
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="fold"):
        run_ccm_rhf_direct(ccm, cderi_storage="block", cderi_build="supercell")
    with pytest.raises(ValueError, match="ws_cutoff_bohr"):
        run_ccm_rhf_direct(ccm, ws_cutoff_bohr=5.0)
    with pytest.raises(ValueError, match="cderi_storage"):
        run_ccm_rhf_direct(ccm, cderi_storage="sparse")
    with pytest.raises(ValueError, match="exchange_form"):
        run_ccm_rhf_direct(ccm, exchange_form="dense")
    with pytest.raises(ValueError, match="storage"):
        ccm_neutral_cderi_fold(ccm, ws_cutoff_bohr=1.0)
    blk, _ = _blocks_and_dense(("h2", (2, 1, 1)), ccm)
    # The open-shell drivers take the block object too (#519, RG-5); their
    # identity to the dense path is pinned in test_ccm_block_cderi_open_shell.py.
    u = run_ccm_uhf_direct(ccm, cderi=blk)
    assert u.converged and u.cderi_info["storage"] == "block"
    with pytest.raises(ValueError, match="apply_ws_cutoff"):
        run_ccm_rhf_direct(ccm, cderi=blk, ws_cutoff_bohr=4.0)
    with pytest.raises(ValueError, match="cutoff"):
        blk.apply_ws_cutoff(-1.0)
    with pytest.raises(ValueError, match="exchange_form"):
        blk.exchange(np.eye(blk.n_sc), form="dense")


# ---------------------------------------------------------------------------
# The prompt's acceptance cells (slow)
# ---------------------------------------------------------------------------

def _bloch(ccm, functional):
    if functional == "hf":
        return run_ccm_rhf_gdf(ccm)
    return run_ccm_rks_gdf(ccm, functional)


def _direct(ccm, functional, cderi):
    if functional == "hf":
        return run_ccm_rhf_direct(ccm, cderi=cderi)
    return run_ccm_rks_direct(ccm, functional, cderi=cderi)


def _identity_rows(key, ccm, functional):
    blk, dense = _blocks_and_dense(key, ccm)
    d = _direct(ccm, functional, dense)
    b = _direct(ccm, functional, blk)
    g = _bloch(ccm, functional)
    assert d.converged and b.converged and g.converged
    assert b.cderi_info["storage"] == "block"
    # Both direct runs receive a precomputed cderi, which the producers stamp
    # "injected" with no tail of their own (#378); the fold built it with the
    # classic fit and the resolved tail, which the Bloch side must match.
    assert b.fit_source == "injected" == d.fit_source
    assert b.rsgdf_tail_ke_cutoff is None and d.rsgdf_tail_ke_cutoff is None
    assert g.fit_source == "classic"
    assert g.rsgdf_tail_ke_cutoff == ccm_neutral_tail_ke_cutoff(ccm)
    if g.parity_held:
        assert g.rsgdf_tail_ke_cutoff is None
    n_c = ccm.n_cells
    dense_resid = (b.energy - d.energy) / n_c
    bloch_resid = g.energy - b.energy / n_c
    return dense_resid, bloch_resid, blk, dense


def _check_bloch(label, bloch_resid):
    """The Theorem-1 row against the Bloch producer, strict for every
    functional: since c9d32436 the real-Gamma KS producers grow their XC
    image domain to the Bloch side's (#512), and the LiH PBE (2,2,2) row
    passes at 1e-10 Ha/cell (#519 RG-1), so the #512 xfail is gone."""
    assert abs(bloch_resid) < ENERGY_TOL_HA_PER_CELL, (
        f"{label}: Bloch - block = {bloch_resid:.3e} Ha/cell")


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["hf", "pbe", "pbe0"])
def test_block_identities_h2_cube_333(functional):
    """H2 6-bohr cube (3,3,3): block against dense at 1e-10 Ha/cell, and the
    Theorem-1 identity against the Bloch producer on the same cell."""
    ccm = CCMSystem(_h2_cube(), (3, 3, 3), "sto-3g")
    dense_resid, bloch_resid, blk, dense = _identity_rows(("h2", (3, 3, 3)), ccm, functional)
    assert abs(dense_resid) < ENERGY_TOL_HA_PER_CELL, (
        f"H2 (3,3,3)/{functional}: block - dense = {dense_resid:.3e} Ha/cell")
    assert blk.nbytes * 4 < dense.nbytes
    _check_bloch(f"H2 (3,3,3)/{functional}", bloch_resid)


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["hf", "pbe", "pbe0"])
@pytest.mark.parametrize("name", ["LiH", "MgO"])
def test_block_identities_rocksalt_222(name, functional):
    """LiH and MgO STO-3G (2,2,2), the prompt's acceptance cells: block
    against dense at 1e-10 Ha/cell for HF, PBE and PBE0, and the Theorem-1
    identity against the Bloch producer."""
    a_ang, z_a, z_b = SYSTEMS[name]
    ccm = CCMSystem(_rocksalt(a_ang, z_a, z_b), (2, 2, 2), "sto-3g")
    dense_resid, bloch_resid, _, _ = _identity_rows((name, (2, 2, 2)), ccm, functional)
    assert abs(dense_resid) < ENERGY_TOL_HA_PER_CELL, (
        f"{name} (2,2,2)/{functional}: block - dense = {dense_resid:.3e} Ha/cell")
    _check_bloch(f"{name} (2,2,2)/{functional}", bloch_resid)
