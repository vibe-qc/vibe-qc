"""Tests for the KS dispatcher (``run_rks_periodic_scf`` /
``run_rks_periodic_gamma_scf``).

Mirrors :mod:`tests.test_periodic_rhf_dispatch` for the Kohn-Sham
side. The DIRECT_TRUNCATED path goes through the existing C++
``run_rks_periodic`` driver; EWALD_3D goes through the Γ-only
Ewald driver shipped in Phase 15c-1 for [1,1,1] meshes; the multi-k
EWALD_3D path (Phase 15c-2) handles denser meshes via
``run_rks_periodic_multi_k_ewald3d``.

Contracts:

1. **DIRECT_TRUNCATED routing reproduces the bare driver** — multi-k
   and Γ-only dispatcher entry points produce the same energy as
   calling ``run_rks_periodic`` directly.

2. **None options accepted** — the convenience default uses a fresh
   ``PeriodicKSOptions()``.

3. **Default LDA functional + small basis converge** through the
   dispatcher.

4. **PeriodicKSOptions copy path** preserves every convergence and
   periodic-XC field relevant to v0.4+ (fock_mixing, level_shift,
   smearing_temperature, quadratic fallback, use_periodic_becke,
   becke_image_radius_bohr).

5. **EWALD_3D routes correctly** through both
   ``run_rks_periodic_gamma_scf`` (Γ-only driver, molecular-limit exact
   exchange) and the multi-k entry point, which lands on
   ``run_rks_periodic_multi_k_ewald3d`` (Phase 15c-2) at every mesh,
   [1,1,1] included (#23), so smearing or a restart density no longer
   switches a hybrid's exchange gauge.

6. **SLAB_EWALD_2D** routes Gamma-only RKS through the rigorous slab
   Ewald driver and dense meshes through the multi-k slab path;
   ``NEUTRALIZED_1D`` still raises ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2(box=30.0):
    """H2 / STO-3G at the centre of an orthorhombic cell; ``box`` is one
    edge length (cube) or three."""
    edges = np.broadcast_to(np.asarray(box, dtype=float), (3,))
    cx, cy, cz = edges / 2
    sysp = vq.PeriodicSystem(
        3, np.diag(edges),
        [vq.Atom(1, [cx, cy, cz - 0.7]),
         vq.Atom(1, [cx, cy, cz + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


# DIRECT_TRUNCATED cells (#430). The direct drivers refuse a cell whose
# |g| <= cutoff_bohr ball holds only g = 0 (#133): a 30-bohr box at
# cutoff 12 would be a free-boundary cluster labelled periodic. An
# 11-bohr cube puts the six nearest images inside both the 12-bohr AO
# ball and the 15-bohr nuclear ball of ``_ks_options`` (the next shell,
# 15.6 bohr, lies outside both), so both lattice sums see the same
# neutral periodic cell.
_DIRECT_BOX = 11.0
# The same for the library defaults (cutoff 15, nuclear cutoff 25): the
# only images are the two along z at 14 bohr; 28 and 30 bohr lie outside
# both balls.
_DEFAULT_CUTOFF_CELL = (30.0, 30.0, 14.0)


def _h2_slab(box: float = 18.0, vacuum: float = 45.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        2, np.diag([box, box, vacuum]),
        [
            vq.Atom(1, [c, c, vacuum / 2 - 0.7]),
            vq.Atom(1, [c, c, vacuum / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _ks_options(method=None):
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    if method is not None:
        o.lattice_opts.coulomb_method = method
    o.damping = 0.3
    o.max_iter = 40
    return o


# ---------------------------------------------------------------------------
# 1. DIRECT_TRUNCATED routing
# ---------------------------------------------------------------------------

def test_dispatcher_routes_direct_truncated_multi_k():
    sysp, basis = _h2(_DIRECT_BOX)
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _ks_options(vq.CoulombMethod.DIRECT_TRUNCATED)

    r_dispatch = vq.run_rks_periodic_scf(sysp, basis, km, opts)
    r_direct = vq.run_rks_periodic(sysp, basis, km, opts)
    assert r_dispatch.converged and r_direct.converged
    assert isinstance(r_dispatch, vq.PeriodicKSResult)
    assert r_dispatch.energy == pytest.approx(r_direct.energy, abs=1e-10)


def test_dispatcher_routes_direct_truncated_gamma():
    sysp, basis = _h2(_DIRECT_BOX)
    opts = _ks_options(vq.CoulombMethod.DIRECT_TRUNCATED)
    r_gamma_dispatch = vq.run_rks_periodic_gamma_scf(sysp, basis, opts)
    # Reference: equivalent multi-k call at [1,1,1] mesh.
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_111 = vq.run_rks_periodic(sysp, basis, km, opts)
    assert r_gamma_dispatch.converged and r_111.converged
    assert isinstance(r_gamma_dispatch, vq.PeriodicKSResult)
    assert r_gamma_dispatch.energy == pytest.approx(r_111.energy, abs=1e-10)


# ---------------------------------------------------------------------------
# 2. None-options accepted
# ---------------------------------------------------------------------------

def test_dispatcher_accepts_none_options():
    sysp, basis = _h2(_DEFAULT_CUTOFF_CELL)
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_scf(sysp, basis, km)
    assert r.converged
    # The default PeriodicKSOptions select DIRECT_TRUNCATED: the native driver.
    assert isinstance(r, vq.PeriodicKSResult)


def test_gamma_dispatcher_accepts_none_options():
    sysp, basis = _h2(_DEFAULT_CUTOFF_CELL)
    r = vq.run_rks_periodic_gamma_scf(sysp, basis)
    assert r.converged
    assert isinstance(r, vq.PeriodicKSResult)


# ---------------------------------------------------------------------------
# 3. Option-translation preserves v0.4 fields
# ---------------------------------------------------------------------------

def test_ks_option_copy_preserves_v0_4_fields():
    """The KS dispatcher's internal ``_copy_ks_options`` helper must
    forward every field added around the v0.4 convergence-control work
    (fock_mixing, level_shift, smearing, quadratic fallback,
    periodic Becke). A field-drop bug here would mirror the RHF
    dispatcher's level_shift regression caught earlier."""
    from vibeqc.periodic_ks_dispatch import _copy_ks_options
    src = vq.PeriodicKSOptions()
    src.fock_mixing = 0.25
    src.level_shift = 0.42
    src.smearing_temperature = 0.013
    src.quadratic_fallback_iter = 23
    src.quadratic_fallback_shift = 0.33
    src.quadratic_fallback_max_step = 0.04
    src.use_periodic_becke = True
    src.becke_image_radius_bohr = 8.5
    src.functional = "PBE"
    src.damping = 0.7
    src.max_iter = 250

    out = _copy_ks_options(src)
    assert out.fock_mixing == pytest.approx(0.25)
    assert out.level_shift == 0.42
    assert out.smearing_temperature == 0.013
    assert out.quadratic_fallback_iter == 23
    assert out.quadratic_fallback_shift == pytest.approx(0.33)
    assert out.quadratic_fallback_max_step == pytest.approx(0.04)
    assert out.use_periodic_becke is True
    assert out.becke_image_radius_bohr == 8.5
    assert out.functional == "PBE"
    assert out.damping == pytest.approx(0.7)
    assert out.max_iter == 250


# ---------------------------------------------------------------------------
# 4. EWALD_3D routing — Γ-only (Phase 15c-1) for [1,1,1], multi-k
#    (Phase 15c-2) for denser meshes.
# ---------------------------------------------------------------------------

def test_ewald_3d_gamma_routes_through_dispatcher():
    """Γ-only dispatcher entry must succeed on EWALD_3D and return a
    PeriodicRKSEwaldResult."""
    sysp, basis = _h2()
    opts = _ks_options(vq.CoulombMethod.EWALD_3D)
    r = vq.run_rks_periodic_gamma_scf(sysp, basis, opts)
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSEwaldResult)


def test_ewald_3d_multi_k_gamma_mesh_routes_to_multi_k_driver():
    """Multi-k dispatcher entry with a [1,1,1] mesh lands on the multi-k
    driver (#23). Until then it delegated to the Γ-only driver unless
    smearing or a restart density was set, and the two build a hybrid's
    exact exchange in different gauges. For a pure functional they are the
    same SCF, so the energy matches the gamma_scf path."""
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _ks_options(vq.CoulombMethod.EWALD_3D)
    r = vq.run_rks_periodic_scf(sysp, basis, km, opts)
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSMultiKEwaldResult)
    r_gamma = vq.run_rks_periodic_gamma_scf(sysp, basis, opts)
    assert isinstance(r_gamma, vq.PeriodicRKSEwaldResult)
    assert r.energy == pytest.approx(r_gamma.energy, abs=1e-8)


def test_gamma_dispatcher_refuses_smearing_with_a_hybrid():
    """Smearing on the Γ-only dispatcher borrows the multi-k driver, whose
    hybrid exchange is the corrected split, not the Γ driver's molecular
    limit. It refuses rather than let smearing change the Hamiltonian."""
    sysp, basis = _h2()
    opts = _ks_options(vq.CoulombMethod.EWALD_3D)
    opts.functional = "PBE0"
    opts.smearing_temperature = 0.01
    with pytest.raises(NotImplementedError, match="run_rks_periodic_scf"):
        vq.run_rks_periodic_gamma_scf(sysp, basis, opts)


def test_ewald_3d_multi_k_dense_mesh_routes_to_new_driver():
    """Dense k-meshes through the multi-k EWALD_3D path lands on
    ``run_rks_periodic_multi_k_ewald3d`` (Phase 15c-2). Was a
    NotImplementedError-asserting test before that phase shipped."""
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _ks_options(vq.CoulombMethod.EWALD_3D)
    r = vq.run_rks_periodic_scf(sysp, basis, km, opts)
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSMultiKEwaldResult)


# ---------------------------------------------------------------------------
# 5. Low-dimensional slab dispatch
# ---------------------------------------------------------------------------

def test_slab_ewald_2d_gamma_routes_through_dispatcher():
    sysp, basis = _h2_slab()
    opts = _ks_options(vq.CoulombMethod.SLAB_EWALD_2D)
    opts.lattice_opts.slab_ewald_alpha = 0.4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5
    opts.max_iter = 35

    r = vq.run_rks_periodic_gamma_scf(sysp, basis, opts)

    assert r.converged
    assert isinstance(r, vq.PeriodicRKSEwaldResult)
    assert r.e_nuclear == pytest.approx(
        vq.nuclear_repulsion_per_cell(sysp, opts.lattice_opts),
        abs=1e-12,
    )
    assert r.energy == pytest.approx(r.e_electronic + r.e_nuclear, abs=1e-10)
    assert r.omega == pytest.approx(opts.lattice_opts.slab_ewald_alpha)
    assert r.grid_shape == (0, 0, 0)
    assert abs(r.e_hf_exchange) < 1e-12


def test_slab_ewald_2d_multi_k_routes_through_dispatcher():
    sysp, basis = _h2_slab()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    opts = _ks_options(vq.CoulombMethod.SLAB_EWALD_2D)
    opts.lattice_opts.slab_ewald_alpha = 0.4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5
    opts.max_iter = 35

    r = vq.run_rks_periodic_scf(sysp, basis, km, opts, progress=False)

    assert r.converged
    assert isinstance(r, vq.PeriodicRKSMultiKEwaldResult)
    assert r.e_nuclear == pytest.approx(
        vq.nuclear_repulsion_per_cell(sysp, opts.lattice_opts),
        abs=1e-12,
    )
    assert r.energy == pytest.approx(r.e_electronic + r.e_nuclear, abs=1e-10)
    assert r.omega == pytest.approx(opts.lattice_opts.slab_ewald_alpha)
    assert r.grid_shape == (0, 0, 0)
    assert abs(r.e_hf_exchange) < 1e-12


def test_neutralized_1d_raises_not_implemented():
    sysp, basis = _h2()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    opts = _ks_options(vq.CoulombMethod.NEUTRALIZED_1D)
    with pytest.raises(NotImplementedError, match="NEUTRALIZED_1D"):
        vq.run_rks_periodic_scf(sysp, basis, km, opts)
