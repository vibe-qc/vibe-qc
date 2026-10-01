"""AO-reach screening of the periodic XC value kernels (vibe-qc#268, #299, #53)
and the grid electron count they report (vibe-qc#385).

``build_xc_periodic`` / ``build_xc_periodic_uks`` evaluate the density on the
periodic Becke grid as a double sum over lattice images of the AOs. They used
to evaluate every AO of every image cell inside ``cutoff_bohr`` at every grid
point and to contract every bra-ket image pair, however far each shell
actually reaches. The screen keeps, per grid batch, only the image shells
whose Gaussian extent (``compute_shell_radial_cutoffs`` at
``ao_screening_tol``) reaches the batch, and skips zero-weight points. The
image domain itself is unchanged, so the screened result must equal the dense
one (``ao_screening_tol=0``) to far below any physical tolerance, while doing
measurably less work. A deliberately loose tolerance must move E_xc: that is
the control showing the screen really drops terms.

The kernels also report ``n_electrons_grid = sum_g w_g rho(r_g)``, the
quantity whose silent shortfall #385 (and #265 before it) reported.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc import monkhorst_pack
from vibeqc._vibeqc_core import (
    GridOptions,
    LatticeSumOptions,
    bloch_sum,
    build_xc_periodic,
    build_xc_periodic_uks,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    periodic_xc_screening_stats,
    real_space_density_from_kpoints,
)
from vibeqc.periodic_grid import build_periodic_becke_grid

_ANG = 1.0 / 0.529177210903
CUTOFF = 12.0
# Screened versus dense: the dropped AO amplitudes are below 1e-12, so the
# energy per cell and every V_xc element agree far below SCF tolerances.
E_TOL = 1e-10
V_TOL = 1e-9


def _fcc(a_bohr, z1, z2):
    lattice = (a_bohr / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    atoms = [vq.Atom(z1, [0.0, 0.0, 0.0]),
             vq.Atom(z2, [a_bohr / 2.0, a_bohr / 2.0, a_bohr / 2.0])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


SYSTEMS = {
    # Dense ionic cell with compact STO-3G shells: the screen prunes.
    "mgo": lambda: _fcc(4.21 * _ANG, 12, 8),
    # Rocksalt LiH (a = 7.72 bohr, the CCM fixture): the diffuse Li 2sp shell
    # (exponent 0.048) reaches past the cutoff, so little can be pruned; the
    # screen must still be exact.
    "lih": lambda: _fcc(7.72, 3, 1),
}


def _lattice_options(image_radius=CUTOFF):
    opts = LatticeSumOptions()
    opts.cutoff_bohr = CUTOFF
    opts.nuclear_cutoff_bohr = CUTOFF
    opts.becke_image_radius_bohr = image_radius
    return opts


def _grid(system):
    gopts = GridOptions()
    gopts.n_radial = 35
    return build_periodic_becke_grid(system, grid_options=gopts,
                                     image_radius_bohr=CUTOFF)


def _density(system, basis, lat_opts):
    """Closed-shell multi-k density from a kinetic-Hamiltonian fold
    (T C = S C eps), as in test_periodic_xc_cross_cell.py: it carries real
    P(g != 0) blocks without any Coulomb build."""
    kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    kpts = list(kmesh.kpoints)
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    T_lat = compute_kinetic_lattice(basis, system, lat_opts)
    n_occ = system.n_electrons() // 2
    C_per_k = []
    for k in kpts:
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        S_k = 0.5 * (S_k + S_k.conj().T)
        T_k = 0.5 * (T_k + T_k.conj().T)
        _, C = sla.eigh(T_k, S_k)
        C_per_k.append(np.asarray(C, dtype=complex))
    cells = list(direct_lattice_cells(system, 2.0 * lat_opts.cutoff_bohr))
    return real_space_density_from_kpoints(
        C_per_k, [n_occ] * len(kpts), kmesh, cells
    )


def _half(D_real):
    return core.make_lattice_matrix_set(
        int(D_real.nbf), list(D_real.cells),
        [0.5 * np.asarray(b, dtype=float) for b in D_real.blocks],
    )


def _max_block_diff(a, b):
    return max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y)), initial=0.0))
        for x, y in zip(a.blocks, b.blocks)
    )


@pytest.fixture(scope="module", params=sorted(SYSTEMS))
def case(request):
    system, basis = SYSTEMS[request.param]()
    lat_opts = _lattice_options()
    return {
        "name": request.param,
        "system": system,
        "basis": basis,
        "grid": _grid(system),
        "opts": lat_opts,
        "density": _density(system, basis, lat_opts),
    }


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_screened_rks_matches_dense(case, functional):
    func = core.Functional(functional)
    args = (case["basis"], case["system"], case["grid"], func,
            case["density"], case["opts"])
    dense = build_xc_periodic(*args, ao_screening_tol=0.0)
    screened = build_xc_periodic(*args)  # default tolerance
    assert abs(screened.e_xc - dense.e_xc) <= E_TOL
    assert _max_block_diff(screened.V_xc, dense.V_xc) <= V_TOL
    assert abs(screened.n_electrons_grid - dense.n_electrons_grid) <= E_TOL
    assert screened.periodic_density_bras and dense.periodic_density_bras


@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_screened_uks_matches_dense_and_rks(case, functional):
    half = _half(case["density"])
    func = core.Functional(functional, 2)
    args = (case["basis"], case["system"], case["grid"], func, half, half,
            case["opts"])
    dense = build_xc_periodic_uks(*args, ao_screening_tol=0.0)
    screened = build_xc_periodic_uks(*args)
    assert abs(screened.e_xc - dense.e_xc) <= E_TOL
    assert _max_block_diff(screened.V_alpha, dense.V_alpha) <= V_TOL
    assert _max_block_diff(screened.V_beta, dense.V_beta) <= V_TOL
    # Same total density as the closed-shell evaluation.
    rks = build_xc_periodic(case["basis"], case["system"], case["grid"],
                            core.Functional(functional), case["density"],
                            case["opts"])
    assert abs(screened.n_electrons_grid - rks.n_electrons_grid) <= E_TOL
    assert abs(screened.e_xc - rks.e_xc) <= 1e-8


def test_screen_prunes_compact_images():
    """Non-vacuity: on MgO/STO-3G the screen evaluates fewer image shells and
    bra-ket pairs than the dense kernel, and skips zero-weight points; with
    the screen disabled the counts are the dense ones."""
    system, basis = SYSTEMS["mgo"]()
    opts = _lattice_options()
    grid = _grid(system)
    D = _density(system, basis, opts)
    st = periodic_xc_screening_stats(basis, system, grid, D, opts)
    assert st.image_shells_live < st.image_shells_dense
    assert st.pairs_live < st.pairs_dense
    assert st.n_points_evaluated < st.n_points
    off = periodic_xc_screening_stats(basis, system, grid, D, opts,
                                      ao_screening_tol=0.0)
    assert off.image_shells_live == off.image_shells_dense
    assert off.pairs_live == off.pairs_dense
    assert off.n_points_evaluated == off.n_points
    # The live fraction is reported for the record (pytest -rA).
    print(f"MgO/STO-3G live pairs {st.pairs_live}/{st.pairs_dense}, "
          f"live image shells {st.image_shells_live}/{st.image_shells_dense}")


def test_loose_screen_moves_exc():
    """Control: an AO amplitude bound of 1e-2 drops shells that matter, so the
    energy must move. If it did not, the exactness test above would be
    vacuous (the screen would not be dropping anything)."""
    system, basis = SYSTEMS["mgo"]()
    opts = _lattice_options()
    grid = _grid(system)
    D = _density(system, basis, opts)
    func = core.Functional("lda")
    dense = build_xc_periodic(basis, system, grid, func, D, opts,
                              ao_screening_tol=0.0)
    loose = build_xc_periodic(basis, system, grid, func, D, opts,
                              ao_screening_tol=1e-2)
    assert abs(loose.e_xc - dense.e_xc) > 1e-6


def test_grid_electron_count_sees_truncated_image_reach():
    """#385: the reported grid integral is the electron count per cell when
    the image reach holds the density, and falls short when it does not --
    the silent failure the issue reported."""
    func = core.Functional("lda")
    opts = _lattice_options()
    # MgO: the kinetic-fold density is compact; cutoff 12 holds all 20
    # electrons (20.0019 on this grid).
    system, basis = SYSTEMS["mgo"]()
    xc = build_xc_periodic(basis, system, _grid(system), func,
                           _density(system, basis, opts), opts)
    assert xc.periodic_density_bras
    assert abs(xc.n_electrons_grid - 20.0) / 20.0 < 5e-3
    # LiH: the kinetic fold occupies the most diffuse Li functions, which
    # reach well past 12 bohr, so the grid visibly misses 4 electrons (3.43).
    # (Cutting the bra reach does not shrink it monotonically: image terms of
    # P(g) carry both signs, so only the count against N is diagnostic.)
    system, basis = SYSTEMS["lih"]()
    full = build_xc_periodic(basis, system, _grid(system), func,
                             _density(system, basis, opts), opts)
    assert abs(full.n_electrons_grid - 4.0) / 4.0 > 1e-2


@pytest.mark.parametrize("functional", ["lda", "pbe"])
def test_screened_potential_of_home_cell_density(case, functional):
    """A home-cell (molecular-limit) density has P(g != 0) = 0 but still a
    V_xc(g) block for every image pair: the potential is dE/dP(g). The
    density-term bound max|P(g)| must not drop those pairs from V_xc (the
    multi-k ROKS Ewald driver's first iteration exposed this)."""
    D = case["density"]
    home = [
        np.asarray(b, dtype=float) if not np.any(np.asarray(c.index))
        else np.zeros_like(np.asarray(b, dtype=float))
        for c, b in zip(D.cells, D.blocks)
    ]
    D_home = core.make_lattice_matrix_set(int(D.nbf), list(D.cells), home)
    half = core.make_lattice_matrix_set(int(D.nbf), list(D.cells),
                                        [0.5 * b for b in home])
    base = (case["basis"], case["system"], case["grid"])
    dense = build_xc_periodic(*base, core.Functional(functional), D_home,
                              case["opts"], ao_screening_tol=0.0)
    screened = build_xc_periodic(*base, core.Functional(functional), D_home,
                                 case["opts"])
    assert not dense.periodic_density_bras
    assert abs(screened.e_xc - dense.e_xc) <= E_TOL
    assert _max_block_diff(screened.V_xc, dense.V_xc) <= V_TOL
    f2 = core.Functional(functional, 2)
    dense_u = build_xc_periodic_uks(*base, f2, half, half, case["opts"],
                                    ao_screening_tol=0.0)
    screened_u = build_xc_periodic_uks(*base, f2, half, half, case["opts"])
    assert _max_block_diff(screened_u.V_alpha, dense_u.V_alpha) <= V_TOL
    assert _max_block_diff(screened_u.V_beta, dense_u.V_beta) <= V_TOL
