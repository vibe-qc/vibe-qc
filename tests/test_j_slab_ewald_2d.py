"""Rigorous 2D (slab) Ewald Hartree ``J``.

Increment 3 of the ``CoulombMethod.SLAB_EWALD_2D`` workstream
(``handovers/HANDOVER_SLAB_EWALD_2D.md``). ``J(D)`` is the electron–electron
partner of the slab ``V_ne``, built by the reciprocal-space FT
(``vibeqc.ewald_composed_slab``): the full periodic Hartree directly in
reciprocal space (in-plane ``g ≠ 0`` × a 1D ``G_z`` integral of ``4π/G²``,
Bloch density) plus the analytic bare ``|z|`` ``g = 0`` slab term. The
Gamma-only and multi-k RHF/RKS/UKS slab SCF paths now consume this kernel.

Correctness witnesses
---------------------
1. **Bilinear-total consistency** — the headline gate. For a fixed neutral
   density, ``E_nn + Tr[D·V_ne] + ½ Tr[D·J]`` (all three slab blocks) equals
   the vacuum-extrapolated 3D-Ewald total to chemical accuracy. This proves
   ``V_ne`` and ``J`` share one 2D-Ewald gauge (the bilinear blocks are exactly
   additive); a gauge mismatch between ``V_ne`` and ``J`` would shift the total
   by ``c·N_elec``.
2. **3D-vacuum limit** — ``J_2D − J_3D(EWALD_3D, large c) = c·S + resid``; the
   residual vanishes as the gap grows, ``c`` drifts with the gap. Ties ``J`` to
   the µHa-validated 3D Hartree.
3. **Mesh convergence / cache** — ke-cutoff converged; cache ↔ uncached ↔
   builder bit-identical; symmetry; the L = 1 (carbon) AO-pair-FT path.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    CoulombMethod,
    EwaldOptions,
    LatticeSumOptions,
    bloch_sum,
    ewald_2d_point_charge_energy_with_background,
    nuclear_repulsion_per_cell,
)
from vibeqc import compute_overlap_lattice
from vibeqc.ewald_composed_slab import (
    build_j_slab_ewald_2d_gamma_cache,
    build_j_slab_ewald_2d_lattice_cache,
    compute_j_slab_ewald_2d_gamma,
    compute_j_slab_ewald_2d_lattice,
    make_slab_ewald_2d_gamma_j_builder,
)

_ATOMS = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
_A = 4.6


def _slab():
    sys2 = vq.PeriodicSystem(2, np.diag([_A, _A, 30.0]), _ATOMS)
    basis = vq.BasisSet(sys2.unit_cell_molecule(), "sto-3g")
    return sys2, basis


def _lat_opts():
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 18.0
    lo.nuclear_cutoff_bohr = 45.0
    lo.coulomb_method = CoulombMethod.SLAB_EWALD_2D
    return lo


def _overlap(basis, system, lo):
    S = np.real(bloch_sum(compute_overlap_lattice(basis, system, lo), np.zeros(3)))
    return 0.5 * (S + S.T)


def _neutral_D(basis, system, lo, n_elec=2.0):
    """A fixed symmetric density with Tr[D·S] = n_elec (charge-neutral cell)."""
    S = _overlap(basis, system, lo)
    v = np.ones(S.shape[0])
    return n_elec * np.outer(v, v) / float(v @ S @ v)


def _polar_hf_slab():
    """A polar HF layer (core element F + net normal dipole, M_z != 0) with a
    realistic molecular-RHF density -- the system the surface-dipole gate needs.
    """
    b = 8.0
    atoms = [vq.Atom(9, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.74])]
    sys2 = vq.PeriodicSystem(2, np.diag([b, b, 40.0]), atoms)
    basis = vq.BasisSet(sys2.unit_cell_molecule(), "sto-3g")
    mol = sys2.unit_cell_molecule()
    res = vq.run_rhf(mol, vq.BasisSet(mol, "sto-3g"))
    C = np.asarray(res.mo_coeffs)
    nocc = 5  # HF: 10 electrons
    D = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    return sys2, basis, D


# ---------------------------------------------------------------------------
# 1. Bilinear-total consistency — V_ne, J and E_nn share one gauge
# ---------------------------------------------------------------------------
def test_bilinear_total_matches_3d_vacuum_limit():
    from vibeqc.periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_gamma
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma
    from vibeqc.ewald_composed import compute_j_ewald_3d_ft_gamma

    sys2, bas2 = _slab()
    lo2 = _lat_opts()
    D = _neutral_D(bas2, sys2, lo2)
    eo = EwaldOptions()
    eo.alpha = 0.45
    eo.real_cutoff_bohr = 45.0

    Vne2 = compute_v_ne_slab_ewald_2d_gamma(bas2, sys2, lo2, ewald_options=eo).v_total
    J2 = compute_j_slab_ewald_2d_gamma(bas2, sys2, D, lattice_opts=lo2)

    # Bare 2D-Ewald E_nn = with-background − the analytic |z| sheet it adds.
    Rn = np.array([list(a.xyz) for a in _ATOMS]).T
    Z = np.array([a.Z for a in _ATOMS], dtype=float)
    nhat = np.array([0.0, 0.0, 1.0])
    area = _A * _A
    z_b = float((Rn.T @ nhat).mean())
    e_wbg = ewald_2d_point_charge_energy_with_background(
        np.diag([_A, _A, 30.0]), Rn, Z, z_b, eo
    )
    sheet = -(2 * np.pi / area) * (-Z.sum()) * np.sum(Z * np.abs(Rn.T @ nhat - z_b))
    Enn2 = e_wbg - sheet
    E2 = Enn2 + float(np.einsum("mn,mn->", D, Vne2)) + 0.5 * float(
        np.einsum("mn,mn->", D, J2)
    )

    # 3D-Ewald total at a large vacuum gap (the validated, gauge-consistent
    # reference). Same basis ordering => same fixed density.
    cz = 200.0
    sys3 = vq.PeriodicSystem(3, np.diag([_A, _A, cz]), _ATOMS)
    bas3 = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
    lo3 = LatticeSumOptions()
    lo3.cutoff_bohr = 18.0
    lo3.nuclear_cutoff_bohr = 45.0
    lo3.coulomb_method = CoulombMethod.EWALD_3D
    D3 = _neutral_D(bas3, sys3, lo3)
    Vne3 = compute_v_ne_ewald_3d_ft_gamma(bas3, sys3, lo3, ke_cutoff=400.0)
    J3 = compute_j_ewald_3d_ft_gamma(bas3, sys3, D3, 0.0, lattice_opts=lo3, ke_cutoff=400.0)
    E3 = (
        nuclear_repulsion_per_cell(sys3, lo3)
        + float(np.einsum("mn,mn->", D3, Vne3))
        + 0.5 * float(np.einsum("mn,mn->", D3, J3))
    )
    # ~18 µHa in practice (mesh/cutoff differences between the 2D and 3D
    # setups); far inside chemical accuracy. A gauge mismatch between V_ne and
    # J would shift this by O(c·N_elec) ~ Ha.
    assert abs(E2 - E3) < 5e-4, (
        f"2D bilinear total {E2:.6f} disagrees with vacuum-extrapolated 3D "
        f"{E3:.6f} by {E2 - E3:.2e} — V_ne / J / E_nn gauge inconsistency."
    )


# ---------------------------------------------------------------------------
# 2. 3D-vacuum limit of J alone (up to gauge ×S)
# ---------------------------------------------------------------------------
def test_j_matches_3d_vacuum_limit_up_to_gauge():
    from vibeqc.ewald_composed import compute_j_ewald_3d_ft_gamma

    sys2, bas2 = _slab()
    lo2 = _lat_opts()
    D = _neutral_D(bas2, sys2, lo2)
    S = _overlap(bas2, sys2, lo2)
    J2 = compute_j_slab_ewald_2d_gamma(bas2, sys2, D, lattice_opts=lo2)

    def fit(cz):
        sys3 = vq.PeriodicSystem(3, np.diag([_A, _A, cz]), _ATOMS)
        bas3 = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
        lo3 = LatticeSumOptions()
        lo3.cutoff_bohr = 18.0
        lo3.coulomb_method = CoulombMethod.EWALD_3D
        D3 = _neutral_D(bas3, sys3, lo3)
        J3 = compute_j_ewald_3d_ft_gamma(bas3, sys3, D3, 0.0, lattice_opts=lo3, ke_cutoff=320.0)
        Dd = J2 - J3
        c = float(np.sum(Dd * S) / np.sum(S * S))
        return c, float(np.max(np.abs(Dd - c * S)))

    c_s, r_s = fit(45.0)
    c_l, r_l = fit(90.0)
    assert r_l < 1e-2
    assert r_l < r_s + 1e-9          # residual converges (3D → 2D limit)
    assert abs(c_l) > abs(c_s) + 0.5  # gauge drifts with the gap


# ---------------------------------------------------------------------------
# 3. Mesh / cache / symmetry
# ---------------------------------------------------------------------------
def test_j_cache_uncached_builder_agree():
    sys2, bas2 = _slab()
    lo = _lat_opts()
    D = _neutral_D(bas2, sys2, lo)
    cache = build_j_slab_ewald_2d_gamma_cache(bas2, sys2, lattice_opts=lo)
    j_cached = compute_j_slab_ewald_2d_gamma(bas2, sys2, D, cache=cache)
    j_unc = compute_j_slab_ewald_2d_gamma(bas2, sys2, D, lattice_opts=lo)
    j_b = make_slab_ewald_2d_gamma_j_builder(bas2, sys2, lattice_opts=lo)(D)
    assert np.max(np.abs(j_cached - j_unc)) < 1e-12
    assert np.max(np.abs(j_cached - j_b)) < 1e-12
    assert np.max(np.abs(j_cached - j_cached.T)) < 1e-12


def test_multi_k_j_lattice_gamma_bloch_matches_gamma_builder():
    sys2, bas2 = _slab()
    lo = _lat_opts()
    D = _neutral_D(bas2, sys2, lo)
    S_lat = compute_overlap_lattice(bas2, sys2, lo)
    for g_idx in range(len(S_lat.cells)):
        S_lat.set_block(g_idx, D)

    cache = build_j_slab_ewald_2d_lattice_cache(
        bas2, sys2, S_lat.cells, lattice_opts=lo, alpha=0.4,
    )
    J_blocks = compute_j_slab_ewald_2d_lattice(
        bas2, sys2, S_lat, lattice_opts=lo, alpha=0.4, cache=cache,
    )
    J_gamma_from_blocks = sum(J_blocks)
    J_gamma = compute_j_slab_ewald_2d_gamma(
        bas2, sys2, D, lattice_opts=lo, alpha=0.4,
    )
    assert np.max(np.abs(J_gamma_from_blocks - J_gamma)) < 1e-10
    by_index = {tuple(cell.index.tolist()): i for i, cell in enumerate(S_lat.cells)}
    for idx, i in by_index.items():
        j = by_index[tuple((-np.array(idx, dtype=int)).tolist())]
        assert np.max(np.abs(J_blocks[i] - J_blocks[j].T)) < 1e-8


def test_j_alpha_split_invariant():
    """``½Tr[D·J]`` is independent of the Ewald split parameter ``alpha``.

    The rigorous gate for the Ewald-split assembly: ``alpha`` only balances the
    real-space ``J_short`` against the damped reciprocal ``J_long`` / screened
    ``J_g0``; the physical Hartree must not move. (The pre-split undamped
    full-reciprocal build was instead ``ke``-converged and under-resolved the
    all-electron core by ~0.1 Ha -- the v0.15 surface-dipole "bug".)
    """
    sys2, bas2 = _slab()
    lo = _lat_opts()
    D = _neutral_D(bas2, sys2, lo)
    vals = [
        0.5 * float(np.einsum("mn,mn->", D, compute_j_slab_ewald_2d_gamma(
            bas2, sys2, D, lattice_opts=lo, alpha=a)))
        for a in (0.30, 0.45, 0.65)
    ]
    assert max(vals) - min(vals) < 1e-5


def test_polar_bilinear_total_alpha_invariant():
    """The polar (M_z != 0) bilinear total ``E_nn + Tr[D·V_ne] + ½Tr[D·J]`` is
    alpha-invariant -- the Sec.7 gauge-consistency gate for a normal dipole.

    A core element (F) + a net surface dipole (HF layer) is precisely what the
    pre-split build got wrong (~0.1 Ha). With V_ne (already split) and J (now
    split) each a-invariant, the assembled total must be too.
    """
    from vibeqc._vibeqc_core import ewald_2d_point_charge_energy_with_background
    from vibeqc.periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_gamma

    sys2, bas2, D = _polar_hf_slab()
    lo = _lat_opts()
    nhat = np.array([0.0, 0.0, 1.0])
    Rn = np.array([list(a.xyz) for a in sys2.unit_cell]).T
    Z = np.array([a.Z for a in sys2.unit_cell], dtype=float)
    area = float(sys2.lattice[0, 0] * sys2.lattice[1, 1])
    z_b = float((Rn.T @ nhat).mean())

    totals = []
    for a in (0.35, 0.50, 0.65):
        eo = EwaldOptions()
        eo.alpha = a
        eo.real_cutoff_bohr = 30.0
        Vne = compute_v_ne_slab_ewald_2d_gamma(bas2, sys2, lo, ewald_options=eo).v_total
        J = compute_j_slab_ewald_2d_gamma(bas2, sys2, D, lattice_opts=lo, alpha=a)
        e_wbg = ewald_2d_point_charge_energy_with_background(
            np.asarray(sys2.lattice), Rn, Z, z_b, eo
        )
        sheet = -(2 * np.pi / area) * (-Z.sum()) * np.sum(
            Z * np.abs(Rn.T @ nhat - z_b)
        )
        e_nn = e_wbg - sheet
        totals.append(
            e_nn
            + float(np.einsum("mn,mn->", D, Vne))
            + 0.5 * float(np.einsum("mn,mn->", D, J))
        )
    assert max(totals) - min(totals) < 1e-5


@pytest.mark.slow
def test_polar_bilinear_matches_core_converged_3d():
    """Polar (M_z != 0) 2D bilinear total == vacuum-extrapolated 3D-Ewald +
    Yeh-Berkowitz ``(2π/V)M_z²``, as the 3D reference's core converges.

    The cross-method absolute gate. The 3D-FT Hartree is undamped, so it
    under-resolves the F core at low ``ke`` and only *approaches* the (already
    core-converged) 2D split value as ``ke`` rises -- this pins both the trend
    and the converged agreement, the direct disproof of the old "99 mHa
    surface-dipole bug" (it was the 3D *and* old-2D core under-convergence).
    """
    from vibeqc._vibeqc_core import (
        ewald_2d_point_charge_energy_with_background,
        nuclear_repulsion_per_cell,
    )
    from vibeqc import compute_dipole
    from vibeqc.periodic_v_ne_slab import compute_v_ne_slab_ewald_2d_gamma
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma
    from vibeqc.ewald_composed import compute_j_ewald_3d_ft_gamma

    sys2, bas2, D = _polar_hf_slab()
    A = float(sys2.lattice[0, 0])
    lo = _lat_opts()
    nhat = np.array([0.0, 0.0, 1.0])
    Rn = np.array([list(a.xyz) for a in sys2.unit_cell]).T
    Z = np.array([a.Z for a in sys2.unit_cell], dtype=float)
    area = A * A
    z_b = float((Rn.T @ nhat).mean())

    eo = EwaldOptions()
    eo.alpha = 0.45
    eo.real_cutoff_bohr = 30.0
    Vne2 = compute_v_ne_slab_ewald_2d_gamma(bas2, sys2, lo, ewald_options=eo).v_total
    J2 = compute_j_slab_ewald_2d_gamma(bas2, sys2, D, lattice_opts=lo, alpha=0.45)
    e_wbg = ewald_2d_point_charge_energy_with_background(
        np.asarray(sys2.lattice), Rn, Z, z_b, eo
    )
    sheet = -(2 * np.pi / area) * (-Z.sum()) * np.sum(Z * np.abs(Rn.T @ nhat - z_b))
    E2 = (e_wbg - sheet) + float(np.einsum("mn,mn->", D, Vne2)) + 0.5 * float(
        np.einsum("mn,mn->", D, J2)
    )

    zmol = np.asarray(compute_dipole(bas2, [0.0, 0.0, 0.0]).z)
    Mz = float(Z @ (Rn.T @ nhat)) - float(np.einsum("mn,mn->", D, zmol))

    cz = 40.0
    sys3 = vq.PeriodicSystem(3, np.diag([A, A, cz]), list(sys2.unit_cell))
    bas3 = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
    lo3 = LatticeSumOptions()
    lo3.cutoff_bohr = lo.cutoff_bohr
    lo3.nuclear_cutoff_bohr = lo.nuclear_cutoff_bohr
    lo3.coulomb_method = CoulombMethod.EWALD_3D
    V = area * cz
    YB = (2 * np.pi / V) * Mz**2

    gaps = []
    for ke in (200.0, 400.0, 600.0):
        Vne3 = compute_v_ne_ewald_3d_ft_gamma(bas3, sys3, lo3, ke_cutoff=ke)
        J3 = compute_j_ewald_3d_ft_gamma(bas3, sys3, D, 0.0, lattice_opts=lo3, ke_cutoff=ke)
        E3 = (
            nuclear_repulsion_per_cell(sys3, lo3)
            + float(np.einsum("mn,mn->", D, Vne3))
            + 0.5 * float(np.einsum("mn,mn->", D, J3))
        )
        gaps.append(abs(E2 - (E3 + YB)))
    # The 3D reference's core converges from below: the gap shrinks monotonically
    # and the high-ke gap is small -- 2D (core-converged by construction) is the
    # limit. (Old undamped-2D vs 3D-ke200 sat ~0.1 Ha apart and never closed.)
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.02


def test_j_rejects_dim3():
    sys3 = vq.PeriodicSystem(3, np.diag([_A, _A, _A]), _ATOMS)
    bas3 = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="dim == 2"):
        build_j_slab_ewald_2d_gamma_cache(bas3, sys3)


def test_j_carbon_slab_l1_path():
    """Carbon sto-3g exercises the L = 1 AO-pair-FT path; J builds + is real."""
    a = 4.6
    lat = np.array([[a, 0, 0], [-0.5 * a, np.sqrt(3) / 2 * a, 0], [0, 0, 28.0]]).T
    sysc = vq.PeriodicSystem(2, lat, [vq.Atom(6, [0, 0, 0])])
    basc = vq.BasisSet(sysc.unit_cell_molecule(), "sto-3g")
    lo = _lat_opts()
    D = _neutral_D(basc, sysc, lo, n_elec=6.0)
    J = compute_j_slab_ewald_2d_gamma(basc, sysc, D, lattice_opts=lo)
    assert J.shape == (basc.nbasis, basc.nbasis)
    assert np.max(np.abs(J - J.T)) < 1e-10
    assert np.all(np.isfinite(J))
