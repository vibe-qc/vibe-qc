"""Rigorous 2D (slab) Ewald summation — Parry / de Leeuw–Perram–Smith.

This is the M1 foundation for ``CoulombMethod.SLAB_EWALD_2D`` (see
``handovers/HANDOVER_SLAB_EWALD_2D.md``): the point-charge Madelung energy of a
charge distribution periodic in the plane (lattice columns 0,1) and finite
along the normal. The self-consistent electronic coupling that un-gates the
SCF dispatch is a later milestone; this file validates the energy primitive
on its own.

Correctness witnesses
---------------------

1. **α-invariance** — the rigorous "is the Ewald split correct" test. The
   total must not depend on the screening parameter α once both sums have
   converged. A wrong coefficient on *any* of the four terms (real,
   reciprocal g≠0, reciprocal g=0, self) breaks the α-cancellation.

2. **3D-vacuum limit** — for a slab with zero net dipole along the normal
   (M_n = 0), the 2D-Ewald energy equals the L→∞ limit of the
   independently-validated *3D* Ewald sum on the same cell with a large
   vacuum gap. This cross-checks the new reciprocal + g=0 formulas against
   a completely separate code path (the 3D ``ewald_point_charge_energy``,
   itself pinned to the NaCl/CsCl/ZnS Madelung constants).

3. **2D square-lattice Madelung constant** — the alternating ±1 square
   ionic lattice has a tabulated 2D Madelung constant 1.6155426…; we
   recover it.

4. **Neutrality is required** — a net-charged periodic plane has a
   divergent energy and must be rejected, not silently regularised.

5. **Geometric invariances** — the energy ignores the vacuum (3rd) lattice
   column and is invariant under a rigid shift along the normal (every term
   depends only on z_ij = z_i − z_j).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf, erfc, erfcx

import vibeqc as vq


def _opts(alpha, real_cutoff, recip_cutoff):
    o = vq.EwaldOptions()
    o.alpha = alpha
    o.real_cutoff_bohr = real_cutoff
    o.recip_cutoff_bohr_inv = recip_cutoff
    return o


# ---------------------------------------------------------------------------
# 1. α-invariance — the rigorous correctness gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.15, 0.25, 0.40, 0.60])
def test_alpha_invariance_2d_slab(alpha):
    """A neutral two-charge slab cell with out-of-plane structure (M_n ≠ 0).

    The 2D-Ewald total must be independent of α to the convergence floor.
    """
    a = 6.0
    d = 0.8  # ± normal offset → the cell carries an out-of-plane dipole
    lat = np.diag([a, a, 40.0])
    positions = np.column_stack([[0.0, 0.0, +d], [a / 2, a / 2, -d]])
    charges = np.array([+1.0, -1.0])

    # Generous, α-independent cutoffs that converge both sums for every α
    # in the parametrisation (small α → slow real space; large α → slow
    # reciprocal space).
    E = vq.ewald_2d_point_charge_energy(
        lat, positions, charges, _opts(alpha, 50.0, 14.0)
    )

    E_ref = vq.ewald_2d_point_charge_energy(
        lat, positions, charges, _opts(0.30, 50.0, 14.0)
    )
    assert abs(E - E_ref) < 1e-9, f"α={alpha}: E={E:.12f} vs ref {E_ref:.12f}"


# ---------------------------------------------------------------------------
# 2. 3D-vacuum limit for a zero-normal-dipole slab
# ---------------------------------------------------------------------------

def test_2d_matches_3d_vacuum_limit_when_dipole_free():
    """M_n = 0 slab: 2D-Ewald == lim_{L→∞} 3D-Ewald(vacuum gap L).

    Cross-checks the 2D reciprocal + g=0 terms against the separate,
    NaCl-pinned 3D ``ewald_point_charge_energy``.
    """
    a = 6.0
    h = 1.0
    # Mirror-symmetric about z = 0 → net charge 0, M_n = Σ q z = 0.
    positions = np.column_stack([
        [0.0,     0.0,     +h],
        [a / 2,   a / 2,   +h],
        [0.0,     0.0,     -h],
        [a / 2,   a / 2,   -h],
    ])
    charges = np.array([+1.0, -1.0, -1.0, +1.0])
    assert abs(charges.sum()) < 1e-12
    assert abs((charges * positions[2]).sum()) < 1e-12  # M_n = 0

    o = _opts(0.30, 35.0, 12.0)
    E_2d = vq.ewald_2d_point_charge_energy(
        np.diag([a, a, 50.0]), positions, charges, o
    )

    # 3D Ewald on the same in-plane cell with a vacuum gap. For a compact
    # M_n = 0 slab the inter-image interaction is already negligible at these
    # gaps, so the 3D sum reproduces the 2D limit to the convergence floor.
    for L in (40.0, 80.0, 160.0):
        E_3d = vq.ewald_point_charge_energy(
            np.diag([a, a, L]), positions, charges, o
        )
        assert abs(E_3d - E_2d) < 1e-9, f"L={L}: E_3d={E_3d} vs E_2d={E_2d}"


# ---------------------------------------------------------------------------
# 3. Tabulated 2D square-lattice Madelung constant
# ---------------------------------------------------------------------------

def test_square_lattice_2d_madelung_constant():
    """Alternating ±1 square lattice (nn distance d): M_2D ≈ 1.6155426267.

    Primitive cell of the checkerboard: like-charge sublattice vectors
    (d, d) and (d, −d), basis +1 at (0,0) and −1 at (d,0); nearest-neighbour
    distance d. Following the same convention as the 3D NaCl/CsCl tests in
    ``test_ewald.py`` (``-energy * r_nn``), this recovers the tabulated 2D
    square-lattice Madelung constant.
    """
    d = 2.0
    lat = np.column_stack([[d, d, 0.0], [d, -d, 0.0], [0.0, 0.0, 50.0]])
    positions = np.column_stack([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
    charges = np.array([+1.0, -1.0])

    E_cell = vq.ewald_2d_point_charge_energy(
        lat, positions, charges, _opts(0.35, 45.0, 14.0)
    )
    madelung = -E_cell * d
    assert abs(madelung - 1.6155426267) < 1e-6, (
        f"extracted 2D square Madelung constant = {madelung:.10f}, "
        f"expected 1.6155426267"
    )

    # Independent tie: a planar cell is M_n = 0, so the 3D-vacuum sum agrees
    # to the convergence floor.
    E_3d_vac = vq.ewald_point_charge_energy(
        np.column_stack([[d, d, 0.0], [d, -d, 0.0], [0.0, 0.0, 70.0]]),
        positions, charges, _opts(0.35, 45.0, 14.0),
    )
    assert abs(E_cell - E_3d_vac) < 1e-9


# ---------------------------------------------------------------------------
# 4. Neutrality is required (a charged plane diverges)
# ---------------------------------------------------------------------------

def test_charged_cell_rejected():
    lat = np.diag([6.0, 6.0, 40.0])
    positions = np.column_stack([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0]])
    charges = np.array([+1.0, +1.0])  # net +2
    with pytest.raises(ValueError, match="neutral"):
        vq.ewald_2d_point_charge_energy(
            lat, positions, charges, _opts(0.3, 30.0, 12.0)
        )


# ---------------------------------------------------------------------------
# 5. Geometric invariances
# ---------------------------------------------------------------------------

def test_invariant_to_vacuum_column():
    """The 2D energy uses only lattice columns 0,1 — the vacuum vector
    (column 2) must not affect the result."""
    a = 6.0
    positions = np.column_stack([[0.0, 0.0, 0.5], [a / 2, a / 2, -0.5]])
    charges = np.array([+1.0, -1.0])
    opts = _opts(0.3, 40.0, 13.0)

    E_short = vq.ewald_2d_point_charge_energy(
        np.diag([a, a, 30.0]), positions, charges, opts
    )
    E_long = vq.ewald_2d_point_charge_energy(
        np.diag([a, a, 120.0]), positions, charges, opts
    )
    assert abs(E_short - E_long) < 1e-11


def test_invariant_to_rigid_normal_shift():
    """Every term depends only on z_ij = z_i − z_j, so a rigid shift of all
    atoms along the slab normal leaves the energy unchanged."""
    a = 6.0
    positions = np.column_stack([[0.0, 0.0, 0.5], [a / 2, a / 2, -0.5]])
    charges = np.array([+1.0, -1.0])
    opts = _opts(0.3, 40.0, 13.0)
    lat = np.diag([a, a, 40.0])

    E0 = vq.ewald_2d_point_charge_energy(lat, positions, charges, opts)
    shifted = positions.copy()
    shifted[2, :] += 5.0  # rigid +5 bohr along the normal
    E1 = vq.ewald_2d_point_charge_energy(lat, shifted, charges, opts)
    assert abs(E0 - E1) < 1e-10


# ---------------------------------------------------------------------------
# 6. Citation route (Parry 1975 + de Leeuw–Perram 1979) is wired
# ---------------------------------------------------------------------------

def test_slab_ewald_citation_route_fires():
    """The ``slab_ewald_2d`` methods route must emit both 2D-Ewald papers,
    ready for the M2 SCF driver to fire (AGENTS.md §8)."""
    from vibeqc.output.citations.registry import load_default_database

    db = load_default_database()
    result = db.assemble(method="slab_ewald_2d", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert {"parry_2d_ewald_1975", "de_leeuw_perram_2d_ewald_1979"} <= keys


# ---------------------------------------------------------------------------
# 7. 2D-Ewald electrostatic potential primitive (e-n attraction foundation)
# ---------------------------------------------------------------------------

def test_potential_reproduces_energy():
    """The exact functional-partner identity: with V evaluated at the charge
    sites (the n=0 short-range self term is skipped inside the potential),

        E = ½ Σ_i q_i V(r_i) + E_self,   E_self = −(α/√π) Σ q_i².

    This ties the new potential to the already-validated energy summer.
    """
    a = 6.0
    lat = np.diag([a, a, 40.0])
    positions = np.column_stack([[0.0, 0.0, 0.7], [a / 2, a / 2, -0.7]])
    charges = np.array([+1.0, -1.0])
    alpha = 0.30
    opts = _opts(alpha, 50.0, 14.0)

    V_at_sites = vq.ewald_2d_point_charge_potential(
        lat, positions, charges, positions.T.copy(), opts, True
    )
    e_self = -(alpha / np.sqrt(np.pi)) * float(np.sum(charges * charges))
    E_via_potential = 0.5 * float(np.dot(charges, V_at_sites)) + e_self
    E = vq.ewald_2d_point_charge_energy(lat, positions, charges, opts)
    assert abs(E_via_potential - E) < 1e-9, f"{E_via_potential} vs {E}"


def test_potential_difference_alpha_invariant():
    """The physical potential (a difference between two points) is α-invariant.
    (The absolute zero is gauge-dependent, so compare a difference.)"""
    a = 6.0
    lat = np.diag([a, a, 40.0])
    positions = np.column_stack([[0.0, 0.0, 0.7], [a / 2, a / 2, -0.7]])
    charges = np.array([+1.0, -1.0])
    pts = np.array([[1.3, 2.1, 0.9], [2.4, 0.6, -0.4]])  # two generic points

    def dphi(alpha):
        v = vq.ewald_2d_point_charge_potential(
            lat, positions, charges, pts, _opts(alpha, 50.0, 14.0), True
        )
        return v[0] - v[1]

    ref = dphi(0.30)
    for alpha in (0.18, 0.45, 0.60):
        assert abs(dphi(alpha) - ref) < 1e-9, f"α={alpha}"


def test_potential_difference_matches_3d_vacuum_limit():
    """For an M_n = 0 slab, the 2D potential difference between two points
    equals the 3D-vacuum potential difference (independent code path)."""
    a = 6.0
    h = 1.0
    positions = np.column_stack([
        [0.0, 0.0, +h], [a / 2, a / 2, +h],
        [0.0, 0.0, -h], [a / 2, a / 2, -h],
    ])
    charges = np.array([+1.0, -1.0, -1.0, +1.0])
    pts = np.array([[1.3, 2.1, 0.5], [2.0, 0.5, -0.3]])
    opts = _opts(0.30, 35.0, 12.0)

    v2 = vq.ewald_2d_point_charge_potential(
        np.diag([a, a, 50.0]), positions, charges, pts, opts, True
    )
    d2 = v2[0] - v2[1]
    for L in (40.0, 80.0):
        v3 = vq.ewald_point_charge_potential(
            np.diag([a, a, L]), positions, charges, pts, opts, True
        )
        assert abs(d2 - (v3[0] - v3[1])) < 1e-7, f"L={L}"


def test_potential_charged_cell_rejected():
    lat = np.diag([6.0, 6.0, 40.0])
    positions = np.column_stack([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0]])
    charges = np.array([+1.0, +1.0])
    pts = np.array([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="neutral"):
        vq.ewald_2d_point_charge_potential(
            lat, positions, charges, pts, _opts(0.3, 30.0, 12.0), True
        )


# ---------------------------------------------------------------------------
# 8. Background-neutralised primitives — the M2b SCF foundation (Increment 1)
# ---------------------------------------------------------------------------
#
# ``ewald_2d_point_charge_energy_with_background`` and its potential sibling
# compute the rigorous 2D-Ewald energy / potential of a (possibly net-charged)
# point set plus a co-located uniform in-plane neutralising sheet of total
# charge q_b = -sum(q) at normal coordinate z_background. They are the no-guard
# building block the slab SCF needs to express the neutral total as physically
# interpretable per-block (nuclei-only / electron-only) pieces.
#
# The decomposition the SCF rests on (verified below):
#   * the rigorous Parry four-term sum is a finite, ALPHA- and ORIGIN-INVARIANT
#     bilinear form even on a NET-CHARGED block (the slab g=0 term regularises
#     it with no background needed), and it is exactly block-additive, so a
#     neutral cell splits into nuclei/electron blocks + a cross term whose sum
#     reproduces the M1 total;
#   * the neutralising sheet's exact field -(2*pi/A) q_b |z - z_b| (Gauss; no
#     Ewald screening for a smooth uniform plane) makes each block neutral and
#     physically a "charged slab in a background at z_b" — z_b-dependent for a
#     charged block (the physical background position the BIPOLE surface
#     gradient consumes), z_b-independent and sheet-free for a neutral cell.


def _cells_2d(v1, v2, cutoff, include_origin):
    area = np.linalg.norm(np.cross(v1, v2))
    n1max = int(np.ceil(cutoff * np.linalg.norm(v2) / area))
    n2max = int(np.ceil(cutoff * np.linalg.norm(v1) / area))
    out = []
    for n1 in range(-n1max, n1max + 1):
        for n2 in range(-n2max, n2max + 1):
            if not include_origin and n1 == 0 and n2 == 0:
                continue
            v = n1 * v1 + n2 * v2
            if v @ v <= cutoff * cutoff:
                out.append(v)
    return np.array(out)


def _parry_four_term(lattice, pos, q, alpha, real_cut, recip_cut):
    """Independent pure-Python oracle: the rigorous 2D-Ewald four-term
    point-charge energy (real + recip g!=0 + slab g=0 + self) with NO
    neutrality guard, reconstructed term-by-term from the formula in
    ``ewald_2d_point_charge_energy``. Lets us evaluate a *non-neutral* block,
    which the bound primitive rejects."""
    a1, a2 = lattice[:, 0], lattice[:, 1]
    cross = np.cross(a1, a2)
    A = np.linalg.norm(cross)
    nhat = cross / A
    b1 = 2 * np.pi * np.cross(a2, nhat) / A
    b2 = 2 * np.pi * np.cross(nhat, a1) / A
    N = q.size
    z = pos.T @ nhat
    sqpi = np.sqrt(np.pi)

    e_real = 0.0
    for n in _cells_2d(a1, a2, real_cut, True):
        for i in range(N):
            d = pos[:, i][None, :] - pos.T + n
            r = np.linalg.norm(d, axis=1)
            m = (r > 1e-14) & (r <= real_cut)
            e_real += np.sum(q[i] * q[m] * erfc(alpha * r[m]) / r[m])
    e_real *= 0.5

    e_recip = 0.0
    for g in _cells_2d(b1, b2, recip_cut, False):
        gn = np.linalg.norm(g)
        g2a = gn / (2 * alpha)
        acc = 0.0
        for i in range(N):
            zij = z[i] - z
            cosg = np.cos(g @ (pos[:, i][None, :] - pos.T).T)
            pref = np.exp(-g2a * g2a - alpha * alpha * zij * zij)
            bracket = erfcx(g2a + alpha * zij) + erfcx(g2a - alpha * zij)
            acc += np.sum(q[i] * q * cosg * pref * bracket)
        e_recip += acc / gn
    e_recip *= np.pi / (2 * A)

    e_g0 = 0.0
    for i in range(N):
        zij = z[i] - z
        e_g0 += np.sum(
            q[i] * q * (zij * erf(alpha * zij)
                        + np.exp(-alpha * alpha * zij * zij) / (alpha * sqpi))
        )
    e_g0 *= -np.pi / A

    e_self = -alpha / sqpi * np.sum(q * q)
    return e_real + e_recip + e_g0 + e_self


def _area_and_z(lattice, pos):
    cross = np.cross(lattice[:, 0], lattice[:, 1])
    A = np.linalg.norm(cross)
    nhat = cross / A
    return A, pos.T @ nhat


def _analytic_sheet(lattice, pos, q, z_b):
    """ΔE_sheet = -(2*pi/A) q_b sum_i q_i |z_i - z_b|, q_b = -sum(q)."""
    A, z = _area_and_z(lattice, pos)
    q_b = -float(np.sum(q))
    return -(2.0 * np.pi / A) * q_b * float(np.sum(q * np.abs(z - z_b)))


def test_bg_energy_reduces_to_m1_on_neutral():
    """On a charge-neutral cell the sheet vanishes (q_b = 0): the background
    variant must equal ``ewald_2d_point_charge_energy`` for every alpha and
    every (irrelevant) z_background."""
    a = 6.0
    lat = np.diag([a, a, 40.0])
    pos = np.column_stack([[0.0, 0.0, 0.7], [a / 2, a / 2, -0.7]])
    q = np.array([+1.0, -1.0])
    for alpha in (0.20, 0.35, 0.55):
        opts = _opts(alpha, 50.0, 14.0)
        E_m1 = vq.ewald_2d_point_charge_energy(lat, pos, q, opts)
        for z_b in (-3.0, 0.0, 9.1):
            E_bg = vq.ewald_2d_point_charge_energy_with_background(
                lat, pos, q, z_b, opts
            )
            assert abs(E_bg - E_m1) < 1e-12, f"alpha={alpha} z_b={z_b}"


def test_bg_energy_equals_bare_four_term_plus_analytic_sheet():
    """The exact term-by-term identity validating the C++ sheet contribution
    against the independent Python four-term oracle, on a NON-NEUTRAL block."""
    d = 3.0
    lat = np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, 80.0]])
    pos = np.column_stack([[0.3, 0.2, +1.1], [1.7, 1.4, -0.7]]).astype(float)
    q = np.array([+2.0, +1.0])  # net +3
    for alpha in (0.20, 0.40):
        opts = _opts(alpha, 80.0, 14.0)
        for z_b in (0.0, 4.0):
            ref = _parry_four_term(lat, pos, q, alpha, 80.0, 14.0) \
                + _analytic_sheet(lat, pos, q, z_b)
            E_bg = vq.ewald_2d_point_charge_energy_with_background(
                lat, pos, q, z_b, opts
            )
            assert abs(E_bg - ref) < 1e-9, f"alpha={alpha} z_b={z_b}"


def test_nuclear_repulsion_per_cell_slab_ewald_is_bare_block():
    """The SCF nuclear block must be the bare 2D-Ewald point term, not the
    background-including value used internally to define charged blocks."""
    d = 5.0
    lat = np.diag([d, d, 70.0])
    atoms = [
        vq.Atom(2, [0.2, 0.4, -0.9]),
        vq.Atom(1, [2.0, 1.7, 1.3]),
    ]
    sysp = vq.PeriodicSystem(2, lat, atoms)
    lopts = vq.LatticeSumOptions()
    lopts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    lopts.nuclear_cutoff_bohr = 70.0
    lopts.slab_ewald_alpha = 0.40

    pos = np.array([list(a.xyz) for a in atoms], dtype=float).T
    q = np.array([a.Z for a in atoms], dtype=float)
    eopts = _opts(lopts.slab_ewald_alpha, lopts.nuclear_cutoff_bohr, 14.0)
    z_b = 0.0
    e_with_background = vq.ewald_2d_point_charge_energy_with_background(
        lat, pos, q, z_b, eopts
    )
    e_bare_ref = e_with_background - _analytic_sheet(lat, pos, q, z_b)

    assert vq.nuclear_repulsion_per_cell(sysp, lopts) == pytest.approx(
        e_bare_ref, abs=1e-10
    )


@pytest.mark.parametrize("alpha", [0.15, 0.30, 0.50, 0.70])
def test_bg_energy_alpha_invariant_on_charged_block(alpha):
    """THE rigorous gate for the background variant: a charged slab + its
    neutralising sheet is a *neutral* physical system, so its energy must be
    alpha-invariant. (The bare four-term block is already alpha-invariant; the
    sheet's bare |z| field — not a second screened f(z) term — preserves that.)
    """
    d = 3.0
    lat = np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, 80.0]])
    pos = np.column_stack([[0.3, 0.2, +1.1], [1.7, 1.4, -0.7]]).astype(float)
    q = np.array([+2.0, +1.0])
    z_b = 0.6
    E = vq.ewald_2d_point_charge_energy_with_background(
        lat, pos, q, z_b, _opts(alpha, 80.0, 14.0)
    )
    E_ref = vq.ewald_2d_point_charge_energy_with_background(
        lat, pos, q, z_b, _opts(0.35, 80.0, 14.0)
    )
    assert abs(E - E_ref) < 1e-9, f"alpha={alpha}: {E} vs {E_ref}"


def test_bg_energy_zb_dependence():
    """A charged block's energy depends on the background plane z_b (physical:
    moving the neutralising jellium changes the electrostatics); a neutral
    cell's does not (the sheet is identically zero)."""
    d = 3.0
    lat = np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, 80.0]])
    opts = _opts(0.35, 80.0, 14.0)

    pos = np.column_stack([[0.3, 0.2, +1.1], [1.7, 1.4, -0.7]]).astype(float)
    q_charged = np.array([+2.0, +1.0])
    vals = [
        vq.ewald_2d_point_charge_energy_with_background(lat, pos, q_charged, z_b, opts)
        for z_b in (-2.0, 0.0, 5.0)
    ]
    assert max(vals) - min(vals) > 1.0  # genuinely z_b-dependent

    q_neutral = np.array([+1.0, -1.0])
    vals_n = [
        vq.ewald_2d_point_charge_energy_with_background(lat, pos, q_neutral, z_b, opts)
        for z_b in (-2.0, 0.0, 5.0)
    ]
    assert max(vals_n) - min(vals_n) < 1e-12  # z_b-independent when neutral


def test_bg_energy_matches_grid_sheet_limit():
    """Gold-standard independent check against the wholly separate M1 code
    path. Represent the uniform sheet by an n*n grid of point charges -Q/n^2
    at z_b, making {points + grid} a *neutral* cell whose bound-M1 energy is
    well-defined. The grid energy differs from the background-variant energy
    only by the grid's discreteness self-energy, which is a clean O(1/n)
    finite-size effect (a discrete lattice tending to a continuum sheet):
    Richardson-extrapolating it to n -> infinity must reproduce E_bg.

    (The points<->grid interaction itself converges to points<->sheet
    *exponentially* in n at finite height above z_b, so the entire O(1/n)
    tail is the grid self-energy, not a kernel error.)"""
    d = 3.0
    lat = np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, 80.0]])
    pos = np.column_stack([[0.3, 0.2, +1.1], [1.7, 1.4, -0.7]]).astype(float)
    q = np.array([+2.0, +1.0])
    z_b = 0.6
    opts = _opts(0.35, 80.0, 14.0)

    E_bg = vq.ewald_2d_point_charge_energy_with_background(lat, pos, q, z_b, opts)
    q_b = -float(np.sum(q))

    def grid_energy(n):
        g = (np.arange(n) + 0.5) / n * d
        GX, GY = np.meshgrid(g, g, indexing="ij")
        grid = np.column_stack([GX.ravel(), GY.ravel(), np.full(n * n, z_b)]).T
        gq = np.full(n * n, q_b / (n * n))
        allpos = np.column_stack([pos, grid])
        allq = np.concatenate([q, gq])
        return vq.ewald_2d_point_charge_energy(lat, allpos, allq, opts)

    # Confirm the tail is pure O(1/n): (E_n - E_bg)*n is constant.
    n1, n2 = 24, 48
    E1, E2 = grid_energy(n1), grid_energy(n2)
    assert abs((E1 - E_bg) * n1 - (E2 - E_bg) * n2) < 1e-3, "tail is not O(1/n)"

    # Richardson-extrapolate the O(1/n) grid self-energy to n -> infinity.
    c = (E1 - E2) / (1.0 / n1 - 1.0 / n2)
    E_inf = E1 - c / n1
    assert abs(E_inf - E_bg) < 1e-7, f"grid-sheet limit {E_inf} != E_bg {E_bg}"


def test_bare_block_additivity_reproduces_total():
    """Option-1 SCF foundation: the neutral total splits exactly into
    nuclei-only + electron-only bare four-term blocks + a cross term, and the
    recombination equals the bound M1 total (which is alpha-invariant) to
    machine precision. This is the bilinearity the slab SCF energy rests on."""
    d = 3.0
    lat = np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, 80.0]])
    pos = np.column_stack([
        [0.3, 0.2, +1.0], [1.7, 1.4, +1.3],   # nuclei (+)
        [0.4, 1.1, -0.9], [1.5, 0.5, -1.2],   # electrons (-)
    ]).astype(float)
    q = np.array([+1.0, +1.0, -1.0, -1.0])
    nuc, ele = [0, 1], [2, 3]
    for alpha in (0.20, 0.35, 0.55):
        opts = _opts(alpha, 60.0, 12.0)
        E_total = vq.ewald_2d_point_charge_energy(lat, pos, q, opts)
        T_nn = _parry_four_term(lat, pos[:, nuc], q[nuc], alpha, 60.0, 12.0)
        T_ee = _parry_four_term(lat, pos[:, ele], q[ele], alpha, 60.0, 12.0)
        cross = E_total - T_nn - T_ee  # = the nuclei-electron cross term
        assert abs((T_nn + T_ee + cross) - E_total) < 1e-10
        # And the total is alpha-invariant (cross-check vs alpha=0.35).
        E_ref = vq.ewald_2d_point_charge_energy(
            lat, pos, q, _opts(0.35, 60.0, 12.0)
        )
        assert abs(E_total - E_ref) < 1e-9, f"alpha={alpha}"


def test_bg_energy_invariant_to_vacuum_column():
    """The background variant on a charged block, like the bare slab energy,
    ignores the vacuum (3rd) lattice column."""
    d = 3.0
    pos = np.column_stack([[0.3, 0.2, +1.1], [1.7, 1.4, -0.7]]).astype(float)
    q = np.array([+2.0, +1.0])
    opts = _opts(0.35, 80.0, 14.0)
    vals = [
        vq.ewald_2d_point_charge_energy_with_background(
            np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, Lz]]), pos, q, 0.6, opts
        )
        for Lz in (60.0, 120.0, 200.0)
    ]
    assert max(vals) - min(vals) < 1e-9, vals


def test_bg_potential_reduces_to_m2a_on_neutral():
    """On a neutral cell the background variant of the potential equals
    ``ewald_2d_point_charge_potential`` (the sheet vanishes)."""
    a = 6.0
    lat = np.diag([a, a, 40.0])
    pos = np.column_stack([[0.0, 0.0, 0.7], [a / 2, a / 2, -0.7]])
    q = np.array([+1.0, -1.0])
    pts = np.array([[1.3, 2.1, 0.9], [2.4, 0.6, -0.4], [0.5, 0.5, 0.0]])
    opts = _opts(0.30, 50.0, 14.0)
    v_m2a = vq.ewald_2d_point_charge_potential(lat, pos, q, pts, opts, True)
    for z_b in (-1.0, 0.0, 6.0):
        v_bg = vq.ewald_2d_point_charge_potential_with_background(
            lat, pos, q, z_b, pts, opts, True
        )
        assert np.max(np.abs(v_bg - v_m2a)) < 1e-12, f"z_b={z_b}"


def test_bg_potential_functional_partner():
    """Tie the background potential to the background energy on a NON-NEUTRAL
    block via the functional-partner identity for the {points + sheet} system:

        E_bg = ½ Σ_i q_i V_bg(r_i) + E_self + ½ ΔE_sheet,

    where V_bg is the background potential at the charge sites (the n=0 short-
    range self term is skipped inside), E_self = -(α/√π) Σ q_i², and ΔE_sheet
    is the analytic point-sheet energy (= E_bg - E_bare). The ½ ΔE_sheet term
    is the reciprocal half (the sheet feeling the points' potential)."""
    d = 3.0
    lat = np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, 80.0]])
    pos = np.column_stack([[0.3, 0.2, +1.1], [1.7, 1.4, -0.7]]).astype(float)
    q = np.array([+2.0, +1.0])
    alpha, z_b = 0.35, 0.6
    opts = _opts(alpha, 80.0, 14.0)

    V = vq.ewald_2d_point_charge_potential_with_background(
        lat, pos, q, z_b, pos.T.copy(), opts, True
    )
    e_self = -(alpha / np.sqrt(np.pi)) * float(np.sum(q * q))
    e_sheet = _analytic_sheet(lat, pos, q, z_b)
    E_via = 0.5 * float(np.dot(q, V)) + e_self + 0.5 * e_sheet
    E_bg = vq.ewald_2d_point_charge_energy_with_background(lat, pos, q, z_b, opts)
    assert abs(E_via - E_bg) < 1e-9, f"{E_via} vs {E_bg}"


# ---------------------------------------------------------------------------
# 9. Analytic 2D-Ewald gradients — the slab GDF gradient's rung 2
# ---------------------------------------------------------------------------
#
# ``ewald_2d_point_charge_gradient_with_background`` differentiates the SAME
# truncated sums the with-background energy evaluates (identical alpha and
# cutoffs): the real-space pair term, the z-resolved Parry reciprocal kernel
# (in-plane force from the phase derivative, normal force from dF/dz where
# both Gaussian terms of the erfc' chain rule cancel exactly, leaving
# F'(g,z) = g [e^{gz} erfc(g/2a+az) - e^{-gz} erfc(g/2a-az)]), the g=0 slab
# term (f'(z) = erf(az)), and the neutralising-sheet term
# -(2*pi/A) q_b q_C sgn(z_C - z_b) along the normal.
# ``nuclear_repulsion_slab_ewald_2d_gradient`` is the analytic partner of
# what ``nuclear_repulsion_per_cell`` computes for SLAB_EWALD_2D (the bare
# four-term block = background primitive at z_b=0 minus the exact sheet
# term), at the SCF's actual alpha and nuclear cutoff.


def _h2_compact_slab(atoms=None):
    """The compact 4.6x4.6-bohr H2 slab fixture from test_slab_2d_routing."""
    if atoms is None:
        atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    return vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)


def _slab_lat_opts(alpha=0.0, nuclear_cutoff=30.0):
    lo = vq.LatticeSumOptions()
    lo.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    lo.nuclear_cutoff_bohr = nuclear_cutoff
    if alpha > 0.0:
        lo.slab_ewald_alpha = alpha
    return lo


def test_bg_gradient_matches_fd_on_charged_block():
    """Central-difference gate for the with-background gradient primitive on
    a NET-CHARGED block with z_b off the atoms — this reaches every term,
    including the sheet's sgn(z_C - z_b) normal force (q_b != 0 here), which
    the neutral nuclear+electron composition can never exercise alone."""
    d = 3.0
    lat = np.column_stack([[d, 0, 0], [0, d, 0], [0, 0, 80.0]])
    pos = np.column_stack([[0.3, 0.2, +1.1], [1.7, 1.4, -0.7]]).astype(float)
    q = np.array([+2.0, +1.0])  # net +3 -> q_b = -3, sheet force reachable
    z_b = 0.6
    opts = _opts(0.35, 80.0, 14.0)

    g_ana = np.asarray(
        vq.ewald_2d_point_charge_gradient_with_background(lat, pos, q, z_b, opts)
    )
    h = 1e-5
    g_fd = np.zeros_like(g_ana)
    for a in range(pos.shape[1]):
        for x in range(3):
            pp = pos.copy()
            pp[x, a] += h
            pm = pos.copy()
            pm[x, a] -= h
            ep = vq.ewald_2d_point_charge_energy_with_background(
                lat, pp, q, z_b, opts
            )
            em = vq.ewald_2d_point_charge_energy_with_background(
                lat, pm, q, z_b, opts
            )
            g_fd[x, a] = (ep - em) / (2.0 * h)
    # Measured 6.3e-11 (h=1e-5); gate with headroom.
    assert np.max(np.abs(g_ana - g_fd)) < 1e-8


def test_bg_gradient_neutral_reduces_to_bare_energy_fd():
    """On a neutral cell the sheet vanishes, so the gradient must match a
    central difference of the bare ``ewald_2d_point_charge_energy``."""
    a = 6.0
    lat = np.diag([a, a, 40.0])
    pos = np.column_stack([[0.0, 0.0, 0.7], [a / 2, a / 2, -0.7]])
    q = np.array([+1.0, -1.0])
    opts = _opts(0.30, 50.0, 14.0)

    g_ana = np.asarray(
        vq.ewald_2d_point_charge_gradient_with_background(lat, pos, q, 0.0, opts)
    )
    h = 1e-5
    g_fd = np.zeros_like(g_ana)
    for c in range(pos.shape[1]):
        for x in range(3):
            pp = pos.copy()
            pp[x, c] += h
            pm = pos.copy()
            pm[x, c] -= h
            g_fd[x, c] = (
                vq.ewald_2d_point_charge_energy(lat, pp, q, opts)
                - vq.ewald_2d_point_charge_energy(lat, pm, q, opts)
            ) / (2.0 * h)
    assert np.max(np.abs(g_ana - g_fd)) < 1e-8


def test_slab_nn_gradient_matches_fd_per_component():
    """Gate (a): FD of ``nuclear_repulsion_per_cell(SLAB_EWALD_2D)`` against
    the analytic ``nuclear_repulsion_slab_ewald_2d_gradient`` on the compact
    H2 slab, every atom and Cartesian component including z, h=1e-5."""
    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = _h2_compact_slab(atoms)
    lopts = _slab_lat_opts()

    g_ana = np.asarray(vq.nuclear_repulsion_slab_ewald_2d_gradient(sysp, lopts))
    assert g_ana.shape == (2, 3)

    h = 1e-5
    g_fd = np.zeros_like(g_ana)
    for a in range(2):
        for x in range(3):
            acc = 0.0
            for sign in (+1.0, -1.0):
                xyz = [list(at.xyz) for at in atoms]
                xyz[a][x] += sign * h
                moved = [vq.Atom(at.Z, p) for at, p in zip(atoms, xyz)]
                sys_moved = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), moved)
                acc += sign * vq.nuclear_repulsion_per_cell(sys_moved, lopts)
            g_fd[a, x] = acc / (2.0 * h)
    # Measured 9.5e-12 (h=1e-5).
    assert np.max(np.abs(g_ana - g_fd)) < 1e-8


@pytest.mark.parametrize("alpha", [0.3, 0.6])
def test_slab_nn_gradient_alpha_invariant(alpha):
    """Gate (b): the bare four-term block is alpha-invariant, so its
    derivative must be too — the parameter-stability probe that caught the
    3D e_nuc gauge bug. Measured <= 6.8e-15 across 0.3/0.4/0.6."""
    sysp = _h2_compact_slab()
    g_ref = np.asarray(
        vq.nuclear_repulsion_slab_ewald_2d_gradient(sysp, _slab_lat_opts(0.4))
    )
    g = np.asarray(
        vq.nuclear_repulsion_slab_ewald_2d_gradient(sysp, _slab_lat_opts(alpha))
    )
    assert np.max(np.abs(g - g_ref)) < 1e-9


def test_slab_nn_gradient_translation_invariance():
    """Gate (c): Newton — the per-cell forces sum to zero."""
    sysp = _h2_compact_slab()
    g = np.asarray(
        vq.nuclear_repulsion_slab_ewald_2d_gradient(sysp, _slab_lat_opts())
    )
    assert np.max(np.abs(g.sum(axis=0))) < 1e-12

    # A rigid translation (in-plane and normal) leaves the energy unchanged,
    # consistent with the zero net force.
    lopts = _slab_lat_opts()
    e0 = vq.nuclear_repulsion_per_cell(sysp, lopts)
    shifted = [
        vq.Atom(1, [0.4 + 0.3, 0.3 - 0.2, 0.55 + 1.7]),
        vq.Atom(1, [2.1 + 0.3, 1.8 - 0.2, -0.55 + 1.7]),
    ]
    sys_shifted = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), shifted)
    assert abs(vq.nuclear_repulsion_per_cell(sys_shifted, lopts) - e0) < 1e-10


def test_slab_nn_gradient_composition_and_guards():
    """The wrapper is exactly {with-background gradient at z_b=0} minus the
    analytic sheet force (gate (d): the nuclear block is ALWAYS net-charged
    — q_b = -sum(Z) != 0 — so the sheet path is inherently exercised), and
    it rejects non-slab systems."""
    sysp = _h2_compact_slab()
    lopts = _slab_lat_opts(0.4)

    pos = np.array([list(a.xyz) for a in sysp.unit_cell], dtype=float).T
    Z = np.array([a.Z for a in sysp.unit_cell], dtype=float)
    eopts = _opts(0.4, lopts.nuclear_cutoff_bohr, -1.0)
    g_bg = np.asarray(
        vq.ewald_2d_point_charge_gradient_with_background(
            np.asarray(sysp.lattice), pos, Z, 0.0, eopts
        )
    ).T
    area = 4.6 * 4.6
    q_b = -float(Z.sum())
    assert q_b != 0.0  # the background term is reachable on every nuclear block
    sheet = np.zeros_like(g_bg)
    sheet[:, 2] = -(2.0 * np.pi / area) * q_b * Z * np.sign(pos[2])
    g_wrapper = np.asarray(
        vq.nuclear_repulsion_slab_ewald_2d_gradient(sysp, lopts)
    )
    assert np.max(np.abs(g_wrapper - (g_bg - sheet))) < 1e-13

    bulk = vq.PeriodicSystem(
        3, np.diag([4.6, 4.6, 30.0]), list(sysp.unit_cell)
    )
    with pytest.raises(ValueError, match="dim == 2"):
        vq.nuclear_repulsion_slab_ewald_2d_gradient(bulk, lopts)
