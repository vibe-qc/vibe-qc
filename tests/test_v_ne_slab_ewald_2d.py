"""Rigorous 2D (slab) Ewald electron–nuclear attraction ``V_ne``.

Increment 2 of the ``CoulombMethod.SLAB_EWALD_2D`` workstream
(``handovers/HANDOVER_SLAB_EWALD_2D.md``). The Γ-point ``V_ne`` matrix is built by the
z-resolved Ewald FT (``vibeqc.periodic_v_ne_slab``): libint ``V_short`` + the
in-plane ``g ≠ 0`` reciprocal sum as a 1D ``G_z`` quadrature of the 3D Coulomb
kernel + the analytic ``g = 0`` Parry slab term against the z-resolved AO-pair
density. Gamma-only and multi-k RHF/RKS/UKS slab SCF paths now consume this
kernel.

Correctness witnesses
---------------------
1. **α-invariance** — the rigorous Ewald-split gate. ``V_short`` / ``V_long`` /
   ``V_g0`` each swing by ~0.6 Ha across α; the total does not (to ~1e-9). A
   wrong coefficient on any term, or a wrong reciprocal/g0 prefactor, breaks
   the α-cancellation.
2. **3D-vacuum limit** — ``V_2D`` and the (µHa-vs-PySCF-validated) 3D-Ewald
   ``V_ne`` on the same cell with a large vacuum gap describe one physics up to
   a gauge constant: ``V_2D − V_3D → c·S`` with the residual vanishing as the
   gap grows, and ``c`` *growing* with the gap (the M_z²/V drift the 2D method
   removes).
3. **Small-cell robustness** — a graphene-scale in-plane cell (a ≈ 4.6 bohr),
   where a molecular-grid quadrature of the Γ-Bloch AO pair fails; the
   reciprocal-space FT is cell-size independent. Carbon sto-3g also exercises
   the L = 1 AO-pair-FT path.
4. **Decomposition / dispatch consistency** — ``total = short + long + g0``;
   the dispatch ``LatticeMatrixSet`` Bloch-sums at Γ to the same matrix.
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
)
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
from vibeqc.periodic_v_ne_slab import (
    build_v_ne_slab_ewald_2d_k_cache,
    compute_v_ne_slab_ewald_2d_gamma,
    compute_v_ne_slab_ewald_2d_k_matrix,
)


def _h2_slab():
    """Two H in a graphene-scale (a = 4.6 bohr) in-plane cell, z-polarised."""
    a = 4.6
    lat = np.diag([a, a, 30.0])
    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    system = vq.PeriodicSystem(2, lat, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _c_slab():
    """One C in a small hexagonal-ish cell — exercises L = 1 (p) AO-pair FT."""
    a = 4.6
    lat = np.array([[a, 0.0, 0.0], [-0.5 * a, np.sqrt(3) / 2 * a, 0.0], [0.0, 0.0, 28.0]]).T
    atoms = [vq.Atom(6, [0.0, 0.0, 0.0])]
    system = vq.PeriodicSystem(2, lat, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lat_opts(cutoff=18.0, nuc_cutoff=45.0):
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    lo.nuclear_cutoff_bohr = nuc_cutoff
    lo.coulomb_method = CoulombMethod.SLAB_EWALD_2D
    return lo


def _vne(system, basis, lo, alpha):
    eo = EwaldOptions()
    eo.alpha = alpha
    eo.real_cutoff_bohr = lo.nuclear_cutoff_bohr
    return compute_v_ne_slab_ewald_2d_gamma(system=system, basis=basis, lat_opts=lo, ewald_options=eo)


# ---------------------------------------------------------------------------
# 1. α-invariance — the rigorous Ewald-split gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("system_basis", [_h2_slab(), _c_slab()], ids=["H2", "C"])
def test_v_ne_alpha_invariance(system_basis):
    system, basis = system_basis
    lo = _lat_opts()
    d3 = _vne(system, basis, lo, 0.30)
    d45 = _vne(system, basis, lo, 0.45)
    d6 = _vne(system, basis, lo, 0.60)
    dev = max(
        float(np.max(np.abs(d3.v_total - d45.v_total))),
        float(np.max(np.abs(d45.v_total - d6.v_total))),
    )
    # The components must genuinely move with α (else the test is vacuous).
    comp_swing = float(np.max(np.abs(d3.v_short - d6.v_short)))
    assert comp_swing > 0.1, f"V_short barely moved with α ({comp_swing:.2e})"
    assert dev < 1e-8, (
        f"2D V_ne must be α-invariant; got {dev:.2e} across α∈{{.30,.45,.60}}. "
        "A wrong reciprocal/g0 coefficient breaks the Ewald-split cancellation."
    )


# ---------------------------------------------------------------------------
# 2. Decomposition / symmetry / dispatch consistency
# ---------------------------------------------------------------------------
def test_v_ne_decomposition_and_symmetry():
    system, basis = _h2_slab()
    lo = _lat_opts()
    d = _vne(system, basis, lo, 0.45)
    assert np.allclose(d.v_total, d.v_short + d.v_long + d.v_g0, atol=1e-14)
    for M in (d.v_total, d.v_short, d.v_long, d.v_g0):
        assert np.max(np.abs(M - M.T)) < 1e-12


def test_dispatch_lattice_bloch_sums_to_gamma():
    """The dispatch LatticeMatrixSet Bloch-sums at Γ to the Γ builder.

    Both use the default α (0.4): the dispatch passes ``ewald_options=None``.
    """
    system, basis = _h2_slab()
    lo = _lat_opts()
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lo)
    V_bloch = np.real(bloch_sum(V_lat, np.zeros(3)))
    V_bloch = 0.5 * (V_bloch + V_bloch.T)
    d = compute_v_ne_slab_ewald_2d_gamma(basis, system, lo)  # default α = 0.4
    assert np.max(np.abs(V_bloch - d.v_total)) < 1e-10


def test_multi_k_v_ne_gamma_matches_gamma_builder_and_nonzero_k_is_hermitian():
    system, basis = _h2_slab()
    lo = _lat_opts()
    cache = build_v_ne_slab_ewald_2d_k_cache(basis, system, lo, alpha=0.4)

    V0 = compute_v_ne_slab_ewald_2d_k_matrix(
        basis, system, lo, np.zeros(3), alpha=0.4, cache=cache,
    )
    Vg = compute_v_ne_slab_ewald_2d_gamma(basis, system, lo).v_total
    assert np.max(np.abs(V0 - Vg)) < 1e-10

    kx = np.array([np.pi / system.lattice[0, 0], 0.0, 0.0])
    Vk = compute_v_ne_slab_ewald_2d_k_matrix(
        basis, system, lo, kx, alpha=0.4, cache=cache,
    )
    assert np.max(np.abs(Vk - Vk.conj().T)) < 1e-12
    assert np.all(np.isfinite(Vk))


# ---------------------------------------------------------------------------
# 3. 3D-vacuum-limit absolute anchor — ties V_2D to the µHa 3D path
# ---------------------------------------------------------------------------
def test_v_ne_matches_3d_vacuum_limit_up_to_gauge():
    """``V_2D − V_3D(EWALD_3D, large c) = c·S + resid``: same physics, gauge
    differs. The residual shrinks and the gauge const ``c`` grows as the gap
    grows — the headline 2D property (3D ``V_ne`` drifts ∝ M_z²/V; 2D is
    fixed). Anchors V_2D to the analytic-FT 3D ``V_ne`` validated to µHa vs
    PySCF GDF."""
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma

    a = 4.6
    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sys2 = vq.PeriodicSystem(2, np.diag([a, a, 30.0]), atoms)
    bas2 = vq.BasisSet(sys2.unit_cell_molecule(), "sto-3g")
    lo2 = _lat_opts()
    V2 = compute_v_ne_slab_ewald_2d_gamma(bas2, sys2, lo2).v_total

    S = np.real(bloch_sum(vq.compute_overlap_lattice(bas2, sys2, lo2), np.zeros(3)))
    S = 0.5 * (S + S.T)

    def gauge_fit(cz):
        sys3 = vq.PeriodicSystem(3, np.diag([a, a, cz]), atoms)
        bas3 = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
        lo3 = LatticeSumOptions()
        lo3.cutoff_bohr = 18.0
        lo3.nuclear_cutoff_bohr = 45.0
        lo3.coulomb_method = CoulombMethod.EWALD_3D
        V3 = compute_v_ne_ewald_3d_ft_gamma(bas3, sys3, lo3, ke_cutoff=300.0)
        D = V2 - V3
        c = float(np.sum(D * S) / np.sum(S * S))
        resid = float(np.max(np.abs(D - c * S)))
        return c, resid

    c_small, resid_small = gauge_fit(45.0)
    c_large, resid_large = gauge_fit(90.0)
    # Same physics up to a gauge constant ×S (loose: ke/gap limited).
    assert resid_large < 1e-3, f"V_2D − V_3D not ∝ S at large gap (resid {resid_large:.2e})"
    # The residual converges (3D → 2D limit) and the gauge const drifts with
    # the gap — the 2D-vs-3D signature.
    assert resid_large < resid_small + 1e-9
    assert c_large > c_small + 0.5, (
        f"3D V_ne gauge must drift with the vacuum gap; c {c_small:.3f}→{c_large:.3f}"
    )


# ---------------------------------------------------------------------------
# 4. Guards
# ---------------------------------------------------------------------------
def test_rejects_dim3():
    a = 4.6
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.4, 0, 0])]
    sys3 = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    bas3 = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
    lo = _lat_opts()
    with pytest.raises(ValueError, match="dim == 2"):
        compute_v_ne_slab_ewald_2d_gamma(bas3, sys3, lo)


def test_z_profile_self_check_raises_when_underresolved():
    """The ∫ n dz = S self-check guards an under-resolved z-grid."""
    system, basis = _h2_slab()
    lo = _lat_opts()
    with pytest.raises(RuntimeError, match="under-resolved"):
        # z_pad far too small to capture the AO-pair density tail.
        compute_v_ne_slab_ewald_2d_gamma(
            basis, system, lo, z_pad=0.05, z_spacing=0.5
        )


# ---------------------------------------------------------------------------
# 5. The z-profile first moment is the BLOCH-summed AO z-dipole (the correct
#    oracle), NOT the home-cell ⟨μ|z|ν⟩.
# ---------------------------------------------------------------------------
def test_z_profile_first_moment_is_bloch_dipole():
    """``∫ z n_μν(z) dz`` equals the **Bloch-summed** AO z-dipole
    ``Σ_g ⟨μ,0|z|ν,R_g⟩``, not the home-cell ``⟨μ|z|ν⟩``.

    A *polar* slab (HF layer, F at z=0, H at z=1.74). The z-resolved AO-pair
    density ``n_μν(z)`` is, by construction, the planar density of the **full
    Γ-Bloch** AO pair (one AO in the home cell, the other summed over all
    in-plane images), so its centroid is the Bloch-summed z-dipole
    (``compute_multipole_moments_lattice`` summed over cells at Γ). For diffuse
    functions at a small in-plane spacing the cross-cell images contribute
    O(7e-3) here -- a real physical term, *not* a bug. An earlier xfail compared
    against the home-cell ``compute_dipole(...).z`` (which omits those images)
    and so mis-flagged the FT as broken; the ~0.1 Ha polar-SCF error it was
    blamed for was in fact the undamped-J core under-convergence, now fixed by
    the Ewald split (see ``handovers/HANDOVER_SLAB_EWALD_2D.md``).
    """
    from vibeqc import direct_lattice_cells
    from vibeqc._vibeqc_core import compute_multipole_moments_lattice
    from vibeqc.aux_basis import _ao_scales_for_rsgdf
    from vibeqc.periodic_v_ne_slab import (
        _slab_geometry,
        _z_resolved_pair_profiles,
    )

    b = 8.0
    atoms = [vq.Atom(9, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.74])]
    system = vq.PeriodicSystem(2, np.diag([b, b, 40.0]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lo = _lat_opts()
    _, nhat, _, _ = _slab_geometry(system)
    cells = direct_lattice_cells(system, lo.cutoff_bohr)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    pair_scales = np.outer(_ao_scales_for_rsgdf(basis), _ao_scales_for_rsgdf(basis))
    z_I = np.array([list(a.xyz) for a in system.unit_cell]) @ nhat
    z_center = 0.5 * (float(z_I.min()) + float(z_I.max()))
    z_half = 0.5 * (float(z_I.max()) - float(z_I.min())) + 8.0
    z_grid, dz, n_prof = _z_resolved_pair_profiles(
        basis, system, lo, nhat=nhat, pair_scales=pair_scales, R_g=R_g,
        z_center=z_center, z_half=z_half,
    )
    dip_from_profile = np.einsum("mnz,z->mn", n_prof, z_grid * dz)

    # Bloch-summed z-dipole Σ_g ⟨μ,0|z|ν,R_g⟩: the lattice multipole z-block
    # (component 3 = z), summed over cells at Γ (k=0 phase = 1).
    M = compute_multipole_moments_lattice(basis, system, lo, 1, (0.0, 0.0, 0.0))
    z_bloch = sum(
        np.asarray(M.blocks[c][3], dtype=float) for c in range(len(M.cells))
    )
    z_bloch = 0.5 * (z_bloch + z_bloch.T)
    assert np.max(np.abs(dip_from_profile - z_bloch)) < 1e-7

    # And it is *not* the home-cell dipole -- the cross-cell images are real.
    from vibeqc import compute_dipole

    z_home = np.asarray(compute_dipole(basis, [0.0, 0.0, 0.0]).z)
    z_home = 0.5 * (z_home + z_home.T)
    assert np.max(np.abs(dip_from_profile - z_home)) > 1e-3


# ---------------------------------------------------------------------------
# 5. k != 0 dispersion parity vs an independent implementation
#    (GitLab #106 -- the SLAB-VNE-K-NONZERO wrong-answer regression)
# ---------------------------------------------------------------------------
#
# The alpha-invariance, 3D-vacuum-limit and Gamma-dispatch witnesses above are
# all *Gamma* statements: they pin the gauge and the Ewald split, and every one
# of them stayed green while V_ne(k) was wrong at every k != 0.
#
# The defect: the g != 0 long-range block needs
#   V_long_muv(k) = S_G v(G) S_g e^{+ik.R_g} conj(rhat_muv(g; G)),
# and the shipped assembly conjugated the *whole Bloch sum*, which conjugates
# the +ik phase along with the AO-pair FT and so evaluates V_long(-k). See the
# derivation and the Parry / de Leeuw-Perram / Sun-Berkelbach citations at
# vibeqc.periodic_v_ne_slab.compute_v_ne_slab_ewald_2d_k_matrix.
#
# k -> -k is invisible wherever k = -k modulo a reciprocal lattice vector, and
# the PySCF-referenced H2 slab gate in tests/test_slab_2d_routing.py runs a
# Gamma-centred (2,2,1) mesh whose four k-points -- (0,0), (0,1/2), (1/2,0),
# (1/2,1/2) in fractional coordinates -- are ALL time-reversal-invariant
# momenta. That gate could not have failed on this defect at any tolerance; it
# is a gauge witness, not a dispersion witness, and it stays exactly as it is.
# The (4,4,1) h-BN mesh below is the smallest one that separates the two: with
# the defect in place its four TRIM points still reproduced the reference to
# <= 1.4e-4 Ha while the twelve non-TRIM points missed by up to 0.68 Ha.

# Monolayer h-BN, a = 2.504 A, 20 A of vacuum. PySCF `cell.a` rows are the
# lattice vectors and PeriodicSystem stores them as COLUMNS, hence every
# transpose below -- the row array handed over directly builds a well-formed
# but different 63.43-degree cell of the same volume (see
# `test_sheet_fixtures_are_the_hexagonal_cells_they_claim_to_be` in
# tests/test_periodic_convergence_auto.py).
_HBN_A_ROWS = np.array(
    [
        [4.731874216062929, 0.0, 0.0],
        [2.3659371080314644, 4.097923278623072, 0.0],
        [0.0, 0.0, 37.794522492515405],
    ]
)
_SHEET_Z = 18.897261246257703

# Physical monolayer h-BN: N at (0,0), B at the (1/3,1/3) honeycomb site of
# this 60-degree cell, B-N = 1.4457 A = a/sqrt(3).
_HBN_ATOMS = (
    (5, (2.3659371080314644, 1.3659744262076907, _SHEET_Z)),
    (7, (0.0, 0.0, _SHEET_Z)),
)

# The cell as staged in the rp235 reproducer and quoted throughout GitLab #106
# and #85. Its boron sits at fractional (1/3, 2/3) -- the second honeycomb site
# of a 120-degree cell -- inside this 60-degree cell, which puts B-N at 0.835 A.
# It is NOT h-BN, and no result measured on it is an h-BN result. It is pinned
# here anyway, under an honest name, for two reasons: it is the exact cell the
# filed evidence was taken on, and its compressed geometry drives the defect
# ~25x harder than the physical cell does, which makes it the sharper witness.
_HBN_RP235_ATOMS = (
    (5, (3.154582810708619, 2.731948852415381, _SHEET_Z)),
    (7, (0.0, 0.0, _SHEET_Z)),
)

# Published target: per-band h(k) dispersions e_b(k) - e_b(Gamma) in Ha for
# h-BN/STO-3G on the Gamma-centred (4,4,1) mesh, from PySCF 2.14.0
# `pbc.dft.KRKS(cell, kpts).density_fit().get_hcore()` with dimension = 2,
# low_dim_ft_type = 'inf_vacuum', precision = 1e-10, run OUT OF PROCESS
# (CLAUDE.md section 10 -- vibe-qc never imports PySCF). Rows are k in
# `cell.make_kpts((4,4,1))` order, columns are bands. Dispersions rather than
# absolute eigenvalues because the two codes use different slab gauges; a
# k-independent constant cancels here and a k-dependent defect does not.
# Regenerate with the runner referenced in GitLab #106.
#
# Physical h-BN (`_HBN_ATOMS`). Pre-fix the assembly missed this by 2.76e-2 Ha.
_HBN_HCORE_DISPERSION_PYSCF = np.array(
    [
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
         0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        [0.00006035, -0.00039828, 0.02115907, -0.02310336, -0.01390706,
         0.08384893, 0.04450478, 0.13660133, -0.07813138, -0.16201479],
        [0.00011866, -0.00076825, 0.06814376, -0.06525096, -0.04776834,
         0.18776479, 0.09361302, 0.34762500, -0.22640930, -0.34809010],
        [0.00006035, -0.00039828, 0.02115907, -0.02310336, -0.01390706,
         0.08384893, 0.04450478, 0.13660133, -0.07813138, -0.16201479],
        [0.00006035, -0.00039828, 0.02115907, -0.02310338, -0.01390706,
         0.08384893, 0.04450477, 0.13660133, -0.07813138, -0.16201479],
        [0.00006035, -0.00039828, 0.02115907, -0.02310337, -0.01390706,
         0.08384893, 0.04450477, 0.13660133, -0.07813138, -0.16201479],
        [0.00009501, -0.00077057, 0.05795402, -0.06772394, -0.03779840,
         0.17937381, 0.14642299, 0.28880345, -0.21665393, -0.34605889],
        [0.00009501, -0.00077057, 0.05795402, -0.06772394, -0.03779840,
         0.17937381, 0.14642299, 0.28880345, -0.21665393, -0.34605889],
        [0.00011866, -0.00076825, 0.06814377, -0.06525096, -0.04776834,
         0.18776479, 0.09361302, 0.34762502, -0.22640931, -0.34809010],
        [0.00009501, -0.00077057, 0.05795402, -0.06772395, -0.03779840,
         0.17937381, 0.14642299, 0.28880346, -0.21665393, -0.34605889],
        [0.00011866, -0.00076825, 0.06814377, -0.06525096, -0.04776834,
         0.18776480, 0.09361302, 0.34762502, -0.22640930, -0.34809010],
        [0.00009501, -0.00077057, 0.05795402, -0.06772395, -0.03779841,
         0.17937381, 0.14642299, 0.28880346, -0.21665393, -0.34605889],
        [0.00006034, -0.00039828, 0.02115907, -0.02310338, -0.01390706,
         0.08384893, 0.04450477, 0.13660134, -0.07813138, -0.16201479],
        [0.00009501, -0.00077057, 0.05795402, -0.06772394, -0.03779840,
         0.17937381, 0.14642299, 0.28880345, -0.21665393, -0.34605889],
        [0.00009502, -0.00077057, 0.05795402, -0.06772394, -0.03779840,
         0.17937381, 0.14642299, 0.28880345, -0.21665393, -0.34605889],
        [0.00006035, -0.00039828, 0.02115907, -0.02310338, -0.01390706,
         0.08384893, 0.04450477, 0.13660134, -0.07813138, -0.16201479],
    ]
)

# The rp235 reproducer cell (`_HBN_RP235_ATOMS`) -- not h-BN, see above.
# Pre-fix the assembly missed this by 6.80e-1 Ha.
_HBN_RP235_HCORE_DISPERSION_PYSCF = np.array(
    [
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
         0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
        [0.00007117, 0.00049463, -0.05107950, 0.03767625, 0.01752007,
         0.09448972, 0.01473466, 0.11091426, -0.20775894, -0.16769607],
        [0.00011770, 0.00120462, -0.08266876, 0.07261998, 0.02887142,
         0.21175582, 0.01480752, 0.29113196, -0.43301207, -0.38521850],
        [0.00007117, 0.00049463, -0.05107950, 0.03767625, 0.01752007,
         0.09448972, 0.01473466, 0.11091426, -0.20775893, -0.16769607],
        [0.00007117, 0.00049463, -0.05107950, 0.03767626, 0.01752007,
         0.09448972, 0.01473465, 0.11091424, -0.20775894, -0.16769608],
        [0.00009133, 0.00046839, 0.01996554, 0.02073036, 0.02831999,
         0.03241046, 0.08528008, 0.10016687, -0.20285346, -0.14479656],
        [0.00016363, 0.00108761, -0.03220761, 0.02387081, 0.05202499,
         0.13991380, 0.14395322, 0.23294381, -0.44238028, -0.38900772],
        [0.00012380, 0.00113664, -0.08102924, 0.03543695, 0.03730189,
         0.19654011, 0.13396238, 0.18054150, -0.42962526, -0.38796559],
        [0.00011770, 0.00120463, -0.08266877, 0.07261998, 0.02887141,
         0.21175582, 0.01480750, 0.29113194, -0.43301207, -0.38521855],
        [0.00016363, 0.00108761, -0.03220760, 0.02387081, 0.05202499,
         0.13991380, 0.14395325, 0.23294380, -0.44238028, -0.38900767],
        [0.00019553, 0.00090025, 0.03227051, -0.01296256, 0.06540020,
         0.06184152, 0.14261776, 0.32176762, -0.45983974, -0.38964571],
        [0.00016363, 0.00108761, -0.03220760, 0.02387081, 0.05202500,
         0.13991380, 0.14395325, 0.23294380, -0.44238028, -0.38900768],
        [0.00007117, 0.00049463, -0.05107950, 0.03767626, 0.01752007,
         0.09448971, 0.01473465, 0.11091424, -0.20775893, -0.16769607],
        [0.00012379, 0.00113664, -0.08102924, 0.03543695, 0.03730189,
         0.19654011, 0.13396238, 0.18054151, -0.42962526, -0.38796559],
        [0.00016363, 0.00108761, -0.03220761, 0.02387080, 0.05202499,
         0.13991380, 0.14395322, 0.23294381, -0.44238028, -0.38900773],
        [0.00009133, 0.00046839, 0.01996554, 0.02073036, 0.02831999,
         0.03241046, 0.08528008, 0.10016687, -0.20285346, -0.14479653],
    ]
)

# Per-band agreement demanded of h(k) = T(k) + V_ne(k). The pre-fix assembly
# missed by 2.76e-2 Ha on physical h-BN and 6.80e-1 Ha on the rp235 cell,
# i.e. by 28x and 680x this tolerance.
_HCORE_DISPERSION_TOL_HA = 1e-3


def _hbn_slab(atoms=_HBN_ATOMS):
    system = vq.PeriodicSystem(
        2,
        _HBN_A_ROWS.T,
        [vq.Atom(z, list(xyz)) for z, xyz in atoms],
        0,
        1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


@pytest.mark.parametrize(
    "atoms, reference",
    [
        (_HBN_ATOMS, _HBN_HCORE_DISPERSION_PYSCF),
        (_HBN_RP235_ATOMS, _HBN_RP235_HCORE_DISPERSION_PYSCF),
    ],
    ids=["hbn-physical", "rp235-reproducer-cell"],
)
def test_slab_hcore_k_dispersion_matches_pyscf_hbn(atoms, reference):
    """h(k) band dispersions match an independent slab implementation.

    SCF-free and density-independent: this is a statement about the
    one-electron Hamiltonian alone, so a failure localises to T(k) or V_ne(k)
    and nothing downstream. Regression for GitLab #106.

    Both cells are pinned. Parity is a statement about two implementations on
    one cell, so it is meaningful on either; physical h-BN is the one that may
    be quoted as an h-BN result, and the rp235 cell is the one the filed
    evidence was measured on and the harsher witness of the two.
    """
    import scipy.linalg

    from vibeqc._vibeqc_core import monkhorst_pack
    from vibeqc.periodic_k_gdf import _clone_slab_lattice_options

    system, basis = _hbn_slab(atoms)
    lat_opts = _clone_slab_lattice_options(vq.PeriodicRHFOptions().lattice_opts)

    kmesh = monkhorst_pack(system, [4, 4, 1], [0, 0, 0], False)
    kpoints = np.asarray(kmesh.kpoints, dtype=float)
    assert kpoints.shape[0] == reference.shape[0]

    S_lat = vq.compute_overlap_lattice(basis, system, lat_opts)
    T_lat = vq.compute_kinetic_lattice(basis, system, lat_opts)
    cache = build_v_ne_slab_ewald_2d_k_cache(basis, system, lat_opts, alpha=0.0)

    eigenvalues = []
    for kpoint in kpoints:
        S = np.asarray(bloch_sum(S_lat, kpoint), dtype=complex)
        S = 0.5 * (S + S.conj().T)
        T = np.asarray(bloch_sum(T_lat, kpoint), dtype=complex)
        V = compute_v_ne_slab_ewald_2d_k_matrix(
            basis, system, lat_opts, kpoint, alpha=0.0, cache=cache,
        )
        H = T + V
        H = 0.5 * (H + H.conj().T)
        eigenvalues.append(scipy.linalg.eigh(H, S, eigvals_only=True))

    eigenvalues = np.asarray(eigenvalues)
    dispersion = eigenvalues - eigenvalues[0][None, :]
    error = np.max(np.abs(dispersion - reference))
    assert error < _HCORE_DISPERSION_TOL_HA, (
        f"slab h(k) dispersion deviates from the PySCF reference by "
        f"{error:.3e} Ha (tolerance {_HCORE_DISPERSION_TOL_HA:.1e} Ha)"
    )


def test_slab_v_ne_k_conjugates_the_pair_ft_not_the_bloch_phase():
    """Freeze the k != 0 assembly convention (GitLab #106).

    Rebuilds V_long(k) from the cache both ways and asserts the shipped matrix
    is the one that conjugates the per-cell AO-pair FT while keeping the +ik
    Bloch phase. The two differ only away from Gamma and away from the
    time-reversal-invariant momenta, which is exactly why every Gamma witness
    in this file stayed green through the defect.
    """
    from vibeqc.periodic_v_ne_slab import _g0_kernel

    system, basis = _hbn_slab()
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 12.0
    lat_opts.nuclear_cutoff_bohr = 18.0
    cache = build_v_ne_slab_ewald_2d_k_cache(basis, system, lat_opts, alpha=0.4)

    # A generic k: neither Gamma nor a TRIM point of this lattice.
    b = 2.0 * np.pi * np.linalg.inv(np.asarray(system.lattice)).T
    k = 0.25 * b[:, 0] + 0.25 * b[:, 1]

    cells = list(cache.v_short_lat.cells)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    phases = np.exp(1j * (R_g @ k))

    def assemble(*, conjugate_pair_ft_per_cell):
        pair = cache.pair_at_cells.conj() if conjugate_pair_ft_per_cell \
            else cache.pair_at_cells
        bloch = np.einsum("g,gmnk->mnk", phases, pair, optimize=True)
        if not conjugate_pair_ft_per_cell:
            bloch = bloch.conj()  # the defective order: conjugates e^{+ik.R}
        V_long = np.einsum("k,mnk->mn", cache.v_long_G, bloch, optimize=True)
        V_short = np.asarray(bloch_sum(cache.v_short_lat, k), dtype=complex)
        n_prof = np.einsum(
            "g,gmnz->mnz", phases, cache.n_prof_at_cells, optimize=True
        )
        V_g0 = np.zeros_like(V_short)
        for Z, z_I in zip(cache.nuclei_Z, cache.z_I):
            ker = _g0_kernel(cache.z_grid - z_I, cache.alpha) * cache.dz
            V_g0 += cache.g0_prefactor * float(Z) * np.einsum(
                "mnz,z->mn", n_prof, ker, optimize=True
            )
        V = V_short + V_long + V_g0
        return 0.5 * (V + V.conj().T)

    shipped = compute_v_ne_slab_ewald_2d_k_matrix(
        basis, system, lat_opts, k, alpha=0.4, cache=cache,
    )
    correct = assemble(conjugate_pair_ft_per_cell=True)
    defective = assemble(conjugate_pair_ft_per_cell=False)

    assert np.max(np.abs(shipped - correct)) < 1e-12
    # The two conventions must actually differ here, or the test is vacuous.
    assert np.max(np.abs(correct - defective)) > 1e-3
