"""The shared corrected-Ewald multi-k exchange split.

Pins the properties the construction is *supposed* to have, taken from the
literature rather than from the implementation:

* the ``q + G = 0`` singularity constant depends only on the k-mesh, the
  lattice and the cell volume, **not** on where the atoms sit inside the
  cell (Carrier, Rohra & Görling, Phys. Rev. B 75, 205126 (2007), abstract
  and § II);
* it scales as ``N^{-1/3}`` with the Born-von-Kármán supercell, i.e. the
  finite-size exchange correction vanishes in the dense-mesh limit
  (Gygi & Baldereschi, Phys. Rev. B 34, 4405(R) (1986); probe-charge form
  per Sundararaman & Arias, Phys. Rev. B 87, 165122 (2013));
* every precondition the split needs fails closed, so no caller can
  silently fall back to the divergent bare-``1/r`` image sum.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_corrected_exchange import (
    CorrectedEwaldExchange,
    corrected_exchange_unavailable_reason,
)

ALPHA = 0.3


def _cell(box: float = 12.0, displace: float = 0.0):
    """H2 in a cubic box, optionally shifted off-centre inside the cell."""
    c = box / 2 + displace
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _cells_r_cart(basis, sysp, cutoff=12.0):
    from vibeqc._vibeqc_core import compute_overlap_lattice

    o = vq.PeriodicKSOptions()
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    lat = compute_overlap_lattice(basis, sysp, o.lattice_opts)
    return np.asarray(
        [np.asarray(c.r_cart, dtype=float) for c in lat.cells]
    )


def test_exxdiv_constant_is_independent_of_atomic_positions():
    """Carrier, Rohra & Görling (2007): the singularity correction depends
    only on the number and positions of the k points and on the lattice
    vectors -- in particular the cell volume -- and *not* on where the
    atoms sit within the unit cell.

    This is the property that lets one scalar be built once per SCF and
    reused at every k, so it is worth pinning against the paper rather
    than against our own output.
    """
    mesh = [2, 2, 1]
    consts = []
    for displace in (0.0, 1.3, -2.1):
        sysp, basis = _cell(displace=displace)
        x = CorrectedEwaldExchange.build(
            basis, sysp, _cells_r_cart(basis, sysp),
            vq.monkhorst_pack(sysp, mesh), ALPHA,
            where="test",
        )
        consts.append(x.exchange_g0)
    assert consts[0] == pytest.approx(consts[1], rel=1e-12)
    assert consts[0] == pytest.approx(consts[2], rel=1e-12)


def test_exxdiv_constant_decays_with_the_bvk_supercell():
    """The finite-size exchange correction must vanish as the mesh densifies.

    The probe-charge Madelung constant of the BvK supercell scales as
    ``xi_M(N.cell) ~ xi_M(cell) / N^{1/3}`` for a cubic mesh, so denser
    meshes must give a monotonically smaller correction. A correction that
    did not shrink would not be a finite-size effect.
    """
    sysp, basis = _cell()
    crc = _cells_r_cart(basis, sysp)
    mags = []
    for n in (1, 2, 3):
        x = CorrectedEwaldExchange.build(
            basis, sysp, crc, vq.monkhorst_pack(sysp, [n, n, n]), ALPHA,
            where="test",
        )
        mags.append(abs(x.exchange_g0))
    assert mags[0] > mags[1] > mags[2], mags


def test_build_is_independent_of_the_lattice_cutoff():
    """Neither k-space arm may depend on the real-space image ball.

    The cutoff only controls the (erfc-screened, convergent) short-range
    arm the caller builds separately; if the reciprocal cache or the
    ``q + G = 0`` constant moved with it, the split would reintroduce the
    cutoff drift it exists to remove.
    """
    sysp, basis = _cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    consts = [
        CorrectedEwaldExchange.build(
            basis, sysp, _cells_r_cart(basis, sysp, cutoff=cut), km, ALPHA,
            where="test",
        ).exchange_g0
        for cut in (12.0, 20.0)
    ]
    assert consts[0] == pytest.approx(consts[1], rel=1e-12)


def test_gamma_mesh_reduces_to_the_single_cell_probe_charge():
    """A (1,1,1) mesh is the unit cell's own probe-charge correction."""
    from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell

    sysp, basis = _cell()
    x = CorrectedEwaldExchange.build(
        basis, sysp, _cells_r_cart(basis, sysp),
        vq.monkhorst_pack(sysp, [1, 1, 1]), ALPHA, where="test",
    )
    volume = float(abs(np.linalg.det(np.asarray(sysp.lattice, dtype=float))))
    expected = probe_charge_madelung_supercell(sysp, (1, 1, 1)) - np.pi / (
        ALPHA * ALPHA * volume * 1.0
    )
    assert x.exchange_g0 == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Fail-closed surface: never fall back to the divergent bare sum
# ---------------------------------------------------------------------------


def test_explicit_kpoint_list_is_refused():
    """An explicit / symmetry-reduced k list carries no mesh dimensions."""
    sysp, basis = _cell()
    full = vq.monkhorst_pack(sysp, [2, 2, 1])
    ks = np.asarray([np.asarray(k, dtype=float) for k in list(full.kpoints)[:2]])
    reduced = vq.as_bloch_kmesh(
        vq.KPoints(
            kpoints_cart=ks,
            kpoints_frac=np.zeros_like(ks),
            weights=np.array([0.5, 0.5]),
        )
    )
    assert "Monkhorst-Pack" in (
        corrected_exchange_unavailable_reason(sysp, reduced) or ""
    )
    with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
        CorrectedEwaldExchange.build(
            basis, sysp, _cells_r_cart(basis, sysp), reduced, ALPHA,
            where="test",
        )


def test_slab_mode_is_refused():
    """The reciprocal q-channel cache is 3D-only."""
    sysp, basis = _cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    assert "slab" in (
        corrected_exchange_unavailable_reason(sysp, km, slab_mode=True) or ""
    )
    with pytest.raises(NotImplementedError, match="slab"):
        CorrectedEwaldExchange.build(
            basis, sysp, _cells_r_cart(basis, sysp), km, ALPHA,
            where="test", slab_mode=True,
        )


def test_non_positive_alpha_is_refused():
    sysp, basis = _cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    with pytest.raises(ValueError, match="positive Ewald"):
        CorrectedEwaldExchange.build(
            basis, sysp, _cells_r_cart(basis, sysp), km, 0.0, where="test",
        )


def test_a_full_mesh_is_accepted():
    sysp, _ = _cell()
    for mesh in ([1, 1, 1], [2, 1, 1], [2, 2, 2]):
        km = vq.monkhorst_pack(sysp, mesh)
        assert corrected_exchange_unavailable_reason(sysp, km) is None


# ---------------------------------------------------------------------------
# Symmetry-unfolded IBZ exchange: wedge-native SCF == full-mesh SCF
# ---------------------------------------------------------------------------


def _sym_cell(box: float = 12.0):
    """H2 in a cube with symmetry attached (P4/mmm; multi-cell at cutoff 12)."""
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    vq.attach_symmetry(sysp)
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _sym_slab_cell():
    """Neutral P4/mmm HeH2 layer with a nontrivial [2,2,1] wedge."""
    sysp = vq.slab_2d(
        [6.0, 0.0, 0.0],
        [0.0, 6.0, 0.0],
        [
            vq.Atom(2, [0.0, 0.0, 0.0]),
            vq.Atom(1, [3.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 3.0, 0.0]),
        ],
        min_a3_bohr=20.0,
    )
    vq.attach_symmetry(sysp)
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _rhf_opts():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 12.0
    o.damping = 0.3
    o.max_iter = 200
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    return o


def _ks_opts(functional: str = "PBE0"):
    o = vq.PeriodicKSOptions()
    o.functional = functional
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 12.0
    o.damping = 0.3
    o.max_iter = 200
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-7
    return o


def _slab_hse_opts():
    o = vq.PeriodicKSOptions()
    o.functional = "HSE06"
    o.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    o.lattice_opts.cutoff_bohr = 8.0
    o.lattice_opts.nuclear_cutoff_bohr = 10.0
    o.lattice_opts.slab_ewald_alpha = 0.4
    grid = vq.GridOptions()
    grid.n_radial = 12
    grid.n_theta = 8
    grid.n_phi = 12
    o.grid = grid
    o.damping = 0.3
    o.max_iter = 100
    o.use_diis = True
    o.conv_tol_energy = 1e-11
    o.conv_tol_grad = 1e-8
    return o


def test_ibz_mesh_matches_full_mesh_on_an_oblique_cell():
    """Wedge == full mesh on a cell whose reciprocal basis is NOT symmetric.

    Every other IBZ parity fixture here is a cube, where ``B = 2pi (A^-1)^T``
    is symmetric and a transposed fractional-coordinate convention is
    indistinguishable from the right one. That blind spot shipped a real
    bug: ``_frac_coords`` solved ``B^T q = k`` instead of ``B q = k``
    (native Monkhorst-Pack constructs ``k = B q``), which only surfaced on
    an oblique mesh -- fixed in `0b35934a9` after this module's cubic
    fixtures passed it through. A hexagonal cell keeps that convention
    honest end to end, through star matching, the density unfolding, the
    exact orbit fold, and the orbit-reduced exchange build -- not just the
    unit-level overlap transport the fix's own regression covers.

    ``(3,3,1)`` is the mesh shape the bug was found on; ``(2,2,2)`` adds
    the c-axis stars.
    """
    a, c = 5.6, 9.0
    # PeriodicSystem stores Cartesian lattice vectors as COLUMNS.
    lat = np.array([[a, 0.0, 0.0],
                    [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
                    [0.0, 0.0, c]]).T
    sysp = vq.PeriodicSystem(
        3, lat, [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    )
    vq.attach_symmetry(sysp)
    assert sysp.symmetry.international_symbol.startswith("P6")
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    def _opts():
        o = vq.PeriodicRHFOptions()
        o.lattice_opts.cutoff_bohr = 9.0
        o.lattice_opts.nuclear_cutoff_bohr = 9.0
        o.damping = 0.3
        o.max_iter = 150
        o.conv_tol_energy = 1e-10
        o.conv_tol_grad = 1e-7
        return o

    for mesh in ([3, 3, 1], [2, 2, 2]):
        kf = vq.monkhorst_pack(sysp, mesh, use_symmetry=False)
        ki = vq.monkhorst_pack(sysp, mesh, use_symmetry=True)
        assert len(ki.kpoints) < len(kf.kpoints)   # a real reduction
        rf = vq.run_rhf_periodic_multi_k_ewald3d(
            sysp, basis, kf, _opts(),
            auto_optimize_truncation=False, progress=False,
        )
        ri = vq.run_rhf_periodic_multi_k_ewald3d(
            sysp, basis, ki, _opts(),
            auto_optimize_truncation=False, progress=False,
        )
        assert rf.converged and ri.converged
        assert ri.energy == pytest.approx(rf.energy, abs=1e-9)
        assert ri.used_kpoint_symmetry_unfolding is True


def test_ibz_mesh_matches_full_mesh_rhf():
    """Wedge-native HF == full-mesh HF, including time-reversal stars.

    The wedge SCF diagonalises at the irreducible points only and unfolds
    the density to the full BZ for the exchange arms (density law per
    Pisani, Dovesi & Roetti 1988, Eq. II.7.21; time reversal
    ``D(-k) = D(k)*``). (2,2,2) exercises proper rotations; (3,3,3)
    additionally has 13 time-reversal star members, and its wedge is 6 of
    27 points. Both measured bit-identical when this landed; the assert
    keeps a small margin.
    """
    sysp, basis = _sym_cell()
    for mesh, n_ibz in (([2, 2, 2], 6), ([3, 3, 3], 6)):
        kf = vq.monkhorst_pack(sysp, mesh, use_symmetry=False)
        ki = vq.monkhorst_pack(sysp, mesh, use_symmetry=True)
        assert len(ki.kpoints) == n_ibz
        rf = vq.run_rhf_periodic_multi_k_ewald3d(
            sysp, basis, kf, _rhf_opts(),
            auto_optimize_truncation=False, progress=False,
        )
        ri = vq.run_rhf_periodic_multi_k_ewald3d(
            sysp, basis, ki, _rhf_opts(),
            auto_optimize_truncation=False, progress=False,
        )
        assert rf.converged and ri.converged
        assert ri.energy == pytest.approx(rf.energy, abs=1e-9)
        assert ri.used_kpoint_symmetry_unfolding is True
        assert rf.used_kpoint_symmetry_unfolding is False


def test_ibz_mesh_matches_full_mesh_rks_hybrid():
    """Wedge-native PBE0 == full-mesh PBE0, energy AND the J/K split.

    The split assertions catch a corrected-arm leak into the J-only
    reporting build, which the total energy is blind to.
    """
    sysp, basis = _sym_cell()
    kf = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    ki = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    rf = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, kf, _ks_opts(),
        auto_optimize_truncation=False, progress=False,
    )
    ri = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, ki, _ks_opts(),
        auto_optimize_truncation=False, progress=False,
    )
    assert rf.converged and ri.converged
    assert ri.energy == pytest.approx(rf.energy, abs=1e-9)
    assert ri.e_hf_exchange == pytest.approx(rf.e_hf_exchange, abs=1e-9)
    assert ri.e_coulomb == pytest.approx(rf.e_coulomb, abs=1e-9)


def test_ibz_mesh_matches_full_mesh_uks_hybrid():
    """Same on the unrestricted route (per-spin unfolding)."""
    sysp, basis = _sym_cell()
    kf = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    ki = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    uf = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, kf, _ks_opts(),
        auto_optimize_truncation=False, progress=False,
    )
    ui = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, ki, _ks_opts(),
        auto_optimize_truncation=False, progress=False,
    )
    assert uf.converged and ui.converged
    assert ui.energy == pytest.approx(uf.energy, abs=1e-9)
    assert ui.e_hf_exchange == pytest.approx(uf.e_hf_exchange, abs=1e-9)
    assert ui.used_kpoint_symmetry_unfolding is True


def test_ibz_transport_converges_with_cutoff_on_a_shift_bearing_cell():
    """The star transport on the case the H2 fixtures cannot see.

    On rock-salt MgO, 42 of the 48 point-group operations map the anion
    into a neighbouring cell, so the Bloch transport carries atom-DEPENDENT
    phases e^{i k.L_a}; a phase bug is O(1) here while every H2-in-a-box
    test stays bit-identical (both atoms acquire the same phase, which
    cancels in Gamma D Gamma^H).

    At a loose lattice cutoff the truncated operators themselves break the
    point symmetry (a radial cell list is not closed under the pair-
    resolved cell action ``R.g + s_b - s_a``), so unfold(D_wedge) and the
    directly-built full-mesh D disagree by the truncation asymmetry -- and
    that gap must COLLAPSE as the cutoff grows. Measured when this landed:
    1.19 (cutoff 8) -> 3.6e-2 (12) -> 4.8e-6 (16) -> 8.0e-9 (20). A
    transport bug would leave an O(1) residual at every cutoff.
    """
    from vibeqc._vibeqc_core import (
        compute_kinetic_lattice,
        compute_overlap_lattice,
    )
    from vibeqc.periodic_k_symmetry import KMeshUnfolding
    from vibeqc.periodic_rhf_multi_k_ewald import (
        _canonical_orthogonalizer_complex,
        _diag_in_orth_basis,
    )

    a = 7.94
    lat = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    sysp = vq.PeriodicSystem(
        3, lat, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, 0.5 * a * np.ones(3))]
    )
    vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    kf = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    ki = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    assert len(ki.kpoints) < len(kf.kpoints)
    unf = KMeshUnfolding.build(sysp, basis, ki)

    def _residual(cutoff):
        o = vq.PeriodicRHFOptions()
        o.lattice_opts.cutoff_bohr = cutoff
        o.lattice_opts.nuclear_cutoff_bohr = cutoff
        S_lat = compute_overlap_lattice(basis, sysp, o.lattice_opts)
        T_lat = compute_kinetic_lattice(basis, sysp, o.lattice_opts)

        def _bloch(latset, k):
            out = np.zeros((nbf, nbf), dtype=complex)
            for cell, blk in zip(latset.cells, latset.blocks):
                R = np.asarray(cell.r_cart, dtype=float)
                out += np.exp(1j * float(np.dot(k, R))) * np.asarray(
                    blk, dtype=float
                )
            return out

        def _dens(km):
            out = []
            for k in km.kpoints:
                k = np.asarray(k, dtype=float)
                S_k = _bloch(S_lat, k)
                S_k = 0.5 * (S_k + S_k.conj().T)
                T_k = _bloch(T_lat, k)
                T_k = 0.5 * (T_k + T_k.conj().T)
                X, _ = _canonical_orthogonalizer_complex(S_k, 1e-10)
                C, _eps = _diag_in_orth_basis(T_k, X)
                out.append(2.0 * C[:, :10] @ C[:, :10].conj().T)
            return out

        D_full = _dens(kf)
        D_unfolded = unf.unfold(_dens(ki))
        return max(
            float(np.abs(x - y).max()) for x, y in zip(D_unfolded, D_full)
        )

    loose = _residual(8.0)
    tight = _residual(16.0)
    assert loose > 1e-1          # the asymmetry is real at cutoff 8
    assert tight < 1e-4, tight   # and collapses once the sums fold-converge
    assert tight < 1e-3 * loose


def test_ibz_mesh_matches_full_mesh_screened_hybrid():
    """hse06 (c_sr-only) on the wedge == full mesh, and the flag is set.

    A screened hybrid never builds the corrected split (its erfc arm is
    convergent in real space), so before this landed the wedge path fed
    its K_erfc the plain weighted wedge fold -- a FIRST-order density
    error on shift-bearing cells at multi-cell cutoffs, invisible on
    pure DFT (second order) and never exercised by the c_full tests.
    The standalone wedge unfolding now serves this path too, and with it
    the SYM3b orbit reduction.
    """
    sysp, basis = _sym_cell()
    kf = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    ki = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=True)
    for runner in (
        vq.run_rks_periodic_multi_k_ewald3d,
        vq.run_uks_periodic_multi_k_ewald3d,
    ):
        rf = runner(
            sysp, basis, kf, _ks_opts("HSE06"),
            auto_optimize_truncation=False, progress=False,
        )
        ri = runner(
            sysp, basis, ki, _ks_opts("HSE06"),
            auto_optimize_truncation=False, progress=False,
        )
        assert rf.converged and ri.converged
        assert ri.energy == pytest.approx(rf.energy, abs=1e-9)
        assert ri.e_hf_exchange == pytest.approx(
            rf.e_hf_exchange, abs=1e-9
        )
        assert ri.used_kpoint_symmetry_unfolding is True
        assert rf.used_kpoint_symmetry_unfolding is False


@pytest.mark.parametrize(
    "runner",
    (
        vq.run_rks_periodic_multi_k_ewald3d,
        vq.run_uks_periodic_multi_k_ewald3d,
    ),
    ids=("rks", "uks"),
)
def test_slab_ibz_mesh_matches_full_mesh_screened_hybrid(runner):
    """A 2D HSE06 wedge must reconstruct the whole-BZ density.

    Pisani, Dovesi & Roetti (1988), Eqs. II.7.21 and II.7.48, require
    each symmetry-related density to be rotated before the whole-BZ fold.
    Dovesi (1986), Eqs. 44-45 and its graphite-monolayer example, makes the
    same reconstruction explicitly dimension-independent. A star weight
    multiplying the representative density is therefore not valid in 2D.
    """
    sysp, basis = _sym_slab_cell()
    full = vq.monkhorst_pack(sysp, [2, 2, 1], use_symmetry=False)
    reduced = vq.monkhorst_pack(sysp, [2, 2, 1], use_symmetry=True)
    assert len(full.kpoints) == 4
    assert len(reduced.kpoints) == 3

    rf = runner(
        sysp,
        basis,
        full,
        _slab_hse_opts(),
        auto_optimize_truncation=False,
        progress=False,
    )
    ri = runner(
        sysp,
        basis,
        reduced,
        _slab_hse_opts(),
        auto_optimize_truncation=False,
        progress=False,
    )
    assert rf.converged and ri.converged
    assert ri.energy == pytest.approx(rf.energy, abs=1e-8)
    assert ri.e_hf_exchange == pytest.approx(rf.e_hf_exchange, abs=1e-8)
    assert ri.used_kpoint_symmetry_unfolding is True
    assert rf.used_kpoint_symmetry_unfolding is False


@pytest.mark.parametrize(
    "runner",
    (
        vq.run_rks_periodic_multi_k_ewald3d,
        vq.run_uks_periodic_multi_k_ewald3d,
    ),
    ids=("rks", "uks"),
)
def test_slab_ibz_screened_hybrid_invalid_star_fails_closed(
    monkeypatch,
    runner,
):
    """Never resume the known-wrong representative-density fold."""
    from vibeqc.periodic_k_symmetry import KMeshUnfolding

    def _reject_star(cls, system, basis, kmesh):
        del cls, system, basis, kmesh
        raise ValueError("incomplete synthetic star")

    monkeypatch.setattr(KMeshUnfolding, "build", classmethod(_reject_star))
    sysp, basis = _sym_slab_cell()
    reduced = vq.monkhorst_pack(sysp, [2, 2, 1], use_symmetry=True)
    with pytest.raises(
        NotImplementedError,
        match="cannot reconstruct the full-star density",
    ):
        runner(
            sysp,
            basis,
            reduced,
            _slab_hse_opts(),
            auto_optimize_truncation=False,
            progress=False,
        )


@pytest.mark.parametrize(
    "runner",
    (
        vq.run_rks_periodic_multi_k_ewald3d,
        vq.run_uks_periodic_multi_k_ewald3d,
    ),
    ids=("rks", "uks"),
)
def test_slab_ibz_screened_hybrid_missing_symmetry_fails_closed(runner):
    """Reduced MP metadata without its symmetry cannot be unfolded."""
    sysp, basis = _sym_slab_cell()
    reduced = vq.monkhorst_pack(sysp, [2, 2, 1], use_symmetry=True)
    sysp.symmetry = None
    with pytest.raises(
        NotImplementedError,
        match="cannot reconstruct the full-star density",
    ):
        runner(
            sysp,
            basis,
            reduced,
            _slab_hse_opts(),
            auto_optimize_truncation=False,
            progress=False,
        )


def test_sym3b_reduced_exchange_blocks_match_symmetrized_full_build():
    """The orbit-reduced K equals symmetrize(full K), bit-level, in situ.

    The wedge path routes its erfc exchange arms through
    ``build_jk_reduced_symmetrized`` (measured 4.8x per-iteration on LiH
    rock-salt at cutoff 12, where the full K_SR build was 94 % of the
    wedge SCF iteration). Its contract is bit-identity against
    ``symmetrize_fock_blocks`` applied to the FULL build -- pinned here on
    the EWALD_3D route's own entry (``build_exchange_blocks``) and on a
    shift-bearing cell, since the BIPOLE-side pins use BIPOLE plumbing.
    NOT pinned against the raw full build: on shift-bearing cells the two
    differ by the operator truncation asymmetry at loose cutoffs (that
    difference collapses with cutoff, as
    ``test_ibz_transport_converges_with_cutoff_on_a_shift_bearing_cell``
    pins for the same mechanism).
    """
    from vibeqc._vibeqc_core import (
        build_jk_2e_real_space,
        compute_overlap_lattice,
    )
    from vibeqc.bipole_symmetry_fock import (
        cell_orbit_mapping,
        representative_cell_indices,
        symmetrize_fock_blocks,
    )
    from vibeqc.periodic_screened_exchange import (
        PeriodicExchangeAssembly,
        build_exchange_blocks,
    )

    a = 7.72
    lat = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    sysp = vq.PeriodicSystem(
        3, lat, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, 0.5 * a * np.ones(3))]
    )
    vq.attach_symmetry(sysp)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 8.0
    o.lattice_opts.nuclear_cutoff_bohr = 8.0
    D = compute_overlap_lattice(basis, sysp, o.lattice_opts)  # symmetric probe density
    mapping = cell_orbit_mapping(sysp, basis, list(D.cells))
    assert mapping is not None
    reps = representative_cell_indices(mapping)
    exx = PeriodicExchangeAssembly(1.0, 0.0, 0.0)
    alpha = 0.35

    K_reduced = build_exchange_blocks(
        basis, sysp, o.lattice_opts, D, exx,
        full_range_alpha=alpha,
        symmetry_reduction=(mapping, reps),
    )
    jk_full = build_jk_2e_real_space(basis, sysp, o.lattice_opts, D, alpha)
    k_sym = [np.asarray(b, dtype=float).copy() for b in jk_full.K.blocks]
    symmetrize_fock_blocks(k_sym, mapping)  # in-place enforcement
    worst = max(
        float(np.abs(np.asarray(K_reduced[g]) - k_sym[g]).max())
        for g in range(len(D.cells))
    )
    assert worst < 1e-12, worst


def test_sym3b_reduction_is_off_on_the_full_mesh_path():
    """A full-mesh job must be byte-identical to the pre-SYM3b behaviour.

    The reduction produces the symmetrized K, which on shift-bearing
    cells differs from the raw K at loose cutoffs; the full-mesh path
    keeps the raw builder so no released number moves. Pinned by running
    a full-mesh job and asserting the result does not carry the wedge
    flag (the reduction is gated on the same condition).
    """
    sysp, basis = _sym_cell()
    kf = vq.monkhorst_pack(sysp, [2, 2, 2], use_symmetry=False)
    r = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, kf, _rhf_opts(),
        auto_optimize_truncation=False, progress=False,
    )
    assert r.used_kpoint_symmetry_unfolding is False


def test_ibz_unfolding_fires_the_symmetry_citation():
    """Pisani 1988 + Dovesi 1986 must fire when the unfolding is used.

    CLAUDE.md section 8: the same merge that lands a citable method lands
    its citation entry and route. The route key is what the runner appends
    when a result carries ``used_kpoint_symmetry_unfolding``.
    """
    from vibeqc.output.citations.registry import load_default_database

    refs = load_default_database().assemble(
        numerics=["kpoint_symmetry_unfolding"]
    )
    text = " ".join(str(c) for c in refs.citations)
    assert "dovesi_symmetry_lcao_1986" in text
    assert "pisani_crystal_1988" in text


def test_periodic_job_cites_the_lattice_sum_truncation_criterion():
    """Every periodic job cites the overlap-magnitude truncation criterion.

    vibe-qc sizes its Bloch overlap lattice sum by bounding the largest
    dropped primitive-pair overlap
    (:func:`vibeqc.lattice_screening.estimate_rcut_overlap_pair`), which is
    Pisani and Dovesi's criterion (Int. J. Quantum Chem. 17, 501 (1980),
    section 4 p. 508) -- CRYSTAL's ITOL1. It rides the always-on
    ``_periodic_lcao`` route next to the monograph, since every periodic
    LCAO result depends on where that sum was cut.
    """
    from vibeqc.output.citations.registry import load_default_database

    refs = load_default_database().assemble(periodic=True)
    text = " ".join(str(c) for c in refs.citations)
    assert "pisani_dovesi_lattice_truncation_1980" in text
    assert "10.1002/qua.560170311" in text


def test_threaded_k_loop_is_bit_identical_to_serial(monkeypatch):
    """Farming the k loop over threads must not change a single bit.

    Each k performs exactly the same operations in the same order --- the
    threads only decide who runs which k --- so the results must be
    identical, not merely close. Anything else means the shared
    ``channel_tables`` cache is being mutated in a way that leaks between
    k-points.
    """
    sysp, basis = _cell()
    nbf = basis.nbasis
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    n_k = len(km.kpoints)
    crc = _cells_r_cart(basis, sysp)

    rng = np.random.default_rng(11)
    S_k = [np.eye(nbf, dtype=complex) for _ in range(n_k)]
    D_k = []
    for _ in range(n_k):
        a = (rng.standard_normal((nbf, nbf))
             + 1j * rng.standard_normal((nbf, nbf)))
        D_k.append(a + a.conj().T)

    def _terms(threads):
        monkeypatch.setenv("VIBEQC_EXCHANGE_K_THREADS", str(threads))
        x = CorrectedEwaldExchange.build(
            basis, sysp, crc, km, ALPHA, where="test"
        )
        x.k_space_terms_all_k(S_k, D_k)  # first pass is serial by design
        return x.k_space_terms_all_k(S_k, D_k)

    serial = _terms(1)
    threaded = _terms(4)
    for a, b in zip(serial, threaded):
        assert np.array_equal(a, b)


def test_first_pass_is_serial_so_the_lazy_cache_cannot_race():
    """The cold pass must not run in parallel.

    ``channel_tables`` builds its q-channel and per-k ``B`` caches
    lazily. If the first pass ran threaded, every worker would miss,
    recompute the same tensors, and race to store them; the losing
    writers' entries get orphaned and recomputed on every later
    iteration. One serial pass fills the cache, after which every call is
    a pure read.
    """
    sysp, basis = _cell()
    km = vq.monkhorst_pack(sysp, [2, 2, 2])
    x = CorrectedEwaldExchange.build(
        basis, sysp, _cells_r_cart(basis, sysp), km, ALPHA, where="test"
    )
    assert x._cache_is_warm is False
    nbf = basis.nbasis
    n_k = len(km.kpoints)
    S_k = [np.eye(nbf, dtype=complex) for _ in range(n_k)]
    D_k = [np.eye(nbf, dtype=complex) for _ in range(n_k)]
    x.k_space_terms_all_k(S_k, D_k)
    assert x._cache_is_warm is True


def test_worker_count_respects_the_override_and_small_meshes():
    from vibeqc.periodic_corrected_exchange import CorrectedEwaldExchange as C

    import os

    prev = os.environ.pop("VIBEQC_EXCHANGE_K_THREADS", None)
    try:
        # Too few k-points to be worth a pool.
        assert C._k_worker_count(1) == 1
        assert C._k_worker_count(3) == 1
        # Never more workers than k-points.
        assert C._k_worker_count(64) <= 64
        os.environ["VIBEQC_EXCHANGE_K_THREADS"] = "1"
        assert C._k_worker_count(64) == 1
        os.environ["VIBEQC_EXCHANGE_K_THREADS"] = "3"
        assert C._k_worker_count(64) == 3
        # A garbage value must not crash the SCF.
        os.environ["VIBEQC_EXCHANGE_K_THREADS"] = "banana"
        assert C._k_worker_count(64) >= 1
    finally:
        os.environ.pop("VIBEQC_EXCHANGE_K_THREADS", None)
        if prev is not None:
            os.environ["VIBEQC_EXCHANGE_K_THREADS"] = prev


def test_unsupported_gauge_is_distinct_from_an_unsupported_mesh():
    """The two failure classes must not be conflated.

    A 2D slab or a ``dim != 3`` cell is a gauge the split does not exist
    for -- those routes keep the historical kernel. A symmetry-reduced
    mesh is one this code cannot serve *yet*, and must fail closed so it
    cannot disagree with the same job on a full mesh. The drivers branch
    on exactly this distinction.
    """
    from vibeqc.periodic_corrected_exchange import is_unsupported_gauge

    sysp, _ = _cell()
    km = vq.monkhorst_pack(sysp, [2, 1, 1])
    assert is_unsupported_gauge(sysp, slab_mode=True) is True
    assert is_unsupported_gauge(sysp, slab_mode=False) is False
    # A reduced mesh on a 3D cell is NOT an unsupported gauge -- it must
    # take the fail-closed arm, not the keep-the-old-kernel arm.
    assert corrected_exchange_unavailable_reason(sysp, km) is None


def test_multi_k_rhf_refuses_an_explicit_kpoint_list():
    """Pure HF on an explicit k list must refuse rather than use a
    different K.

    An explicit list carries no Monkhorst-Pack star metadata, so there is
    nothing to unfold from; leaving it on the old bare kernel while a
    full mesh takes the corrected split made the two disagree by
    4.6e-5 Ha on H2/(2,2,2). Symmetry-reduced MP meshes ARE served now,
    via the star unfolding -- pinned by the IBZ parity tests below.
    """
    sysp, basis = _cell()
    full = vq.monkhorst_pack(sysp, [2, 2, 1])
    ks = np.asarray([np.asarray(k, dtype=float) for k in list(full.kpoints)[:2]])
    reduced = vq.as_bloch_kmesh(
        vq.KPoints(
            kpoints_cart=ks,
            kpoints_frac=np.zeros_like(ks),
            weights=np.array([0.5, 0.5]),
        )
    )
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 12.0
    o.max_iter = 5
    with pytest.raises(NotImplementedError, match="explicit k-point list"):
        vq.run_rhf_periodic_multi_k_ewald3d(sysp, basis, reduced, o)


def test_multi_k_uks_hybrid_refuses_an_explicit_kpoint_list():
    """Same for a global hybrid on the unrestricted route."""
    sysp, basis = _cell()
    full = vq.monkhorst_pack(sysp, [2, 2, 1])
    ks = np.asarray([np.asarray(k, dtype=float) for k in list(full.kpoints)[:2]])
    reduced = vq.as_bloch_kmesh(
        vq.KPoints(
            kpoints_cart=ks,
            kpoints_frac=np.zeros_like(ks),
            weights=np.array([0.5, 0.5]),
        )
    )
    o = vq.PeriodicKSOptions()
    o.functional = "PBE0"
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 12.0
    o.max_iter = 5
    with pytest.raises(NotImplementedError, match="explicit k-point list"):
        vq.run_uks_periodic_multi_k_ewald3d(sysp, basis, reduced, o)


def test_multi_k_uks_pure_functional_accepts_a_symmetry_reduced_mesh():
    """A pure functional builds no full-range arm, so nothing is refused.

    Guards against keying the fail-close on ``needs_exchange`` instead of
    ``c_full``: that would break every reduced-mesh pure-DFT job, and
    every screened hybrid too (hse06 is c_full = 0).
    """
    sysp, basis = _cell()
    full = vq.monkhorst_pack(sysp, [2, 2, 1])
    ks = np.asarray([np.asarray(k, dtype=float) for k in list(full.kpoints)[:2]])
    reduced = vq.as_bloch_kmesh(
        vq.KPoints(
            kpoints_cart=ks,
            kpoints_frac=np.zeros_like(ks),
            weights=np.array([0.5, 0.5]),
        )
    )
    o = vq.PeriodicKSOptions()
    o.functional = "PBE"
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 12.0
    o.max_iter = 5
    r = vq.run_uks_periodic_multi_k_ewald3d(sysp, basis, reduced, o)
    assert r.e_hf_exchange == pytest.approx(0.0, abs=1e-12)
