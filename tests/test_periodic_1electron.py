"""Phase 12a: periodic one-electron infrastructure.

Covers:
  - PeriodicSystem constructor + reciprocal lattice + electron count.
  - direct_lattice_cells cutoff behavior across dim ∈ {1, 2, 3}.
  - One-electron lattice sums S(g), T(g), V(g).
  - Bloch-sum Hermiticity, time-reversal, isolated-molecule-limit recovery.
  - **Bloch ↔ supercell folding equivalence**: the union of band energies
    over a K-point MP mesh on one unit cell equals the Γ-point band
    energies of a K-fold supercell. Tested for the exponentially-
    convergent T operator (S + V involve 1/r tails and converge only
    polynomially, so lattice-sum truncation produces the expected ~1e-2
    Ha boundary mismatch between the two partitionings — that is physics,
    not a bug).
  - Monkhorst-Pack mesh construction, auto-clamping of non-periodic axes,
    IBZ reduction via spglib.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_unit_cell():
    return [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])]


def _tight_opts(cutoff: float = 80.0):
    o = vq.LatticeSumOptions()
    o.cutoff_bohr = cutoff
    o.nuclear_cutoff_bohr = 200.0
    return o


# ---------------------------------------------------------------------------
# PeriodicSystem + direct-lattice iteration
# ---------------------------------------------------------------------------

def test_periodic_system_reciprocal_lattice():
    lat = np.diag([4.0, 5.0, 6.0])
    sysp = vq.PeriodicSystem(3, lat, _h2_unit_cell())
    B = sysp.reciprocal_lattice()
    # a_i · b_j = 2π δ_ij
    for i in range(3):
        for j in range(3):
            expected = 2 * np.pi if i == j else 0.0
            assert abs(lat[:, i] @ B[:, j] - expected) < 1e-12


def test_periodic_system_n_electrons():
    uc = [vq.Atom(8, [0, 0, 0]), vq.Atom(1, [0, 0, 1.8]), vq.Atom(1, [0, 0, -1.8])]
    s = vq.PeriodicSystem(3, np.diag([10.0, 10.0, 10.0]), uc)
    assert s.n_electrons() == 10           # H2O neutral
    s.charge = +1
    assert s.n_electrons() == 9


def test_unit_cell_molecule_uses_valid_minimal_multiplicity_for_odd_electron_cell():
    a = 4.0853 / 0.529177210903
    lat = (a / 2.0) * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    sysp = vq.PeriodicSystem(3, lat, [vq.Atom(47, [0, 0, 0])])

    mol = sysp.unit_cell_molecule()

    assert mol.n_electrons() == 47
    assert mol.multiplicity == 2


def test_periodic_system_rejects_invalid_dim():
    with pytest.raises(RuntimeError, match="dim"):
        vq.PeriodicSystem(0, np.eye(3) * 5, _h2_unit_cell())
    with pytest.raises(RuntimeError, match="dim"):
        vq.PeriodicSystem(4, np.eye(3) * 5, _h2_unit_cell())


@pytest.mark.parametrize("cutoff", [-1.0, np.nan, np.inf])
def test_direct_lattice_cells_rejects_invalid_cutoff(cutoff):
    system = vq.PeriodicSystem(3, np.eye(3) * 5.0, _h2_unit_cell())
    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        vq.direct_lattice_cells(system, cutoff)


def test_direct_lattice_cells_rejects_nonfinite_lattice():
    lattice = np.eye(3) * 5.0
    lattice[0, 0] = np.nan
    system = vq.PeriodicSystem(3, lattice, _h2_unit_cell())
    with pytest.raises(RuntimeError, match="finite values"):
        vq.direct_lattice_cells(system, 10.0)


def test_direct_lattice_cells_rejects_unrepresentable_extent():
    system = vq.PeriodicSystem(
        3, np.diag([1.0, 1.0, 1.0e-9]), _h2_unit_cell()
    )
    with pytest.raises(RuntimeError, match="too large"):
        vq.direct_lattice_cells(system, 10.0)


@pytest.mark.parametrize("dim,mesh_shape", [
    (1, (3, 1, 1)),
    (2, (3, 3, 1)),
    (3, (3, 3, 3)),
])
def test_direct_lattice_cells_matches_dim(dim, mesh_shape):
    """The number of cells within cutoff must scale with dim."""
    a = 4.0
    lat = np.diag([a, a, a]) if dim == 3 else np.diag([a, a, 30.0 if dim == 1 else a])
    if dim == 1:
        lat = np.diag([a, 30.0, 30.0])
    if dim == 2:
        lat = np.diag([a, a, 30.0])
    sysp = vq.PeriodicSystem(dim, lat, _h2_unit_cell())
    cutoff = a * 1.5   # include only n in {-1, 0, 1} along periodic axes
    cells = vq.direct_lattice_cells(sysp, cutoff)
    # Expected: all (n1, n2, n3) with |n_i| <= 1 on periodic axes and 0
    # on vacuum axes, filtered to |r| <= cutoff.
    expected = 0
    for n1 in range(-1, 2):
        for n2 in range(-1, 2):
            for n3 in range(-1, 2):
                if dim < 3 and n3 != 0: continue
                if dim < 2 and n2 != 0: continue
                r = n1 * lat[:, 0] + n2 * lat[:, 1] + n3 * lat[:, 2]
                if np.linalg.norm(r) <= cutoff: expected += 1
    assert len(cells) == expected
    # First cell is always the origin, sorted by |r|.
    assert tuple(cells[0].index.tolist()) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Lattice sums + Bloch invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_isolated_molecule_limit(dim):
    """With a lattice so large that only g=0 is in the cutoff, the Bloch-
    summed Hcore at any k must reproduce the molecular Hcore exactly."""
    a = 50.0
    if dim == 1: lat = np.diag([a, 30.0, 30.0])
    elif dim == 2: lat = np.diag([a, a, 30.0])
    else: lat = np.diag([a, a, a])

    sysp = vq.PeriodicSystem(dim, lat, _h2_unit_cell())
    mol = sysp.unit_cell_molecule()
    basis = vq.BasisSet(mol, "sto-3g")

    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 10.0   # only g=0 nuclei
    S = vq.compute_overlap_lattice(basis, sysp, opts)
    T = vq.compute_kinetic_lattice(basis, sysp, opts)
    V = vq.compute_nuclear_lattice(basis, sysp, opts)
    assert len(S) == 1

    # Molecular reference
    Sm = vq.compute_overlap(basis)
    Tm = vq.compute_kinetic(basis)
    Vm = vq.compute_nuclear(basis, mol)
    import scipy.linalg
    ref = scipy.linalg.eigh(Tm + Vm, Sm, eigvals_only=True)

    B = sysp.reciprocal_lattice()
    for frac in (0.0, 0.3, 0.5):
        k_frac = np.zeros(3)
        k_frac[:dim] = frac if dim == 1 else (
            frac if True else 0.0  # same frac per axis for 2D/3D
        )
        if dim >= 2: k_frac[1] = frac * 0.7
        if dim >= 3: k_frac[2] = frac * 0.5
        k_cart = B @ k_frac
        Sk = vq.bloch_sum(S, k_cart)
        Hk = vq.bloch_sum(T, k_cart) + vq.bloch_sum(V, k_cart)
        bd = vq.diagonalize_bloch(Hk, Sk)
        assert np.max(np.abs(bd.energies - ref)) < 1e-12


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_bloch_matrices_are_hermitian(dim):
    a = 4.0
    if dim == 1: lat = np.diag([a, 30.0, 30.0])
    elif dim == 2: lat = np.diag([a, a, 30.0])
    else: lat = np.diag([a, a, a])
    sysp = vq.PeriodicSystem(dim, lat, _h2_unit_cell())
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    S = vq.compute_overlap_lattice(basis, sysp, _tight_opts())
    T = vq.compute_kinetic_lattice(basis, sysp, _tight_opts())

    B = sysp.reciprocal_lattice()
    # A handful of non-Γ k-points (fractional coordinates)
    ks = [np.array([0.1, 0.2, 0.3])[:dim].tolist() + [0.0]*(3-dim),
          [0.5, 0.0, 0.0][:3],
          [-0.25, 0.25, 0.0][:3]]
    for kf in ks:
        k_cart = B @ np.array(kf)
        Sk = vq.bloch_sum(S, k_cart)
        Tk = vq.bloch_sum(T, k_cart)
        assert np.max(np.abs(Sk - Sk.conj().T)) < 1e-12
        assert np.max(np.abs(Tk - Tk.conj().T)) < 1e-12


def test_time_reversal_symmetry():
    """Eigenvalues at +k and -k must coincide for a system with time-
    reversal symmetry (all one-electron operators here)."""
    a = 4.0
    sysp = vq.PeriodicSystem(1, np.diag([a, 30.0, 30.0]), _h2_unit_cell())
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    S = vq.compute_overlap_lattice(basis, sysp, _tight_opts())
    T = vq.compute_kinetic_lattice(basis, sysp, _tight_opts())
    V = vq.compute_nuclear_lattice(basis, sysp, _tight_opts())
    B = sysp.reciprocal_lattice()
    for frac in (0.1, 0.2, 0.4):
        for sign in (+1, -1):
            k_cart = B @ np.array([sign * frac, 0, 0])
            bd = vq.diagonalize_bloch(
                vq.bloch_sum(T, k_cart) + vq.bloch_sum(V, k_cart),
                vq.bloch_sum(S, k_cart),
            )
            if sign == +1:
                ref = bd.energies.copy()
            else:
                assert np.max(np.abs(bd.energies - ref)) < 1e-12


def test_bloch_supercell_folding_kinetic():
    """Union of K MP-mesh eigenvalues on the 1-cell system equals the
    Γ-point eigenvalues of the K-cell supercell. Tested on T (exponentially
    convergent lattice sum → machine-precision agreement)."""
    a, K = 4.0, 3
    uc = _h2_unit_cell()
    sysp = vq.PeriodicSystem(1, np.diag([a, 30.0, 30.0]), uc)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _tight_opts()

    S_g = vq.compute_overlap_lattice(basis, sysp, opts)
    T_g = vq.compute_kinetic_lattice(basis, sysp, opts)

    B = sysp.reciprocal_lattice()
    eig_p = []
    for m in range(K):
        k_cart = B @ np.array([m/K, 0, 0])
        bd = vq.diagonalize_bloch(vq.bloch_sum(T_g, k_cart),
                                  vq.bloch_sum(S_g, k_cart))
        eig_p.extend(bd.energies.tolist())
    eig_p = np.sort(np.array(eig_p))

    supercell = []
    for i in range(K):
        for at in uc:
            supercell.append(vq.Atom(at.Z, [at.xyz[0]+i*a, at.xyz[1], at.xyz[2]]))
    sysp_s = vq.PeriodicSystem(1, np.diag([K*a, 30.0, 30.0]), supercell)
    basis_s = vq.BasisSet(sysp_s.unit_cell_molecule(), "sto-3g")
    S_gs = vq.compute_overlap_lattice(basis_s, sysp_s, opts)
    T_gs = vq.compute_kinetic_lattice(basis_s, sysp_s, opts)
    bds = vq.diagonalize_bloch(
        vq.bloch_sum(T_gs, np.zeros(3)),
        vq.bloch_sum(S_gs, np.zeros(3)),
    )
    eig_s = np.sort(bds.energies)
    assert np.max(np.abs(eig_p - eig_s)) < 1e-12


# ---------------------------------------------------------------------------
# k-mesh + IBZ
# ---------------------------------------------------------------------------

def test_monkhorst_pack_full_mesh_weights_sum_to_one():
    sysp = vq.PeriodicSystem(3, np.eye(3)*5, _h2_unit_cell())
    km = vq.monkhorst_pack(sysp, [4, 4, 4])
    assert len(km) == 64
    assert abs(sum(km.weights) - 1.0) < 1e-12


def test_monkhorst_pack_clamps_vacuum_axes():
    """Non-periodic axes must collapse to a single k=0 point regardless
    of the requested mesh size."""
    sysp = vq.PeriodicSystem(1, np.diag([4.0, 30.0, 30.0]), _h2_unit_cell())
    km = vq.monkhorst_pack(sysp, [8, 5, 5])
    assert len(km) == 8
    for k in km.kpoints:
        assert abs(k[1]) < 1e-14 and abs(k[2]) < 1e-14

    shifted = vq.monkhorst_pack(sysp, [8, 1, 1], [0, 1, 1])
    assert shifted.is_shift == [0, 0, 0]
    for k in shifted.kpoints:
        assert abs(k[1]) < 1e-14 and abs(k[2]) < 1e-14


def test_monkhorst_pack_ibz_reduction():
    """Cubic H2 → IBZ reduction of a 4×4×4 mesh must have fewer points
    than the full 64-point mesh, with weights still summing to 1."""
    a = 5.0
    sysp = vq.PeriodicSystem(3, np.eye(3)*a, _h2_unit_cell())
    vq.attach_symmetry(sysp)
    km_full = vq.monkhorst_pack(sysp, [4, 4, 4])
    km_ir   = vq.monkhorst_pack(sysp, [4, 4, 4], use_symmetry=True)
    assert len(km_ir) < len(km_full)
    assert abs(sum(km_ir.weights) - 1.0) < 1e-12
    # Every full-mesh point maps to a representative
    assert len(km_ir.ir_mapping) == 64
    assert max(km_ir.ir_mapping) == len(km_ir) - 1


def test_ibz_requires_symmetry_attached():
    sysp = vq.PeriodicSystem(3, np.eye(3)*5, _h2_unit_cell())
    with pytest.raises(RuntimeError, match="attach_symmetry"):
        vq.monkhorst_pack(sysp, [4, 4, 4], use_symmetry=True)


# ---------------------------------------------------------------------------
# Diagonaliser guards
# ---------------------------------------------------------------------------

def test_diagonalize_bloch_rejects_singular_overlap():
    """A "basis" with two identical functions has rank-deficient S."""
    import numpy as np
    S = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
    F = np.eye(2, dtype=complex)
    with pytest.raises(RuntimeError, match="near-singular"):
        vq.diagonalize_bloch(F, S)


# ---------------------------------------------------------------------------
# Image enumeration: the |g| bound is not translation invariant
# ---------------------------------------------------------------------------
#
# A periodic two-centre lattice sum runs over translations g of the ket
# centre, and whether a term matters is set by the PHYSICAL separation
# |O_mu - O_nu - g|, so the translations that contribute form a ball centred
# on the intra-cell offset, not on the origin. Sharma & Beylkin,
# J. Chem. Theory Comput. (2021), doi:10.1021/acs.jctc.0c01195, make this
# explicit: their Eq. 17 writes <a|K|b>^per = sum_P <a|K|b_P>, and the
# lattice-dependent factor of Eq. 18 depends on P only through
# T(P) = rho * ||A - B - P||^2. ||A - B - P||, not ||P||.
#
# vibe-qc's one-electron builders bound |g| instead, so the matrix depends on
# WHICH equally valid description of the same crystal you hand them. These
# tests pin that defect as a strict xfail with its measured size, so that the
# fix -- which has to move a whole family of kernels and their consumers
# together, see cpp/include/vibeqc/lattice_pair_cells.hpp and
# handovers/HANDOVER_OPEN_BUGS_V015.md -- flips them to XPASS rather than
# landing silently.

_BOHR = 0.529177210903


def _rocksalt(z_a, z_b, a_ang, shift_frac=(0, 0, 0)):
    """FCC binary with the second atom at (1/2, 1/2, 1/2) + shift_frac."""
    a = a_ang / _BOHR
    h = 0.5 * a
    lat = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    frac = np.array([0.5, 0.5, 0.5]) + np.asarray(shift_frac, dtype=float)
    return vq.PeriodicSystem(
        3, lat, [vq.Atom(z_a, [0, 0, 0]), vq.Atom(z_b, list(lat.T @ frac))]
    )


def _gamma_sum(system, basis_name, builder, cutoff=15.0):
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = cutoff          # the shipped default
    opts.nuclear_cutoff_bohr = 25.0    # the shipped default
    return np.asarray(builder(basis, system, opts).blocks).sum(axis=0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "one-electron lattice sums bound |g| rather than the physical pair "
        "separation, so the Gamma matrices are not translation invariant; "
        "see handovers/HANDOVER_OPEN_BUGS_V015.md. Since #429 stage 2 the "
        "pair-complete enumeration exists behind "
        "LatticeSumOptions.pair_complete_1e (default off, see the passing "
        "sibling below); this strict xfail pins that the default has not "
        "flipped without its consumer families being re-measured"
    ),
)
@pytest.mark.parametrize(
    "z_a,z_b,a_ang,basis_name",
    [
        (3, 1, 4.084, "def2-svp"),   # LiH rocksalt: drift 7.2e-01 in S
        (12, 8, 4.21, "def2-svp"),   # MgO rocksalt:  drift 8.7e-02 in S
        (12, 8, 4.21, "sto-3g"),
    ],
)
@pytest.mark.parametrize(
    "builder", [vq.compute_overlap_lattice, vq.compute_kinetic_lattice]
)
def test_one_electron_lattice_sums_are_translation_invariant(
    z_a, z_b, a_ang, basis_name, builder
):
    """The same crystal, described two ways, must give the same matrix.

    Shifting the second atom by a lattice vector leaves the crystal
    untouched; only the Cartesian coordinates of one representative move,
    and both descriptions here keep it inside the cell. An overlap matrix
    element is O(1), so the measured drifts are wrong answers rather than
    tolerance questions. They grow with the intra-cell offset and with
    basis diffuseness, and decay to round-off only once the cutoff exceeds
    range + offset: on MgO/sto-3g the S drift falls 3.7e-01 (cutoff 8) ->
    5.0e-03 (12) -> 2.3e-09 (20) -> 3.3e-16 (26).
    """
    ref = _gamma_sum(_rocksalt(z_a, z_b, a_ang), basis_name, builder)
    scale = float(np.max(np.abs(ref)))
    for shift in ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (-1, -1, 0)):
        moved = _gamma_sum(_rocksalt(z_a, z_b, a_ang, shift), basis_name,
                           builder)
        drift = float(np.max(np.abs(moved - ref)))
        assert drift < 1e-10 * max(scale, 1.0), (
            f"{builder.__name__} moved by {drift:.3e} (matrix scale "
            f"{scale:.3e}) when the atom was re-described {shift} lattice "
            "vectors away -- the crystal is unchanged"
        )


def test_direct_lattice_cells_is_a_prefix_under_cutoff_growth():
    """A cutoff's cell list is an exact prefix of any larger cutoff's.

    ``direct_lattice_cells`` stable-sorts by |r|. A crystal lattice has
    large groups of exactly equal |r|, and an unstable sort could order
    them differently between two cutoffs -- or between standard library
    versions, which would make cell-resolved results platform-dependent.
    Anything that pairs a matrix built at one cutoff with one built at
    another, index for index, depends on this.
    """
    system = _rocksalt(12, 8, 4.21)
    for small, large in ((10.0, 15.0), (15.0, 20.0), (12.0, 25.0)):
        a = vq.direct_lattice_cells(system, small)
        b = vq.direct_lattice_cells(system, large)
        assert len(b) > len(a)
        for i, cell in enumerate(a):
            assert tuple(cell.index) == tuple(b[i].index), (
                f"cutoff {small} cell {i} is {tuple(cell.index)} but "
                f"cutoff {large} has {tuple(b[i].index)} there"
            )


# ---------------------------------------------------------------------------
# GitLab #429 stage 2 (Family 1): the pair-complete one-electron enumeration
# behind LatticeSumOptions.pair_complete_1e.
# ---------------------------------------------------------------------------


def _gamma_sum_pair_complete(system, basis_name, builder, cutoff=15.0):
    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    opts.nuclear_cutoff_bohr = 25.0
    opts.pair_complete_1e = True
    return np.asarray(builder(basis, system, opts).blocks).sum(axis=0)


@pytest.mark.parametrize(
    "z_a,z_b,a_ang,basis_name",
    [
        (3, 1, 4.084, "def2-svp"),   # LiH rocksalt: drift 7.2e-01 in S off
        (12, 8, 4.21, "def2-svp"),   # MgO rocksalt:  drift 8.7e-02 in S off
        (12, 8, 4.21, "sto-3g"),
    ],
)
@pytest.mark.parametrize(
    "builder", [vq.compute_overlap_lattice, vq.compute_kinetic_lattice]
)
def test_pair_complete_one_electron_sums_are_translation_invariant(
    z_a, z_b, a_ang, basis_name, builder
):
    """With the switch on, re-describing the crystal by moving an atom one
    lattice vector leaves the Gamma sum unchanged to round-off: the term set
    {(mu, nu, g) : |O_mu - O_nu - g| <= cutoff} is translation invariant by
    construction (Sharma & Beylkin, JCTC 17, 3916 (2021), Eqs. 17-18)."""
    ref = _gamma_sum_pair_complete(_rocksalt(z_a, z_b, a_ang), basis_name, builder)
    scale = float(np.max(np.abs(ref)))
    for shift in ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (-1, -1, 0)):
        moved = _gamma_sum_pair_complete(
            _rocksalt(z_a, z_b, a_ang, shift), basis_name, builder
        )
        drift = float(np.max(np.abs(moved - ref)))
        assert drift < 1e-10 * max(scale, 1.0), (
            f"{builder.__name__} with pair_complete_1e moved by {drift:.3e} "
            f"(matrix scale {scale:.3e}) under the lattice relabelling {shift}"
        )


def test_pair_complete_1e_is_off_by_default_and_bit_identical():
    """The switch defaults to off, the default reproduces the historical
    sum bit for bit, and on the worst audited row the two enumerations
    differ by the filed drift order, so the switch is doing real work."""
    system = _rocksalt(3, 1, 4.084)
    basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
    opts = vq.LatticeSumOptions()
    assert opts.pair_complete_1e is False
    opts.cutoff_bohr = 15.0
    opts.nuclear_cutoff_bohr = 25.0
    s_default = np.asarray(vq.compute_overlap_lattice(basis, system, opts).blocks)
    opts_off = vq.LatticeSumOptions()
    opts_off.cutoff_bohr = 15.0
    opts_off.nuclear_cutoff_bohr = 25.0
    opts_off.pair_complete_1e = False
    s_off = np.asarray(vq.compute_overlap_lattice(basis, system, opts_off).blocks)
    assert s_default.shape == s_off.shape
    assert np.array_equal(s_default, s_off)
    s_on = _gamma_sum_pair_complete(system, "def2-svp", vq.compute_overlap_lattice)
    assert float(np.max(np.abs(s_on - s_default.sum(axis=0)))) > 1e-3


def test_pair_complete_lattice_cells_extend_the_plain_ball_as_a_prefix():
    """The padded list is a superset whose leading entries are exactly the
    plain |g| ball in the same order, so a consumer that indexes the plain
    list's prefix into the padded one reads the same cells."""
    system = _rocksalt(12, 8, 4.21)
    basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
    plain = vq.direct_lattice_cells(system, 15.0)
    padded = vq.pair_complete_lattice_cells(basis, system, 15.0)
    assert len(padded) > len(plain)
    for a, b in zip(plain, padded):
        assert tuple(int(i) for i in a.index) == tuple(int(i) for i in b.index)
    # The zero cell stays first.
    assert tuple(int(i) for i in padded[0].index) == (0, 0, 0)
    # The builders enumerate that list under the switch.
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 15.0
    opts.pair_complete_1e = True
    s_on = vq.compute_overlap_lattice(basis, system, opts)
    assert len(s_on.cells) == len(padded)
    opts.pair_complete_1e = False
    s_off = vq.compute_overlap_lattice(basis, system, opts)
    assert len(s_off.cells) == len(plain)


def test_pair_complete_overlap_gradient_differentiates_the_pair_complete_sum():
    """Energy/gradient consistency of the switched-on term set: with a fixed
    symmetric W(g) on the pair-complete cell list, the analytic
    -sum_g W(g).dS(g)/dR contraction matches central differences of
    E(R) = sum_g Tr[W(g) S(g)] built with the same switch."""
    from vibeqc._vibeqc_core import make_lattice_matrix_set

    def system_at(dz):
        a = 4.084 * 1.8897261245650618
        lat = (a / 2.0) * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        frac = np.array([0.5, 0.5, 0.5])
        pos = list(lat.T @ frac)
        pos[2] += dz
        return vq.PeriodicSystem(3, lat, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, pos)])

    def opts_on():
        o = vq.LatticeSumOptions()
        o.cutoff_bohr = 8.0
        o.nuclear_cutoff_bohr = 8.0
        o.pair_complete_1e = True
        return o

    sysp = system_at(0.15)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    s_ref = vq.compute_overlap_lattice(basis, sysp, opts_on())
    rng = np.random.default_rng(429)
    nbf = int(basis.nbasis)
    w_blocks = []
    for _ in s_ref.cells:
        w = rng.standard_normal((nbf, nbf))
        w_blocks.append(0.5 * (w + w.T))
    W = make_lattice_matrix_set(nbf, list(s_ref.cells), w_blocks)

    def energy(dz):
        sys_d = system_at(dz)
        basis_d = vq.BasisSet(sys_d.unit_cell_molecule(), "sto-3g")
        s_d = vq.compute_overlap_lattice(basis_d, sys_d, opts_on())
        assert len(s_d.cells) == len(s_ref.cells)
        return float(sum(np.sum(np.asarray(w) * np.asarray(b)) for w, b in zip(w_blocks, s_d.blocks)))

    from vibeqc._vibeqc_core import overlap_lattice_gradient_contribution

    g_an = np.asarray(
        overlap_lattice_gradient_contribution(basis, sysp, W, opts_on())
    )
    h = 1e-4
    g_fd = (energy(0.15 + h) - energy(0.15 - h)) / (2.0 * h)
    # The contraction returns -dE/dR; compare the H atom's z component.
    assert g_an[1, 2] == pytest.approx(-g_fd, abs=1e-6)
    assert abs(g_fd) > 1e-3
