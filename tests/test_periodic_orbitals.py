"""Periodic Bloch-orbital writers + evaluator.

The four required physics checks for Phase V3:

* **Bloch identity** ψ_{n,k}(r + T) ≈ e^{i k·T} ψ_{n,k}(r) for sampled
  (k, T) pairs — the kernel obeys the lattice phase relation regardless
  of MO character.
* **Γ-limit equivalence** at Γ on a long-cell H₂ chain, the Bloch sum
  reduces to the molecular MO of an isolated H₂ (extra cells contribute
  only AO tails, which decay).
* **Density consistency** Σ_k w_k Σ_n^occ |ψ_{n,k}|² integrates to
  ``n_electrons / 2`` per cell on a converged Γ-only RHF.
* **Density cross-check** the same density rebuilt from per-k Bloch
  orbitals matches :func:`evaluate_periodic_density_on_grid` on the
  same orthorhombic cell.

Plus structural checks on the writers themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# H₂ chain: 1D lattice along x with two H atoms per cell at the cell
# center. Vacuum padding 30 bohr along y and z. Atoms in the cell
# *interior* (not the corner) so that integrating |ψ|² over one
# primitive cell captures the full AO support (avoids the half-support
# pitfall when atoms sit on the (0, 0, 0) boundary).
_A = 6.0
_D = 1.4
_Y0 = _Z0 = 15.0


def _h2_chain():
    return vq.PeriodicSystem(
        1,
        [[_A, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [(_A - _D) / 2, _Y0, _Z0]),
         vq.Atom(1, [(_A + _D) / 2, _Y0, _Z0])],
    )


def _hcore_C_at_k(system, basis, k_cart):
    """Solve the Bloch Hcore eigenproblem at one k. Returns (eps, C).
    Hcore is enough for the structural / convention tests; the SCF
    convergence story is unrelated to V3 correctness."""
    opts = vq.LatticeSumOptions()
    S_lat = vq.compute_overlap_lattice(basis, system, opts)
    T_lat = vq.compute_kinetic_lattice(basis, system, opts)
    V_lat = vq.compute_nuclear_lattice(basis, system, opts)
    S_k = vq.bloch_sum(S_lat, k_cart)
    F_k = vq.bloch_sum(T_lat, k_cart) + vq.bloch_sum(V_lat, k_cart)
    sol = vq.diagonalize_bloch(F_k, S_k)
    return np.asarray(sol.energies), np.asarray(sol.coefficients)


# ---------------------------------------------------------------------------
# Primitive-cell grid factory
# ---------------------------------------------------------------------------

def test_primitive_cell_grid_shape_and_origin():
    # Use a longer 1D cell than the shared structural fixture so AO-image
    # overlap is negligible and the density integral is a crisp electron
    # count check rather than an overlap-convention check.
    a_long = 10.0
    system = vq.PeriodicSystem(
        1,
        [[a_long, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [4.3, _Y0, _Z0]), vq.Atom(1, [5.7, _Y0, _Z0])],
    )
    grid = vq.make_primitive_cell_grid(system, spacing_bohr=0.2)
    # Auto-shape: ceil(|a_i| / spacing) along each lattice direction.
    assert grid.shape == (50, 150, 150)
    assert grid.points.shape == (50 * 150 * 150, 3)
    assert np.allclose(grid.origin, [0.0, 0.0, 0.0])

    # The grid must NOT include the duplicate boundary voxel (XSF spec).
    # Largest x coordinate is (n_a - 1) / n_a · |a_x|.
    x_max = grid.points[:, 0].max()
    assert x_max == pytest.approx((50 - 1) / 50 * a_long, rel=1e-12)


def test_primitive_cell_grid_explicit_shape():
    system = _h2_chain()
    grid = vq.make_primitive_cell_grid(system, grid_shape=(12, 24, 24))
    assert grid.shape == (12, 24, 24)
    assert grid.points.shape == (12 * 24 * 24, 3)


# ---------------------------------------------------------------------------
# (A) Bloch identity ψ(r + T) = exp(i k·T) ψ(r)
# ---------------------------------------------------------------------------

def test_bloch_identity_at_general_k():
    """ψ_{n,k}(r + T) = e^{i k·T} ψ_{n,k}(r) — the defining lattice
    phase relation. Sampled at a generic k away from time-reversal-
    invariant momenta so the orbital is genuinely complex."""
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    # Pick k = 0.4 · π / a along x — far from Γ and X.
    k = np.array([0.4 * np.pi / _A, 0.0, 0.0])
    _, C = _hcore_C_at_k(system, basis, k)

    rng = np.random.default_rng(1234)
    # 50 random sample points scattered through the cell.
    pts = np.column_stack([
        rng.uniform(0.0, _A,  size=50),
        rng.uniform(5.0, 25.0, size=50),
        rng.uniform(5.0, 25.0, size=50),
    ])

    psi = vq.evaluate_bloch_orbital(basis, system, pts, C, k, band_index=0)

    # Translate by T = 1·a along x and re-evaluate.
    T = np.array([_A, 0.0, 0.0])
    psi_T = vq.evaluate_bloch_orbital(
        basis, system, pts + T, C, k, band_index=0,
    )

    expected = np.exp(1j * np.dot(k, T)) * psi
    # The default 15 bohr lattice cutoff drops AO contributions at the
    # boundary inconsistently between r and r+T (different cells fall
    # off the truncation sphere). Tightening the cutoff drives the
    # residual into FP noise — verify the relation at machine precision
    # with a generous cutoff that matches both samples.
    psi_far = vq.evaluate_bloch_orbital(
        basis, system, pts, C, k, band_index=0,
        lattice_cutoff_bohr=60.0,
    )
    psi_T_far = vq.evaluate_bloch_orbital(
        basis, system, pts + T, C, k, band_index=0,
        lattice_cutoff_bohr=60.0,
    )
    expected_far = np.exp(1j * np.dot(k, T)) * psi_far
    assert np.allclose(psi_T_far, expected_far, atol=1e-12, rtol=1e-12)
    # The default-cutoff variant is still good to a few parts in 1e6.
    assert np.allclose(psi_T, expected, atol=1e-6, rtol=1e-6)


def test_bloch_phase_real_at_gamma_x():
    """At Γ and X (1D), the Bloch phase exp(i k·T) is real (±1), so the
    orbital is real up to a single global complex phase (the gauge
    arbitrariness of the eigenvector). Re-phase to absorb that gauge,
    then verify the imaginary part vanishes."""
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    grid = vq.make_primitive_cell_grid(system, spacing_bohr=0.4)
    for k_x in (0.0, np.pi / _A):
        k = np.array([k_x, 0.0, 0.0])
        _, C = _hcore_C_at_k(system, basis, k)
        psi = vq.evaluate_bloch_orbital(
            basis, system, grid.points, C, k, band_index=0,
        )
        # Pick the gauge that makes the maximum-amplitude voxel real
        # positive, then verify the rest of ψ is real to FP noise.
        idx_max = np.argmax(np.abs(psi))
        gauge = np.conj(psi[idx_max]) / np.abs(psi[idx_max])
        psi_gauged = psi * gauge
        scale = float(np.max(np.abs(psi_gauged.real)))
        assert scale > 0.05
        assert np.max(np.abs(psi_gauged.imag)) < 1e-10 * max(scale, 1.0)


# ---------------------------------------------------------------------------
# (B) Γ-limit: a long-cell Bloch sum reduces to the molecular MO
# ---------------------------------------------------------------------------

def test_gamma_limit_matches_molecular_mo_long_cell():
    """When the cell length ≫ AO decay scale, the Γ Bloch orbital
    coincides with the molecular MO of the same atoms in vacuum (extra
    cells contribute only AO tails). We pin both the eigenvalues and
    the orbital values on a coarse grid spanning the central cell."""
    a_long = 30.0   # bohr — well past STO-3G H1s decay (~3 bohr)
    system = vq.PeriodicSystem(
        1,
        [[a_long, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [(a_long - _D) / 2, _Y0, _Z0]),
         vq.Atom(1, [(a_long + _D) / 2, _Y0, _Z0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    eps, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))

    # Molecular reference: same two atoms in vacuum. Use the standard
    # symmetric-orthogonalisation diagonalisation so that C^T S C = I
    # (matches diagonalize_bloch's normalisation convention).
    mol = vq.Molecule(
        [vq.Atom(1, [(a_long - _D) / 2, _Y0, _Z0]),
         vq.Atom(1, [(a_long + _D) / 2, _Y0, _Z0])],
    )
    mbasis = vq.BasisSet(mol, "sto-3g")
    H = np.asarray(vq.compute_kinetic(mbasis)) + np.asarray(vq.compute_nuclear(mbasis, mol))
    S = np.asarray(vq.compute_overlap(mbasis))
    s_eig, U = np.linalg.eigh(S)
    X = U @ np.diag(s_eig ** -0.5)              # S^(-½)
    eps_mol, V = np.linalg.eigh(X.T @ H @ X)
    C_mol = X @ V                               # C^T S C = I

    # Bonding orbital eigenvalue.
    assert eps[0] == pytest.approx(eps_mol[0], abs=1e-6)

    # Spot-check the orbital VALUE at the bond midpoint (z=Z0, y=Y0):
    pt_mid = np.array([[a_long / 2, _Y0, _Z0]])
    psi_periodic = vq.evaluate_bloch_orbital(
        basis, system, pt_mid, C, np.array([0.0, 0.0, 0.0]), 0,
    )
    chi_mol = vq.evaluate_ao(mbasis, pt_mid)
    psi_mol = chi_mol @ C_mol[:, 0]
    # Both up to a global sign.
    ratio = float(psi_periodic[0].real) / float(psi_mol[0])
    assert abs(abs(ratio) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# (C) Density consistency: ∫_cell Σ_k w_k Σ_n^occ |ψ|² ≈ n_electrons / 2
# ---------------------------------------------------------------------------

def test_density_normalization_gamma_only():
    """Closed-shell H₂ chain at Γ — one occupied band. The integrated
    density per cell should be n_e / 2 = 1 (one doubly-occupied MO)."""
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))

    grid = vq.make_primitive_cell_grid(system, spacing_bohr=0.15)
    volume = abs(np.linalg.det(np.asarray(system.lattice)))
    dV = volume / np.prod(grid.shape)

    psi = vq.evaluate_bloch_orbital(
        basis, system, grid.points, C, np.array([0.0, 0.0, 0.0]), 0,
    )
    integral = float(np.sum(np.abs(psi) ** 2) * dV)
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_density_normalization_multi_k():
    """Multi-k sum on the H₂ chain. Σ_k w_k Σ_n^occ |ψ_{n,k}|² × 2
    integrates to n_electrons per cell."""
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    n_e = system.n_electrons()
    n_occ = n_e // 2

    kmesh = vq.monkhorst_pack(system, [4, 1, 1])

    grid = vq.make_primitive_cell_grid(system, spacing_bohr=0.20)
    volume = abs(np.linalg.det(np.asarray(system.lattice)))
    dV = volume / np.prod(grid.shape)

    rho = np.zeros(grid.points.shape[0], dtype=float)
    for k_cart, w in zip(kmesh.kpoints, kmesh.weights):
        _, C = _hcore_C_at_k(system, basis, np.asarray(k_cart, dtype=float))
        for n in range(n_occ):
            psi = vq.evaluate_bloch_orbital(
                basis, system, grid.points, C, np.asarray(k_cart, dtype=float), n,
            )
            rho += 2.0 * float(w) * (psi.real ** 2 + psi.imag ** 2)

    integral = float(np.sum(rho) * dV)
    assert integral == pytest.approx(float(n_e), abs=1e-2)


# ---------------------------------------------------------------------------
# (D) Cross-check vs evaluate_periodic_density_on_grid
# ---------------------------------------------------------------------------

def test_density_matches_periodic_density_grid():
    """For the same Γ-only Hcore wavefunction, the BZ-sum density built
    from Bloch orbitals matches the LatticeMatrixSet-driven density
    builder on an orthorhombic cell.

    Note on convention: ``evaluate_periodic_density_on_grid`` implements
    ``ρ(r) = Σ_g Σ_{μν} D(g)_{μν} χ_μ(r) χ_ν(r − g)`` — only one of
    the two AO factors carries a lattice shift. The Bloch-orbital
    formula ``ρ(r) = 2 Σ_n^occ |ψ_n|²`` symmetrises the lattice sum
    across both AO factors, which equals the grid formula only in the
    limit where AO periodic images do not overlap (i.e. cell larger
    than the AO decay scale). We test on a long cell (a = 20 bohr ≫
    STO-3G H1s ~3 bohr) so both converge to the same density."""
    a_long = 20.0
    system = vq.PeriodicSystem(
        1,
        [[a_long, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [(a_long - _D) / 2, _Y0, _Z0]),
         vq.Atom(1, [(a_long + _D) / 2, _Y0, _Z0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))

    n_occ = system.n_electrons() // 2

    # Modest grid: enough to resolve the AO at the bond midpoint.
    grid_shape = (40, 60, 60)

    # (1) Bloch-orbital density: ρ(r) = 2 |ψ_HOMO|² at Γ.
    grid = vq.make_primitive_cell_grid(system, grid_shape=grid_shape)
    rho_bloch = np.zeros(grid.points.shape[0], dtype=float)
    for n in range(n_occ):
        psi = vq.evaluate_bloch_orbital(
            basis, system, grid.points, C, np.array([0.0, 0.0, 0.0]), n,
        )
        rho_bloch += 2.0 * (psi.real ** 2 + psi.imag ** 2)
    rho_bloch = rho_bloch.reshape(grid_shape)

    # (2) LatticeMatrixSet density via evaluate_periodic_density_on_grid.
    # For a single-k Γ system, real_space_density_from_kpoints rebuilds
    # D(g) given the per-k coefficient list and the occupied-band count.
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    coeffs_list = [C.astype(complex)]
    opts = vq.LatticeSumOptions()
    cells_S = vq.compute_overlap_lattice(basis, system, opts).cells
    D_real = vq.real_space_density_from_kpoints(
        coeffs_list, [n_occ], kmesh, cells_S,
    )
    rho_grid, _ = vq.evaluate_periodic_density_on_grid(
        basis, system, D_real, grid_shape=grid_shape,
    )

    # Both paths integrate to n_electrons per cell (within grid
    # discretisation) and agree pointwise wherever the density is
    # numerically meaningful. Compare on the support set where
    # ρ_grid > 1e-8; outside that the values are FP-noise underflows
    # whose relative discrepancy is unphysical.
    volume = abs(np.linalg.det(np.asarray(system.lattice)))
    dV = volume / np.prod(grid_shape)
    n_e = float(system.n_electrons())
    assert float(rho_bloch.sum()) * dV == pytest.approx(n_e, abs=1e-1)
    assert float(rho_grid.sum())  * dV == pytest.approx(n_e, abs=1e-1)
    mask = rho_grid > 1e-8
    assert mask.any()
    assert np.allclose(rho_bloch[mask], rho_grid[mask], rtol=2e-2, atol=1e-6)


# ---------------------------------------------------------------------------
# Writer structural checks
# ---------------------------------------------------------------------------

def test_write_xsf_mo_round_trip(tmp_path):
    """Periodic XSF MO file: header is well-formed and the data block
    has exactly n1·n2·n3 floats."""
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))

    p = vq.write_xsf_mo(
        tmp_path / "mo.xsf", system, basis, C,
        np.array([0.0, 0.0, 0.0]), 0,
        grid_shape=(8, 16, 16),
        component="real",
    )
    text = p.read_text()
    # _h2_chain() is dim=1, so the block keyword names it a polymer: only the
    # first PRIMVEC row is a real lattice vector.
    assert text.splitlines()[0] == "POLYMER"
    assert "PRIMVEC" in text
    assert "BEGIN_DATAGRID_3D_bloch_orbital_0_real" in text
    assert "END_DATAGRID_3D_bloch_orbital_0_real" in text

    lines = text.splitlines()
    start = lines.index("BEGIN_DATAGRID_3D_bloch_orbital_0_real")
    end = lines.index("END_DATAGRID_3D_bloch_orbital_0_real")
    # Header inside block: shape + origin + 3 spans = 5 lines.
    n_floats = sum(len(L.split()) for L in lines[start + 6:end])
    assert n_floats == 8 * 16 * 16


def test_write_xsf_mo_components(tmp_path):
    """Component selector: 'density' must be everywhere ≥ 0; 'imag'
    at Γ must be everywhere ≈ 0 (eigenvector is real)."""
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))

    for comp, predicate in [
        ("real", lambda v: True),
        ("imag", lambda v: np.max(np.abs(v)) < 1e-10),
        ("abs", lambda v: np.all(v >= -1e-12)),
        ("density", lambda v: np.all(v >= -1e-12)),
    ]:
        p = vq.write_xsf_mo(
            tmp_path / f"mo_{comp}.xsf", system, basis, C,
            np.array([0.0, 0.0, 0.0]), 0,
            grid_shape=(8, 16, 16), component=comp,
        )
        text = p.read_text().splitlines()
        start = text.index(f"BEGIN_DATAGRID_3D_bloch_orbital_0_{comp}")
        end = text.index(f"END_DATAGRID_3D_bloch_orbital_0_{comp}")
        vals = np.array([float(t) for L in text[start + 6:end] for t in L.split()])
        assert predicate(vals)


def test_write_xsf_mo_invalid_component(tmp_path):
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="component"):
        vq.write_xsf_mo(
            tmp_path / "bad.xsf", system, basis, C,
            np.array([0.0, 0.0, 0.0]), 0,
            grid_shape=(4, 4, 4), component="nope",
        )


def test_write_cube_mo_periodic_roundtrip(tmp_path):
    """Supercell cube file: replicated atoms + axis-aligned voxels."""
    system = _h2_chain()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))

    n_replica = (3, 1, 1)
    p = vq.write_cube_mo_periodic(
        tmp_path / "psi.cube", system, basis, C,
        np.array([0.0, 0.0, 0.0]), 0,
        n_replica=n_replica, spacing_bohr=0.4,
    )
    text = p.read_text().splitlines()
    n_atoms = int(text[2].split()[0])
    assert n_atoms == 2 * np.prod(n_replica)   # 6 atoms in 3-cell view


def test_write_cube_mo_periodic_rejects_nonorthorhombic(tmp_path):
    """Non-orthogonal cells must raise — vibe-qc's cube writer is
    axis-aligned."""
    skewed = vq.PeriodicSystem(
        3,
        [[5.0, 0.5, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        [vq.Atom(1, [2.0, 2.5, 2.5]),
         vq.Atom(1, [3.0, 2.5, 2.5])],
    )
    basis = vq.BasisSet(skewed.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(skewed, basis, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="orthorhombic"):
        vq.write_cube_mo_periodic(
            tmp_path / "bad.cube", skewed, basis, C,
            np.array([0.0, 0.0, 0.0]), 0,
            n_replica=(2, 2, 2),
        )


def test_write_xsf_density_round_trip(tmp_path):
    """Convenience density wrapper: build D(g) from a Γ-only solution,
    write the XSF, integrate to recover n_electrons."""
    a_long = 10.0
    system = vq.PeriodicSystem(
        1,
        [[a_long, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [4.3, _Y0, _Z0]), vq.Atom(1, [5.7, _Y0, _Z0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    _, C = _hcore_C_at_k(system, basis, np.array([0.0, 0.0, 0.0]))
    n_occ = system.n_electrons() // 2

    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    opts = vq.LatticeSumOptions()
    cells_S = vq.compute_overlap_lattice(basis, system, opts).cells
    D_real = vq.real_space_density_from_kpoints(
        [C.astype(complex)], [n_occ], kmesh, cells_S,
    )

    grid_shape = (100, 100, 100)  # spacing 0.1 / 0.3 / 0.3: fine enough
    p = vq.write_xsf_density(
        tmp_path / "rho.xsf", system, basis, D_real,
        grid_shape=grid_shape, name="rho",
    )
    text = p.read_text()
    assert "BEGIN_DATAGRID_3D_rho" in text

    # Re-parse the data block and integrate; expect n_electrons within
    # the grid-discretisation error of the periodic-density formula.
    lines = text.splitlines()
    start = lines.index("BEGIN_DATAGRID_3D_rho")
    end = lines.index("END_DATAGRID_3D_rho")
    vals = np.array([float(t) for L in lines[start + 6:end] for t in L.split()])
    assert vals.size == int(np.prod(grid_shape))
    volume = abs(np.linalg.det(np.asarray(system.lattice)))
    dV = volume / vals.size
    integral = float(vals.sum()) * dV
    assert integral == pytest.approx(float(system.n_electrons()), rel=5e-2)
