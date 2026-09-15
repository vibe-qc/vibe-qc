"""Gamma RHF compcell GDF analytic-gradient regression tests.

Validates the C++ periodic 2c/3c gradient kernels and the DF J/K
gradient formulas at fixed density, the fully analytic Ewald ``V_ne``
derivative on H2 and LiH, and the public full-SCF gradient route against
a central difference of the converged energy.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    CoulombMethod,
    LatticeSumOptions,
    compute_2c_eri_lattice_gradient_weighted,
    compute_3c_eri_lattice_gradient_weighted,
    nuclear_repulsion_gradient_per_cell,
)
from vibeqc.aux_basis import (
    _build_lpq_compcell_state,
    fuse_transform_matrix,
    make_aux_basis_set,
    make_compensating_basis,
    make_fused_basis,
    make_modrho_aux_basis,
)
from vibeqc.periodic_gdf_gradient import (
    _build_compcell_gradient_cache,
    _CompcellGradientCache,
    _compute_j_gradient_compcell,
    _compute_k_gradient_compcell,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2_12bohr_system():
    """H2 in a 12-bohr cubic box, STO-3G."""
    box = 12.0
    c = box / 2.0
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    system = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _diatomic_12bohr_system(z_a: int, z_b: int):
    """Centred diatomic in the same 12-bohr cell as the GDF fixture."""
    box = 12.0
    c = box / 2.0
    atoms = [
        vq.Atom(z_a, [c, c, c - 0.7]),
        vq.Atom(z_b, [c, c, c + 0.7]),
    ]
    system = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lat_opts():
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    opts.nuclear_cutoff_bohr = 12.0
    return opts


def _build_aux_fused(system, basis):
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    chg = make_compensating_basis(modrho, mol, eta=1.0)
    fused = make_fused_basis(modrho, chg, mol)
    A_mat = fuse_transform_matrix(modrho, chg)
    return aux, modrho, chg, fused, A_mat


# ---------------------------------------------------------------------------
# Tests — C++ gradient kernels
# ---------------------------------------------------------------------------


def test_2c_gradient_kernel_shape():
    """2c lattice gradient kernel returns (n_atoms, 3) shape."""
    system, basis = _h2_12bohr_system()
    _, _, _, fused, _ = _build_aux_fused(system, basis)
    lat_opts = _lat_opts()
    n_atoms = len(system.unit_cell)
    omega = np.eye(fused.nbasis)
    grad = np.asarray(
        compute_2c_eri_lattice_gradient_weighted(fused, system, lat_opts, omega)
    )
    assert grad.shape == (n_atoms, 3)


def test_2c_gradient_kernel_newton3():
    """2c lattice gradient satisfies Newton's 3rd law (sum=0)."""
    system, basis = _h2_12bohr_system()
    _, _, _, fused, _ = _build_aux_fused(system, basis)
    lat_opts = _lat_opts()
    omega = np.ones((fused.nbasis, fused.nbasis))
    grad = np.asarray(
        compute_2c_eri_lattice_gradient_weighted(fused, system, lat_opts, omega)
    )
    total_force = grad.sum(axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-12)


def test_3c_gradient_kernel_shape():
    """3c lattice gradient kernel returns (n_atoms, 3) shape."""
    system, basis = _h2_12bohr_system()
    _, _, _, fused, _ = _build_aux_fused(system, basis)
    lat_opts = _lat_opts()
    n_atoms = len(system.unit_cell)
    W = np.ones(
        (fused.nbasis, basis.nbasis * basis.nbasis), dtype=np.float64, order="C"
    )
    grad = np.asarray(
        compute_3c_eri_lattice_gradient_weighted(basis, fused, system, lat_opts, W)
    )
    assert grad.shape == (n_atoms, 3)


def test_3c_gradient_kernel_newton3():
    """3c lattice gradient satisfies Newton's 3rd law (sum=0)."""
    system, basis = _h2_12bohr_system()
    _, _, _, fused, _ = _build_aux_fused(system, basis)
    lat_opts = _lat_opts()
    W = np.ones(
        (fused.nbasis, basis.nbasis * basis.nbasis), dtype=np.float64, order="C"
    )
    grad = np.asarray(
        compute_3c_eri_lattice_gradient_weighted(basis, fused, system, lat_opts, W)
    )
    total_force = grad.sum(axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests — DF J gradient at fixed D
# ---------------------------------------------------------------------------


def test_j_gradient_fixed_d_vs_fd():
    """Thresholded DF-J gradient includes the retained-space response."""
    system, basis = _h2_12bohr_system()
    lat_opts = _lat_opts()
    aux, _, _, _, _ = _build_aux_fused(system, basis)

    np.random.seed(42)
    D = np.random.randn(basis.nbasis, basis.nbasis)
    D = 0.5 * (D + D.T)

    linear_dep_thr = 1.0e-3
    cache = _build_compcell_gradient_cache(
        system,
        basis,
        aux,
        compcell_eta=1.0,
        lattice_opts=lat_opts,
        linear_dep_thr=linear_dep_thr,
        rcut_strategy="pyscf_auto",
        rcut_precision=1e-8,
    )
    assert cache.n_fit < cache.n_aux
    assert cache.lat_opts_2c.cutoff_bohr != cache.lat_opts_3c.cutoff_bohr
    g_J = _compute_j_gradient_compcell(system, basis, D, cache)

    # FD: rebuild at displaced geometry
    delta = 1e-4
    atoms = list(system.unit_cell)
    for a in range(len(atoms)):
        for d in range(3):
            a_p = []
            a_m = []
            for i, atom in enumerate(atoms):
                xyz_p = list(atom.xyz)
                xyz_m = list(atom.xyz)
                if i == a:
                    xyz_p[d] += delta
                    xyz_m[d] -= delta
                a_p.append(vq.Atom(int(atom.Z), xyz_p))
                a_m.append(vq.Atom(int(atom.Z), xyz_m))
            s_p = vq.PeriodicSystem(3, np.eye(3) * 12.0, a_p)
            s_m = vq.PeriodicSystem(3, np.eye(3) * 12.0, a_m)

            # Rebuild E_J at displaced geometry (same D)
            def _e_j(sys_d):
                mol_d = sys_d.unit_cell_molecule()
                basis_d = vq.BasisSet(mol_d, "sto-3g")
                aux_d = make_aux_basis_set(mol_d, aux_name="def2-svp-jk")
                state = _build_lpq_compcell_state(
                    sys_d,
                    basis_d,
                    aux_d,
                    molecule=mol_d,
                    lat_opts=lat_opts,
                    linear_dep_thr=linear_dep_thr,
                    compcell_eta=1.0,
                    apply_aft_correction=False,
                    rcut_strategy="pyscf_auto",
                    rcut_precision=1e-8,
                )
                A_d = state.A
                M_f = state.M_fused
                T_f = state.T_fused
                M_c = A_d @ M_f @ A_d.T
                T_c = np.einsum("iP,Pmn->imn", A_d, T_f)
                T_flat = T_c.reshape(T_c.shape[0], -1)
                rho = T_flat @ D.ravel()
                eigvals, U = np.linalg.eigh(M_c)
                keep = eigvals > linear_dep_thr * eigvals[-1]
                gamma = U[:, keep] @ (
                    (U[:, keep].T @ rho) / eigvals[keep]
                )
                return 0.5 * float(rho @ gamma)

            Ep = _e_j(s_p)
            Em = _e_j(s_m)
            fd_val = (Ep - Em) / (2 * delta)
            assert abs(fd_val - g_J[a, d]) < 1e-7, (
                f"J gradient mismatch at atom {a}, dir {d}: "
                f"analytic={g_J[a, d]:.10f}, FD={fd_val:.10f}"
            )


# ---------------------------------------------------------------------------
# Tests — DF K gradient at fixed C_occ
# ---------------------------------------------------------------------------


def test_k_gradient_fixed_c_vs_fd():
    """Thresholded DF-K gradient includes the retained-space response."""
    system, basis = _h2_12bohr_system()
    lat_opts = _lat_opts()
    aux, _, _, _, _ = _build_aux_fused(system, basis)

    rng = np.random.default_rng(123)
    C_occ, _ = np.linalg.qr(rng.normal(size=(basis.nbasis, 2)))
    assert C_occ.shape[1] == 2
    D = 2.0 * C_occ @ C_occ.T

    linear_dep_thr = 1.0e-3
    cache = _build_compcell_gradient_cache(
        system,
        basis,
        aux,
        compcell_eta=1.0,
        lattice_opts=lat_opts,
        linear_dep_thr=linear_dep_thr,
    )
    assert cache.n_fit < cache.n_aux
    g_K = _compute_k_gradient_compcell(
        system,
        basis,
        D,
        C_occ,
        1.0,
        cache,
    )

    delta = 1e-4
    atoms = list(system.unit_cell)
    for a in [0]:  # test one atom to keep runtime reasonable
        for d in [2]:  # test z-component
            a_p = []
            a_m = []
            for i, atom in enumerate(atoms):
                xyz_p = list(atom.xyz)
                xyz_m = list(atom.xyz)
                if i == a:
                    xyz_p[d] += delta
                    xyz_m[d] -= delta
                a_p.append(vq.Atom(int(atom.Z), xyz_p))
                a_m.append(vq.Atom(int(atom.Z), xyz_m))
            s_p = vq.PeriodicSystem(3, np.eye(3) * 12.0, a_p)
            s_m = vq.PeriodicSystem(3, np.eye(3) * 12.0, a_m)

            def _e_k(sys_d):
                mol_d = sys_d.unit_cell_molecule()
                basis_d = vq.BasisSet(mol_d, "sto-3g")
                aux_d = make_aux_basis_set(mol_d, aux_name="def2-svp-jk")
                modrho_d = make_modrho_aux_basis(aux_d, mol_d)
                chg_d = make_compensating_basis(modrho_d, mol_d, eta=1.0)
                fused_d = make_fused_basis(modrho_d, chg_d, mol_d)
                A_d = fuse_transform_matrix(modrho_d, chg_d)
                from vibeqc._vibeqc_core import (
                    compute_2c_eri_lattice,
                    compute_3c_eri_lattice,
                )

                M_f = np.asarray(compute_2c_eri_lattice(fused_d, sys_d, lat_opts))
                M_f = 0.5 * (M_f + M_f.T)
                T_f = np.asarray(
                    compute_3c_eri_lattice(basis_d, fused_d, sys_d, lat_opts)
                )
                M_c = A_d @ M_f @ A_d.T
                T_c = np.einsum("iP,Pmn->imn", A_d, T_f)
                eigvals, U = np.linalg.eigh(M_c)
                keep = eigvals > linear_dep_thr * eigvals[-1]
                T_flat = T_c.reshape(T_c.shape[0], -1)
                Lpq = (U[:, keep].T @ T_flat) / np.sqrt(eigvals[keep])[:, None]
                Lpq = Lpq.reshape(keep.sum(), basis_d.nbasis, basis_d.nbasis)
                from vibeqc.pbc_gdf import _build_k_from_lpq

                K = _build_k_from_lpq(Lpq, D)
                return -0.25 * float(np.einsum("ij,ij->", D, K))

            Ep = _e_k(s_p)
            Em = _e_k(s_m)
            fd_val = (Ep - Em) / (2 * delta)
            assert abs(fd_val - g_K[a, d]) < 1e-7, (
                f"K gradient mismatch at atom {a}, dir {d}: "
                f"analytic={g_K[a, d]:.10f}, FD={fd_val:.10f}"
            )


def test_gradient_cache_uses_scf_resolved_compcell_cutoffs():
    """The gradient fit resolves the same distinct 2c/3c cutoffs as SCF.

    Both pins grew by exactly sqrt(2) on 2026-08-02 when RCUT-PYSCF-MISPORT
    was fixed (6.9635305883 -> 9.8479194000 and 9.1474643134 ->
    12.9364680326): the estimator had been decaying with the single-shell
    exponent alpha instead of the pair reduced exponent alpha/2. A future
    diff that divides these by 1.4142 has reintroduced that bug rather than
    found a better cutoff.
    """
    system, basis = _h2_12bohr_system()
    aux, _, _, _, _ = _build_aux_fused(system, basis)
    cache = _build_compcell_gradient_cache(
        system,
        basis,
        aux,
        compcell_eta=1.0,
        lattice_opts=_lat_opts(),
        linear_dep_thr=1e-9,
        rcut_strategy="pyscf_auto",
        rcut_precision=1e-8,
    )

    assert cache.lat_opts_2c.cutoff_bohr == pytest.approx(9.8479194000)
    assert cache.lat_opts_3c.cutoff_bohr == pytest.approx(12.9364680326)
    assert cache.lat_opts_2c.cutoff_bohr != cache.lat_opts_3c.cutoff_bohr


# ---------------------------------------------------------------------------
# Tests — nuclear repulsion gradient (1-e sanity)
# ---------------------------------------------------------------------------


def test_nuclear_repulsion_gradient_newton3():
    """Nuclear repulsion gradient satisfies Newton's 3rd law."""
    system, basis = _h2_12bohr_system()
    lat_opts = _lat_opts()
    grad = np.asarray(nuclear_repulsion_gradient_per_cell(system, lat_opts))
    total_force = grad.sum(axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-12)


@pytest.mark.parametrize("atomic_numbers", [(1, 1), (3, 1)], ids=["h2", "lih"])
def test_v_ne_analytic_gradient_matches_fixed_density_fd(
    atomic_numbers,
    monkeypatch,
):
    """The complete FT-based Ewald V_ne derivative matches its value route.

    H2 exercises the s-only AO-pair derivative; LiH adds the calibrated
    p-shell path.  The modest reciprocal cutoff keeps this component test
    cheap while retaining the same analytic formula used at the production
    cutoff.
    """
    from vibeqc.periodic_v_ne import compute_v_ne_ewald_3d_ft_gamma
    import vibeqc.periodic_v_ne_gradient as v_ne_gradient

    system, basis = _diatomic_12bohr_system(*atomic_numbers)
    lat_opts = _lat_opts()
    ke_cutoff = 8.0
    if atomic_numbers == (3, 1):
        monkeypatch.setenv("VIBEQC_VNE_EWALD3D_FT_CHUNK_MIB", "1.0")

    rng = np.random.default_rng(42)
    D = rng.normal(size=(basis.nbasis, basis.nbasis))
    D = 0.5 * (D + D.T)

    import vibeqc._aopair_ft as aopair_ft

    native_gradient = (
        v_ne_gradient.ao_pair_fourier_transform_gamma_gradient_weighted
    )
    native_calls = 0

    def observe_native_derivative(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_gradient(*args, **kwargs)

    def reject_python_derivative(*args, **kwargs):
        raise AssertionError("the production gradient must use native contraction")

    with monkeypatch.context() as patch:
        patch.setattr(
            v_ne_gradient,
            "ao_pair_fourier_transform_gamma_gradient_weighted",
            observe_native_derivative,
        )
        patch.setattr(
            aopair_ft,
            "ao_pair_fourier_transform_grad_at_cells",
            reject_python_derivative,
        )
        analytic = v_ne_gradient.compute_v_ne_ewald_3d_ft_gamma_gradient(
            basis,
            system,
            lat_opts,
            D,
            ke_cutoff=ke_cutoff,
        )
    minimum_native_calls = 2 if atomic_numbers == (3, 1) else 1
    assert native_calls >= minimum_native_calls

    delta = 2.0e-5
    atoms = list(system.unit_cell)
    lattice = np.asarray(system.lattice, dtype=float)
    fd = np.zeros_like(analytic)
    for atom_idx in range(len(atoms)):
        for axis in range(3):
            energies = []
            for sign in (1.0, -1.0):
                displaced = []
                for idx, atom in enumerate(atoms):
                    xyz = np.asarray(atom.xyz, dtype=float).copy()
                    if idx == atom_idx:
                        xyz[axis] += sign * delta
                    displaced.append(vq.Atom(int(atom.Z), xyz.tolist()))
                displaced_system = vq.PeriodicSystem(
                    3,
                    lattice,
                    displaced,
                    charge=system.charge,
                    multiplicity=system.multiplicity,
                )
                displaced_basis = vq.BasisSet(
                    displaced_system.unit_cell_molecule(),
                    basis.name,
                )
                v_ne = compute_v_ne_ewald_3d_ft_gamma(
                    displaced_basis,
                    displaced_system,
                    lat_opts,
                    ke_cutoff=ke_cutoff,
                )
                energies.append(float(np.einsum("ij,ij->", D, v_ne)))
            fd[atom_idx, axis] = (energies[0] - energies[1]) / (2.0 * delta)

    np.testing.assert_allclose(analytic, fd, atol=1.0e-5, rtol=0.0)
    np.testing.assert_allclose(analytic.sum(axis=0), 0.0, atol=1.0e-10, rtol=0.0)


# ---------------------------------------------------------------------------
# Tests — full SCF gradient vs FD (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exxdiv", ["ewald", "none"])
def test_full_scf_gradient_vs_fd_h2(monkeypatch, exxdiv):
    """Full GDF SCF gradient matches FD on H2/STO-3G/12-bohr (z-component).

    Runs a GDF SCF, computes the analytic gradient via the convenience
    function, and compares against central-difference FD of the SCF
    total energy at one atom×direction.  The value builder is disabled
    while evaluating the gradient, pinning the analytic V_ne route and
    preventing a regression to 6N whole-matrix finite differences.
    """
    system, basis = _h2_12bohr_system()
    lat_opts = _lat_opts()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = lat_opts
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")

    import vibeqc.periodic_v_ne as periodic_v_ne

    def reject_numerical_v_ne(*args, **kwargs):
        raise AssertionError("the GDF gradient must not rebuild V_ne")

    with monkeypatch.context() as patch:
        patch.setattr(
            periodic_v_ne,
            "compute_v_ne_ewald_3d_ft_gamma",
            reject_numerical_v_ne,
        )
        result = vq.run_pbc_gdf_rhf(
            system,
            basis,
            options=options,
            aux_basis="def2-svp-jk",
            gdf_method="compcell",
            compcell_eta=1.0,
            apply_aft_correction=False,
            gdf_linear_dep_threshold=1e-9,
            exxdiv=exxdiv,
            compute_gradient=True,
        )
    assert result.converged
    assert result.gradient is not None
    assert result.v_ne_backend == "analytic_ft"
    assert result.v_ne_ke_cutoff == pytest.approx(8.0)
    assert result.exxdiv == exxdiv
    assert result.gdf_rcut_strategy == "pyscf_auto"
    assert result.gdf_fit_cutoff_2c == pytest.approx(9.8479194000)
    assert result.gdf_fit_cutoff_3c == pytest.approx(12.9364680326)
    assert len(result.aux_basis_fingerprint) == 64
    grad = np.asarray(result.gradient)

    if exxdiv == "ewald":
        # The exported route must also reconstruct the fit from scalar
        # provenance when the SCF's transient private state is unavailable.
        direct = np.asarray(vq.compute_gdf_gradient(system, basis, result))
        np.testing.assert_allclose(direct, grad, rtol=0.0, atol=1e-12)

        # The lower-level exported route must enforce the same scalar
        # provenance instead of accepting caller-selected fit/gauge meshes.
        from vibeqc.periodic_gdf_gradient import (
            compute_gdf_gradient_rhf_gamma,
        )

        aux = make_aux_basis_set(
            system.unit_cell_molecule(), aux_name="def2-svp-jk"
        )
        with pytest.raises(ValueError, match="compcell_eta must match"):
            compute_gdf_gradient_rhf_gamma(
                system, basis, result, aux_basis=aux, compcell_eta=0.8
            )
        with pytest.raises(ValueError, match="v_ne_ke_cutoff must match"):
            compute_gdf_gradient_rhf_gamma(
                system, basis, result, aux_basis=aux, v_ne_ke_cutoff=9.0
            )

        bad_gauge_opts = _lat_opts()
        bad_gauge_opts.cutoff_bohr = 4.0
        bad_gauge_opts.nuclear_cutoff_bohr = 4.0
        bad_gauge_opts.coulomb_method = CoulombMethod.EWALD_3D
        with pytest.raises(ValueError, match="gauge_lat_opts cutoffs"):
            compute_gdf_gradient_rhf_gamma(
                system,
                basis,
                result,
                aux_basis=aux,
                gauge_lat_opts=bad_gauge_opts,
            )

        # Changing the environment after the SCF cannot change the default
        # V_ne mesh used by this entry point. Stop at the V_ne call so the
        # test does not need a second compcell fit rebuild.
        low_level_cache = _build_compcell_gradient_cache(
            system,
            basis,
            aux,
            compcell_eta=result.compcell_eta,
            lattice_opts=lat_opts,
            linear_dep_thr=result.gdf_linear_dep_threshold,
            rcut_strategy=result.gdf_rcut_strategy,
            rcut_precision=result.gdf_rcut_precision,
        )

        displaced_atoms = list(system.unit_cell)
        shifted_xyz = list(displaced_atoms[0].xyz)
        shifted_xyz[2] += 1.0e-3
        displaced_atoms[0] = vq.Atom(
            int(displaced_atoms[0].Z), shifted_xyz
        )
        displaced_system = vq.PeriodicSystem(
            3, np.asarray(system.lattice), displaced_atoms
        )
        displaced_basis = vq.BasisSet(
            displaced_system.unit_cell_molecule(), "sto-3g"
        )
        displaced_aux = make_aux_basis_set(
            displaced_system.unit_cell_molecule(),
            aux_name="def2-svp-jk",
        )
        displaced_cache = _build_compcell_gradient_cache(
            displaced_system,
            displaced_basis,
            displaced_aux,
            compcell_eta=result.compcell_eta,
            lattice_opts=lat_opts,
            linear_dep_thr=result.gdf_linear_dep_threshold,
            rcut_strategy=result.gdf_rcut_strategy,
            rcut_precision=result.gdf_rcut_precision,
        )
        with pytest.raises(ValueError, match="cache provenance"):
            compute_gdf_gradient_rhf_gamma(
                system,
                basis,
                result,
                aux_basis=aux,
                cache=displaced_cache,
            )

        observed_cutoffs = []

        def capture_v_ne_cutoff(*args, ke_cutoff, **kwargs):
            observed_cutoffs.append(ke_cutoff)
            raise RuntimeError("captured retained V_ne cutoff")

        import vibeqc.periodic_v_ne_gradient as v_ne_gradient

        with monkeypatch.context() as patch:
            patch.setenv("VIBEQC_VNE_EWALD3D_KE", "9.0")
            patch.setattr(
                v_ne_gradient,
                "compute_v_ne_ewald_3d_ft_gamma_gradient",
                capture_v_ne_cutoff,
            )
            with pytest.raises(RuntimeError, match="captured retained"):
                compute_gdf_gradient_rhf_gamma(
                    system,
                    basis,
                    result,
                    aux_basis=aux,
                    cache=low_level_cache,
                )
        assert observed_cutoffs == [pytest.approx(result.v_ne_ke_cutoff)]

    # Verify equal-and-opposite forces (Newton's 3rd law)
    total_force = grad.sum(axis=0)
    assert np.allclose(total_force, 0.0, atol=1e-10), f"∑F ≠ 0: {total_force}"

    # FD check on z-component of atom 0
    delta = 1e-4
    a, d = 0, 2
    atoms = list(system.unit_cell)
    box = 12.0
    a_p = []
    a_m = []
    for i, atom in enumerate(atoms):
        xyz_p = list(atom.xyz)
        xyz_m = list(atom.xyz)
        if i == a:
            xyz_p[d] += delta
            xyz_m[d] -= delta
        a_p.append(vq.Atom(int(atom.Z), xyz_p))
        a_m.append(vq.Atom(int(atom.Z), xyz_m))

    s_p = vq.PeriodicSystem(3, np.eye(3) * box, a_p)
    s_m = vq.PeriodicSystem(3, np.eye(3) * box, a_m)
    b_p = vq.BasisSet(s_p.unit_cell_molecule(), "sto-3g")
    b_m = vq.BasisSet(s_m.unit_cell_molecule(), "sto-3g")

    rp = vq.run_pbc_gdf_rhf(
        s_p, b_p,
        options=options,
        aux_basis="def2-svp-jk",
        gdf_method="compcell",
        compcell_eta=1.0,
        apply_aft_correction=False,
        gdf_linear_dep_threshold=1e-9,
        exxdiv=exxdiv,
    )
    rm = vq.run_pbc_gdf_rhf(
        s_m, b_m,
        options=options,
        aux_basis="def2-svp-jk",
        gdf_method="compcell",
        compcell_eta=1.0,
        apply_aft_correction=False,
        gdf_linear_dep_threshold=1e-9,
        exxdiv=exxdiv,
    )
    fd_val = (rp.energy - rm.energy) / (2.0 * delta)

    # Tightened from the historical 5e-3 gate after the G-PBC-002
    # milestone-3b identification: the long-undiagnosed flat 3.394e-3
    # residual was the truncated-direct-sum nuclear gradient paired with
    # the Ewald-gauge e_nuc; with the matching Ewald nuclear gradient
    # the observed FD residual on this control is ~3e-9.
    assert abs(fd_val - grad[a, d]) < 1e-6, (
        f"Full SCF gradient mismatch at atom {a}, dir {d}: "
        f"analytic={grad[a, d]:.10f}, FD={fd_val:.10f}, "
        f"diff={abs(fd_val - grad[a, d]):.3e}"
    )


def test_public_gradient_route_rejects_unsupported_fit_derivatives():
    """The public driver fails before SCF instead of relabeling a gradient.

    (AFT-on compcell left this list at milestone 3b, and rsgdf at
    milestone 6 rung 4 — both fit derivatives are implemented and
    FD-gated. The remaining unsupported fit is MDF; the rsgdf-specific
    envelope pins live in
    ``test_rsgdf_gradient_rejects_unsupported_envelope``.)
    """
    system, basis = _h2_12bohr_system()
    with pytest.raises(NotImplementedError, match="MDF fit derivative"):
        vq.run_pbc_gdf_rhf(
            system,
            basis,
            compute_gradient=True,
            gdf_method="mdf",
        )


@pytest.mark.parametrize(
    "result_update, gradient_kwargs, exception, message",
    [
        ({"converged": False}, {}, ValueError, "not converged"),
        (
            {"backend": "pbc-gdf-compcell-uhf"},
            {},
            NotImplementedError,
            "Gamma RHF compcell",
        ),
        (
            {"functional": "pbe"},
            {},
            NotImplementedError,
            "Gamma RHF compcell",
        ),
        (
            {"v_ne_backend": "grid"},
            {},
            NotImplementedError,
            "analytical FT Ewald V_ne",
        ),
        ({}, {"alpha_hf": 0.25}, ValueError, "alpha_hf=1.0"),
    ],
)
def test_low_level_gradient_route_enforces_rhf_boundary(
    result_update,
    gradient_kwargs,
    exception,
    message,
):
    """The module-level component API cannot omit unsupported terms."""
    from types import SimpleNamespace

    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient_rhf_gamma

    system, basis = _h2_12bohr_system()
    values = {
        "converged": True,
        "backend": "pbc-gdf-compcell",
        "functional": "",
        "apply_aft_correction": False,
        "v_ne_backend": "analytic_ft",
    }
    values.update(result_update)
    result = SimpleNamespace(**values)
    with pytest.raises(exception, match=message):
        compute_gdf_gradient_rhf_gamma(
            system,
            basis,
            result,
            aux_basis=basis,
            **gradient_kwargs,
        )


def test_public_gradient_route_rejects_grid_v_ne(monkeypatch):
    """The grid diagnostic cannot be relabeled as an FT gradient."""
    system, basis = _h2_12bohr_system()
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_BACKEND", "grid")
    with pytest.raises(NotImplementedError, match="analytical FT Ewald V_ne"):
        vq.run_pbc_gdf_rhf(
            system,
            basis,
            apply_aft_correction=False,
            compute_gradient=True,
        )


def test_direct_gradient_route_requires_scf_v_ne_cutoff():
    """A direct call cannot silently change the converged V_ne mesh."""
    from types import SimpleNamespace

    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient

    system, basis = _h2_12bohr_system()
    result = SimpleNamespace(
        converged=True,
        apply_aft_correction=False,
        backend="pbc-gdf-compcell",
        functional="",
        v_ne_backend="analytic_ft",
        v_ne_ke_cutoff=8.0,
    )
    with pytest.raises(ValueError, match="must match the converged SCF"):
        compute_gdf_gradient(system, basis, result, v_ne_ke_cutoff=9.0)


@pytest.mark.parametrize(
    "gradient_kwargs, message",
    [
        ({"aux_basis_name": "sto-3g"}, "aux_basis_name must match"),
        ({"compcell_eta": 1.0}, "compcell_eta must match"),
    ],
)
def test_direct_gradient_route_requires_scf_fit_provenance(
    gradient_kwargs,
    message,
):
    """A direct call cannot silently rebuild a different GDF fit."""
    from types import SimpleNamespace

    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient

    system, basis = _h2_12bohr_system()
    result = SimpleNamespace(
        converged=True,
        apply_aft_correction=False,
        backend="pbc-gdf-compcell",
        functional="",
        v_ne_backend="analytic_ft",
        v_ne_ke_cutoff=8.0,
        aux_basis_name="def2-svp-jk",
        aux_basis_fingerprint="retained",
        compcell_eta=0.25,
    )
    with pytest.raises(ValueError, match=message):
        compute_gdf_gradient(system, basis, result, **gradient_kwargs)


def test_direct_gradient_route_requires_scf_rcut_provenance():
    """A direct call cannot guess the SCF's compcell cutoff strategy."""
    from types import SimpleNamespace

    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient

    system, basis = _h2_12bohr_system()
    result = SimpleNamespace(
        converged=True,
        apply_aft_correction=False,
        backend="pbc-gdf-compcell",
        functional="",
        v_ne_backend="analytic_ft",
        v_ne_ke_cutoff=8.0,
        aux_basis_name="def2-svp-jk",
        aux_basis_fingerprint="retained",
        compcell_eta=1.0,
    )
    with pytest.raises(ValueError, match="rcut strategy"):
        compute_gdf_gradient(system, basis, result)


@pytest.mark.parametrize(
    "exxdiv, madelung, message",
    [
        ("none", 0.1, "requires a zero"),
        ("ewald", 0.0, "does not match the periodic cell"),
        ("ewald", 0.1, "does not match the periodic cell"),
        ("invalid", 0.0, "invalid exxdiv/Madelung"),
    ],
)
def test_direct_gradient_route_validates_exxdiv_provenance(
    exxdiv,
    madelung,
    message,
):
    """The wrapper uses the SCF's recorded exchange-divergence gauge."""
    from types import SimpleNamespace

    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient

    system, basis = _h2_12bohr_system()
    result = SimpleNamespace(
        converged=True,
        apply_aft_correction=False,
        backend="pbc-gdf-compcell",
        functional="",
        v_ne_backend="analytic_ft",
        v_ne_ke_cutoff=8.0,
        aux_basis_name="def2-svp-jk",
        aux_basis_fingerprint="retained",
        compcell_eta=1.0,
        gdf_rcut_strategy="pyscf_auto",
        gdf_rcut_precision=1e-8,
        gdf_linear_dep_threshold=1e-9,
        gdf_lattice_cutoff_bohr=12.0,
        gdf_nuclear_cutoff_bohr=12.0,
        gdf_fit_cutoff_2c=9.8479194000,
        gdf_fit_cutoff_3c=12.9364680326,
        exxdiv=exxdiv,
        madelung_constant=madelung,
    )
    with pytest.raises(ValueError, match=message):
        compute_gdf_gradient(system, basis, result)


def test_low_level_gradient_route_cannot_override_scf_madelung():
    """The component entry point cannot bypass exxdiv provenance."""
    from types import SimpleNamespace

    from vibeqc.madelung import madelung_constant_for_cell
    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient_rhf_gamma

    system, basis = _h2_12bohr_system()
    result = SimpleNamespace(
        converged=True,
        backend="pbc-gdf-compcell",
        functional="",
        apply_aft_correction=False,
        v_ne_backend="analytic_ft",
        exxdiv="ewald",
        madelung_constant=madelung_constant_for_cell(system),
    )
    with pytest.raises(ValueError, match="must match the converged SCF"):
        compute_gdf_gradient_rhf_gamma(
            system,
            basis,
            result,
            aux_basis=basis,
            madelung=0.0,
        )


def test_direct_gradient_rejects_changed_same_name_aux_basis(monkeypatch):
    """An auxiliary-basis name cannot stand in for content provenance."""
    from vibeqc._vibeqc_core import ShellInfo
    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient

    system, basis = _h2_12bohr_system()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = _lat_opts()
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    result = vq.run_pbc_gdf_rhf(
        system,
        basis,
        options=options,
        aux_basis="def2-svp-jk",
        gdf_method="compcell",
        compcell_eta=1.0,
        apply_aft_correction=False,
        gdf_linear_dep_threshold=1e-9,
    )
    assert result.converged

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    changed_shells = []
    for shell_idx, shell in enumerate(aux.shells()):
        exponents = list(shell.exponents)
        if shell_idx == 0:
            exponents[0] *= 1.1
        changed_shells.append(
            ShellInfo(
                int(shell.atom_index),
                int(shell.l),
                bool(shell.pure),
                exponents,
                list(shell.coefficients),
                list(shell.origin),
            )
        )
    changed_aux = vq.BasisSet(mol, changed_shells, aux.name, True)
    assert changed_aux.name == aux.name

    with pytest.raises(ValueError, match="auxiliary basis content"):
        compute_gdf_gradient(system, basis, result, aux_basis=changed_aux)


def test_uhf_gradient_m1_matches_rhf(monkeypatch):
    """UHF(M=1) analytic gradient == RHF analytic gradient (algebra gate).

    At multiplicity 1 the per-spin assembly must collapse exactly onto the
    closed-shell one: D_a = D_b = D/2 and C_a = C_b = C_occ make the two
    half-weighted exchange kernels sum to the single RHF kernel call, the
    occupation-1 W equal the closed-shell 2*C.eps.C^T, and the per-spin
    exxdiv weight xi*(D_a S D_a + D_b S D_b) equal the RHF xi/2*D S D.
    This pins every per-spin factor in compute_gdf_gradient_uhf_gamma
    against the FD-validated closed-shell route (G-PBC-002 milestone 1).
    """
    system, basis = _h2_12bohr_system()
    lat_opts = _lat_opts()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = lat_opts
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    common = dict(
        options=options,
        aux_basis="def2-svp-jk",
        gdf_method="compcell",
        compcell_eta=1.0,
        apply_aft_correction=False,
        gdf_linear_dep_threshold=1e-9,
        exxdiv="ewald",
        compute_gradient=True,
    )
    r_rhf = vq.run_pbc_gdf_rhf(system, basis, **common)
    assert r_rhf.converged
    # The public UHF wiring is released (G-PBC-002 milestone 3b): the
    # M = 1 algebra gate now runs for real and must collapse exactly
    # onto the closed-shell route.
    r_uhf = vq.run_pbc_gdf_uhf(system, basis, **common)
    assert r_uhf.converged
    assert r_uhf.gradient is not None
    np.testing.assert_allclose(
        np.asarray(r_uhf.gradient),
        np.asarray(r_rhf.gradient),
        rtol=0.0,
        atol=5e-7,
    )


def test_uhf_gradient_rejects_unsupported_routes():
    """The UHF gradient preview fails closed outside its envelope."""
    system, basis = _h2_12bohr_system()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = _lat_opts()
    with pytest.raises(NotImplementedError, match="MDF fit derivative"):
        vq.run_pbc_gdf_uhf(
            system, basis, options=options, aux_basis="def2-svp-jk",
            gdf_method="mdf", compute_gradient=True,
        )


def _run_rks_gdf_h2(functional, dz=0.0, compute_gradient=False):
    """KS (or RHF when functional is None) compcell/AFT-off SCF on the
    H2/12-bohr fixture, atom 0 displaced by ``dz`` along z."""
    box = 12.0
    c = box / 2.0
    atoms = [
        vq.Atom(1, [c, c, c - 0.7 + dz]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    system = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = (
        vq.PeriodicKSOptions() if functional else vq.PeriodicRHFOptions()
    )
    options.lattice_opts = _lat_opts()
    options.max_iter = 80
    options.conv_tol_energy = 1e-11
    result = vq.run_pbc_gdf_rhf(
        system,
        basis,
        options=options,
        functional=functional,
        aux_basis="def2-svp-jk",
        gdf_method="compcell",
        compcell_eta=1.0,
        apply_aft_correction=False,
        gdf_linear_dep_threshold=1e-9,
        exxdiv="ewald",
        compute_gradient=compute_gradient,
        progress=False,
    )
    assert result.converged
    return result


def test_rks_gdf_gradient_vs_fd_lda_pbe_pbe0(monkeypatch):
    """G-PBC-002 milestone 2: Γ KS compcell gradient matches FD.

    Three functionals cover the three KS-specific terms: lda exercises
    the alpha_hf=0 no-K branch + LDA XC Pulay, pbe the GGA s-piece of
    ``xc_lattice_gradient_contribution``, pbe0 the alpha-scaled K and
    exxdiv-shift W-term.

    Tolerance history: this test originally used the landed 5e-3 gate
    plus a differential gate against the RHF baseline, because every
    route shared a flat, then-unidentified ~3.394e-3 residual. The
    milestone-3b identification (the truncated-direct-sum nuclear
    gradient vs the Ewald-gauge e_nuc) removed that floor, so each
    functional is now gated absolutely at 1e-6 (observed ~1e-8).
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4

    for functional in ("lda", "pbe", "pbe0"):
        r = _run_rks_gdf_h2(functional, compute_gradient=True)
        assert r.backend == "pbc-gdf-compcell-rks"
        assert r.gradient is not None
        grad = np.asarray(r.gradient)
        # Newton's third law.
        assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-10)
        fd = (
            _run_rks_gdf_h2(functional, dz=+delta).energy
            - _run_rks_gdf_h2(functional, dz=-delta).energy
        ) / (2.0 * delta)
        assert abs(grad[0, 2] - fd) < 1e-6, (
            f"{functional}: analytic={grad[0, 2]:.8f} FD={fd:.8f} "
            f"diff={abs(grad[0, 2] - fd):.3e}"
        )


def test_rks_gradient_entry_rejects_rhf_result():
    """The KS gradient entry point fails closed on a plain RHF result."""
    from types import SimpleNamespace

    from vibeqc.periodic_gdf_gradient import compute_gdf_gradient_rks_gamma

    system, basis = _h2_12bohr_system()
    result = SimpleNamespace(
        converged=True,
        backend="pbc-gdf-compcell",
        functional="",
        apply_aft_correction=False,
        v_ne_backend="analytic_ft",
    )
    with pytest.raises(NotImplementedError, match="closed-shell KS"):
        compute_gdf_gradient_rks_gamma(
            system, basis, result, aux_basis=basis
        )


def test_rks_gradient_rejects_unsupported_routes():
    """The KS gradient preview keeps the RHF wiring's fail-closed knobs."""
    system, basis = _h2_12bohr_system()
    options = vq.PeriodicKSOptions()
    options.lattice_opts = _lat_opts()
    common = dict(
        options=options,
        functional="pbe",
        aux_basis="def2-svp-jk",
        compute_gradient=True,
    )
    with pytest.raises(NotImplementedError, match="MDF fit derivative"):
        vq.run_pbc_gdf_rhf(system, basis, gdf_method="mdf", **common)


def test_aft_2c_weighted_gradient_vs_fd():
    """G-PBC-002 milestone 3a: the weighted 2c AFT derivative is exact.

    Isolation gate: build the SCF's exact fused (modrho-aux + chg) basis
    on the H2 fixture, contract j2c_p with a fixed random weight, and
    central-difference that scalar against the analytic
    ``_compcell_aft_correction_2c_gradient_weighted`` for every atom and
    axis. The only atom-position dependence of j2c_p is the exp(-iG.R)
    phase of each shell FT (mesh, Coulomb kernel, and radial content are
    lattice/exponent-only), so the analytic derivative must match FD at
    the kernel tolerance, in both FT conventions.
    """
    from vibeqc.aux_basis import (
        _compcell_aft_correction,
        _compcell_aft_correction_2c_gradient_weighted,
        make_aux_basis_set,
        make_modrho_aux_basis,
        make_compensating_basis,
        make_fused_basis,
    )

    eta = 1.0
    delta = 1e-5

    def fused_for(system):
        mol = system.unit_cell_molecule()
        aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
        modrho = make_modrho_aux_basis(aux, mol)
        chg = make_compensating_basis(modrho, mol, eta=eta)
        fused = make_fused_basis(modrho, chg, mol)
        return fused, modrho.nbasis

    def displaced(a, d, step):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        base[a][d] += step
        atoms = [vq.Atom(1, xyz) for xyz in base]
        return vq.PeriodicSystem(3, np.eye(3) * box, atoms)

    system = displaced(0, 0, 0.0)
    fused, n_aux = fused_for(system)
    n_chg = fused.nbasis - n_aux
    rng = np.random.default_rng(7)
    W = rng.standard_normal((n_chg, fused.nbasis))

    for convention in ("libint", "libcint"):
        grad = _compcell_aft_correction_2c_gradient_weighted(
            fused,
            n_aux,
            system,
            eta=eta,
            weight=W,
            ft_convention=convention,
        )
        assert grad.shape == (2, 3)
        for a in range(2):
            for d in range(3):
                vals = []
                for step in (+delta, -delta):
                    s = displaced(a, d, step)
                    f_s, n_aux_s = fused_for(s)
                    assert n_aux_s == n_aux
                    j2c = _compcell_aft_correction(
                        f_s,
                        n_aux,
                        s,
                        eta=eta,
                        ft_convention=convention,
                    )
                    vals.append(float(np.sum(W * j2c)))
                fd = (vals[0] - vals[1]) / (2.0 * delta)
                assert abs(fd - grad[a, d]) < 1e-7, (
                    f"{convention}: atom {a} axis {d}: "
                    f"analytic={grad[a, d]:.12f} FD={fd:.12f} "
                    f"diff={abs(fd - grad[a, d]):.3e}"
                )


def test_aft_3c_weighted_gradient_vs_fd():
    """G-PBC-002 milestone 3a/b: the weighted 3c AFT derivative is exact.

    Isolation gate, sibling of ``test_aft_2c_weighted_gradient_vs_fd``:
    contract j3c_p with a fixed random weight and central-difference the
    scalar against the analytic
    ``_compcell_aft_correction_3c_gradient_weighted`` for every atom and
    axis, in both FT conventions. The derivative has two pieces — the
    chg-function phase and the Bloch AO-pair FT centres (via the
    existing weighted C++ kernel) — and the mesh/kernel/cell list carry
    no atom-position dependence, so both pieces together must match FD
    at the kernel tolerance.
    """
    from vibeqc.aux_basis import (
        _compcell_aft_correction_3c,
        _compcell_aft_correction_3c_gradient_weighted,
        make_aux_basis_set,
        make_modrho_aux_basis,
        make_compensating_basis,
        make_fused_basis,
    )

    eta = 1.0
    delta = 1e-5
    lat = _lat_opts()

    def bases_for(system):
        mol = system.unit_cell_molecule()
        aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
        modrho = make_modrho_aux_basis(aux, mol)
        chg = make_compensating_basis(modrho, mol, eta=eta)
        fused = make_fused_basis(modrho, chg, mol)
        ao = vq.BasisSet(mol, "sto-3g")
        return fused, modrho.nbasis, ao

    def displaced(a, d, step):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        base[a][d] += step
        atoms = [vq.Atom(1, xyz) for xyz in base]
        return vq.PeriodicSystem(3, np.eye(3) * box, atoms)

    system = displaced(0, 0, 0.0)
    fused, n_aux, ao = bases_for(system)
    n_chg = fused.nbasis - n_aux
    rng = np.random.default_rng(11)
    W3 = rng.standard_normal((n_chg, ao.nbasis, ao.nbasis))

    for convention in ("libint", "libcint"):
        grad = _compcell_aft_correction_3c_gradient_weighted(
            fused,
            ao,
            n_aux,
            system,
            eta=eta,
            weight=W3,
            ft_convention=convention,
            lat_opts=lat,
        )
        assert grad.shape == (2, 3)
        for a in range(2):
            for d in range(3):
                vals = []
                for step in (+delta, -delta):
                    s = displaced(a, d, step)
                    f_s, n_aux_s, ao_s = bases_for(s)
                    assert n_aux_s == n_aux
                    j3c = _compcell_aft_correction_3c(
                        f_s,
                        ao_s,
                        n_aux,
                        s,
                        eta=eta,
                        ft_convention=convention,
                        lat_opts=lat,
                    )
                    vals.append(float(np.sum(W3 * j3c)))
                fd = (vals[0] - vals[1]) / (2.0 * delta)
                assert abs(fd - grad[a, d]) < 1e-7, (
                    f"{convention}: atom {a} axis {d}: "
                    f"analytic={grad[a, d]:.12f} FD={fd:.12f} "
                    f"diff={abs(fd - grad[a, d]):.3e}"
                )


def test_full_scf_gradient_vs_fd_aft_on(monkeypatch):
    """G-PBC-002 milestone 3b: gradients on the production AFT-on fit.

    The compcell AFT metric derivative (2c + 3c) makes
    ``compute_gradient=True`` valid on the default
    ``apply_aft_correction=True`` surface for RHF, closed-shell KS, and
    the newly released public UHF wiring. Each route's analytic
    z-component must match a central difference of the same driver's
    AFT-on energy at 1e-6 (observed ~3-6e-9 on this control). The RHF
    energy here differs from the AFT-off control by ~70 mHa, so a
    missing or wrong AFT force term cannot hide inside the gate.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4
    lat = _lat_opts()

    def build(dz=0.0, mult=1):
        box = 12.0
        c = box / 2.0
        atoms = [
            vq.Atom(1, [c, c, c - 0.7 + dz]),
            vq.Atom(1, [c, c, c + 0.7]),
        ]
        system = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
        if mult != 1:
            system.multiplicity = mult
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        return system, basis

    common = dict(
        aux_basis="def2-svp-jk",
        gdf_method="compcell",
        compcell_eta=1.0,
        apply_aft_correction=True,
        gdf_linear_dep_threshold=1e-9,
        exxdiv="ewald",
        progress=False,
    )

    def run_rhf_like(functional, dz, grad):
        system, basis = build(dz)
        options = (
            vq.PeriodicKSOptions() if functional else vq.PeriodicRHFOptions()
        )
        options.lattice_opts = lat
        options.max_iter = 80
        options.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_rhf(
            system, basis, options=options, functional=functional,
            compute_gradient=grad, **common,
        )

    def run_uhf(dz, grad):
        system, basis = build(dz, mult=3)
        options = vq.PeriodicRHFOptions()
        options.lattice_opts = lat
        options.max_iter = 80
        options.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_uhf(
            system, basis, options=options, compute_gradient=grad, **common,
        )

    cases = [
        ("RHF", lambda dz, grad=False: run_rhf_like(None, dz, grad)),
        ("pbe0", lambda dz, grad=False: run_rhf_like("pbe0", dz, grad)),
        ("UHF-triplet", lambda dz, grad=False: run_uhf(dz, grad)),
    ]
    for label, runner in cases:
        r = runner(0.0, grad=True)
        assert r.converged
        assert r.gradient is not None
        # The closed-shell results retain the AFT flag; the UHF result
        # has no retention field (its in-driver gradient consumes the
        # SCF's fit state directly).
        assert bool(getattr(r, "apply_aft_correction", True)) is True
        grad = np.asarray(r.gradient)
        assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-10)
        fd = (runner(+delta).energy - runner(-delta).energy) / (2.0 * delta)
        assert abs(grad[0, 2] - fd) < 1e-6, (
            f"{label}: analytic={grad[0, 2]:.10f} FD={fd:.10f} "
            f"diff={abs(grad[0, 2] - fd):.3e}"
        )


def test_uks_gdf_gradient_m1_and_triplet_fd(monkeypatch):
    """G-PBC-002 milestone 4: Γ UKS compcell gradient, both gates.

    Gate 1 (algebra): UKS(M=1, functional) must collapse onto the
    FD-validated RKS gradient at machine precision — this pins the
    alpha_hf-scaled per-spin exchange, the per-spin exxdiv W term, and
    the spin-polarised XC Pulay against the closed-shell route.
    Gate 2 (physics): the H2 triplet passes FD on the production AFT-on
    surface at 1e-6 for LDA (alpha = 0, no exchange gradient) and PBE0
    (alpha-scaled exchange + shift); observed 4e-8 / 4e-7.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4
    lat = _lat_opts()

    def build(dz=0.0, mult=1):
        box = 12.0
        c = box / 2.0
        atoms = [
            vq.Atom(1, [c, c, c - 0.7 + dz]),
            vq.Atom(1, [c, c, c + 0.7]),
        ]
        system = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
        if mult != 1:
            system.multiplicity = mult
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        return system, basis

    common = dict(
        aux_basis="def2-svp-jk",
        gdf_method="compcell",
        compcell_eta=1.0,
        apply_aft_correction=True,
        gdf_linear_dep_threshold=1e-9,
        exxdiv="ewald",
        progress=False,
    )

    def run_uks(functional, dz=0.0, mult=3, grad=False):
        system, basis = build(dz, mult)
        options = vq.PeriodicKSOptions()
        options.lattice_opts = lat
        options.max_iter = 100
        options.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_uks(
            system, basis, options=options, functional=functional,
            compute_gradient=grad, **common,
        )

    def run_rks(functional, dz=0.0, grad=False):
        system, basis = build(dz, 1)
        options = vq.PeriodicKSOptions()
        options.lattice_opts = lat
        options.max_iter = 100
        options.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_rhf(
            system, basis, options=options, functional=functional,
            compute_gradient=grad, **common,
        )

    for functional in ("lda", "pbe0"):
        ru = run_uks(functional, mult=1, grad=True)
        rr = run_rks(functional, grad=True)
        assert ru.converged and rr.converged
        assert ru.gradient is not None
        np.testing.assert_allclose(
            np.asarray(ru.gradient),
            np.asarray(rr.gradient),
            rtol=0.0,
            atol=1e-12,
        )

    for functional in ("lda", "pbe0"):
        r = run_uks(functional, grad=True)
        assert r.converged
        grad = np.asarray(r.gradient)
        assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-10)
        fd = (
            run_uks(functional, dz=+delta).energy
            - run_uks(functional, dz=-delta).energy
        ) / (2.0 * delta)
        assert abs(grad[0, 2] - fd) < 1e-6, (
            f"{functional}: analytic={grad[0, 2]:.10f} FD={fd:.10f} "
            f"diff={abs(grad[0, 2] - fd):.3e}"
        )


def test_uks_gradient_rejects_unsupported_routes():
    """The UKS gradient preview fails closed outside its envelope."""
    system, basis = _h2_12bohr_system()
    options = vq.PeriodicKSOptions()
    options.lattice_opts = _lat_opts()
    with pytest.raises(NotImplementedError, match="MDF fit derivative"):
        vq.run_pbc_gdf_uks(
            system, basis, options=options, functional="pbe",
            aux_basis="def2-svp-jk", gdf_method="mdf",
            compute_gradient=True,
        )
    # (The finite-temperature smearing reject was lifted 2026-07-30:
    # Fermi-Dirac smeared gradients are the Mermin free-energy forces —
    # see the smeared FD gates below. Non-Fermi-Dirac flavors still
    # fail closed: test_gamma_gradient_rejects_non_fermi_dirac_flavor.)


def test_rsgdf_weighted_2c_metric_gradient_vs_fd():
    """G-PBC-002 milestone 6 (Item-2 rung 1): the rsgdf weighted 2c
    metric derivative is exact.

    Isolation gate, full-aux sibling of
    ``test_aft_2c_weighted_gradient_vs_fd``: contract the dense-mesh
    rsgdf 2c metric (base + high-|G| tail, exactly the
    ``build_lpq_native_fft`` formula) with a fixed random weight and
    central-difference that scalar against the analytic
    ``_rsgdf_weighted_2c_metric_gradient`` for every atom and axis. The
    modrho-rescaled aux is rebuilt per displaced geometry, exactly as a
    future rsgdf gradient cache will do.
    """
    from vibeqc.aux_basis import (
        _rsgdf_weighted_2c_metric_gradient,
        make_aux_basis_set,
        make_modrho_aux_basis,
        rsgdf_aux_fourier_transform,
        rsgdf_dense_g_mesh,
    )

    ke = 60.0
    tail_ke = 120.0
    delta = 1e-5

    def modrho_for(system):
        mol = system.unit_cell_molecule()
        aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
        return make_modrho_aux_basis(aux, mol)

    def displaced(a, d, step):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        base[a][d] += step
        atoms = [vq.Atom(1, xyz) for xyz in base]
        return vq.PeriodicSystem(3, np.eye(3) * box, atoms)

    def metric_value(system, W):
        """Base+tail rsgdf 2c metric contracted with W (the builder's
        exact formula)."""
        aux = modrho_for(system)
        V = float(abs(np.linalg.det(np.asarray(system.lattice))))
        total = 0.0
        blocks = [rsgdf_dense_g_mesh(system, ke)]
        G_tail_all = rsgdf_dense_g_mesh(system, tail_ke)
        norms = np.linalg.norm(G_tail_all, axis=1)
        blocks.append(G_tail_all[norms > float(np.sqrt(2.0 * ke))])
        for G_block in blocks:
            G2 = (G_block**2).sum(axis=1)
            nz = G2 > 0
            Gc = G_block[nz]
            if Gc.shape[0] == 0:
                continue
            coul = (4.0 * np.pi) / G2[nz] / V
            F = rsgdf_aux_fourier_transform(aux, Gc)
            M = np.real(F.conj() @ (F * coul[None, :]).T)
            total += float(np.sum(W * M))
        return total

    system = displaced(0, 0, 0.0)
    aux0 = modrho_for(system)
    rng = np.random.default_rng(13)
    W = rng.standard_normal((aux0.nbasis, aux0.nbasis))

    grad = _rsgdf_weighted_2c_metric_gradient(
        aux0,
        system,
        ke_cutoff=ke,
        weight=W,
        tail_ke_cutoff=tail_ke,
    )
    assert grad.shape == (2, 3)
    for a in range(2):
        for d in range(3):
            vp = metric_value(displaced(a, d, +delta), W)
            vm = metric_value(displaced(a, d, -delta), W)
            fd = (vp - vm) / (2.0 * delta)
            assert abs(fd - grad[a, d]) < 1e-7, (
                f"atom {a} axis {d}: analytic={grad[a, d]:.12f} "
                f"FD={fd:.12f} diff={abs(fd - grad[a, d]):.3e}"
            )


def test_rsgdf_weighted_3c_tensor_gradient_vs_fd():
    """G-PBC-002 milestone 6 (Item-2 rung 2): the rsgdf weighted 3c
    tensor derivative is exact.

    Isolation gate, dense-mesh sibling of
    ``test_aft_3c_weighted_gradient_vs_fd``: contract the
    ``build_lpq_native_fft`` 3c tensor (Bloch pair FT over the lat_opts
    cell list, rsgdf pair scales, base + tail) with a fixed random
    weight and central-difference that scalar against the analytic
    ``_rsgdf_weighted_3c_tensor_gradient`` for every atom and axis.
    """
    from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch
    from vibeqc._vibeqc_core import direct_lattice_cells
    from vibeqc.aux_basis import (
        _ao_scales_for_rsgdf,
        _rsgdf_weighted_3c_tensor_gradient,
        make_aux_basis_set,
        make_modrho_aux_basis,
        rsgdf_aux_fourier_transform,
        rsgdf_dense_g_mesh,
    )

    ke = 40.0
    tail_ke = 80.0
    delta = 1e-5
    lat = _lat_opts()

    def bases_for(system):
        mol = system.unit_cell_molecule()
        aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
        return make_modrho_aux_basis(aux, mol), vq.BasisSet(mol, "sto-3g")

    def displaced(a, d, step):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        base[a][d] += step
        atoms = [vq.Atom(1, xyz) for xyz in base]
        return vq.PeriodicSystem(3, np.eye(3) * box, atoms)

    def tensor_value(system, W3):
        aux, ao = bases_for(system)
        n_orb = ao.nbasis
        V = float(abs(np.linalg.det(np.asarray(system.lattice))))
        cells = direct_lattice_cells(system, float(lat.cutoff_bohr))
        R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
        scales = _ao_scales_for_rsgdf(ao)
        pair_scales = np.outer(scales, scales)
        total = 0.0
        blocks = [rsgdf_dense_g_mesh(system, ke)]
        G_tail_all = rsgdf_dense_g_mesh(system, tail_ke)
        norms = np.linalg.norm(G_tail_all, axis=1)
        blocks.append(G_tail_all[norms > float(np.sqrt(2.0 * ke))])
        for G_block in blocks:
            G2 = (G_block**2).sum(axis=1)
            nz = G2 > 0
            Gc = np.ascontiguousarray(G_block[nz])
            if Gc.shape[0] == 0:
                continue
            coul = (4.0 * np.pi) / G2[nz] / V
            F = rsgdf_aux_fourier_transform(aux, Gc)
            rho = ao_pair_fourier_transform_bloch(
                ao, Gc, R_g, k_cart=np.zeros(3)
            ) * pair_scales[:, :, None]
            aux_w = F.conj() * coul[None, :]
            T = np.real(np.einsum("Pk,mnk->Pmn", aux_w, rho))
            total += float(np.sum(W3 * T))
        return total

    system = displaced(0, 0, 0.0)
    aux0, ao0 = bases_for(system)
    rng = np.random.default_rng(17)
    W3 = rng.standard_normal((aux0.nbasis, ao0.nbasis, ao0.nbasis))
    # μν-symmetric weight, as every gradient-algebra consumer supplies
    # (D-outer and C.eta.C^T weights are symmetric in the pair indices).
    W3 = 0.5 * (W3 + W3.transpose(0, 2, 1))

    grad = _rsgdf_weighted_3c_tensor_gradient(
        aux0,
        ao0,
        system,
        ke_cutoff=ke,
        weight=W3,
        lat_opts=lat,
        tail_ke_cutoff=tail_ke,
    )
    assert grad.shape == (2, 3)
    for a in range(2):
        for d in range(3):
            vp = tensor_value(displaced(a, d, +delta), W3)
            vm = tensor_value(displaced(a, d, -delta), W3)
            fd = (vp - vm) / (2.0 * delta)
            assert abs(fd - grad[a, d]) < 1e-7, (
                f"atom {a} axis {d}: analytic={grad[a, d]:.12f} "
                f"FD={fd:.12f} diff={abs(fd - grad[a, d]):.3e}"
            )


def _rsgdf_cache_for(system, basis, ke, tail_ke, lat, thr):
    from vibeqc.aux_basis import make_aux_basis_set, make_modrho_aux_basis
    from vibeqc.periodic_gdf_gradient import _build_rsgdf_gradient_cache

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    return _build_rsgdf_gradient_cache(
        system, basis, modrho,
        ke_cutoff=ke, tail_ke_cutoff=tail_ke, lat_opts=lat,
        linear_dep_thr=thr,
    )


def test_rsgdf_j_and_k_gradient_fixed_density_vs_fd():
    """G-PBC-002 milestone 6 (rung 3): rsgdf DF-J and DF-K gradients.

    The compcell J/K weight algebra on the rsgdf fit (identity
    compensation, ABSOLUTE eigen threshold), contracted with the
    reciprocal-space M/T derivative kernels from rungs 1+2. Fixed
    density / fixed C_occ: FD rebuilds the rsgdf metric+tensor at
    displaced geometries and re-evaluates E_J = 1/2 rho.f(M).rho and
    E_K = -alpha/4 Tr[D K_fit[D]] with the same absolute threshold.
    A coarse threshold forces at least one dropped mode so the
    Frechet retained/dropped response is exercised, exactly like the
    compcell gates.
    """
    from vibeqc.periodic_gdf_gradient import (
        _compute_j_gradient_rsgdf,
        _compute_k_gradient_rsgdf,
    )

    ke = 40.0
    tail_ke = 80.0
    thr = 1.0e-3  # coarse ABSOLUTE threshold: forces dropped modes
    delta = 1e-5
    lat = _lat_opts()
    system, basis = _h2_12bohr_system()

    rng = np.random.default_rng(19)
    D = rng.standard_normal((basis.nbasis, basis.nbasis))
    D = 0.5 * (D + D.T)
    C_occ, _ = np.linalg.qr(rng.normal(size=(basis.nbasis, 2)))
    D_k = 2.0 * C_occ @ C_occ.T
    alpha = 1.0

    cache = _rsgdf_cache_for(system, basis, ke, tail_ke, lat, thr)
    assert cache.n_fit < cache.n_aux  # dropped modes present
    g_J = _compute_j_gradient_rsgdf(system, basis, D, cache)
    g_K = _compute_k_gradient_rsgdf(system, basis, D_k, C_occ, alpha, cache)

    def displaced(a, d, step):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        base[a][d] += step
        atoms = [vq.Atom(1, xyz) for xyz in base]
        s = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
        return s, vq.BasisSet(s.unit_cell_molecule(), "sto-3g")

    def energies(sys_d, basis_d):
        c = _rsgdf_cache_for(sys_d, basis_d, ke, tail_ke, lat, thr)
        T_flat = c.T.reshape(c.n_aux, -1)
        rho = T_flat @ D.ravel()
        U = c.eigvecs
        gamma = U @ (c.inverse_eigvals * (U.T @ rho))
        e_j = 0.5 * float(rho @ gamma)
        # Fitted Lpq with the same absolute threshold; E_K as the
        # compcell gate builds it.
        keep = c.keep_mask
        Lpq = (
            (U[:, keep].T @ T_flat)
            / np.sqrt(c.eigvals[keep])[:, None]
        ).reshape(-1, c.n_orb, c.n_orb)
        from vibeqc.pbc_gdf import _build_k_from_lpq

        K = _build_k_from_lpq(Lpq, D_k)
        e_k = -0.25 * alpha * float(np.einsum("ij,ij->", D_k, K))
        return e_j, e_k

    for a in range(2):
        for d in range(3):
            sp, bp = displaced(a, d, +delta)
            sm, bm = displaced(a, d, -delta)
            ejp, ekp = energies(sp, bp)
            ejm, ekm = energies(sm, bm)
            fd_j = (ejp - ejm) / (2.0 * delta)
            fd_k = (ekp - ekm) / (2.0 * delta)
            assert abs(fd_j - g_J[a, d]) < 1e-7, (
                f"J atom {a} axis {d}: analytic={g_J[a, d]:.10f} "
                f"FD={fd_j:.10f} diff={abs(fd_j - g_J[a, d]):.3e}"
            )
            assert abs(fd_k - g_K[a, d]) < 1e-7, (
                f"K atom {a} axis {d}: analytic={g_K[a, d]:.10f} "
                f"FD={fd_k:.10f} diff={abs(fd_k - g_K[a, d]):.3e}"
            )


def test_rsgdf_full_scf_gradient_vs_fd(monkeypatch):
    """G-PBC-002 milestone 6 (rung 4): the production-path rsgdf RHF
    gradient matches FD.

    run_pbc_gdf_rhf(gdf_method='rsgdf', compute_gradient=True) — the
    PySCF-µHa-parity production Γ route — assembles the shared
    one-electron/W/nn/V_ne terms with the rung-3 rsgdf J/K. Central
    difference of the same driver's energy at ke=60 (observed ~4e-9
    against the 1e-6 gate).
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4
    lat = _lat_opts()

    def run(dz=0.0, grad=False):
        box = 12.0
        c = box / 2.0
        s = vq.PeriodicSystem(3, np.eye(3) * box,
            [vq.Atom(1, [c, c, c - 0.7 + dz]), vq.Atom(1, [c, c, c + 0.7])])
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = lat
        o.max_iter = 80
        o.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_rhf(
            s, b, options=o, aux_basis="def2-svp-jk", gdf_method="rsgdf",
            rsgdf_ke_cutoff=60.0, exxdiv="ewald",
            compute_gradient=grad, progress=False)

    r = run(grad=True)
    assert r.converged
    assert r.gradient is not None
    assert r.backend.startswith("pbc-gdf-rsgdf")
    grad = np.asarray(r.gradient)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-10)
    fd = (run(dz=+delta).energy - run(dz=-delta).energy) / (2.0 * delta)
    assert abs(grad[0, 2] - fd) < 1e-6, (
        f"analytic={grad[0, 2]:.10f} FD={fd:.10f} "
        f"diff={abs(grad[0, 2] - fd):.3e}"
    )


@pytest.mark.slow
def test_rsgdf_full_scf_gradient_vs_fd_production_default(monkeypatch):
    """Rung-4 production-default smoke: the same gate at ke=200
    (observed ~4e-9; the gradient costs ~35 s on 18 cores)."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4
    lat = _lat_opts()

    def run(dz=0.0, grad=False):
        box = 12.0
        c = box / 2.0
        s = vq.PeriodicSystem(3, np.eye(3) * box,
            [vq.Atom(1, [c, c, c - 0.7 + dz]), vq.Atom(1, [c, c, c + 0.7])])
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = lat
        o.max_iter = 80
        o.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_rhf(
            s, b, options=o, aux_basis="def2-svp-jk", gdf_method="rsgdf",
            exxdiv="ewald", compute_gradient=grad, progress=False)

    r = run(grad=True)
    assert r.converged
    grad = np.asarray(r.gradient)
    fd = (run(dz=+delta).energy - run(dz=-delta).energy) / (2.0 * delta)
    assert abs(grad[0, 2] - fd) < 1e-6


def test_rsgdf_gradient_rejects_unsupported_envelope():
    """The rsgdf gradient's remaining fail-closed pin: mdf.

    The tailed dense-core hold was lifted 2026-07-29 (the dense-core FD
    "inconsistency" was an e_nuc truncation artefact in the energy; see
    the closed dense-core entry in HANDOVER_OPEN_BUGS_V015.md), and the
    Schwarz-screened hold was lifted 2026-07-30 (the gradient cache
    mirrors the SCF's pair mask — screened FD gates below).
    """
    system, basis = _h2_12bohr_system()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = _lat_opts()
    common = dict(options=options, aux_basis="def2-svp-jk",
                  compute_gradient=True)
    with pytest.raises(NotImplementedError, match="MDF fit derivative"):
        vq.run_pbc_gdf_rhf(system, basis, gdf_method="mdf", **common)


def _rsgdf_screen_keep_mask(system, basis, modrho, *, ke, thr, lat, q=None):
    """The SCF's Schwarz AO-pair keep mask at one momentum transfer.

    Recomputes exactly the mask input the screened fit derives it from
    (base-mesh metric on the shifted q sphere) and calls THE SAME
    ``_rsgdf_fit_pair_keep_mask`` the SCF and gradient-cache builders
    call — used by the screened FD gates to pin that the mask is
    non-trivial and IDENTICAL at the displaced FD geometries (the
    fixed-mask requirement: a mask flip across ±h would differentiate
    a different objective on each side)."""
    from vibeqc._vibeqc_core import direct_lattice_cells
    from vibeqc.aux_basis import (
        _rsgdf_fit_pair_keep_mask,
        _rsgdf_shifted_dense_g_mesh,
        rsgdf_aux_fourier_transform,
    )

    q = np.zeros(3) if q is None else np.asarray(q, dtype=float).reshape(3)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    cells = direct_lattice_cells(system, float(lat.cutoff_bohr))
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    Gq = _rsgdf_shifted_dense_g_mesh(system, q, float(ke))
    Gq2 = (Gq**2).sum(axis=1)
    nz = Gq2 > 1e-12
    Gq, Gq2 = Gq[nz], Gq2[nz]
    coul = (4.0 * np.pi) / Gq2 / V
    M = np.zeros((modrho.nbasis, modrho.nbasis), dtype=np.complex128)
    for lo in range(0, Gq.shape[0], 65536):
        F = rsgdf_aux_fourier_transform(modrho, Gq[lo : lo + 65536])
        M += (F.conj() * coul[lo : lo + 65536][None, :]) @ F.T
    return _rsgdf_fit_pair_keep_mask(
        basis, R_g, Gq2, coul, M,
        fit_screen_threshold=float(thr), fit_pair_list=None, progress=None,
    )


# Mid-gap Schwarz thresholds for the screened gates, valid at BOTH
# momentum transfers of the (2,1,1) mesh (the bounds are q-dependent —
# larger at the BZ-boundary transfer where the Coulomb weights peak).
# H2/STO-3G at ke=60: inter-atom AO pairs bound 0.368 (q=0) / 0.480
# (q=-b1/2), on-atom 0.596 / 0.764 — 0.54 drops the 2 inter-atom pairs
# of 4 in every group with min |bound - threshold| margin ~0.056, which
# dwarfs any h=1e-4 displacement effect on the bounds. LiH (p shells):
# 0.9 drops the <= 0.65 (q=0) / <= 0.73 (q=-b1/2) levels and keeps
# >= 1.026, min margin ~0.13.
_H2_SCREEN_THR = 0.54
_LIH_SCREEN_THR = 0.9


def test_rsgdf_screened_gradient_cache_bit_consistent_with_scf_fit():
    """Screened-fit envelope gate (2026-07-30): the Γ rsgdf gradient
    cache mirrors the SCF's Schwarz-screened fit bit-for-bit.

    The screened Γ SCF fit is ``build_lpq_native_fft`` delegating to
    the Bloch builder at ``k = 0``; the cache mirrors that route (same
    mask builder on the same base-mesh metric, masked pair-FT zeroing
    at the identical sweep point, q = 0 real projection, no T
    symmetrisation). Reconstructing Lpq from the cache through the
    SCF's own complex fit contraction must reproduce the SCF Lpq
    EXACTLY (np.array_equal — the M6-rung-9 lesson: any fit mismatch
    is Fréchet-amplified near threshold), on a config where the screen
    actually drops pairs."""
    from vibeqc.aux_basis import build_lpq_native_fft
    from vibeqc.periodic_gdf_gradient import _build_rsgdf_gradient_cache

    system, basis = _h2_12bohr_system()
    lat = _lat_opts()
    ke = 60.0
    ld_thr = 1e-9
    thr = _H2_SCREEN_THR

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)

    cache = _build_rsgdf_gradient_cache(
        system, basis, modrho,
        ke_cutoff=ke, tail_ke_cutoff=None, lat_opts=lat,
        linear_dep_thr=ld_thr, fit_screen_threshold=thr,
    )
    # Non-trivial mask: some pairs masked, some kept (H2/STO-3G at
    # thr=0.54 drops exactly the 2 inter-atom AO pairs of 4).
    assert cache.pair_keep_ao is not None
    n_masked = int(np.count_nonzero(~cache.pair_keep_ao))
    n_total = basis.nbasis * basis.nbasis
    assert 0 < n_masked < n_total
    assert n_masked == 2 and n_total == 4

    lpq_scf = build_lpq_native_fft(
        system, basis, modrho,
        ke_cutoff=ke, lat_opts=lat, linear_dep_thr=ld_thr,
        fit_screen_threshold=thr,
    )
    assert cache.n_fit == int(lpq_scf.shape[0])
    # Masked pairs are exact hard zeros in the SCF fit.
    assert np.all(lpq_scf[:, ~cache.pair_keep_ao] == 0.0)

    # The SCF's screened Γ Lpq is np.real of the complex Bloch fit; at
    # Γ the eigenvectors are real, so re-running the fit contraction in
    # complex arithmetic on the cache's real-projected T reproduces it
    # bit-for-bit (the zero-imaginary products cannot perturb the real
    # accumulation).
    keep = cache.keep_mask
    T_flat = cache.T.reshape(cache.n_aux, -1).astype(np.complex128)
    lpq_cache = np.real(
        (cache.eigvecs[:, keep].conj().T @ T_flat)
        / np.sqrt(cache.eigvals[keep])[:, None]
    ).reshape(cache.n_fit, cache.n_orb, cache.n_orb)
    assert np.array_equal(lpq_cache, lpq_scf)

    # The mask helper used by the FD gates reproduces the cache's mask.
    mask = _rsgdf_screen_keep_mask(
        system, basis, modrho, ke=ke, thr=thr, lat=lat
    )
    assert np.array_equal(mask, cache.pair_keep_ao)


def test_rsgdf_screened_full_scf_gradient_vs_fd(monkeypatch):
    """Screened-fit envelope gate (2026-07-30): the production-path
    screened rsgdf RHF gradient matches FD AT FIXED MASK.

    Same H2/ke=60 anchor as ``test_rsgdf_full_scf_gradient_vs_fd`` with
    ``fit_screen_threshold=0.54`` (drops the 2 inter-atom AO pairs of
    4). The analytic gradient differentiates the screened objective at
    the base geometry's mask, so the gate additionally pins that the
    Schwarz mask is non-trivial and IDENTICAL at the three FD
    geometries (mid-gap threshold: min |bound - threshold| ~ 0.17
    versus an O(1e-4) displacement effect). Observed ~2.7e-9 against
    the 1e-6 gate."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4
    lat = _lat_opts()
    thr = _H2_SCREEN_THR

    def build(dz=0.0):
        box = 12.0
        c = box / 2.0
        s = vq.PeriodicSystem(3, np.eye(3) * box,
            [vq.Atom(1, [c, c, c - 0.7 + dz]), vq.Atom(1, [c, c, c + 0.7])])
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        return s, b

    def run(dz=0.0, grad=False):
        s, b = build(dz)
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = lat
        o.max_iter = 80
        o.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_rhf(
            s, b, options=o, aux_basis="def2-svp-jk", gdf_method="rsgdf",
            rsgdf_ke_cutoff=60.0, exxdiv="ewald",
            fit_screen_threshold=thr,
            compute_gradient=grad, progress=False)

    # Fixed-mask pin: the Schwarz keep mask is non-trivial and does not
    # flip across the FD displacements.
    masks = []
    for dz in (-delta, 0.0, +delta):
        s, b = build(dz)
        mol = s.unit_cell_molecule()
        modrho = make_modrho_aux_basis(
            make_aux_basis_set(mol, aux_name="def2-svp-jk"), mol
        )
        masks.append(_rsgdf_screen_keep_mask(
            s, b, modrho, ke=60.0, thr=thr, lat=lat
        ))
    assert masks[1] is not None
    assert 0 < int(np.count_nonzero(~masks[1])) < b.nbasis**2
    assert np.array_equal(masks[0], masks[1])
    assert np.array_equal(masks[2], masks[1])

    r = run(grad=True)
    assert r.converged
    assert r.gradient is not None
    grad = np.asarray(r.gradient)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-10)
    fd = (run(dz=+delta).energy - run(dz=-delta).energy) / (2.0 * delta)
    assert abs(grad[0, 2] - fd) < 1e-6, (
        f"analytic={grad[0, 2]:.10f} FD={fd:.10f} "
        f"diff={abs(grad[0, 2] - fd):.3e}"
    )


def test_rsgdf_ks_and_open_shell_gradients_vs_fd(monkeypatch):
    """G-PBC-002 milestone 6 (rung 5): rsgdf KS + open-shell gradients.

    (a) Closed-shell KS on the rsgdf fit: alpha-scaled rsgdf exchange +
    the XC Pulay on the SCF's own quadrature; LDA and PBE0 FD at 1e-6
    (observed ~1e-8). (b) Open-shell: the per-spin rsgdf assembly —
    UHF-triplet FD (its energy is the documented PySCF Gamma pin to
    sub-uHa), UKS(M=1) collapses onto the rsgdf RKS gradient at machine
    precision, and the UKS triplet passes FD for LDA and PBE0.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    delta = 1e-4
    lat = _lat_opts()

    def build(dz=0.0, mult=1):
        box = 12.0
        c = box / 2.0
        s = vq.PeriodicSystem(3, np.eye(3) * box,
            [vq.Atom(1, [c, c, c - 0.7 + dz]), vq.Atom(1, [c, c, c + 0.7])])
        if mult != 1:
            s.multiplicity = mult
        return s, vq.BasisSet(s.unit_cell_molecule(), "sto-3g")

    common = dict(aux_basis="def2-svp-jk", gdf_method="rsgdf",
                  rsgdf_ke_cutoff=60.0, exxdiv="ewald", progress=False)

    def run_rks(func, dz=0.0, grad=False):
        s, b = build(dz, 1)
        o = vq.PeriodicKSOptions()
        o.lattice_opts = lat
        o.max_iter = 100
        o.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_rhf(s, b, options=o, functional=func,
                                  compute_gradient=grad, **common)

    def run_uhf(dz=0.0, grad=False):
        s, b = build(dz, 3)
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = lat
        o.max_iter = 100
        o.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_uhf(s, b, options=o,
                                  compute_gradient=grad, **common)

    def run_uks(func, dz=0.0, mult=3, grad=False):
        s, b = build(dz, mult)
        o = vq.PeriodicKSOptions()
        o.lattice_opts = lat
        o.max_iter = 100
        o.conv_tol_energy = 1e-11
        return vq.run_pbc_gdf_uks(s, b, options=o, functional=func,
                                  compute_gradient=grad, **common)

    # (a) closed-shell KS
    for func in ("lda", "pbe0"):
        r = run_rks(func, grad=True)
        assert r.converged
        g = np.asarray(r.gradient)
        assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)
        fd = (run_rks(func, dz=+delta).energy
              - run_rks(func, dz=-delta).energy) / (2.0 * delta)
        assert abs(g[0, 2] - fd) < 1e-6, f"rks {func}: {abs(g[0,2]-fd):.3e}"

    # (b) UHF triplet
    r = run_uhf(grad=True)
    assert r.converged
    g = np.asarray(r.gradient)
    fd = (run_uhf(dz=+delta).energy - run_uhf(dz=-delta).energy) / (
        2.0 * delta
    )
    assert abs(g[0, 2] - fd) < 1e-6, f"uhf: {abs(g[0,2]-fd):.3e}"

    # (b) UKS: M=1 identity + triplet FD
    for func in ("lda", "pbe0"):
        ru = run_uks(func, mult=1, grad=True)
        rr = run_rks(func, grad=True)
        np.testing.assert_allclose(
            np.asarray(ru.gradient), np.asarray(rr.gradient),
            rtol=0.0, atol=1e-12,
        )
        r = run_uks(func, grad=True)
        assert r.converged
        g = np.asarray(r.gradient)
        fd = (run_uks(func, dz=+delta).energy
              - run_uks(func, dz=-delta).energy) / (2.0 * delta)
        assert abs(g[0, 2] - fd) < 1e-6, f"uks {func}: {abs(g[0,2]-fd):.3e}"


# ---------------------------------------------------------------------------
# Tests — G-resolved-weighted AO-pair FT gradient kernel
# ---------------------------------------------------------------------------


def test_gweighted_gradient_kernel_matches_looped_scalar_kernel():
    """The G-resolved-weight kernel reproduces summed scalar-kernel calls.

    For a separable Q[mu, nu, k] = sum_p pw_p[mu, nu] * rw_p[k] the
    gweighted kernel must equal the sum of scalar-kernel calls over p up
    to floating-point reassociation.  H2/STO-3G pins the closed-form s-s
    fast path; LiH/STO-3G adds Li p shells and pins the general-L
    McMurchie-Davidson branch with its per-G Cartesian weight fold.
    """
    from vibeqc._vibeqc_core import (
        ao_pair_fourier_transform_gamma_gradient_gweighted,
        direct_lattice_cells,
    )
    from vibeqc.aux_basis import rsgdf_dense_g_mesh
    from vibeqc.periodic_v_ne_gradient import (
        ao_pair_fourier_transform_gamma_gradient_weighted,
    )

    n_w = 3
    for system, basis in (
        _h2_12bohr_system(),
        _diatomic_12bohr_system(3, 1),  # LiH: p shells on Li (L > 0 branch)
    ):
        n_orb = basis.nbasis
        n_atoms = len(system.unit_cell)

        G_all = np.asarray(rsgdf_dense_g_mesh(system, 30.0), dtype=float)
        G = np.ascontiguousarray(G_all[(G_all**2).sum(axis=1) > 0.0])
        n_G = G.shape[0]

        cells = direct_lattice_cells(system, 12.0)
        R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
        if R_g.size == 0:
            R_g = np.zeros((1, 3), dtype=float)

        rng = np.random.default_rng(7)
        pw = rng.normal(size=(n_w, n_orb, n_orb))
        rw = rng.normal(size=(n_w, n_G)) + 1j * rng.normal(size=(n_w, n_G))

        ref = np.zeros((n_atoms, 3))
        for p in range(n_w):
            ref += np.asarray(
                ao_pair_fourier_transform_gamma_gradient_weighted(
                    basis,
                    G,
                    R_g,
                    np.ascontiguousarray(pw[p]),
                    np.ascontiguousarray(rw[p]),
                    n_atoms,
                )
            )

        Q = np.ascontiguousarray(np.einsum("pmn,pk->mnk", pw, rw))
        assert Q.shape == (n_orb, n_orb, n_G)
        new = np.asarray(
            ao_pair_fourier_transform_gamma_gradient_gweighted(
                basis, G, R_g, Q, n_atoms
            )
        )

        assert new.shape == (n_atoms, 3)
        assert np.allclose(
            ref, new, rtol=0.0, atol=1e-10 * max(1.0, np.abs(ref).max())
        )


def _bloch_gweighted_kernel_fixture(z_a, z_b, seed):
    """Shared fixture for the Bloch gweighted gradient kernel tests.

    Same mesh/cell-list/weight construction as the gweighted parity
    test above, with the complex Q drawn directly (the Bloch kernel's
    contract is G-resolved complex weights, no separability assumed).
    """
    from vibeqc._vibeqc_core import direct_lattice_cells
    from vibeqc.aux_basis import rsgdf_dense_g_mesh

    system, basis = _diatomic_12bohr_system(z_a, z_b)
    n_orb = basis.nbasis
    n_atoms = len(system.unit_cell)

    G_all = np.asarray(rsgdf_dense_g_mesh(system, 30.0), dtype=float)
    G = np.ascontiguousarray(G_all[(G_all**2).sum(axis=1) > 0.0])
    n_G = G.shape[0]

    cells = direct_lattice_cells(system, 12.0)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    if R_g.size == 0:
        R_g = np.zeros((1, 3), dtype=float)

    rng = np.random.default_rng(seed)
    Q = np.ascontiguousarray(
        rng.normal(size=(n_orb, n_orb, n_G))
        + 1j * rng.normal(size=(n_orb, n_orb, n_G))
    )
    return system, basis, n_atoms, G, R_g, Q


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_bloch_gweighted_gradient_kernel_k0_reduces_to_gamma(
    atomic_numbers,
):
    """The Bloch gweighted kernel at k = 0 reproduces the Gamma kernel.

    HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 1 gate: at k_cart = 0
    the Bloch phase is exactly (1, 0), so the new kernel must agree
    with ``ao_pair_fourier_transform_gamma_gradient_gweighted``
    bit-for-bit on the same complex Q.  H2/STO-3G pins the s-s fast
    path; LiH/STO-3G pins the general-L McMurchie-Davidson branch.
    """
    from vibeqc._aopair_ft import (
        ao_pair_fourier_transform_bloch_gradient_gweighted,
    )
    from vibeqc._vibeqc_core import (
        ao_pair_fourier_transform_gamma_gradient_gweighted,
    )

    z_a, z_b = atomic_numbers
    _, basis, n_atoms, G, R_g, Q = _bloch_gweighted_kernel_fixture(
        z_a, z_b, seed=11
    )

    gamma = np.asarray(
        ao_pair_fourier_transform_gamma_gradient_gweighted(
            basis, G, R_g, Q, n_atoms
        )
    )
    bloch = ao_pair_fourier_transform_bloch_gradient_gweighted(
        basis, G, R_g, np.zeros(3), Q, n_atoms
    )

    assert bloch.shape == (n_atoms, 3)
    assert np.abs(bloch - gamma).max() <= 1e-14


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_bloch_gweighted_gradient_kernel_vs_fd_nonzero_k(atomic_numbers):
    """FD gate for the Bloch gweighted kernel at k != 0.

    Central-differences the value objective
    ``f(R) = Re sum Q conj(ao_pair_fourier_transform_bloch(...))``
    with the BasisSet rebuilt from the displaced system per step, so
    the analytic kernel is pinned against exactly the Bloch sum the
    value kernel computes (same +ik ket phase, conjugated in the
    objective).  Generic k exercises complex phases on every cell.
    """
    from vibeqc._aopair_ft import (
        ao_pair_fourier_transform_bloch,
        ao_pair_fourier_transform_bloch_gradient_gweighted,
    )

    z_a, z_b = atomic_numbers
    _, basis, n_atoms, G, R_g, Q = _bloch_gweighted_kernel_fixture(
        z_a, z_b, seed=11
    )
    k_cart = np.array([0.1, 0.05, -0.03])

    analytic = ao_pair_fourier_transform_bloch_gradient_gweighted(
        basis, G, R_g, k_cart, Q, n_atoms
    )

    box = 12.0
    c = box / 2.0
    base_positions = [[c, c, c - 0.7], [c, c, c + 0.7]]
    atomic_zs = (z_a, z_b)

    def objective(positions):
        atoms = [vq.Atom(z, p) for z, p in zip(atomic_zs, positions)]
        displaced = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
        displaced_basis = vq.BasisSet(
            displaced.unit_cell_molecule(), "sto-3g"
        )
        ft = ao_pair_fourier_transform_bloch(
            displaced_basis, G, R_g, k_cart
        )
        return float(np.real(np.sum(Q * np.conj(ft))))

    h = 1e-5
    fd = np.zeros((n_atoms, 3))
    for atom in range(n_atoms):
        for axis in range(3):
            for sign in (+1.0, -1.0):
                positions = [list(p) for p in base_positions]
                positions[atom][axis] += sign * h
                fd[atom, axis] += sign * objective(positions)
            fd[atom, axis] /= 2.0 * h

    assert np.abs(analytic - fd).max() <= 1e-7


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_multik_rsgdf_gradient_cache_bit_consistent_with_scf_fit(
    atomic_numbers,
):
    """HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 2 gate: the multi-k
    rsgdf gradient cache mirrors the SCF's shared-q fit bit-for-bit.

    Builds the cache on the Γ-centered (2,1,1) mesh (the exact k points
    run_krhf_periodic_gdf derives via _kmesh_to_kpoints_weights) and,
    for EVERY unique-q group and every bra k, reconstructs Lpq from the
    cache's eigensystem+T exactly as build_lpq_bloch_native_fft_shared_q
    does, then calls that SCF builder directly with the same inputs and
    demands np.array_equal — the M6-rung-9 lesson: ANY fit mismatch is
    Fréchet-amplified near threshold, so closeness gates are not enough.
    The (2,1,1) mesh also pins the BZ-boundary canonicalisation edge:
    +b1/2 and -b1/2 transfers share one canonical -b1/2 group whose ket
    momenta are k_bra + q_canonical, not k_j. H2 pins the s-only pair-FT
    branch; LiH (p shells on Li) the general-L branch.
    """
    from vibeqc.aux_basis import (
        build_lpq_bloch_native_fft_shared_q,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic_gdf_gradient import (
        _build_multik_rsgdf_gradient_cache,
    )
    from vibeqc.periodic_k_gdf import _kmesh_to_kpoints_weights

    system, basis = _diatomic_12bohr_system(*atomic_numbers)
    lat = _lat_opts()
    ke = 60.0
    thr = 1e-9
    kpts, _weights = _kmesh_to_kpoints_weights(system, (2, 1, 1))
    assert kpts.shape == (2, 3)

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)

    cache = _build_multik_rsgdf_gradient_cache(
        system,
        basis,
        modrho,
        kpts,
        ke_cutoff=ke,
        tail_ke_cutoff=None,
        lat_opts=lat,
        linear_dep_thr=thr,
    )
    assert cache.n_aux == modrho.nbasis
    assert cache.n_orb == basis.nbasis

    # Full (k_i, k_j) pair coverage: J's q=0 diagonal AND K's off-q.
    covered = sorted(p for g in cache.groups.values() for p in g.pairs)
    assert covered == [(i, j) for i in range(2) for j in range(2)]
    q_norms = [
        float(np.linalg.norm(g.q)) for g in cache.groups.values()
    ]
    assert any(n < 1e-12 for n in q_norms)  # q=0 group present
    assert any(n > 1e-12 for n in q_norms)  # BZ-boundary group present

    for _q_key, group in sorted(cache.groups.items()):
        assert len(group.T_list) == len(group.bra_k_indices)
        k_bras = kpts[np.asarray(group.bra_k_indices, dtype=int)]
        lpq_scf = build_lpq_bloch_native_fft_shared_q(
            system,
            basis,
            modrho,
            k_bras,
            group.q,
            ke_cutoff=ke,
            tail_ke_cutoff=None,
            lat_opts=lat,
            linear_dep_thr=thr,
        )
        # Per-q fit rank matches the SCF's.
        assert group.n_fit == int(lpq_scf[0].shape[0])
        assert group.n_fit == int(np.count_nonzero(group.keep_mask))
        if float(np.linalg.norm(group.q)) < 1e-12:
            # q=0 metric is real-projected (float64), like the SCF's —
            # required so eigh dispatches to the same LAPACK driver.
            assert np.isrealobj(group.M)
        assert group.inverse_frechet.dtype == np.complex128
        for T, lpq_ref in zip(group.T_list, lpq_scf):
            T_flat = T.reshape(cache.n_aux, -1)
            keep = group.keep_mask
            lpq_cache = (
                (group.eigvecs[:, keep].conj().T @ T_flat)
                / np.sqrt(group.eigvals[keep])[:, None]
            ).reshape(group.n_fit, cache.n_orb, cache.n_orb)
            assert np.array_equal(lpq_cache, lpq_ref)


@pytest.mark.parametrize(
    "atomic_numbers, screen_thr",
    [((1, 1), _H2_SCREEN_THR), ((3, 1), _LIH_SCREEN_THR)],
    ids=["h2_ss", "lih_general_l"],
)
def test_multik_screened_rsgdf_gradient_cache_bit_consistent_with_scf_fit(
    atomic_numbers, screen_thr
):
    """Screened-fit envelope gate (2026-07-30): the multi-k rsgdf
    gradient cache mirrors the SCF's SCHWARZ-SCREENED shared-q fit
    bit-for-bit.

    The screened sibling of the unscreened gate above: same (2,1,1)
    mesh and rebuild-vs-SCF ``np.array_equal`` demand, with
    ``fit_screen_threshold`` chosen mid-gap in each anchor's Schwarz
    bound spectrum so EVERY q group's pair mask is non-trivial (some
    pairs masked, some kept). Pins that the cache derives the identical
    per-q mask from the same mask builder, zeroes the masked pairs at
    the same sweep point, and that masked Lpq entries are exact hard
    zeros."""
    from vibeqc.aux_basis import build_lpq_bloch_native_fft_shared_q
    from vibeqc.periodic_gdf_gradient import (
        _build_multik_rsgdf_gradient_cache,
    )
    from vibeqc.periodic_k_gdf import _kmesh_to_kpoints_weights

    system, basis = _diatomic_12bohr_system(*atomic_numbers)
    lat = _lat_opts()
    ke = 60.0
    thr = 1e-9
    kpts, _weights = _kmesh_to_kpoints_weights(system, (2, 1, 1))

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)

    cache = _build_multik_rsgdf_gradient_cache(
        system, basis, modrho, kpts,
        ke_cutoff=ke, tail_ke_cutoff=None, lat_opts=lat,
        linear_dep_thr=thr, fit_screen_threshold=float(screen_thr),
    )
    assert cache.fit_screen_threshold == float(screen_thr)
    assert len(cache.groups) == 2  # q=0 + the BZ-boundary transfer

    n_total = basis.nbasis * basis.nbasis
    for _q_key, group in sorted(cache.groups.items()):
        # Non-trivial per-q mask: some pairs masked, some kept.
        assert group.pair_keep_ao is not None
        n_masked = int(np.count_nonzero(~group.pair_keep_ao))
        assert 0 < n_masked < n_total
        k_bras = kpts[np.asarray(group.bra_k_indices, dtype=int)]
        lpq_scf = build_lpq_bloch_native_fft_shared_q(
            system, basis, modrho, k_bras, group.q,
            ke_cutoff=ke, tail_ke_cutoff=None, lat_opts=lat,
            linear_dep_thr=thr, fit_screen_threshold=float(screen_thr),
        )
        assert group.n_fit == int(lpq_scf[0].shape[0])
        for T, lpq_ref in zip(group.T_list, lpq_scf):
            # Masked pairs are exact hard zeros on both sides.
            assert np.all(T[:, ~group.pair_keep_ao] == 0.0)
            assert np.all(lpq_ref[:, ~group.pair_keep_ao] == 0.0)
            T_flat = T.reshape(cache.n_aux, -1)
            keep = group.keep_mask
            lpq_cache = (
                (group.eigvecs[:, keep].conj().T @ T_flat)
                / np.sqrt(group.eigvals[keep])[:, None]
            ).reshape(group.n_fit, cache.n_orb, cache.n_orb)
            assert np.array_equal(lpq_cache, lpq_ref)


def _scf_multik_source_for(system, basis, kpts, ke, tail_ke, lat, thr):
    """Rebuild the production SR/LR source for independent gradient assembly.

    The legacy raw-tensor derivative tests below intentionally keep their
    old cache. Full-SCF finite differences must differentiate the source
    that now produces the energy, including its physical auxiliary basis.
    """
    from vibeqc.aux_basis import make_aux_basis_set
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    assert tail_ke is None
    aux = make_aux_basis_set(system.unit_cell_molecule(), aux_name="def2-svp-jk")
    return _build_range_separated_lpq_cache(
        system, basis, aux, kpts, True,
        omega=.4, raw_integral_error=1e-10,
        pair_cutoff=float(lat.cutoff_bohr), ke_cutoff=ke,
        linear_dep_thr=thr, memory_byte_cap=512 * 1024**2,
        native_workspace_byte_cap=64 * 1024**2,
        image_candidate_cap=10_000_000, reciprocal_candidate_cap=10_000_000,
    )


def _multik_j_cache_for(system, basis, kpts, ke, tail_ke, lat, thr):
    from vibeqc.aux_basis import make_aux_basis_set, make_modrho_aux_basis
    from vibeqc.periodic_gdf_gradient import (
        _build_multik_rsgdf_gradient_cache,
    )

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    return _build_multik_rsgdf_gradient_cache(
        system, basis, modrho, kpts,
        ke_cutoff=ke, tail_ke_cutoff=tail_ke, lat_opts=lat,
        linear_dep_thr=thr,
    )


def _multik_ej_from_cache(cache, D_k, weights):
    """E_J = 0.5 Re[t^T f(M0) t] from the cache's q=0 group — the
    unfactorized form of the SCF's ``_build_j_from_lpq`` +
    ``E_coulomb = 0.5 S_i w_i Re Tr[D(k_i) J(k_i)]`` (no conjugation
    anywhere, PySCF convention). Returns (E_J, t)."""
    from vibeqc.periodic_gdf_gradient import _multik_rsgdf_q0_group

    group = _multik_rsgdf_q0_group(cache)
    t = np.zeros(cache.n_aux, dtype=np.complex128)
    for bra_pos, k_idx in enumerate(group.bra_k_indices):
        T_flat = group.T_list[bra_pos].reshape(cache.n_aux, -1)
        t += float(weights[k_idx]) * (
            T_flat @ np.asarray(D_k[k_idx], dtype=np.complex128).T.ravel()
        )
    t_eig = group.eigvecs.T @ t
    e_j = 0.5 * float(np.real(np.sum(group.inverse_eigvals * t_eig * t_eig)))
    return e_j, t


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_multik_j_gradient_fixed_density_vs_fd(atomic_numbers):
    """HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 3a gate: the multi-k
    KRHF DF-J fit-derivative (q=0 blocks) matches fixed-density FD.

    Converged D(k) from the production (2,1,1) rsgdf cderi route
    (use_compcell=True), then FROZEN: the FD side rebuilds
    system+basis+cache per displaced geometry and re-evaluates
    E_J = 0.5 Re[t^T f(M0) t] with the frozen D — isolating the fit
    derivative from the (later-rung) K/one-electron/W terms. Measured
    |analytic - FD| at h=1e-4: H2 5.7e-10 (bond axis z) / 1.1e-12
    (perpendicular x); LiH 1.2e-9 / 4.4e-12 — the Γ fixed-D J anchors
    sit at the same 1e-9-class. Also pinned: t is real to ~6e-17
    (k/-k pairing of Hermitian D on the Γ-centered mesh; the algebra
    does NOT rely on it), and E_J(cache) reproduces the SCF's
    e_coulomb (exactly the same objective; loose gate only because
    e_coulomb is evaluated at the last pre-diagonalization density
    while result.density is post-diagonalization). LiH covers
    general-L aux/AO shells (its dense-core PARITY_HELD warning is an
    ABSOLUTE-energy caveat, irrelevant to a fixed-density isolation
    gate).
    """
    from vibeqc.periodic_gdf_gradient import (
        _compute_j_gradient_multik_rsgdf,
    )
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    z_a, z_b = atomic_numbers
    ke = 60.0
    thr = 1e-9  # matches the driver's gdf_linear_dep_threshold default
    lat = _lat_opts()
    h = 1e-4

    def displaced(atom=None, axis=None, step=0.0):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        if atom is not None:
            base[atom][axis] += step
        atoms = [vq.Atom(z, xyz) for z, xyz in zip((z_a, z_b), base)]
        s = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
        return s, vq.BasisSet(s.unit_cell_molecule(), "sto-3g")

    system, basis = displaced()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = lat
    options.max_iter = 120
    options.conv_tol_energy = 1e-10
    r = run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), options, aux_basis="def2-svp-jk",
        use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=ke,
        progress=False,
    )
    assert r.converged
    D_k = [np.asarray(D) for D in r.density]  # FROZEN below
    weights = np.asarray(r.kpoint_weights, dtype=float)
    kpts = np.asarray(r.kpoints_cart, dtype=float)

    cache = _multik_j_cache_for(system, basis, kpts, ke, None, lat, thr)
    e_j0, t = _multik_ej_from_cache(cache, D_k, weights)
    assert np.abs(t.imag).max() <= 1e-14
    assert abs(e_j0 - float(r.e_coulomb)) < 1e-5

    grad = _compute_j_gradient_multik_rsgdf(
        system, basis, D_k, weights, cache
    )
    assert grad.shape == (2, 3)
    # Fixed-D E_J is invariant under rigid translation (every phase in
    # M/T depends on centre differences only), so Newton's 3rd law
    # holds for the J fit term alone.
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-12)

    for a, d in [(0, 2), (0, 0)]:
        vals = {}
        for sign in (+1, -1):
            s_d, b_d = displaced(atom=a, axis=d, step=sign * h)
            c_d = _multik_j_cache_for(s_d, b_d, kpts, ke, None, lat, thr)
            vals[sign], _ = _multik_ej_from_cache(c_d, D_k, weights)
        fd = (vals[+1] - vals[-1]) / (2.0 * h)
        assert abs(grad[a, d] - fd) <= 1e-7, (
            f"atom {a} axis {d}: analytic={grad[a, d]:.12f} "
            f"FD={fd:.12f} diff={abs(grad[a, d] - fd):.3e}"
        )


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_multik_j_gradient_reduces_to_gamma_at_1k(atomic_numbers):
    """Consistency reduction (rung 3a): at kmesh=(1,1,1) the multi-k J
    assembly matches the Γ ``_compute_j_gradient_rsgdf`` on the same
    system/weight algebra to <= 1e-12.

    Same fixed random symmetric D as the Γ rung-3 gate; base + tail
    meshes both exercised. The two paths differ mechanically (batched
    vs per-k pair-FT kernel, complex unsymmetrized vs real-projected
    T, Bloch vs Γ 3c derivative kernel at k=0), so this pins the whole
    complex-weight algebra against the FD-anchored Γ template.
    Measured max diff: H2 5.9e-17, LiH 9.4e-16.
    """
    from vibeqc.aux_basis import make_aux_basis_set, make_modrho_aux_basis
    from vibeqc.periodic_gdf_gradient import (
        _build_multik_rsgdf_gradient_cache,
        _build_rsgdf_gradient_cache,
        _compute_j_gradient_multik_rsgdf,
        _compute_j_gradient_rsgdf,
    )

    ke = 40.0
    tail_ke = 80.0
    thr = 1e-9
    lat = _lat_opts()
    system, basis = _diatomic_12bohr_system(*atomic_numbers)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)

    rng = np.random.default_rng(19)
    D = rng.standard_normal((basis.nbasis, basis.nbasis))
    D = 0.5 * (D + D.T)

    gamma_cache = _build_rsgdf_gradient_cache(
        system, basis, modrho,
        ke_cutoff=ke, tail_ke_cutoff=tail_ke, lat_opts=lat,
        linear_dep_thr=thr,
    )
    g_gamma = _compute_j_gradient_rsgdf(system, basis, D, gamma_cache)

    multik_cache = _build_multik_rsgdf_gradient_cache(
        system, basis, modrho, np.zeros((1, 3)),
        ke_cutoff=ke, tail_ke_cutoff=tail_ke, lat_opts=lat,
        linear_dep_thr=thr,
    )
    g_multik = _compute_j_gradient_multik_rsgdf(
        system, basis, [D.astype(np.complex128)], [1.0], multik_cache
    )

    assert np.abs(g_multik - g_gamma).max() <= 1e-12


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_multik_oneel_w_gradient_fixed_density_vs_fd(atomic_numbers):
    """HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 3b gates: the multi-k
    kinetic, overlap-Lagrangian, split-Ewald V_ne, nuclear-repulsion,
    and exxdiv-shift gradient terms each match a fixed-density FD of
    EXACTLY the term the multi-k SCF energy uses (the e_nuc-gauge
    lesson: FD the SCF's own matrix build, never a plausible
    substitute).

    Converged D(k)/C(k)/eps(k) from the production (2,1,1) rsgdf route,
    then FROZEN. Each FD side rebuilds system+basis per displacement
    and re-derives the SCF's own lattice options (``_oneel_lattice_
    opts`` for S/T, ``_gauge_lat_opts_for_v_ne_and_e_nuc`` +
    ``compute_nuclear_lattice_dispatch`` for V_ne — the split
    erfc-short + analytic-FT-long + G=0-correction build, NOT the
    Γ-only all-in-one routine) and re-evaluates the per-term energy:

      E_T   = S_k w_k Re Tr[D(k) T(k)]
      E_S   = S_k w_k Re Tr[W(k) S(k)]         (helper returns -dE_S/dR)
      E_V   = S_k w_k Re Tr[D(k) V(k)]
      E_nn  = ewald_nuclear_repulsion          (cheap re-pin; partner
                                                gradient proven at 2.6e-9)
      E_exx = -1/4 a xi(R) S_k w_k Re Tr[D S D S]  (xi recomputed per
              displacement: position-independence is thereby FD-verified,
              not assumed)

    Measured |analytic - FD| at h=1e-4 (bond axis z):
      H2:  kinetic 9.2e-11, overlap-W 2.9e-10, v_ne 2.8e-10,
           nn 2.6e-9, exxdiv 7.0e-11
      LiH: kinetic 1.0e-9, overlap-W 1.4e-10, v_ne 2.6e-9,
           nn 7.8e-9, exxdiv 8.5e-11
    Gates 1e-7 (nn 5e-8). The perpendicular components vanish to
    <= 2e-11 on both sides. The assembled 1e+W+nn helper is pinned to
    the sum of its terms and to Newton's third law.
    """
    from vibeqc._vibeqc_core import (
        bloch_sum,
        compute_kinetic_lattice,
        compute_overlap_lattice,
        ewald_nuclear_repulsion,
        ewald_nuclear_repulsion_gradient,
    )
    from vibeqc.periodic_gdf_gradient import (
        _compute_exxdiv_w_gradient_multik,
        _compute_kinetic_gradient_multik,
        _compute_oneel_w_gradient_multik,
        _compute_overlap_w_gradient_multik,
        _compute_v_ne_gradient_multik,
    )
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        _oneel_lattice_opts,
        run_krhf_periodic_gdf,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    z_a, z_b = atomic_numbers
    lat = _lat_opts()
    h = 1e-4
    alpha_hf = 1.0

    def displaced(atom=None, axis=None, step=0.0):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        if atom is not None:
            base[atom][axis] += step
        atoms = [vq.Atom(z, xyz) for z, xyz in zip((z_a, z_b), base)]
        s = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
        return s, vq.BasisSet(s.unit_cell_molecule(), "sto-3g")

    system, basis = displaced()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = lat
    options.max_iter = 120
    options.conv_tol_energy = 1e-10
    r = run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), options, aux_basis="def2-svp-jk",
        use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=60.0,
        progress=False,
    )
    assert r.converged
    n_k = 2
    D_k = [np.asarray(D) for D in r.density]          # FROZEN below
    S_k = [np.asarray(S) for S in r.overlap]          # FROZEN below
    weights = np.asarray(r.kpoint_weights, dtype=float)
    kpts = np.asarray(r.kpoints_cart, dtype=float)
    n_occ = system.n_electrons() // 2
    # W(k) = 2 C_occ(k) diag(eps_occ(k)) C_occ(k)^H — the
    # _bloch_fold_w_per_k closed-shell convention. FROZEN below.
    W_k = []
    for ik in range(n_k):
        C_occ = np.asarray(r.mo_coeffs[ik])[:, :n_occ]
        eps_occ = np.asarray(r.mo_energies[ik])[:n_occ]
        W_k.append(2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T)

    oneel = _oneel_lattice_opts(
        system, basis, lat, rcut_strategy="pyscf_auto", k_points_cart=kpts
    )
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    xi = _madelung_for_kmesh(system, (2, 1, 1))

    g_T = _compute_kinetic_gradient_multik(
        system, basis, D_k, weights, kpts, oneel
    )
    g_S = _compute_overlap_w_gradient_multik(
        system, basis, W_k, weights, kpts, oneel
    )
    g_V = _compute_v_ne_gradient_multik(
        system, basis, D_k, weights, kpts, gauge
    )
    g_nn = np.asarray(ewald_nuclear_repulsion_gradient(system))
    g_exx = _compute_exxdiv_w_gradient_multik(
        system, basis, D_k, S_k, weights, kpts, alpha_hf, xi, oneel
    )
    g_all = _compute_oneel_w_gradient_multik(
        system, basis, D_k, W_k, weights, kpts,
        oneel_lat_opts=oneel, gauge_lat_opts=gauge,
    )
    # Assembly == sum of its terms, and Newton's 3rd law on the
    # assembled fixed-density 1e+W+nn gradient (every term depends on
    # centre differences only under rigid translation).
    assert np.abs(g_all - (g_nn + g_T + g_S + g_V)).max() <= 1e-14
    assert np.allclose(g_all.sum(axis=0), 0.0, atol=1e-9)

    def e_terms(s_d, b_d):
        oneel_d = _oneel_lattice_opts(
            s_d, b_d, lat, rcut_strategy="pyscf_auto", k_points_cart=kpts
        )
        gauge_d = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, s_d)
        T_lat = compute_kinetic_lattice(b_d, s_d, oneel_d)
        S_lat = compute_overlap_lattice(b_d, s_d, oneel_d)
        V_lat = compute_nuclear_lattice_dispatch(b_d, s_d, gauge_d)
        xi_d = _madelung_for_kmesh(s_d, (2, 1, 1))
        E = np.zeros(5)
        for ik in range(n_k):
            w = float(weights[ik])
            Tk = np.asarray(bloch_sum(T_lat, kpts[ik]))
            Sk = np.asarray(bloch_sum(S_lat, kpts[ik]))
            Vk = np.asarray(bloch_sum(V_lat, kpts[ik]))
            E[0] += w * float(np.real(np.trace(D_k[ik] @ Tk)))
            E[1] += w * float(np.real(np.trace(W_k[ik] @ Sk)))
            E[2] += w * float(np.real(np.trace(D_k[ik] @ Vk)))
            E[4] += (
                -0.25 * alpha_hf * xi_d * w
                * float(np.real(np.trace(D_k[ik] @ Sk @ D_k[ik] @ Sk)))
            )
        E[3] = float(ewald_nuclear_repulsion(s_d))
        return E

    gates = {
        "kinetic": 1e-7,
        "overlap-W": 1e-7,
        "v_ne": 1e-7,
        "nn": 5e-8,
        "exxdiv": 1e-7,
    }
    for a, d in [(0, 2), (0, 0)]:
        vals = {}
        for sign in (+1, -1):
            s_d, b_d = displaced(atom=a, axis=d, step=sign * h)
            vals[sign] = e_terms(s_d, b_d)
        fd = (vals[+1] - vals[-1]) / (2.0 * h)
        # The overlap helper returns the gradient CONTRIBUTION
        # -dE_S/dR; compare its negation against the FD of E_S.
        analytic = [
            g_T[a, d], -g_S[a, d], g_V[a, d], g_nn[a, d], g_exx[a, d]
        ]
        for name, ana, f in zip(gates, analytic, fd):
            assert abs(ana - f) <= gates[name], (
                f"{name} atom {a} axis {d}: analytic={ana:.12f} "
                f"FD={f:.12f} diff={abs(ana - f):.3e}"
            )


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_multik_oneel_w_gradient_reduces_to_gamma_at_1k(atomic_numbers):
    """Γ-reduction (rung 3b): at kmesh=(1,1,1) each multi-k 1e/W/exxdiv
    term matches the corresponding piece of the Γ assembly
    (``compute_gdf_gradient_rhf_gamma``'s term structure: homogeneous
    Γ lattice sets into the k-blind kernels; V_ne via
    ``compute_v_ne_ewald_3d_ft_gamma_gradient``) to <= 1e-10.

    Random symmetric D/W/S at fixed geometry — the reduction is an
    algebraic identity of the fold + kernels, not a property of a
    converged density. Measured max diffs: kinetic 8.7e-19 / 1.1e-16,
    overlap-W 5.6e-17 / 3.3e-16, v_ne 5.6e-17 / 1.8e-15, exxdiv
    9.9e-28 / 5.6e-17 (H2 / LiH). The v_ne comparison pins the Bloch
    gweighted gradient kernel at k=0 + complex weights against the
    Γ-only real-weight kernel route end-to-end. The nn term is the
    identical ``ewald_nuclear_repulsion_gradient`` call on both paths
    (nothing to compare).
    """
    from vibeqc._vibeqc_core import (
        compute_overlap_lattice,
        kinetic_lattice_gradient_contribution,
        overlap_lattice_gradient_contribution,
    )
    from vibeqc.periodic_gdf_gradient import (
        _compute_exxdiv_w_gradient_multik,
        _compute_kinetic_gradient_multik,
        _compute_overlap_w_gradient_multik,
        _compute_v_ne_gradient_multik,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
    from vibeqc.periodic_v_ne_gradient import (
        compute_v_ne_ewald_3d_ft_gamma_gradient,
    )

    system, basis = _diatomic_12bohr_system(*atomic_numbers)
    lat = _lat_opts()
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    n = basis.nbasis
    rng = np.random.default_rng(7)
    D = rng.standard_normal((n, n))
    D = 0.5 * (D + D.T)
    W = rng.standard_normal((n, n))
    W = 0.5 * (W + W.T)
    S = rng.standard_normal((n, n))
    S = 0.5 * (S + S.T)

    kpts = np.zeros((1, 3))
    w = [1.0]

    def gamma_set(M):
        # The Γ assembly's homogeneous lattice-set construction.
        st = compute_overlap_lattice(basis, system, lat)
        home_only = all(
            tuple(int(v) for v in np.asarray(cc.index).reshape(3))
            == (0, 0, 0)
            for cc in st.cells
        )
        zero = np.zeros_like(M)
        for ci in range(len(st.cells)):
            idx = tuple(
                int(v) for v in np.asarray(st.cells[ci].index).reshape(3)
            )
            st.set_block(
                ci, M if (idx == (0, 0, 0) or not home_only) else zero
            )
        return st

    g_T_m = _compute_kinetic_gradient_multik(
        system, basis, [D.astype(np.complex128)], w, kpts, lat
    )
    g_T_g = np.asarray(
        kinetic_lattice_gradient_contribution(
            basis, system, gamma_set(D), lat
        )
    )
    assert np.abs(g_T_m - g_T_g).max() <= 1e-10

    g_S_m = _compute_overlap_w_gradient_multik(
        system, basis, [W.astype(np.complex128)], w, kpts, lat
    )
    g_S_g = np.asarray(
        overlap_lattice_gradient_contribution(
            basis, system, gamma_set(W), lat
        )
    )
    assert np.abs(g_S_m - g_S_g).max() <= 1e-10

    g_V_m = _compute_v_ne_gradient_multik(
        system, basis, [D.astype(np.complex128)], w, kpts, gauge
    )
    g_V_g = compute_v_ne_ewald_3d_ft_gamma_gradient(
        basis, system, gauge, D, ke_cutoff=200.0
    )
    assert np.abs(g_V_m - g_V_g).max() <= 1e-10

    xi, a_hf = 0.3, 1.0
    g_E_m = _compute_exxdiv_w_gradient_multik(
        system, basis,
        [D.astype(np.complex128)], [S.astype(np.complex128)],
        w, kpts, a_hf, xi, lat,
    )
    W_exx = 0.5 * a_hf * xi * (D @ S @ D)
    g_E_g = np.asarray(
        overlap_lattice_gradient_contribution(
            basis, system, gamma_set(W_exx), lat
        )
    )
    assert np.abs(g_E_m - g_E_g).max() <= 1e-10


def _multik_ek_from_cache(cache, D_k, weights):
    """E_K = -0.25 S_i w_i Re Tr[D(k_i) K(k_i)] from the cache, with
    K built by the SCF's own ``_build_k_from_lpq_cache`` on per-pair
    Lpq reconstructed from each q group's eigensystem + T exactly as
    ``build_lpq_bloch_native_fft_shared_q`` factorizes them
    (``L = U_keep^H T / sqrt(λ_keep)``) — the unfactorized objective
    ``-0.25 S_q Re Tr[G(q) H(q)]`` the K gradient differentiates."""
    from vibeqc.periodic_k_gdf import _build_k_from_lpq_cache

    n = cache.n_orb
    lpq = {}
    for group in cache.groups.values():
        keep = group.keep_mask
        scale = np.sqrt(group.eigvals[keep])[:, None]
        for pair_pos, (i, j) in enumerate(group.pairs):
            T_flat = group.T_list[pair_pos].reshape(cache.n_aux, -1)
            lpq[(i, j)] = (
                (group.eigvecs[:, keep].conj().T @ T_flat) / scale
            ).reshape(-1, n, n)
    K_k = _build_k_from_lpq_cache(
        lpq, [np.asarray(D) for D in D_k], list(weights), nbasis=n
    )
    return -0.25 * sum(
        float(weights[i])
        * float(np.real(np.trace(np.asarray(D_k[i]) @ K_k[i])))
        for i in range(len(D_k))
    )


@pytest.mark.parametrize(
    "atomic_numbers", [(1, 1), (3, 1)], ids=["h2_ss", "lih_general_l"]
)
def test_multik_k_gradient_fixed_density_vs_fd(atomic_numbers):
    """HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 4 gate: the multi-k
    KRHF DF-K fit-derivative over ALL q groups matches fixed-density
    FD — including the (2,1,1) mesh's q != 0 BZ-boundary group, whose
    complex Hermitian metric weight exercises the new complex-weight
    2c Bloch kernel and the off-diagonal (k_i, k_j) 3c weights.

    Converged D(k) from the production (2,1,1) rsgdf route, then
    FROZEN: the FD side rebuilds system+basis+cache per displaced
    geometry and re-evaluates E_K = -0.25 S_i w_i Re Tr[D(k_i) K(k_i)]
    with K from the SCF's own ``_build_k_from_lpq_cache`` on
    cache-reconstructed per-pair Lpq — exactly the objective the
    analytic weight algebra differentiates. Measured |analytic - FD|
    at h=1e-4: H2 3.3e-10 (bond axis z) / 1.1e-12 (perpendicular x);
    LiH 4.8e-10 / 2.2e-12 — the Γ fixed-D K anchors sit at the same
    1e-9-class. Also pinned: E_K(cache) + the exxdiv shift reproduces
    the SCF's e_hf_exchange (H2 bitwise 0.0 — the rung-2
    bit-consistency at work; LiH 9.1e-8, loose gate only because the
    SCF energy decomposition is evaluated at the last
    pre-diagonalization density), and Newton's third law on the
    fixed-D fit term (every phase depends on centre differences only
    under rigid translation; measured <= 2.1e-14).
    """
    from vibeqc.periodic_gdf_gradient import (
        _compute_k_gradient_multik_rsgdf,
    )
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        run_krhf_periodic_gdf,
    )

    z_a, z_b = atomic_numbers
    ke = 60.0
    thr = 1e-9  # matches the driver's gdf_linear_dep_threshold default
    lat = _lat_opts()
    h = 1e-4

    def displaced(atom=None, axis=None, step=0.0):
        box = 12.0
        c = box / 2.0
        base = [[c, c, c - 0.7], [c, c, c + 0.7]]
        if atom is not None:
            base[atom][axis] += step
        atoms = [vq.Atom(z, xyz) for z, xyz in zip((z_a, z_b), base)]
        s = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
        return s, vq.BasisSet(s.unit_cell_molecule(), "sto-3g")

    system, basis = displaced()
    options = vq.PeriodicRHFOptions()
    options.lattice_opts = lat
    options.max_iter = 120
    options.conv_tol_energy = 1e-10
    r = run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), options, aux_basis="def2-svp-jk",
        use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=ke,
        progress=False,
    )
    assert r.converged
    D_k = [np.asarray(D) for D in r.density]  # FROZEN below
    S_k = [np.asarray(S) for S in r.overlap]
    weights = np.asarray(r.kpoint_weights, dtype=float)
    kpts = np.asarray(r.kpoints_cart, dtype=float)

    cache = _multik_j_cache_for(system, basis, kpts, ke, None, lat, thr)
    # The K gate must exercise a q != 0 group (task contract): the
    # Γ-centered (2,1,1) mesh folds both ±b1/2 transfers into ONE
    # canonical BZ-boundary group with pairs (0,1) and (1,0).
    q_norms = [float(np.linalg.norm(g.q)) for g in cache.groups.values()]
    assert any(n > 1e-12 for n in q_norms)

    e_k0 = _multik_ek_from_cache(cache, D_k, weights)
    # Unfactorized cache E_K + exxdiv shift == the SCF's shifted
    # exchange energy (alpha = 1 KRHF).
    xi = _madelung_for_kmesh(system, (2, 1, 1))
    e_shift = -0.25 * xi * sum(
        float(weights[i])
        * float(np.real(np.trace(D_k[i] @ S_k[i] @ D_k[i] @ S_k[i])))
        for i in range(len(D_k))
    )
    assert abs(e_k0 + e_shift - float(r.e_hf_exchange)) < 1e-5

    grad = _compute_k_gradient_multik_rsgdf(
        system, basis, D_k, weights, cache
    )
    assert grad.shape == (2, 3)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-12)

    for a, d in [(0, 2), (0, 0)]:
        vals = {}
        for sign in (+1, -1):
            s_d, b_d = displaced(atom=a, axis=d, step=sign * h)
            c_d = _multik_j_cache_for(s_d, b_d, kpts, ke, None, lat, thr)
            vals[sign] = _multik_ek_from_cache(c_d, D_k, weights)
        fd = (vals[+1] - vals[-1]) / (2.0 * h)
        assert abs(grad[a, d] - fd) <= 1e-7, (
            f"atom {a} axis {d}: analytic={grad[a, d]:.12f} "
            f"FD={fd:.12f} diff={abs(grad[a, d] - fd):.3e}"
        )


@pytest.mark.parametrize(
    "atomic_numbers, assembly_gate",
    [((1, 1), 1e-12), ((3, 1), 2e-11)],
    ids=["h2_ss", "lih_general_l"],
)
def test_multik_k_gradient_reduces_to_gamma_at_1k(
    atomic_numbers, assembly_gate
):
    """Consistency reduction (rung 4): at kmesh=(1,1,1) the multi-k K
    assembly matches the Γ ``_compute_k_gradient_rsgdf`` at
    ``alpha_hf = 1``, and the new complex-weight 2c Bloch metric
    kernel reduces to the real Γ kernel at q = 0 with a real weight
    to <= 1e-13 (unit consistency of the C_d contraction:
    ``Im(W_real ∘ C_d) = W_real ∘ Im(C_d)`` elementwise and the
    shifted mesh IS the unshifted mesh at q = 0).

    Same idempotent-D construction as the Γ rung-3 gate
    (D = 2 C_occ C_occ^T): the Γ template's occupied-space weight
    algebra (A = C^T T C Gram, -alpha / -2 alpha prefactors) and the
    multi-k density-space algebra (G(q) Gram over pairs, -0.25 /
    -0.5 w_i w_j prefactors) coincide exactly there — the docstring
    Γ-reduction identity ``G = 4 outer-gram(A)``. Base + tail meshes
    both exercised. Measured max diffs: 2c kernel 0.0 / 0.0
    (byte-identical); K assembly H2 4.4e-14, LiH 6.8e-12. The LiH
    assembly gate is 2e-11, NOT 1e-12: the two caches accumulate the
    same metric with different rounding (Γ real-accumulated
    per chunk, multi-k complex-accumulated then real-projected;
    |ΔM| ~ 2e-15), and the K weight's truncated-inverse response
    amplifies the resulting eigenvector rounding by the fit-response
    conditioning eps·λ_max/λ_min_kept (7.1e-9 for LiH/def2-svp-jk at
    ke=40 vs 1.3e-11 for H2 — the
    ``rsgdf_fit_response_conditioning`` discriminator). The linear
    J weight has no such quadratic response and its reduction sits at
    1e-15 for both systems.
    """
    from vibeqc.aux_basis import (
        _rsgdf_weighted_2c_metric_gradient,
        _rsgdf_weighted_2c_metric_gradient_bloch,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic_gdf_gradient import (
        _build_multik_rsgdf_gradient_cache,
        _build_rsgdf_gradient_cache,
        _compute_k_gradient_multik_rsgdf,
        _compute_k_gradient_rsgdf,
    )

    ke = 40.0
    tail_ke = 80.0
    thr = 1e-9
    lat = _lat_opts()
    system, basis = _diatomic_12bohr_system(*atomic_numbers)
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)

    rng = np.random.default_rng(19)
    C_occ, _ = np.linalg.qr(rng.normal(size=(basis.nbasis, 2)))
    D = 2.0 * C_occ @ C_occ.T

    gamma_cache = _build_rsgdf_gradient_cache(
        system, basis, modrho,
        ke_cutoff=ke, tail_ke_cutoff=tail_ke, lat_opts=lat,
        linear_dep_thr=thr,
    )
    g_gamma = _compute_k_gradient_rsgdf(
        system, basis, D, C_occ, 1.0, gamma_cache
    )

    multik_cache = _build_multik_rsgdf_gradient_cache(
        system, basis, modrho, np.zeros((1, 3)),
        ke_cutoff=ke, tail_ke_cutoff=tail_ke, lat_opts=lat,
        linear_dep_thr=thr,
    )
    g_multik = _compute_k_gradient_multik_rsgdf(
        system, basis, [D.astype(np.complex128)], [1.0], multik_cache
    )
    assert np.abs(g_multik - g_gamma).max() <= assembly_gate

    # Unit consistency of the new 2c kernel at q = 0 with a real
    # weight against the Γ-only real-weight kernel.
    W = rng.standard_normal((modrho.nbasis, modrho.nbasis))
    W = 0.5 * (W + W.T)
    g2_gamma = _rsgdf_weighted_2c_metric_gradient(
        modrho, system, ke_cutoff=ke, weight=W, tail_ke_cutoff=tail_ke
    )
    g2_bloch = _rsgdf_weighted_2c_metric_gradient_bloch(
        modrho, system, ke_cutoff=ke, q_cart=np.zeros(3),
        weight=W.astype(np.complex128), tail_ke_cutoff=tail_ke,
    )
    assert np.abs(g2_bloch - g2_gamma).max() <= 1e-13


def test_multik_krhf_full_scf_gradient_vs_fd(monkeypatch):
    """HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 4 — THE composition
    gate: the assembled multi-k KRHF gradient
    (``_compute_krhf_gradient_multik`` = oneel/W/nn + DF-J + DF-K +
    exxdiv shift, all FD-gated per term) matches a central difference
    of the DRIVER's converged total energy at (2,1,1).

    Converged KRHF (exxdiv='ewald' automatic on the KRHF path) at
    ke=60, conv_tol 1e-11; analytic side from the converged
    D(k)/C_occ(k)/eps_occ(k)/S(k) with the SCF's own resolved lattice
    options (``_oneel_lattice_opts`` / ``_gauge_lat_opts_for_v_ne_
    and_e_nuc``) and the BvK ``_madelung_for_kmesh`` xi; FD side reruns
    run_krhf_periodic_gdf at ±h = 2e-4 with identical settings.
    VIBEQC_VNE_EWALD3D_KE=8.0 speeds the reruns; SCF and gradient
    resolve the same env value, so the differentiated functional is
    consistent. Measured: bond-axis |analytic - FD| = 1.1e-8 at
    h=2e-4 with Newton's-3rd-law residual 2.2e-16 (the Γ full-SCF
    anchor sits at ~4e-9); gate 1e-6. If this gate ever fails, bisect
    with the per-term fixed-D gates above — do NOT paper over
    (CLAUDE.md § 7).
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_gdf_gradient import (
        _compute_krhf_gradient_multik,
    )
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        _oneel_lattice_opts,
        run_krhf_periodic_gdf,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc

    ke = 60.0
    thr = 1e-9
    lat = _lat_opts()
    h = 2e-4

    def run(dz=0.0):
        box = 12.0
        c = box / 2.0
        s = vq.PeriodicSystem(3, np.eye(3) * box, [
            vq.Atom(1, [c, c, c - 0.7 + dz]),
            vq.Atom(1, [c, c, c + 0.7]),
        ])
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = lat
        o.max_iter = 160
        o.conv_tol_energy = 1e-11
        res = run_krhf_periodic_gdf(
            s, b, (2, 1, 1), o, aux_basis="def2-svp-jk",
            use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=ke,
            progress=False,
        )
        assert res.converged
        return s, b, res

    system, basis, r = run()
    n_k = 2
    D_k = [np.asarray(D) for D in r.density]
    S_k = [np.asarray(S) for S in r.overlap]
    weights = np.asarray(r.kpoint_weights, dtype=float)
    kpts = np.asarray(r.kpoints_cart, dtype=float)
    n_occ = system.n_electrons() // 2
    W_k = []
    for ik in range(n_k):
        C_occ = np.asarray(r.mo_coeffs[ik])[:, :n_occ]
        eps_occ = np.asarray(r.mo_energies[ik])[:n_occ]
        W_k.append(2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T)

    oneel = _oneel_lattice_opts(
        system, basis, lat, rcut_strategy="pyscf_auto", k_points_cart=kpts
    )
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    xi = _madelung_for_kmesh(system, (2, 1, 1))
    cache = _scf_multik_source_for(system, basis, kpts, ke, None, lat, thr)

    grad = _compute_krhf_gradient_multik(
        system, basis, D_k, W_k, S_k, weights, kpts, cache,
        alpha_hf=1.0, madelung=xi,
        oneel_lat_opts=oneel, gauge_lat_opts=gauge,
    )
    assert grad.shape == (2, 3)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-7)

    fd = (run(dz=+h)[2].energy - run(dz=-h)[2].energy) / (2.0 * h)
    assert abs(grad[0, 2] - fd) <= 1e-6, (
        f"analytic={grad[0, 2]:.12f} FD={fd:.12f} "
        f"diff={abs(grad[0, 2] - fd):.3e}"
    )


def test_multik_screened_krhf_full_scf_gradient_vs_fd(monkeypatch):
    """Screened-fit envelope gate (2026-07-30): the multi-k KRHF driver
    gradient with a Schwarz-screened fit matches FD AT FIXED MASK.

    ``run_krhf_periodic_gdf(compute_gradient=True)`` at (2,1,1)/ke=60
    with ``fit_screen_threshold=0.54`` (H2: drops the 2 inter-atom AO
    pairs of 4 in every q group) versus a central difference of the
    same driver's converged energy. The gate additionally pins that the
    per-q Schwarz masks are non-trivial and IDENTICAL at the three FD
    geometries (mid-gap threshold; a flip would differentiate a
    different objective on each side — the M6-rung-9 lesson class).
    Observed ~1.2e-8 against the 1e-6 gate."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_k_gdf import (
        _kmesh_to_kpoints_weights,
        run_krhf_periodic_gdf,
    )
    from vibeqc.aux_basis import _canonical_reciprocal_transfer

    ke = 60.0
    thr = _H2_SCREEN_THR
    lat = _lat_opts()
    h = 2e-4

    def run(dz=0.0, grad=False):
        s, b = _multik_h2_211_system(dz=dz)
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = lat
        o.max_iter = 160
        o.conv_tol_energy = 1e-11
        res = run_krhf_periodic_gdf(
            s, b, (2, 1, 1), o, aux_basis="def2-svp-jk",
            use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=ke,
            fit_screen_threshold=thr,
            compute_gradient=grad, progress=False,
        )
        assert res.converged
        return s, b, res

    # Fixed-mask pin at BOTH momentum transfers of the (2,1,1) mesh.
    system0, basis0 = _multik_h2_211_system()
    kpts, _w = _kmesh_to_kpoints_weights(system0, (2, 1, 1))
    q_list = [
        np.zeros(3),
        _canonical_reciprocal_transfer(system0, kpts[1] - kpts[0]),
    ]
    for q in q_list:
        masks = []
        for dz in (-h, 0.0, +h):
            s, b = _multik_h2_211_system(dz=dz)
            mol = s.unit_cell_molecule()
            modrho = make_modrho_aux_basis(
                make_aux_basis_set(mol, aux_name="def2-svp-jk"), mol
            )
            masks.append(_rsgdf_screen_keep_mask(
                s, b, modrho, ke=ke, thr=thr, lat=lat, q=q
            ))
        assert masks[1] is not None
        assert 0 < int(np.count_nonzero(~masks[1])) < basis0.nbasis**2
        assert np.array_equal(masks[0], masks[1])
        assert np.array_equal(masks[2], masks[1])

    _s, _b, r = run(grad=True)
    grad = np.asarray(r.gradient)
    assert grad.shape == (2, 3)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-7)
    fd = (run(dz=+h)[2].energy - run(dz=-h)[2].energy) / (2.0 * h)
    assert abs(grad[0, 2] - fd) <= 1e-6, (
        f"analytic={grad[0, 2]:.12f} FD={fd:.12f} "
        f"diff={abs(grad[0, 2] - fd):.3e}"
    )


def _multik_uhf_spin_data(r, n_alpha, n_beta):
    """Per-spin D/W/S/weights/kpts from a converged KUHF/KUKS result.

    ``W(k) = S_s C_s,occ(k) diag(eps_s,occ(k)) C_s,occ(k)^H`` — the
    occupation-1 open-shell convention (no closed-shell factor 2)."""
    n_k = len(r.kpoints_cart)
    D_a = [np.asarray(D) for D in r.density_alpha]
    D_b = [np.asarray(D) for D in r.density_beta]
    S_k = [np.asarray(S) for S in r.overlap]
    W_k = []
    for ik in range(n_k):
        W = np.zeros_like(S_k[ik], dtype=np.complex128)
        for C_s, e_s, n_s in (
            (r.mo_coeffs_alpha[ik], r.mo_energies_alpha[ik], n_alpha),
            (r.mo_coeffs_beta[ik], r.mo_energies_beta[ik], n_beta),
        ):
            if n_s == 0:
                continue
            C_occ = np.asarray(C_s)[:, :n_s]
            eps_occ = np.asarray(e_s)[:n_s]
            W = W + (C_occ * eps_occ[None, :]) @ C_occ.conj().T
        W_k.append(W)
    weights = np.asarray(r.kpoint_weights, dtype=float)
    kpts = np.asarray(r.kpoints_cart, dtype=float)
    return D_a, D_b, W_k, S_k, weights, kpts


def test_multik_kuhf_full_scf_gradient_vs_fd(monkeypatch):
    """HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 5 — open-shell
    composition gate: the assembled multi-k KUHF gradient
    (``_compute_kuhf_gradient_multik`` = oneel/W/nn on the spin-summed
    density + DF-J on D_t + per-spin 2a-weighted DF-K + per-spin
    exxdiv shift) matches a central difference of the DRIVER's
    converged total energy at (2,1,1).

    H2 triplet (n_alpha = 2, n_beta = 0 — the Γ ladder's open-shell
    anchor; the empty beta channel also pins the zero-density spin
    skip), rsgdf ke=60, conv_tol 1e-11, exxdiv='ewald' automatic on
    the KUHF path with the BvK ``_madelung_for_kmesh`` xi. FD side
    reruns run_kuhf_periodic_gdf at ±h = 2e-4 with identical settings
    (VIBEQC_VNE_EWALD3D_KE=8.0 on both sides). Measured: bond-axis
    |analytic - FD| = 1.6e-8 at h=2e-4, Newton's-3rd-law residual
    1.2e-15; gate 1e-6. If this fails, bisect with the per-term
    fixed-D gates above — do NOT paper over (CLAUDE.md § 7).
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_gdf_gradient import (
        _compute_kuhf_gradient_multik,
    )
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        _oneel_lattice_opts,
        run_kuhf_periodic_gdf,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc

    ke = 60.0
    thr = 1e-9
    lat = _lat_opts()
    h = 2e-4

    def run(dz=0.0):
        box = 12.0
        c = box / 2.0
        s = vq.PeriodicSystem(3, np.eye(3) * box, [
            vq.Atom(1, [c, c, c - 0.7 + dz]),
            vq.Atom(1, [c, c, c + 0.7]),
        ])
        s.multiplicity = 3
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = lat
        o.max_iter = 200
        o.conv_tol_energy = 1e-11
        res = run_kuhf_periodic_gdf(
            s, b, (2, 1, 1), o, aux_basis="def2-svp-jk",
            gdf_method="rsgdf", rsgdf_ke_cutoff=ke, progress=False,
        )
        assert res.converged
        return s, b, res

    system, basis, r = run()
    D_a, D_b, W_k, S_k, weights, kpts = _multik_uhf_spin_data(r, 2, 0)

    oneel = _oneel_lattice_opts(
        system, basis, lat, rcut_strategy="pyscf_auto", k_points_cart=kpts
    )
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    xi = _madelung_for_kmesh(system, (2, 1, 1))
    cache = _scf_multik_source_for(system, basis, kpts, ke, None, lat, thr)

    grad = _compute_kuhf_gradient_multik(
        system, basis, D_a, D_b, W_k, S_k, weights, kpts, cache,
        alpha_hf=1.0, madelung=xi,
        oneel_lat_opts=oneel, gauge_lat_opts=gauge,
    )
    assert grad.shape == (2, 3)
    assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-7)

    fd = (run(dz=+h)[2].energy - run(dz=-h)[2].energy) / (2.0 * h)
    assert abs(grad[0, 2] - fd) <= 1e-6, (
        f"analytic={grad[0, 2]:.12f} FD={fd:.12f} "
        f"diff={abs(grad[0, 2] - fd):.3e}"
    )


def test_multik_kuhf_gradient_reduces_to_gamma_at_1k(monkeypatch):
    """Γ-reduction (rung 5): at kmesh=(1,1,1) the multi-k KUHF assembly
    matches the FD-validated Γ rsgdf open-shell assembly
    ``compute_gdf_gradient_rsgdf_uhf_gamma`` on the same converged
    per-spin data to <= 1e-10.

    This pins the per-spin composition factors against the Γ template:
    the multi-k ``2 a x`` helper-on-D_s exchange weight vs the Γ
    ``0.5 x kernel(alpha)`` occupied-orbital weight, and the per-spin
    doubled exxdiv helper vs the Γ ``W_exx = a xi S_s D_s S D_s``.
    The J/oneel/V_ne/K mechanical reductions are pinned by the rung
    3a/3b/4 gates; the same lattice options are passed on both sides
    (oneel_lat_opts = the Γ template's lattice_opts) so this isolates
    the spin algebra. run_kuhf_periodic_gdf has no Γ fast-path
    delegation, so (1,1,1) runs the true multi-k loop. Measured max
    diff 1.7e-14.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.aux_basis import make_aux_basis_set, make_modrho_aux_basis
    from vibeqc.periodic_gdf_gradient import (
        _build_rsgdf_gradient_cache,
        _compute_kuhf_gradient_multik,
        compute_gdf_gradient_rsgdf_uhf_gamma,
    )
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        run_kuhf_periodic_gdf,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc

    ke = 60.0
    thr = 1e-9
    lat = _lat_opts()
    v_ne_ke = 8.0

    system, basis = _h2_12bohr_system()
    system.multiplicity = 3
    o = vq.PeriodicRHFOptions()
    o.lattice_opts = lat
    o.max_iter = 200
    o.conv_tol_energy = 1e-11
    r = run_kuhf_periodic_gdf(
        system, basis, (1, 1, 1), o, aux_basis="def2-svp-jk",
        gdf_method="rsgdf", rsgdf_ke_cutoff=ke, progress=False,
    )
    assert r.converged
    n_alpha, n_beta = 2, 0
    D_a, D_b, W_k, S_k, weights, kpts = _multik_uhf_spin_data(
        r, n_alpha, n_beta
    )
    # Γ densities/overlaps are real to machine precision; the complex
    # per-k diagonalization leaves an arbitrary U(1) phase on each MO
    # column, so fix the phase (the Γ K objective depends on the
    # projector C C^H only) before realifying for the Γ assembly.
    for M in (D_a[0], S_k[0]):
        assert float(np.abs(np.imag(M)).max()) <= 1e-12
    C_a = np.asarray(r.mo_coeffs_alpha[0])[:, :n_alpha]
    lead = np.argmax(np.abs(C_a), axis=0)
    C_a = C_a * np.exp(
        -1j * np.angle(C_a[lead, np.arange(C_a.shape[1])])
    )[None, :]
    assert float(np.abs(np.imag(C_a)).max()) <= 1e-10

    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    xi = _madelung_for_kmesh(system, (1, 1, 1))

    cache = _multik_j_cache_for(system, basis, kpts, ke, None, lat, thr)
    g_multik = _compute_kuhf_gradient_multik(
        system, basis, D_a, D_b, W_k, S_k, weights, kpts, cache,
        alpha_hf=1.0, madelung=xi,
        oneel_lat_opts=lat, gauge_lat_opts=gauge,
        v_ne_ke_cutoff=v_ne_ke,
    )

    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk")
    modrho = make_modrho_aux_basis(aux, mol)
    gamma_cache = _build_rsgdf_gradient_cache(
        system, basis, modrho,
        ke_cutoff=ke, tail_ke_cutoff=None, lat_opts=lat,
        linear_dep_thr=thr,
    )
    g_gamma = compute_gdf_gradient_rsgdf_uhf_gamma(
        system, basis,
        D_alpha=np.real(D_a[0]),
        D_beta=np.real(D_b[0]),
        C_alpha_occ=np.real(C_a),
        C_beta_occ=np.zeros((basis.nbasis, 0)),
        eps_alpha_occ=np.asarray(r.mo_energies_alpha[0])[:n_alpha],
        eps_beta_occ=np.zeros(0),
        S=np.real(S_k[0]),
        cache=gamma_cache,
        lattice_opts=lat,
        gauge_lat_opts=gauge,
        madelung=xi,
        v_ne_ke_cutoff=v_ne_ke,
        alpha_hf=1.0,
    )
    assert np.abs(g_multik - g_gamma).max() <= 1e-10


def _run_multik_krks(functional, dz=0.0):
    """Converged (2,1,1) KRKS on the H2/12-bohr fixture, atom 0
    displaced by ``dz`` along z. PeriodicKSOptions defaults
    (use_periodic_becke=True) — the gradient side must rebuild the
    same quadrature."""
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    box = 12.0
    c = box / 2.0
    s = vq.PeriodicSystem(3, np.eye(3) * box, [
        vq.Atom(1, [c, c, c - 0.7 + dz]),
        vq.Atom(1, [c, c, c + 0.7]),
    ])
    b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
    o = vq.PeriodicKSOptions()
    o.lattice_opts = _lat_opts()
    o.max_iter = 200
    o.conv_tol_energy = 1e-11
    res = run_krks_periodic_gdf(
        s, b, (2, 1, 1), o, functional=functional,
        aux_basis="def2-svp-jk", use_compcell=True, gdf_method="rsgdf",
        rsgdf_ke_cutoff=60.0, progress=False,
    )
    assert res.converged
    return s, b, o, res


def _multik_krks_grad_inputs(system, basis, opts, r):
    """The gradient-side rebuild of the (2,1,1) KRKS SCF's own
    quadrature/options: periodic-Becke grid per the KS options, the
    driver's kmesh_bloch/lat_opts for the XC fold, resolved
    oneel/gauge options, and the BvK xi."""
    from vibeqc._vibeqc_core import GridOptions
    from vibeqc._vibeqc_core import monkhorst_pack as mp_native
    from vibeqc.periodic_grid import build_periodic_becke_grid
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        _oneel_lattice_opts,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc

    lat = opts.lattice_opts
    kpts = np.asarray(r.kpoints_cart, dtype=float)
    weights = np.asarray(r.kpoint_weights, dtype=float)
    assert bool(opts.use_periodic_becke)  # PeriodicKSOptions default
    grid = build_periodic_becke_grid(
        system,
        grid_options=(getattr(opts, "grid", None) or GridOptions()),
        image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
    )
    kmesh_bloch = mp_native(system, [2, 1, 1], [0, 0, 0], False)
    oneel = _oneel_lattice_opts(
        system, basis, lat, rcut_strategy="pyscf_auto", k_points_cart=kpts
    )
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    xi = _madelung_for_kmesh(system, (2, 1, 1))
    return lat, kpts, weights, grid, kmesh_bloch, oneel, gauge, xi


def test_multik_krks_full_scf_gradient_vs_fd(monkeypatch):
    """Rung-5 KS composition gate: the assembled multi-k KRKS gradient
    (``_compute_krks_gradient_multik`` = KRHF assembler at the
    functional's a_x + XC Pulay on the SCF's own folded density and
    periodic-Becke quadrature) matches a central difference of the
    DRIVER's converged total energy at (2,1,1).

    Two functionals split the KS-specific terms: lda exercises the
    a_x = 0 no-K branch + the LDA XC Pulay of the folded multi-k
    density; pbe0 the a_x-scaled DF-K, the per-k exxdiv shift, and the
    GGA s-piece. The fixed-grid XC Pulay is the same lattice-summed
    primitive used by the BIPOLE route, and the moving-grid correction
    rebuilds that same SCF quadrature with fixed density.
    Measured |analytic - FD| at h=2e-4: lda 4.1e-10, pbe0 1.5e-8;
    gate 1e-6. Newton's-3rd-law residual observed ~4e-15 (gated
    loosely at 1e-7: the XC grid-motion term is only required to be
    translationally consistent at quadrature accuracy).
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_gdf_gradient import (
        _compute_krks_gradient_multik,
    )

    thr = 1e-9
    h = 2e-4

    for functional in ("lda", "pbe0"):
        system, basis, opts, r = _run_multik_krks(functional)
        (lat, kpts, weights, grid, kmesh_bloch, oneel, gauge, xi) = (
            _multik_krks_grad_inputs(system, basis, opts, r)
        )
        n_k = 2
        D_k = [np.asarray(D) for D in r.density]
        S_k = [np.asarray(S) for S in r.overlap]
        n_occ = system.n_electrons() // 2
        W_k = []
        for ik in range(n_k):
            C_occ = np.asarray(r.mo_coeffs[ik])[:, :n_occ]
            eps_occ = np.asarray(r.mo_energies[ik])[:n_occ]
            W_k.append(2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T)
        cache = _scf_multik_source_for(
            system, basis, kpts, 60.0, None, lat, thr
        )
        grad = _compute_krks_gradient_multik(
            system, basis, D_k, W_k, S_k, weights, kpts, cache,
            functional=functional, xc_grid=grid, kmesh_bloch=kmesh_bloch,
            xc_lat_opts=lat, madelung=xi,
            oneel_lat_opts=oneel, gauge_lat_opts=gauge,
            xc_grid_options=opts.grid,
            xc_use_periodic_becke=bool(opts.use_periodic_becke),
            xc_becke_image_radius_bohr=float(opts.becke_image_radius_bohr),
        )
        assert grad.shape == (2, 3)
        assert np.allclose(grad.sum(axis=0), 0.0, atol=1e-7)

        fd = (
            _run_multik_krks(functional, dz=+h)[3].energy
            - _run_multik_krks(functional, dz=-h)[3].energy
        ) / (2.0 * h)
        assert abs(grad[0, 2] - fd) <= 1e-6, (
            f"{functional}: analytic={grad[0, 2]:.12f} FD={fd:.12f} "
            f"diff={abs(grad[0, 2] - fd):.3e}"
        )


def test_multik_kuks_open_shell_full_scf_gradient_vs_fd(monkeypatch):
    """Issue #158: genuine open-shell KUKS includes atom-centred XC grid
    motion and matches central FD of its own converged energy.

    The old fixed-grid-only assembly missed 2.680e-3 Ha/bohr on H-z and
    violated translational invariance by 7.143e-3 Ha/bohr on this triplet.
    """
    from vibeqc.periodic_k_gdf import run_kuks_periodic_gdf

    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    h = 2e-4

    def run(dz: float, *, compute_gradient: bool = False):
        system = vq.PeriodicSystem(
            3,
            np.eye(3) * 7.0,
            [
                vq.Atom(3, [3.5, 3.5, 2.6]),
                vq.Atom(1, [3.5, 3.5, 5.6 + dz]),
            ],
            charge=0,
            multiplicity=3,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicKSOptions()
        opts.max_iter = 200
        opts.conv_tol_energy = 1e-10
        opts.use_periodic_becke = True
        opts.grid.n_radial = 20
        opts.grid.angular = "lebedev"
        opts.grid.lebedev_order = 17
        return run_kuks_periodic_gdf(
            system,
            basis,
            (1, 1, 1),
            opts,
            functional="pbe",
            aux_basis="def2-svp-jk",
            gdf_method="rsgdf",
            rsgdf_ke_cutoff=60.0,
            compute_gradient=compute_gradient,
            progress=False,
        )

    center = run(0.0, compute_gradient=True)
    plus = run(+h)
    minus = run(-h)
    assert center.converged and plus.converged and minus.converged
    assert center.gradient is not None
    fd = (float(plus.energy) - float(minus.energy)) / (2.0 * h)
    analytic = float(np.asarray(center.gradient)[1, 2])
    assert abs(analytic - fd) <= 1e-6, (
        f"KUKS analytic={analytic:.12f} FD={fd:.12f} "
        f"diff={abs(analytic - fd):.3e}"
    )
    assert np.max(np.abs(np.asarray(center.gradient).sum(axis=0))) <= 1e-7


def test_multik_kuks_m1_collapses_onto_krks(monkeypatch):
    """Rung-5 KUKS algebra gate: on the same converged (2,1,1) KRKS
    pbe0 data, the per-spin KUKS assembler at D_a = D_b = D/2 with the
    identical occupation-1 W collapses onto the closed-shell KRKS
    assembler at machine precision.

    This pins every per-spin rung-5 factor against the FD-gated
    closed-shell composition: 2 a_x x helper(D/2) per spin == a_x x
    helper(D) (the K helper is quadratic in D), the doubled per-spin
    exxdiv helper == the closed-shell one, the spin-summed J/oneel
    terms, and the spin-paired ``xc_lattice_gradient_contribution_uks``
    == the closed-shell XC primitive on the pointwise-identical folded
    density (the KUKS(M=1) == KRKS SCF construction). pbe0 (not lda)
    so the exchange and shift factors are actually exercised.
    Measured max diff 2.8e-17; gate 1e-10.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_gdf_gradient import (
        _compute_krks_gradient_multik,
        _compute_kuks_gradient_multik,
    )

    thr = 1e-9
    functional = "pbe0"
    system, basis, opts, r = _run_multik_krks(functional)
    (lat, kpts, weights, grid, kmesh_bloch, oneel, gauge, xi) = (
        _multik_krks_grad_inputs(system, basis, opts, r)
    )
    n_k = 2
    D_k = [np.asarray(D) for D in r.density]
    S_k = [np.asarray(S) for S in r.overlap]
    n_occ = system.n_electrons() // 2
    W_k = []
    for ik in range(n_k):
        C_occ = np.asarray(r.mo_coeffs[ik])[:, :n_occ]
        eps_occ = np.asarray(r.mo_energies[ik])[:n_occ]
        W_k.append(2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T)
    cache = _scf_multik_source_for(system, basis, kpts, 60.0, None, lat, thr)

    common = dict(
        functional=functional, xc_grid=grid, kmesh_bloch=kmesh_bloch,
        xc_lat_opts=lat, madelung=xi,
        oneel_lat_opts=oneel, gauge_lat_opts=gauge,
        xc_grid_options=opts.grid,
        xc_use_periodic_becke=bool(opts.use_periodic_becke),
        xc_becke_image_radius_bohr=float(opts.becke_image_radius_bohr),
    )
    g_rks = _compute_krks_gradient_multik(
        system, basis, D_k, W_k, S_k, weights, kpts, cache, **common
    )
    D_half = [0.5 * D for D in D_k]
    g_uks = _compute_kuks_gradient_multik(
        system, basis, D_half, D_half, W_k, S_k, weights, kpts, cache,
        **common,
    )
    assert np.abs(g_uks - g_rks).max() <= 1e-10


def test_rsgdf_gradient_conditioning_diagnostic(monkeypatch):
    """The fit-response conditioning estimate separates the dense-core
    class from vacuum anchors, and no gradient-accuracy warning fires.

    Validated formula (gamma-split discriminator, dense-core gradient
    entry in HANDOVER_OPEN_BUGS_V015.md): eps * lam_max / lam_min_kept.
    MgO at the driver threshold sits ~3e-7; the H2 vacuum anchor sits
    ~1e-11. Since the 2026-07-29 e_nuc Ewald-gauge fix the analytic
    dense-core gradient matches full-SCF FD to < 5e-9 Ha/bohr, so the
    indicator is informational only -- a RuntimeWarning here would be
    misleading and must NOT be raised.
    """
    import warnings as _warnings

    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")

    ang2bohr = 1.8897259886
    a = 4.211 * ang2bohr
    prim = 0.5 * a * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float
    )
    o = 0.5 * (prim[0] + prim[1] + prim[2])
    mgo = vq.PeriodicSystem(
        3, prim, [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, o.tolist())]
    )
    mgo_basis = vq.BasisSet(mgo.unit_cell_molecule(), "sto-3g")
    lat_mgo = LatticeSumOptions()
    lat_mgo.cutoff_bohr = 18.0
    lat_mgo.nuclear_cutoff_bohr = 18.0

    h2, h2_basis = _h2_12bohr_system()
    lat_h2 = _lat_opts()

    def run(system, basis, lat):
        options = vq.PeriodicRHFOptions()
        options.lattice_opts = lat
        options.max_iter = 80
        options.conv_tol_energy = 1e-10
        return vq.run_pbc_gdf_rhf(
            system, basis, options=options, aux_basis="def2-svp-jk",
            gdf_method="rsgdf", rsgdf_ke_cutoff=60.0,
            rsgdf_tail_ke_cutoff=0.0, exxdiv="ewald",
            compute_gradient=True, progress=False,
        )

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        r_mgo = run(mgo, mgo_basis, lat_mgo)
        mgo_warns = [
            x for x in w if "dense-core fit class" in str(x.message)
        ]
    assert r_mgo.gradient_conditioning > 1e-8
    assert not mgo_warns

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        r_h2 = run(h2, h2_basis, lat_h2)
        h2_warns = [
            x for x in w if "dense-core fit class" in str(x.message)
        ]
    assert r_h2.gradient_conditioning < 1e-9
    assert not h2_warns


def _mgo_primitive_system():
    """MgO primitive FCC (a = 4.211 Ang), the dense-core anchor."""
    ang2bohr = 1.8897259886
    a = 4.211 * ang2bohr
    prim = 0.5 * a * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float
    )
    o = 0.5 * (prim[0] + prim[1] + prim[2])
    system = vq.PeriodicSystem(
        3, prim, [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, o.tolist())]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_gdf_e_nuc_is_converged_ewald(monkeypatch):
    """The 3D GDF drivers' nuclear energy is the converged Ewald sum.

    Pins the 2026-07-29 e_nuc gauge fix: nuclear_repulsion_per_cell
    under the EWALD_3D gauge truncates its real-space sum at
    lat_opts.nuclear_cutoff_bohr and on MgO at 18 bohr is unconverged
    by 2.1e-5 Ha WITH a spurious geometry dependence -- the source of
    the historical "dense-core gradient discrepancy" (RESOLVED entry,
    HANDOVER_OPEN_BUGS_V015.md). The driver e_nuclear must equal
    ewald_nuclear_repulsion(system) exactly (same function, no
    truncation), on both the Gamma and the multi-k route.
    """
    from vibeqc._vibeqc_core import ewald_nuclear_repulsion

    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    system, basis = _mgo_primitive_system()
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 18.0
    lat.nuclear_cutoff_bohr = 18.0
    e_ewald = float(ewald_nuclear_repulsion(system))

    options = vq.PeriodicRHFOptions()
    options.lattice_opts = lat
    options.max_iter = 80
    options.conv_tol_energy = 1e-10
    r_gamma = vq.run_pbc_gdf_rhf(
        system, basis, options=options, aux_basis="def2-svp-jk",
        gdf_method="rsgdf", rsgdf_ke_cutoff=60.0,
        rsgdf_tail_ke_cutoff=0.0, exxdiv="ewald", progress=False,
    )
    assert r_gamma.e_nuclear == pytest.approx(e_ewald, abs=1e-12)

    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    r_mk = run_krhf_periodic_gdf(
        system, basis, kmesh=(1, 1, 1), options=options,
        aux_basis="def2-svp-jk", progress=False,
    )
    assert r_mk.e_nuclear == pytest.approx(e_ewald, abs=1e-12)


@pytest.mark.slow
def test_dense_core_full_scf_gradient_vs_fd(monkeypatch):
    """Dense-core (MgO) full-SCF FD regression gate.

    The historical symptom this pins against returning: a FLAT
    2.544e-5 Ha/bohr analytic-vs-FD gap on MgO/STO-3G, invisible on
    vacuum-box anchors, caused by the truncated e_nuc (see
    test_gdf_e_nuc_is_converged_ewald). Post-fix measurement: 3.3e-10
    at h=2e-4 (and 3e-10..5e-9 across h=1e-4..8e-4); the 1e-6 gate
    fails loudly at the historical 2.5e-5.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    ang2bohr = 1.8897259886
    a = 4.211 * ang2bohr
    prim = 0.5 * a * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float
    )
    o = 0.5 * (prim[0] + prim[1] + prim[2])
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 18.0
    lat.nuclear_cutoff_bohr = 18.0

    def run(dz=0.0, grad=False):
        s = vq.PeriodicSystem(
            3, prim, [vq.Atom(12, [0.0, 0.0, dz]), vq.Atom(8, o.tolist())]
        )
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        options = vq.PeriodicRHFOptions()
        options.lattice_opts = lat
        options.max_iter = 120
        options.conv_tol_energy = 1e-12
        return vq.run_pbc_gdf_rhf(
            s, b, options=options, aux_basis="def2-svp-jk",
            gdf_method="rsgdf", rsgdf_ke_cutoff=60.0,
            rsgdf_tail_ke_cutoff=0.0, exxdiv="ewald",
            compute_gradient=grad, progress=False,
        )

    r0 = run(grad=True)
    g = float(np.asarray(r0.gradient)[0, 2])
    h = 2e-4
    fd = (run(dz=+h).energy - run(dz=-h).energy) / (2.0 * h)
    assert abs(g - fd) < 1e-6


# ---------------------------------------------------------------------------
# Item-4 rung 6: public multi-k driver wiring + fail-closed guards
# ---------------------------------------------------------------------------


def _multik_h2_211_system(dz=0.0, multiplicity=1):
    box = 12.0
    c = box / 2.0
    s = vq.PeriodicSystem(3, np.eye(3) * box, [
        vq.Atom(1, [c, c, c - 0.7 + dz]),
        vq.Atom(1, [c, c, c + 0.7]),
    ])
    if multiplicity != 1:
        s.multiplicity = multiplicity
    b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
    return s, b


def test_multik_krhf_driver_gradient_wiring(monkeypatch):
    """Rung 6 public wiring: ``run_krhf_periodic_gdf(compute_gradient=
    True)`` populates ``result.gradient`` with the value the FD-gated
    private assembler produces on the same converged run (identical to
    the OMP reduction-noise floor, ~1e-23).

    The driver's gradient block hands its own converged D(k)/C(k)/
    eps(k), resolved lattice options, BvK xi, and a fresh
    deterministic cache rebuild to ``_compute_krhf_gradient_multik``;
    recomputing here with the same inputs must agree to 1e-15 (any
    real drift means the wiring passes different provenance than the
    composition gate validated). The bond-axis value is pinned against
    the rung-4 FD gate's measurement (analytic - FD = 1.1e-8 at
    h=2e-4): g_z(atom 0) = -0.036128 Ha/bohr.
    """
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_gdf_gradient import _compute_krhf_gradient_multik
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        _oneel_lattice_opts,
        run_krhf_periodic_gdf,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc

    ke = 60.0
    thr = 1e-9
    lat = _lat_opts()
    system, basis = _multik_h2_211_system()
    o = vq.PeriodicRHFOptions()
    o.lattice_opts = lat
    o.max_iter = 160
    o.conv_tol_energy = 1e-11
    r = run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), o, aux_basis="def2-svp-jk",
        use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=ke,
        compute_gradient=True, progress=False,
    )
    assert r.converged
    assert r.gradient is not None and r.gradient.shape == (2, 3)
    assert np.abs(r.gradient.sum(axis=0)).max() <= 1e-12  # Newton 3
    # Pinned against the rung-4 full-SCF FD gate on this exact fixture.
    assert abs(r.gradient[0, 2] - (-0.0361276)) < 1e-5

    # Bit-identical recomposition from the returned converged state.
    n_k = 2
    n_occ = system.n_electrons() // 2
    D_k = [np.asarray(D) for D in r.density]
    S_k = [np.asarray(S) for S in r.overlap]
    weights = np.asarray(r.kpoint_weights, dtype=float)
    kpts = np.asarray(r.kpoints_cart, dtype=float)
    W_k = []
    for ik in range(n_k):
        C_occ = np.asarray(r.mo_coeffs[ik])[:, :n_occ]
        eps_occ = np.asarray(r.mo_energies[ik])[:n_occ]
        W_k.append(2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T)
    oneel = _oneel_lattice_opts(
        system, basis, lat, rcut_strategy="pyscf_auto", k_points_cart=kpts
    )
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    xi = _madelung_for_kmesh(system, (2, 1, 1))
    cache = _scf_multik_source_for(system, basis, kpts, ke, None, lat, thr)
    g_ref = _compute_krhf_gradient_multik(
        system, basis, D_k, W_k, S_k, weights, kpts, cache,
        alpha_hf=1.0, madelung=xi,
        oneel_lat_opts=oneel, gauge_lat_opts=gauge,
    )
    # Identical provenance up to the ~1e-23 OMP reduction-order noise
    # of the C++ kernels (measured: same-input reruns differ at
    # 2e-23; a provenance mismatch -- wrong lattice options, xi, or
    # fit parameters -- shows at >= 1e-9). Bit-identity is therefore
    # not a stable gate; 1e-15 is.
    assert np.abs(np.asarray(r.gradient) - np.asarray(g_ref)).max() <= 1e-15


def test_multik_kuhf_driver_gradient_wiring(monkeypatch):
    """Rung 6 open-shell wiring: ``run_kuhf_periodic_gdf(
    compute_gradient=True)`` on the H2 triplet (n_beta = 0) matches
    the FD-gated per-spin assembler to 1e-15 on the same converged
    run; bond-axis value pinned against the rung-5 FD gate
    (analytic - FD = 1.6e-8): g_z(atom 0) = +0.642252 Ha/bohr."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_gdf_gradient import _compute_kuhf_gradient_multik
    from vibeqc.periodic_k_gdf import (
        _madelung_for_kmesh,
        _oneel_lattice_opts,
        run_kuhf_periodic_gdf,
    )
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc

    ke = 60.0
    thr = 1e-9
    lat = _lat_opts()
    system, basis = _multik_h2_211_system(multiplicity=3)
    o = vq.PeriodicRHFOptions()
    o.lattice_opts = lat
    o.max_iter = 200
    o.conv_tol_energy = 1e-11
    r = run_kuhf_periodic_gdf(
        system, basis, (2, 1, 1), o, aux_basis="def2-svp-jk",
        gdf_method="rsgdf", rsgdf_ke_cutoff=ke,
        compute_gradient=True, progress=False,
    )
    assert r.converged
    assert r.gradient is not None and r.gradient.shape == (2, 3)
    assert np.abs(r.gradient.sum(axis=0)).max() <= 1e-12
    assert abs(r.gradient[0, 2] - 0.642252) < 1e-5

    D_a, D_b, W_k, S_k, weights, kpts = _multik_uhf_spin_data(r, 2, 0)
    oneel = _oneel_lattice_opts(
        system, basis, lat, rcut_strategy="pyscf_auto", k_points_cart=kpts
    )
    gauge = _gauge_lat_opts_for_v_ne_and_e_nuc(lat, system)
    xi = _madelung_for_kmesh(system, (2, 1, 1))
    cache = _scf_multik_source_for(system, basis, kpts, ke, None, lat, thr)
    g_ref = _compute_kuhf_gradient_multik(
        system, basis, D_a, D_b, W_k, S_k, weights, kpts, cache,
        alpha_hf=1.0, madelung=xi,
        oneel_lat_opts=oneel, gauge_lat_opts=gauge,
    )
    # Same-provenance gate at 1e-15 (see the KRHF wiring test for the
    # measured ~1e-23 OMP reduction-order noise that rules out strict
    # bit-identity).
    assert np.abs(np.asarray(r.gradient) - np.asarray(g_ref)).max() <= 1e-15


def test_multik_krks_driver_gradient_wiring(monkeypatch):
    """Rung 6 KS wiring: ``run_krks_periodic_gdf(functional='lda',
    compute_gradient=True)`` returns the actual FD-gated assembler output
    with the accepted density and its gradient provenance. The bond-axis
    value pinned against the rung-5 FD gate (analytic - FD = 4.1e-10):
    g_z(atom 0) = -0.012945 Ha/bohr."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    import vibeqc.periodic_gdf_gradient as gradient_module
    original = gradient_module._compute_krks_gradient_multik
    captured = []

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append((args, kwargs, np.array(value, copy=True)))
        return value

    monkeypatch.setattr(gradient_module, '_compute_krks_gradient_multik', capture)
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    ke = 60.0
    thr = 1e-9
    system, basis = _multik_h2_211_system()
    o = vq.PeriodicKSOptions()
    o.lattice_opts = _lat_opts()
    o.max_iter = 200
    o.conv_tol_energy = 1e-11
    r = run_krks_periodic_gdf(
        system, basis, (2, 1, 1), o, functional="lda",
        aux_basis="def2-svp-jk", use_compcell=True, gdf_method="rsgdf",
        rsgdf_ke_cutoff=ke, compute_gradient=True, progress=False,
    )
    assert r.converged
    assert r.gradient is not None and r.gradient.shape == (2, 3)
    assert np.abs(r.gradient.sum(axis=0)).max() <= 1e-12
    assert abs(r.gradient[0, 2] - (-0.012945)) < 1e-5

    lat, kpts, weights, grid, kmesh_bloch, oneel, gauge, xi = (
        _multik_krks_grad_inputs(system, basis, o, r)
    )
    n_k = 2
    n_occ = system.n_electrons() // 2
    D_k = [np.asarray(D) for D in r.density]
    S_k = [np.asarray(S) for S in r.overlap]
    W_k = []
    for ik in range(n_k):
        C_occ = np.asarray(r.mo_coeffs[ik])[:, :n_occ]
        eps_occ = np.asarray(r.mo_energies[ik])[:n_occ]
        W_k.append(2.0 * (C_occ * eps_occ[None, :]) @ C_occ.conj().T)
    assert len(captured) == 1
    args, kwargs, returned = captured[0]
    assert args[0] is system and args[1] is basis
    for actual, expected in zip(args[2:7], (D_k, W_k, S_k, weights, kpts)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    assert kwargs['functional'] == 'lda'
    assert kwargs['madelung'] == 0.
    assert kwargs['xc_lat_opts'].cutoff_bohr == lat.cutoff_bohr
    assert kwargs['oneel_lat_opts'].cutoff_bohr == oneel.cutoff_bohr
    assert kwargs['gauge_lat_opts'].nuclear_cutoff_bohr == gauge.nuclear_cutoff_bohr
    assert kwargs['xc_use_periodic_becke'] == bool(o.use_periodic_becke)
    assert kwargs['xc_becke_image_radius_bohr'] == float(o.becke_image_radius_bohr)
    # This wiring assertion is exact. Repeating a native parallel reduction
    # tested its summation order at 1e-15 rather than the driver's wiring;
    # independent finite-difference accuracy gates remain elsewhere above.
    np.testing.assert_array_equal(r.gradient, returned)


def test_multik_gradient_fail_closed_guards():
    """Rung 6 fail-closed envelope: every unsupported configuration
    raises ``NotImplementedError`` naming its reason BEFORE the SCF
    runs (cheap: no integrals are built). One case per named guard of
    HANDOVER_GDF_GRADIENT_DEFERRED.md § 4 rung 6 plus the
    driver-specific ones (use_compcell, cosx, Γ legacy fallback,
    dim=2 slab)."""
    from vibeqc.kpoints import KPoints
    from vibeqc.periodic_k_gdf import (
        run_krhf_periodic_gdf,
        run_krks_periodic_gdf,
    )

    system, basis = _multik_h2_211_system()
    o = vq.PeriodicRHFOptions()
    o.lattice_opts = _lat_opts()
    common = dict(
        aux_basis="def2-svp-jk", use_compcell=True, gdf_method="rsgdf",
        rsgdf_ke_cutoff=60.0, compute_gradient=True, progress=False,
    )

    # (The Schwarz-screened case was lifted 2026-07-30: screened fits
    # are differentiated at the SCF's fixed pair mask — see the
    # screened bit-consistency + FD gates above.)

    # (The finite-temperature Fermi-Dirac smearing rejects were lifted
    # 2026-07-30: the smeared gradient is dA/dR of the Mermin
    # free_energy — see the smeared FD gates below. What remains
    # guarded: non-Fermi-Dirac flavors (the entropy-conjugacy
    # restriction, tested directly on the guard helper) and every
    # bz_integration other than the literal 'smearing' alias.)
    from vibeqc.periodic_k_gdf import _reject_unsupported_multik_gradient

    with pytest.raises(
        NotImplementedError, match="Fermi-Dirac / Mermin smearing only"
    ):
        _reject_unsupported_multik_gradient(
            "run_krhf_periodic_gdf",
            dim=3,
            gdf_method="rsgdf",
            smearing_temperature=0.01,
            k_exchange="gdf",
            screened_omega=None,
            functional_is_range_separated=False,
            weights=np.array([0.5, 0.5]),
            smearing_flavor="methfessel-paxton",
        )

    # bz_integration='gilat' (sharp-Fermi occupation backend) has no
    # differentiated occupation response; 'smearing' is the one alias
    # accepted (it is literally the default occupation path).
    with pytest.raises(NotImplementedError, match="no gradient"):
        run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), o,
            **common, bz_integration="gilat",
        )

    # ibz_native (2026-08-01 increment 2): the SCF converges the
    # symmetry-TRANSPORTED exchange objective while the assemblers
    # differentiate the full-BZ one — the mismatch is the transport
    # residual, so the combination fails closed by name.
    with pytest.raises(NotImplementedError, match="ibz_native"):
        run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), o,
            **common, ibz_native=True,
        )

    # Non-rsgdf fit methods.
    with pytest.raises(
        NotImplementedError, match=r"gdf_method='rsgdf' only"
    ):
        run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), o,
            **{**common, "gdf_method": "compcell"},
        )
    with pytest.raises(
        NotImplementedError, match=r"gdf_method='rsgdf' only"
    ):
        run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), o, **{**common, "gdf_method": "mdf"}
        )

    # Non-uniform (IBZ-style) k weights: an explicit KPoints carrying
    # custom weights (an ir-mapped reduction would be expanded to the
    # uniform full BZ upstream and is supported).
    kp = KPoints.monkhorst_pack(system, (2, 1, 1), symmetry=False)
    kp.weights = np.array([0.75, 0.25])
    kp.mesh = None  # explicit list: no parent-mesh metadata
    with pytest.raises(NotImplementedError, match="uniform full-BZ"):
        run_krhf_periodic_gdf(system, basis, kp, o, **common)

    # Range-separated / screened hybrids (the cosx backend is the one
    # route that runs them for energies; the fitted-K derivative has
    # no screened arm).
    with pytest.raises(NotImplementedError, match="range-separated"):
        run_krks_periodic_gdf(
            system, basis, (2, 1, 1), None, functional="hse06",
            **{**common, "k_exchange": "cosx"},
        )

    # COSX exchange backend (full-range functional).
    with pytest.raises(NotImplementedError, match=r"k_exchange='gdf'"):
        run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), o, **{**common, "k_exchange": "cosx"}
        )

    # Pure-DFT Ewald-3D J path (no cached-Lpq fit to differentiate).
    with pytest.raises(NotImplementedError, match=r"use_compcell=True"):
        run_krks_periodic_gdf(
            system, basis, (2, 1, 1), None, functional="lda",
            **{**common, "use_compcell": False},
        )

    # DFT+U energy term.
    with pytest.raises(NotImplementedError, match=r"DFT\+U"):
        run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), o,
            **{
                **common,
                "dft_plus_u_sites": [vq.HubbardSite(0, 0, 0.1)],
            },
        )

    # Γ tuple mesh outside the pure fast path falls back to the legacy
    # molecular-limit driver, which has no gradient.
    with pytest.raises(NotImplementedError, match="legacy"):
        run_krhf_periodic_gdf(system, basis, (1, 1, 1), o, **common)

    # dim=1 wire: the remaining dim != 3 case on this entry (dim=2
    # closed-shell slabs route to the dedicated slab gradient since
    # 2026-07-30, § 6 rung 5 — gated in tests/test_slab_2d_routing.py).
    wire = vq.PeriodicSystem(
        1, np.eye(3) * 12.0, [
            vq.Atom(1, [6.0, 6.0, 6.0 - 0.7]),
            vq.Atom(1, [6.0, 6.0, 6.0 + 0.7]),
        ]
    )
    wire_basis = vq.BasisSet(wire.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="3D periodic"):
        run_krhf_periodic_gdf(
            wire, wire_basis, (2, 1, 1), o, **common
        )
    # dim=2 on the slab route: the gradient rides the slab envelope,
    # so an option outside it (compcell) still fails closed by name.
    slab = vq.PeriodicSystem(
        2, np.eye(3) * 12.0, [
            vq.Atom(1, [6.0, 6.0, 6.0 - 0.7]),
            vq.Atom(1, [6.0, 6.0, 6.0 + 0.7]),
        ]
    )
    slab_basis = vq.BasisSet(slab.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="signed truncated"):
        run_krhf_periodic_gdf(
            slab, slab_basis, (2, 1, 1), o, **common
        )


# ---------------------------------------------------------------------------
# Tests — finite-temperature (Fermi-Dirac) smeared free-energy gradients
# ---------------------------------------------------------------------------
#
# The differentiated objective is the DRIVER's Mermin free energy
# A = E - T.S (result.free_energy), NOT result.energy: Mermin, Phys.
# Rev. 137, A1441 (1965) makes A stationary at the self-consistent
# finite-T solution, so the occupation- and mu-response terms in dA/dR
# vanish and the T = 0 assemblers apply at the fractional-occupation
# D/W (Marzari-Vanderbilt, PRL 82, 3296 (1999) — the smeared-force
# statement). FD-differencing result.energy instead disagrees by
# ~T.dS/dR (measured 0.10 Ha/bohr on the stretched fixture below), so
# a gate that accidentally differences E would fail loudly.
#
# use_diis=False on every smeared fixture here, deliberately: on these
# symmetry-locked H2 fixtures the FDS-SDF commutator vanishes
# IDENTICALLY (sigma_g/sigma_u eigenvectors are fixed by symmetry), so
# DIIS receives zero error vectors from iteration 1. Pre-fix that
# terminated the smeared SCF after ~3 iterations at a state whose
# occupations came from a stale mixed Fock: stored occ [0.998955,
# 0.001045] vs the converged Fock's own Fermi filling [0.999126,
# 0.000874] on the 2.4-bohr scout (measured 2026-07-30; analytic-vs-FD
# then off by 3.3e-3). FIXED 2026-08-01: the accelerators now skip
# extrapolation (and record nothing) when the commutator error is at
# the numerical floor, and the smeared convergence tests additionally
# require the stored occupations to be the Fermi filling of the
# current Fock's own eigenvalues — with DIIS on these fixtures now
# converge to the same fixed point as plain iteration (regression:
# tests/test_periodic_gdf_smearing_diis.py). The gates below keep
# use_diis=False as the historical plain-iteration reference
# trajectory (stored == recomputed filling to ~1e-12; gates ~4e-10).


def _stretched_h2_box(bond=3.0, dz=0.0):
    """H2 stretched to ``bond`` bohr in the 12-bohr box: the sigma/
    sigma* gap (~0.48 Ha at 3.0 bohr) is a few k_B T at T = 0.1 Ha, so
    the Fermi-Dirac occupations are meaningfully fractional."""
    box = 12.0
    c = box / 2.0
    s = vq.PeriodicSystem(3, np.eye(3) * box, [
        vq.Atom(1, [c, c, c - bond / 2.0 + dz]),
        vq.Atom(1, [c, c, c + bond / 2.0]),
    ])
    b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
    return s, b


def _run_gamma_uhf_smeared(gdf_method, dz=0.0, smearing_T=0.10,
                           compute_gradient=False):
    system, basis = _stretched_h2_box(dz=dz)
    o = vq.PeriodicRHFOptions()
    o.lattice_opts = _lat_opts()
    o.max_iter = 200
    o.conv_tol_energy = 1e-11
    o.smearing_temperature = smearing_T
    o.use_diis = False  # zero-commutator DIIS trap; see block comment
    kwargs = dict(
        options=o, aux_basis="def2-svp-jk",
        gdf_linear_dep_threshold=1e-9, exxdiv="ewald",
        compute_gradient=compute_gradient, progress=False,
    )
    if gdf_method == "compcell":
        kwargs.update(
            gdf_method="compcell", compcell_eta=1.0,
            apply_aft_correction=False,
        )
    else:
        kwargs.update(gdf_method="rsgdf", rsgdf_ke_cutoff=60.0)
    r = vq.run_pbc_gdf_uhf(system, basis, **kwargs)
    assert r.converged
    return r


@pytest.mark.parametrize("gdf_method", ["compcell", "rsgdf"])
def test_gamma_uhf_smeared_free_energy_gradient_vs_fd(
    monkeypatch, gdf_method
):
    """Γ UHF Fermi-Dirac smeared analytic gradient == central FD of the
    driver's free_energy (A = E - T.S) on stretched H2 at kBT = 0.1 Ha.

    Fractional occupations are genuinely active (per-spin f =
    [0.918, 0.082] compcell / [0.926, 0.074] rsgdf — asserted), so
    this pins the sqrt(f)-scaled orbital blocks: fractional W, the
    f_i.f_j-weighted exchange, and the fractional-D exxdiv D S D term.
    Measured |analytic - FD(A)| at h = 2e-4: 3.8e-10 (compcell) /
    1.9e-10 (rsgdf); FD of result.energy instead disagrees by ~0.1
    Ha/bohr (the T.dS/dR term), asserted as a cross-check. Gate 1e-6.
    If this fails, bisect with the T = 0 gates + the T->0 consistency
    test below — do NOT paper over (CLAUDE.md § 7)."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    if gdf_method == "compcell":
        with pytest.warns(UserWarning, match="apply_aft_correction"):
            r = _run_gamma_uhf_smeared(gdf_method, compute_gradient=True)
    else:
        r = _run_gamma_uhf_smeared(gdf_method, compute_gradient=True)
    assert r.gradient is not None and r.gradient.shape == (2, 3)
    occ = np.concatenate([r.occupations_alpha, r.occupations_beta])
    assert np.any((occ > 0.05) & (occ < 0.95)), (
        f"fixture lost its fractional occupations: {occ}"
    )
    assert np.abs(np.asarray(r.gradient).sum(axis=0)).max() <= 1e-10

    h = 2e-4
    if gdf_method == "compcell":
        with pytest.warns(UserWarning, match="apply_aft_correction"):
            rp = _run_gamma_uhf_smeared(gdf_method, dz=+h)
            rm = _run_gamma_uhf_smeared(gdf_method, dz=-h)
    else:
        rp = _run_gamma_uhf_smeared(gdf_method, dz=+h)
        rm = _run_gamma_uhf_smeared(gdf_method, dz=-h)
    fd_a = (rp.free_energy - rm.free_energy) / (2.0 * h)
    fd_e = (rp.energy - rm.energy) / (2.0 * h)
    g = float(r.gradient[0, 2])
    assert abs(g - fd_a) <= 1e-6, (
        f"analytic={g:.12f} FD(A)={fd_a:.12f} diff={abs(g - fd_a):.3e}"
    )
    # The objective is A, not E: T.dS/dR is macroscopic here.
    assert abs(g - fd_e) > 1e-3, (
        "FD of .energy unexpectedly matches — smearing inactive?"
    )


def test_multik_krhf_smeared_free_energy_gradient_vs_fd(monkeypatch):
    """Multi-k (2,1,1) KRHF Fermi-Dirac smeared analytic gradient ==
    central FD of the driver's free_energy on stretched H2 at
    kBT = 0.1 Ha.

    Pins the smeared multi-k wiring: W(k) = sum_i f_i(k) eps_i(k)
    c c^H from result.occupations (in [0, 2]; measured [1.855, 0.145]
    per k — asserted fractional), fractional D(k) through DF-J/K and
    the exxdiv shift. Measured |analytic - FD(A)| at h = 2e-4:
    3.5e-10; FD of result.energy disagrees by ~0.1 Ha/bohr. Gate 1e-6.
    If this fails, bisect with the T = 0 multi-k composition gates —
    do NOT paper over (CLAUDE.md § 7)."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    def run(dz=0.0, compute_gradient=False):
        s, b = _stretched_h2_box(dz=dz)
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = _lat_opts()
        o.max_iter = 300
        o.conv_tol_energy = 1e-11
        o.smearing_temperature = 0.10
        o.use_diis = False  # zero-commutator DIIS trap; see block comment
        r = run_krhf_periodic_gdf(
            s, b, (2, 1, 1), o, aux_basis="def2-svp-jk",
            use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=60.0,
            # 'smearing' is the accepted bz_integration alias — it is
            # literally the default occupation path, so passing it here
            # additionally pins that the guard admits it.
            bz_integration="smearing",
            compute_gradient=compute_gradient, progress=False,
        )
        assert r.converged
        return r

    r = run(compute_gradient=True)
    assert r.gradient is not None and r.gradient.shape == (2, 3)
    occ = np.concatenate([np.asarray(o) for o in r.occupations])
    assert np.any((occ > 0.05) & (occ < 1.95)), (
        f"fixture lost its fractional occupations: {occ}"
    )
    assert np.abs(np.asarray(r.gradient).sum(axis=0)).max() <= 1e-10

    h = 2e-4
    rp, rm = run(dz=+h), run(dz=-h)
    fd_a = (rp.free_energy - rm.free_energy) / (2.0 * h)
    fd_e = (rp.energy - rm.energy) / (2.0 * h)
    g = float(r.gradient[0, 2])
    assert abs(g - fd_a) <= 1e-6, (
        f"analytic={g:.12f} FD(A)={fd_a:.12f} diff={abs(g - fd_a):.3e}"
    )
    assert abs(g - fd_e) > 1e-3, (
        "FD of .energy unexpectedly matches — smearing inactive?"
    )


def test_smeared_gradient_matches_t0_at_tiny_temperature(monkeypatch):
    """T -> 0 consistency: on the gapped H2/1.4-bohr fixture at
    kBT = 1e-4 the Fermi-Dirac occupations are integer to < 1e-12 and
    the smeared gradient path (sqrt(f)-scaled blocks at Γ, occupation-
    weighted W(k) on multi-k) must reproduce the T = 0 gradient.
    Measured max |g(T) - g(0)|: 7.9e-16 (Γ UHF compcell) / 1.2e-15
    (multi-k KRHF rsgdf); gate 1e-9."""
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_KE", "8.0")
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    def gamma(T):
        system, basis = _h2_12bohr_system()
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = _lat_opts()
        o.max_iter = 200
        o.conv_tol_energy = 1e-11
        o.smearing_temperature = T
        o.use_diis = False
        with pytest.warns(UserWarning, match="apply_aft_correction"):
            r = vq.run_pbc_gdf_uhf(
                system, basis, options=o, aux_basis="def2-svp-jk",
                gdf_method="compcell", compcell_eta=1.0,
                apply_aft_correction=False,
                gdf_linear_dep_threshold=1e-9, exxdiv="ewald",
                compute_gradient=True, progress=False,
            )
        assert r.converged
        return r

    def multik(T):
        s, b = _multik_h2_211_system()
        o = vq.PeriodicRHFOptions()
        o.lattice_opts = _lat_opts()
        o.max_iter = 300
        o.conv_tol_energy = 1e-11
        o.smearing_temperature = T
        o.use_diis = False
        r = run_krhf_periodic_gdf(
            s, b, (2, 1, 1), o, aux_basis="def2-svp-jk",
            use_compcell=True, gdf_method="rsgdf", rsgdf_ke_cutoff=60.0,
            compute_gradient=True, progress=False,
        )
        assert r.converged
        return r

    for name, fn in (("gamma-uhf", gamma), ("multik-krhf", multik)):
        r0 = fn(0.0)
        rt = fn(1e-4)
        occ = (
            np.concatenate([rt.occupations_alpha, rt.occupations_beta])
            if name == "gamma-uhf"
            else np.concatenate([np.asarray(o) for o in rt.occupations])
        )
        assert np.abs(occ - np.round(occ)).max() < 1e-12
        diff = np.abs(
            np.asarray(rt.gradient) - np.asarray(r0.gradient)
        ).max()
        assert diff <= 1e-9, f"{name}: T->0 gradient drift {diff:.3e}"


def test_gamma_gradient_rejects_non_fermi_dirac_flavor_pending_force_validation():
    """Non-FD/non-Mermin GDF gradients stay fail-closed pending their own force gates."""
    from vibeqc.pbc_gdf import _reject_non_fermi_dirac_gradient
    from vibeqc.smearing import SmearingOptions

    for flavor in ("methfessel-paxton", "marzari-vanderbilt"):
        with pytest.raises(
            NotImplementedError, match="Fermi-Dirac / Mermin smearing only"
        ):
            _reject_non_fermi_dirac_gradient(
                "run_pbc_gdf_uhf",
                SmearingOptions(temperature=0.01, flavor=flavor),
            )
    # Fermi-Dirac, Mermin, and disabled smearing pass through.
    _reject_non_fermi_dirac_gradient(
        "run_pbc_gdf_uhf", SmearingOptions(temperature=0.01)
    )
    _reject_non_fermi_dirac_gradient(
        "run_pbc_gdf_uhf", SmearingOptions(temperature=0.01, flavor="mermin")
    )
    _reject_non_fermi_dirac_gradient(
        "run_pbc_gdf_uhf",
        SmearingOptions(temperature=0.0, flavor="marzari-vanderbilt"),
    )


@pytest.mark.parametrize('threshold', [1e-10, .5])
def test_coulomb_gradient_weights_match_complex_metric_and_tensor_variation(monkeypatch, threshold):
    """Raw-source variations test the actual Coulomb assembly's derivative."""
    from types import SimpleNamespace
    import vibeqc.aux_basis as aux_module
    from vibeqc.periodic_gdf_gradient import _compute_j_gradient_multik_rsgdf

    rng = np.random.default_rng(142)
    def random(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    unitary, _ = np.linalg.qr(random((3, 3)))
    eigenvalues = np.array([.2, 2., 4.])
    metric = (unitary * eigenvalues) @ unitary.conj().T
    tensors = random((2, 3, 2, 2))
    density = random((2, 2, 2))
    density = density @ density.conj().transpose(0, 2, 1)
    weights = np.array([.3, .7])
    dm = random((3, 3))
    dm += dm.conj().T.copy()
    dt = random(tensors.shape)
    inverse = np.where(eigenvalues > threshold, 1 / eigenvalues, 0.)
    frechet = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            frechet[i, j] = (
                -inverse[i]**2 if i == j
                else (inverse[i] - inverse[j]) / (eigenvalues[i] - eigenvalues[j])
            )
    group = SimpleNamespace(
        q=np.zeros(3), pairs=[(0, 0), (1, 1)], bra_k_indices=[0, 1],
        T_list=tensors, eigvecs=unitary, inverse_eigvals=inverse,
        inverse_frechet=frechet, pair_keep_ao=None,
    )
    cache = SimpleNamespace(
        n_aux=3, n_orb=2, k_cart_list=np.zeros((2, 3)), groups={0: group},
        aux_basis=None, ke_cutoff=1., tail_ke_cutoff=None, lat_opts=None,
    )
    def metric_variation(*args, weight, **kwargs):
        return np.array([[np.sum(weight * dm).real, 0., 0.]])
    tensor_index = 0
    def tensor_variation(*args, weight, **kwargs):
        nonlocal tensor_index
        value = np.sum(weight * dt[tensor_index]).real
        tensor_index += 1
        return np.array([[value, 0., 0.]])
    monkeypatch.setattr(aux_module, '_rsgdf_weighted_2c_metric_gradient_bloch', metric_variation)
    monkeypatch.setattr(aux_module, '_rsgdf_weighted_3c_tensor_gradient_bloch', tensor_variation)
    analytic = _compute_j_gradient_multik_rsgdf(None, None, density, weights, cache)[0, 0]

    def energy(step):
        values, vectors = np.linalg.eigh(metric + step * dm)
        keep = values > threshold
        whitener = vectors[:, keep].conj().T / np.sqrt(values[keep])[:, None]
        factors = [(whitener @ t.reshape(3, -1)).reshape(-1, 2, 2)
                   for t in tensors + step * dt]
        coulomb = aux_module._build_coulomb_from_diagonal_factors(factors, density, weights)
        return .5 * sum(w * np.trace(d @ j).real for w, d, j in zip(weights, density, coulomb))
    h = 1e-5
    # A five-point derivative removes the O(h^2) inverse-metric curvature
    # error without shrinking the step into cancellation of large energies.
    finite_difference = (energy(-2*h) - 8*energy(-h)
                         + 8*energy(h) - energy(2*h)) / (12*h)
    assert analytic == pytest.approx(finite_difference, rel=0, abs=5e-7)
    assert tensor_index == 2


@pytest.mark.parametrize("threshold", [1e-10, .5])
@pytest.mark.parametrize("scales,off_diagonal", [
    ((1., 0.), False), ((0., 1.), False), ((1., .25), False), ((0., 1.), True),
])
def test_sr_lr_fit_response_differentiates_actual_jk_assembly(threshold, scales, off_diagonal):
    from vibeqc.aux_basis import (
        _RangeSeparatedGdfBatch, _RangeSeparatedGdfFitState,
        _build_coulomb_from_diagonal_factors, _accumulate_exchange_from_factors,
    )
    from vibeqc.periodic_gdf_gradient import _range_separated_jk_response_weights

    rng = np.random.default_rng(664)
    def random(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    unitary, _ = np.linalg.qr(random((3, 3)))
    values = np.array([.2, 2., 4.])
    metric = (unitary * values) @ unitary.conj().T
    tensors = random((2, 3, 2, 2))
    density = random((2, 2, 2))
    density = density @ density.conj().transpose(0, 2, 1)
    weights, pairs = np.array([.3, .7]), [(0, 0), (1, 1)]
    if off_diagonal:
        pairs = [(0, 1), (1, 0)]
    dm = random((3, 3))
    dm += dm.conj().T.copy()
    dt = random(tensors.shape)
    state = _RangeSeparatedGdfFitState(
        tensors, unitary, values > threshold, np.zeros((1, 3)), np.zeros((2, 3)),
    )
    q = np.array([.2, 0., 0.]) if off_diagonal else np.zeros(3)
    batch = _RangeSeparatedGdfBatch(None, q, values, 0, 0, state)
    kwargs = dict(linear_dep_thr=threshold, workspace_byte_cap=2**20,
                  coulomb_scale=scales[0], exchange_scale=scales[1])
    wm, wt = _range_separated_jk_response_weights(batch, pairs, density, weights, **kwargs)
    analytic = (np.sum(wm * dm) + np.sum(wt * dt)).real
    assert wm.flags.f_contiguous and wt.flags.c_contiguous

    # A memory-limited q group can be rebuilt a batch at a time. Hartree
    # still uses the density summed over the whole group, including its
    # components along dropped auxiliary modes. Exchange weights add per
    # pair. The q=0 metric response must be counted exactly once.
    raw_density = sum(
        weights[i] * (t.reshape(3, -1) @ density[i].T.reshape(-1))
        for i, t in enumerate(tensors)
    ) if scales[0] else None
    split_metric = np.zeros_like(wm)
    split_tensors = []
    for position, pair in enumerate(pairs):
        selected_state = state._replace(
            three_center=tensors[position:position + 1],
            ket_kpoints=state.ket_kpoints[position:position + 1],
        )
        selected_batch = batch._replace(fit_state=selected_state)
        sm, st = _range_separated_jk_response_weights(
            selected_batch, [pair], density, weights, **kwargs,
            coulomb_source=raw_density, include_coulomb_metric=position == 0,
        )
        split_metric += sm
        split_tensors.append(st[0])
        if scales[0]:
            with pytest.raises(ValueError, match="every q=0 diagonal"):
                _range_separated_jk_response_weights(
                    selected_batch, [pair], density, weights, **kwargs,
                )
    np.testing.assert_allclose(split_metric, wm, rtol=0, atol=2e-11)
    np.testing.assert_allclose(split_tensors, wt, rtol=0, atol=2e-11)

    def energy(step):
        eigenvalues, vectors = np.linalg.eigh(metric + step * dm)
        keep = eigenvalues > threshold
        whitener = vectors[:, keep].conj().T / np.sqrt(eigenvalues[keep])[:, None]
        factors = [(whitener @ t.reshape(3, -1)).reshape(-1, 2, 2)
                   for t in tensors + step * dt]
        coulomb = _build_coulomb_from_diagonal_factors(factors, density, weights)
        ej = .5 * sum(w * np.trace(d @ j).real
                      for w, d, j in zip(weights, density, coulomb))
        ek = 0.
        for (i, j), factor in zip(pairs, factors):
            matrix = np.zeros((2, 2), dtype=complex)
            _accumulate_exchange_from_factors(matrix, factor, density[j], weights[j], 2**20)
            ek -= .25 * weights[i] * np.trace(density[i] @ matrix).real
        return scales[0] * ej + scales[1] * ek
    # Retaining the 0.2 metric mode makes the inverse response large.
    # A fourth-order stencil resolves it without loosening the derivative
    # gate to the O(h^2) error of a two-point difference.
    for h in (1e-4, 5e-5):
        fd = (energy(-2*h) - 8*energy(-h) + 8*energy(h) - energy(2*h)) / (12*h)
        assert analytic == pytest.approx(fd, abs=5e-7, rel=0)
    with pytest.raises(MemoryError, match="workspace"):
        _range_separated_jk_response_weights(
            batch, pairs, density, weights, **dict(kwargs, workspace_byte_cap=1),
        )
    with pytest.raises(ValueError, match="threshold differs"):
        _range_separated_jk_response_weights(
            batch, pairs, density, weights,
            **dict(kwargs, linear_dep_thr=.5 if threshold < .5 else 1e-10),
        )
