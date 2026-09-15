"""dim=2 routing: AUTO -> direct 2D Ewald; explicit closed-shell slab GDF.

Background (the bug this pins). Before this landed, ``pick_jk_method`` had no
dimensionality branch and returned GDF unconditionally for closed-shell SCF.
GDF is a bulk (dim=3) J/K builder: its Gamma drivers raise for ``dim != 3``,
but the **multi-k** entries silently ran a ``dim=2`` slab as a 3-D crystal of
sheets stacked ``a3`` apart (normal-axis mesh pinned to one k-point). The
observable symptom on graphene/HSE06 was a nuclear repulsion that *depended on
the k-mesh* (``E_nuclear`` = +321.39 Ha at k=(1,1,1) vs +557.97 Ha at
k=(8,8,1), identical geometry) and a total energy off by thousands of Ha. A
purely geometric Madelung sum cannot depend on the k-mesh -- CLAUDE.md Sec. 7
says that is a bug, not a convergence problem.

AUTO routes ``dim=2`` to the rigorous vacuum-free Parry / de Leeuw-Perram
gauge (``CoulombMethod.SLAB_EWALD_2D``), whose total is invariant to the
bookkeeping ``a3`` and whose ``E_nuclear`` is k-independent. Explicit
closed-shell GDF uses a separately validated signed slab-truncated metric;
unsupported bulk and open-shell routes still fail closed.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_jk_method import PeriodicJKMethod, pick_jk_method

_LAT = np.diag([18.0, 18.0, 31.4])


def _pick(dim: int, scf_method: str, method="auto") -> PeriodicJKMethod:
    return pick_jk_method(
        method,
        lattice=_LAT,
        basis_name="sto-3g",
        n_atoms=2,
        scf_method=scf_method,
        dim=dim,
    )


def _h2_slab():
    """A neutral H2 layer as a genuine dim=2 slab (real z, no vacuum box)."""
    atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
    sysp = vq.slab_2d([18.0, 0.0, 0.0], [0.0, 18.0, 0.0], atoms)
    return sysp, vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")


def _ks_opts():
    o = vq.PeriodicKSOptions()
    o.functional = "PBE"
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    o.damping = 0.3
    o.max_iter = 60
    o.use_diis = True
    return o


# ---------------------------------------------------------------------------
# 1. AUTO picker: dimensionality branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scf_method", ["RHF", "RKS", "UKS"])
def test_auto_dim2_routes_to_slab_ewald_2d(scf_method):
    assert _pick(2, scf_method) == PeriodicJKMethod.SLAB_EWALD_2D


def test_auto_dim3_unchanged():
    # The established bulk heuristic must not regress.
    assert _pick(3, "RKS") == PeriodicJKMethod.GDF
    assert _pick(3, "RHF") == PeriodicJKMethod.GDF
    assert _pick(3, "UKS") == PeriodicJKMethod.BIPOLE


def test_auto_dim2_uhf_fails_closed():
    # Open-shell HF on a slab is a follow-on; must not silently fall to a bulk
    # builder.
    with pytest.raises(NotImplementedError, match="RHF/RKS/UKS"):
        _pick(2, "UHF")


# ---------------------------------------------------------------------------
# 2. Explicit slab GDF boundary; unsupported builders fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bulk", ["bipole", "gpw", "gapw", "rijcosx"])
def test_explicit_bulk_method_on_dim2_fails_closed(bulk):
    with pytest.raises(NotImplementedError, match="bulk"):
        _pick(2, "RKS", bulk)


@pytest.mark.parametrize("scf_method", ["RHF", "RKS"])
def test_explicit_gdf_on_dim2_allows_closed_shell(scf_method):
    assert _pick(2, scf_method, "gdf") == PeriodicJKMethod.GDF


@pytest.mark.parametrize("scf_method", ["UHF", "UKS"])
def test_explicit_gdf_on_dim2_keeps_open_shell_closed(scf_method):
    with pytest.raises(NotImplementedError, match="closed-shell RHF/RKS"):
        _pick(2, scf_method, "gdf")


def test_slab_ewald_2d_on_bulk_cell_fails_closed():
    with pytest.raises(NotImplementedError, match="requires a dim=2 slab"):
        _pick(3, "RKS", "slab_ewald_2d")


def test_multi_k_gdf_guard_allows_dim1_polymer():
    """`dim=1` must NOT be rejected -- guarding `dim != 3` would revert a fix.

    There is no rigorous 1-D Coulomb gauge yet (`NEUTRALIZED_1D` still raises),
    so the cached-Lpq GDF Hartree is the *supported* polymer/wire route: main
    `5fe4d021` ("Lpq Hartree for multi-k dim<3 KS") moved the dim=1 H2-chain
    from -3.0897 to the variational -3.89 Ha/cell through exactly these entries.
    Only the slab has a rigorous vacuum-free alternative, so only the slab fails
    closed.
    """
    from vibeqc.periodic_k_gdf import _reject_slab_dim

    chain = vq.PeriodicSystem(
        1,
        np.diag([5.0, 30.0, 30.0]),
        [vq.Atom(1, [0.0, 15.0, 15.0]), vq.Atom(1, [1.4, 15.0, 15.0])],
    )
    _reject_slab_dim(chain, "run_krhf_periodic_gdf")  # must not raise

    sysp, _ = _h2_slab()
    with pytest.raises(NotImplementedError):
        _reject_slab_dim(sysp, "run_krhf_periodic_gdf")

    # AUTO on a polymer keeps the established GDF route (unchanged by this work).
    assert _pick(1, "RKS") == PeriodicJKMethod.GDF


@pytest.mark.parametrize("scf_method", ["UHF", "UKS"])
def test_auto_dim1_open_shell_uses_supported_gdf_route(scf_method):
    assert _pick(1, scf_method) == PeriodicJKMethod.GDF


def test_auto_dim1_rohf_fails_with_accurate_route_diagnostic():
    with pytest.raises(NotImplementedError, match="no maintained dim=1 ROHF"):
        _pick(1, "ROHF")


def test_open_shell_multi_k_gdf_entries_fail_closed_on_slab():
    """The fitted slab route is closed-shell until its spin gauge is proven."""
    from vibeqc.periodic_k_gdf import (
        run_kuhf_periodic_gdf,
        run_kuks_periodic_gdf,
    )

    sysp, basis = _h2_slab()
    for fn, kw in [
        (run_kuhf_periodic_gdf, {}),
        (run_kuks_periodic_gdf, {"functional": "PBE"}),
    ]:
        with pytest.raises(NotImplementedError, match=r"bulk \(dim=3\)"):
            fn(sysp, basis, (2, 2, 1), **kw)


# ---------------------------------------------------------------------------
# 3. Inc-D private foundation: exact slab-truncated reciprocal kernel
# ---------------------------------------------------------------------------


def test_slab_truncated_coulomb_kernel_matches_eq_a6():
    """Pin the nonzero branches and the finite slab G=0 gauge separately."""
    from vibeqc.aux_basis import _slab_truncated_coulomb_kernel

    sysp, _ = _h2_slab()
    length = float(np.linalg.norm(np.asarray(sysp.lattice)[:, 2]))
    gx = 2.0 * np.pi / 18.0
    gz = 2.0 * np.pi / length
    G = np.array(
        [
            [0.0, 0.0, 0.0],
            [gx, 0.0, 0.0],
            [0.0, 0.0, gz],
            [gx, 0.0, gz],
        ]
    )

    got = _slab_truncated_coulomb_kernel(sysp, G)
    expected = np.array(
        [
            -0.5 * np.pi * length**2,
            4.0 * np.pi / gx**2 * (1.0 - np.exp(-gx * length / 2.0)),
            8.0 * np.pi / gz**2,
            4.0
            * np.pi
            / (gx**2 + gz**2)
            * (1.0 + np.exp(-gx * length / 2.0)),
        ]
    )
    assert np.allclose(got, expected, rtol=2e-14, atol=2e-14)
    assert np.allclose(
        _slab_truncated_coulomb_kernel(sysp, -G), got, rtol=0.0, atol=2e-14
    )


def test_slab_truncated_coulomb_kernel_is_rotation_covariant():
    from vibeqc.aux_basis import _slab_truncated_coulomb_kernel

    sysp, _ = _h2_slab()
    length = float(np.linalg.norm(np.asarray(sysp.lattice)[:, 2]))
    G = np.array(
        [
            [0.13, -0.27, 0.19],
            [-0.31, 0.07, -0.11],
            [0.0, 0.0, 2.0 * np.pi / length],
        ]
    )
    q = np.array([0.04, -0.02, 0.0])
    reference = _slab_truncated_coulomb_kernel(sysp, G, q_cart=q)

    angle = 0.37
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    rotated_lattice = rotation @ np.asarray(sysp.lattice, dtype=float)
    rotated = vq.PeriodicSystem(2, rotated_lattice, list(sysp.unit_cell))
    got = _slab_truncated_coulomb_kernel(
        rotated, G @ rotation.T, q_cart=q @ rotation.T
    )
    assert np.allclose(got, reference, rtol=2e-14, atol=2e-14)


def test_slab_truncated_coulomb_kernel_rejects_wrong_geometry():
    from vibeqc.aux_basis import _slab_truncated_coulomb_kernel

    sysp, _ = _h2_slab()
    G = np.zeros((1, 3))
    bulk = vq.PeriodicSystem(3, np.asarray(sysp.lattice), list(sysp.unit_cell))
    with pytest.raises(ValueError, match="genuine dim=2"):
        _slab_truncated_coulomb_kernel(bulk, G)

    skewed_lattice = np.asarray(sysp.lattice, dtype=float).copy()
    skewed_lattice[:, 2] += np.array([1.0, 0.0, 0.0])
    skewed = vq.PeriodicSystem(2, skewed_lattice, list(sysp.unit_cell))
    with pytest.raises(ValueError, match="normal to the periodic slab plane"):
        _slab_truncated_coulomb_kernel(skewed, G)

    with pytest.raises(ValueError, match="q_cart to lie"):
        _slab_truncated_coulomb_kernel(sysp, G, q_cart=np.array([0.0, 0.0, 0.1]))


def test_slab_truncated_gdf_contractions_keep_zero_mode():
    """The private 2c/3c seam uses Eq. A6 weights, not bulk G=0 omission."""
    from vibeqc.aux_basis import (
        _slab_truncated_coulomb_kernel,
        _slab_truncated_gdf_contractions,
    )

    sysp, _ = _h2_slab()
    G = np.array([[0.0, 0.0, 0.0], [2.0 * np.pi / 18.0, 0.0, 0.0]])
    aux_ft = np.array([[1.0, 0.5 + 0.2j], [-0.3j, 1.2]])
    pair_ft = np.array([[[2.0, -0.4j]]])
    metric, three_center = _slab_truncated_gdf_contractions(
        sysp, G, aux_ft, pair_ft
    )

    volume = float(abs(np.linalg.det(np.asarray(sysp.lattice, dtype=float))))
    weights = _slab_truncated_coulomb_kernel(sysp, G) / volume
    expected_metric = (aux_ft.conj() * weights[None, :]) @ aux_ft.T
    expected_metric = 0.5 * (expected_metric + expected_metric.conj().T)
    expected_three_center = np.einsum(
        "PG,G,mnG->Pmn", aux_ft.conj(), weights, pair_ft
    )
    assert np.allclose(metric, expected_metric, rtol=0.0, atol=1e-14)
    assert np.allclose(three_center, expected_three_center, rtol=0.0, atol=1e-14)
    assert np.allclose(metric, metric.conj().T, rtol=0.0, atol=1e-14)

    zero_weight = weights[0]
    assert zero_weight < 0.0
    assert not np.allclose(
        metric,
        (aux_ft[:, 1:].conj() * weights[None, 1:]) @ aux_ft[:, 1:].T,
    )


def test_slab_truncated_dense_mesh_keeps_normal_fourier_support():
    """The truncated kernel needs full G-normal resolution, not a 2D sheet."""
    from vibeqc.aux_basis import _slab_truncated_dense_g_mesh

    sysp, _ = _h2_slab()
    G = _slab_truncated_dense_g_mesh(sysp, 1.0)
    normal = np.asarray(sysp.lattice, dtype=float)[:, 2]
    normal = normal / np.linalg.norm(normal)
    G_normal = G @ normal
    G_plane = G - G_normal[:, None] * normal[None, :]

    assert np.allclose(G[0], 0.0, rtol=0.0, atol=0.0)
    assert np.any(np.abs(G_normal) > 1e-12)
    assert np.any(np.linalg.norm(G_plane, axis=1) > 1e-12)
    assert np.max(np.linalg.norm(G, axis=1)) <= np.sqrt(2.0) + 1e-12


def test_slab_truncated_dense_mesh_transfer_labels_and_normal_guard():
    from vibeqc.aux_basis import _slab_truncated_dense_g_mesh

    sysp, _ = _h2_slab()
    lattice = np.asarray(sysp.lattice, dtype=float)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    q = 0.23 * reciprocal[:, 0]
    reference = _slab_truncated_dense_g_mesh(sysp, 1.0, q_cart=q)
    relabelled = _slab_truncated_dense_g_mesh(
        sysp, 1.0, q_cart=q + reciprocal[:, 0]
    )
    assert np.allclose(relabelled, reference, rtol=0.0, atol=1e-13)

    normal = lattice[:, 2] / np.linalg.norm(lattice[:, 2])
    with pytest.raises(ValueError, match="q_cart to lie"):
        _slab_truncated_dense_g_mesh(sysp, 1.0, q_cart=0.1 * normal)


def test_factor_slab_truncated_gdf_metric_retains_signature():
    from vibeqc.aux_basis import _factor_slab_truncated_gdf_metric

    metric = np.diag([4.0, 1.0, 1e-12])
    three_center = np.arange(12.0).reshape(3, 2, 2)
    fit = _factor_slab_truncated_gdf_metric(
        metric, three_center, linear_dep_thr=1e-10
    )
    expected = np.stack([three_center[1], 0.5 * three_center[0]])
    assert np.allclose(fit.factors, expected, rtol=0.0, atol=1e-14)
    assert np.array_equal(fit.signs, [1, 1])
    assert np.allclose(
        fit.eigenvalues, [1e-12, 1.0, 4.0], rtol=0.0, atol=1e-15
    )

    signed_metric = np.diag([-0.25, 2.0])
    signed_three_center = np.arange(8.0).reshape(2, 2, 2)
    signed_fit = _factor_slab_truncated_gdf_metric(
        signed_metric, signed_three_center, linear_dep_thr=1e-10
    )
    assert np.array_equal(signed_fit.signs, [-1, 1])
    factors = signed_fit.factors.reshape(2, -1)
    reconstructed = factors.conj().T @ (signed_fit.signs[:, None] * factors)
    direct = signed_three_center.reshape(2, -1).T @ np.linalg.solve(
        signed_metric, signed_three_center.reshape(2, -1)
    )
    assert np.allclose(reconstructed, direct, rtol=0.0, atol=1e-13)


def test_slab_truncated_aux_metric_is_positive_and_a3_stable():
    """Real modrho auxiliary spectra stay positive as bookkeeping a3 changes."""
    from vibeqc.aux_basis import (
        _factor_slab_truncated_gdf_metric,
        _slab_truncated_dense_g_mesh,
        _slab_truncated_gdf_contractions,
        make_aux_basis_set,
        make_modrho_aux_basis,
        rsgdf_aux_fourier_transform,
    )

    atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
    base = vq.slab_2d([18.0, 0.0, 0.0], [0.0, 18.0, 0.0], atoms)
    molecule = base.unit_cell_molecule()
    raw_aux = make_aux_basis_set(molecule, aux_name="def2-svp-jk")
    aux = make_modrho_aux_basis(raw_aux, molecule)

    spectra = []
    for padding in (15.0, 40.0):
        sysp = vq.slab_2d(
            [18.0, 0.0, 0.0],
            [0.0, 18.0, 0.0],
            atoms,
            normal_padding_bohr=padding,
        )
        G = _slab_truncated_dense_g_mesh(sysp, 10.0)
        aux_ft = rsgdf_aux_fourier_transform(aux, G)
        metric, three_center = _slab_truncated_gdf_contractions(
            sysp,
            G,
            aux_ft,
            np.zeros((1, 1, G.shape[0]), dtype=np.complex128),
        )
        fit = _factor_slab_truncated_gdf_metric(
            metric, three_center
        )
        assert np.all(np.isfinite(fit.factors))
        assert np.all(fit.signs == 1)
        assert float(fit.eigenvalues[0]) > 0.0
        spectra.append(fit.eigenvalues)

    assert np.allclose(spectra[0], spectra[1], rtol=2e-3, atol=3e-8)


def test_slab_truncated_bloch_fit_is_nonzero_and_a3_stable():
    """The private builder forms a real AO-pair fit without opening a route."""
    from vibeqc.aux_basis import (
        _build_lpq_bloch_slab_truncated,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )

    atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
    base = vq.slab_2d([18.0, 0.0, 0.0], [0.0, 18.0, 0.0], atoms)
    molecule = base.unit_cell_molecule()
    ao_basis = vq.BasisSet(molecule, "sto-3g")
    aux_basis = make_modrho_aux_basis(
        make_aux_basis_set(molecule, aux_name="def2-svp-jk"), molecule
    )
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 12.0

    pair_grams = []
    first_fit = None
    first_transfer = None
    for padding in (15.0, 40.0):
        sysp = vq.slab_2d(
            [18.0, 0.0, 0.0],
            [0.0, 18.0, 0.0],
            atoms,
            normal_padding_bohr=padding,
        )
        reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(sysp.lattice)).T
        transfer = 0.17 * reciprocal[:, 0]
        fit = _build_lpq_bloch_slab_truncated(
            sysp,
            ao_basis,
            aux_basis,
            np.zeros(3),
            transfer,
            ke_cutoff=10.0,
            lat_opts=lat_opts,
            g_chunk=2048,
        )
        assert np.all(np.isfinite(fit.factors))
        assert float(np.linalg.norm(fit.factors)) > 1.0
        assert fit.includes_finite_coulomb_zero_mode
        assert np.isfinite(fit.primitive_exchange_probe_charge_madelung)
        pair_grams.append(
            np.einsum(
                "P,Pmn,Prs->mnrs",
                fit.metric_signs,
                fit.factors.conj(),
                fit.factors,
            )
        )
        if first_fit is None:
            first_fit = fit
            first_transfer = transfer

    assert np.allclose(pair_grams[0], pair_grams[1], rtol=2e-5, atol=2e-8)

    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(base.lattice)).T
    relabelled = _build_lpq_bloch_slab_truncated(
        base,
        ao_basis,
        aux_basis,
        np.zeros(3),
        first_transfer + reciprocal[:, 0],
        ke_cutoff=10.0,
        lat_opts=lat_opts,
        g_chunk=2048,
    )
    assert np.allclose(relabelled.factors, first_fit.factors, rtol=0.0, atol=2e-12)
    assert np.array_equal(relabelled.metric_signs, first_fit.metric_signs)
    assert np.allclose(relabelled.q_cart, first_fit.q_cart, rtol=0.0, atol=1e-14)

    normal = np.asarray(base.lattice)[:, 2]
    normal = normal / np.linalg.norm(normal)
    with pytest.raises(ValueError, match="Bloch momenta in the periodic slab plane"):
        _build_lpq_bloch_slab_truncated(
            base, ao_basis, aux_basis, np.zeros(3), 0.1 * normal
        )


def test_slab_probe_charge_and_gamma_fit_match_private_jk_envelope():
    """Raw J and probe-corrected K match independent slab implementations."""
    from vibeqc._vibeqc_core import bloch_sum, build_jk_2e_real_space
    from vibeqc.aux_basis import (
        _build_lpq_bloch_slab_truncated,
        _contract_slab_gdf_gamma,
        _slab_probe_charge_madelung,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.ewald_composed_slab import compute_j_slab_ewald_2d_gamma

    sysp, ao_basis = _h2_slab()
    molecule = sysp.unit_cell_molecule()
    aux_basis = make_modrho_aux_basis(
        make_aux_basis_set(molecule, aux_name="def2-svp-jk"), molecule
    )
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 12.0
    lat_opts.nuclear_cutoff_bohr = 30.0
    lat_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D

    overlap_lattice = vq.compute_overlap_lattice(ao_basis, sysp, lat_opts)
    overlap = np.real(bloch_sum(overlap_lattice, np.zeros(3)))
    overlap = 0.5 * (overlap + overlap.T)
    vector = np.ones(ao_basis.nbasis)
    density = 2.0 * np.outer(vector, vector) / float(vector @ overlap @ vector)

    fit = _build_lpq_bloch_slab_truncated(
        sysp,
        ao_basis,
        aux_basis,
        np.zeros(3),
        np.zeros(3),
        ke_cutoff=30.0,
        lat_opts=lat_opts,
        g_chunk=2048,
    )
    j_fit, k_fit = _contract_slab_gdf_gamma(fit, density, overlap)
    k_raw = np.einsum(
        "L,Lmk,kl,Lnl->mn",
        fit.metric_signs,
        fit.factors,
        density,
        fit.factors.conj(),
        optimize=True,
    ).real
    k_raw = 0.5 * (k_raw + k_raw.T)

    xi = _slab_probe_charge_madelung(sysp)
    assert np.isclose(
        xi,
        _slab_probe_charge_madelung(sysp, alpha=0.30),
        rtol=0.0,
        atol=2e-12,
    )
    assert np.isclose(fit.primitive_exchange_probe_charge_madelung, xi)

    j_reference = compute_j_slab_ewald_2d_gamma(
        ao_basis, sysp, density, lattice_opts=lat_opts
    )
    density_lattice = vq.compute_overlap_lattice(ao_basis, sysp, lat_opts)
    for index in range(len(density_lattice.cells)):
        density_lattice.set_block(index, density)
    k_reference = sum(
        np.asarray(block)
        for block in build_jk_2e_real_space(
            ao_basis, sysp, lat_opts, density_lattice, 0.0
        ).K.blocks
    )

    assert np.max(np.abs(j_fit - j_reference)) < 1.5e-4
    assert np.max(np.abs(k_raw - k_reference)) > 0.3
    assert np.max(np.abs(k_fit - k_reference)) < 1.5e-3
    exchange_energy_error = -0.25 * float(
        np.einsum("ij,ij->", density, k_fit - k_reference)
    )
    assert abs(exchange_energy_error) < 7e-4


def test_compact_slab_signed_fit_matches_independent_coulomb():
    """The compact-cell negative gauge mode contributes to the fitted J."""
    from vibeqc._vibeqc_core import bloch_sum
    from vibeqc.aux_basis import (
        _build_lpq_bloch_slab_truncated,
        _contract_slab_gdf_gamma,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.ewald_composed_slab import compute_j_slab_ewald_2d_gamma

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)
    molecule = sysp.unit_cell_molecule()
    ao_basis = vq.BasisSet(molecule, "sto-3g")
    aux_basis = make_modrho_aux_basis(
        make_aux_basis_set(molecule, aux_name="def2-svp-jk"), molecule
    )
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 18.0
    lat_opts.nuclear_cutoff_bohr = 30.0
    lat_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D

    overlap_lattice = vq.compute_overlap_lattice(ao_basis, sysp, lat_opts)
    overlap = np.real(bloch_sum(overlap_lattice, np.zeros(3)))
    overlap = 0.5 * (overlap + overlap.T)
    vector = np.ones(ao_basis.nbasis)
    density = 2.0 * np.outer(vector, vector) / float(vector @ overlap @ vector)

    fit = _build_lpq_bloch_slab_truncated(
        sysp,
        ao_basis,
        aux_basis,
        np.zeros(3),
        np.zeros(3),
        ke_cutoff=30.0,
        lat_opts=lat_opts,
    )
    assert np.count_nonzero(fit.metric_signs < 0) == 1
    assert float(fit.metric_eigenvalues[0]) < -1.0

    j_fit, k_fit = _contract_slab_gdf_gamma(fit, density, overlap)
    j_reference = compute_j_slab_ewald_2d_gamma(
        ao_basis, sysp, density, lattice_opts=lat_opts
    )
    assert np.max(np.abs(j_fit - j_reference)) < 1.5e-4
    assert np.all(np.isfinite(k_fit))


def test_compact_slab_multik_exchange_matches_pyscf_inf_vacuum():
    """Signed K plus the BvK probe shift matches an out-of-process oracle.

    Reference: PySCF 2.13.1, ``dimension=2``,
    ``low_dim_ft_type='inf_vacuum'``, ``exxdiv='ewald'``, STO-3G AO basis,
    def2-SVP-JKFIT auxiliary basis, and the exact density matrices assembled
    below on the Gamma-centered (2,2,1) mesh. PySCF was invoked only as a
    separate validation process; it is not a vibe-qc runtime dependency.
    """
    from vibeqc._vibeqc_core import bloch_sum
    from vibeqc.aux_basis import (
        _build_lpq_bloch_slab_truncated,
        _contract_slab_gdf_multik_exchange,
        _slab_probe_charge_madelung,
        _slab_probe_charge_madelung_for_kmesh,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)
    molecule = sysp.unit_cell_molecule()
    ao_basis = vq.BasisSet(molecule, "sto-3g")
    aux_basis = make_modrho_aux_basis(
        make_aux_basis_set(molecule, aux_name="def2-svp-jk"), molecule
    )
    lat_opts = vq.LatticeSumOptions()
    lat_opts.cutoff_bohr = 18.0
    mesh = (2, 2, 1)
    kmesh = vq.monkhorst_pack(sysp, list(mesh))
    kpoints = np.asarray(kmesh.kpoints)
    weights = np.asarray(kmesh.weights)

    overlap_lattice = vq.compute_overlap_lattice(ao_basis, sysp, lat_opts)
    overlaps = []
    densities = []
    for kpoint in kpoints:
        overlap = np.asarray(bloch_sum(overlap_lattice, kpoint))
        overlap = 0.5 * (overlap + overlap.conj().T)
        vector = np.ones(ao_basis.nbasis, dtype=complex)
        density = 2.0 * np.outer(vector, vector.conj()) / float(
            np.real(vector.conj() @ overlap @ vector)
        )
        overlaps.append(overlap)
        densities.append(density)

    fits = {}
    for i, k_bra in enumerate(kpoints):
        for j, k_ket in enumerate(kpoints):
            fits[(i, j)] = _build_lpq_bloch_slab_truncated(
                sysp,
                ao_basis,
                aux_basis,
                k_bra,
                k_ket,
                ke_cutoff=30.0,
                lat_opts=lat_opts,
                g_chunk=4096,
            )

    xi_bvk = _slab_probe_charge_madelung_for_kmesh(sysp, mesh)
    assert np.isclose(
        xi_bvk,
        0.5 * _slab_probe_charge_madelung(sysp),
        rtol=0.0,
        atol=2e-12,
    )
    with pytest.raises(ValueError, match="positive integer"):
        _slab_probe_charge_madelung_for_kmesh(sysp, (2.5, 2, 1))
    with pytest.raises(ValueError, match="n3 == 1"):
        _slab_probe_charge_madelung_for_kmesh(sysp, (2, 2, 2))
    exchange = _contract_slab_gdf_multik_exchange(
        fits,
        densities,
        overlaps,
        weights,
        bvk_probe_charge_madelung=xi_bvk,
    )
    pyscf_reference = np.array(
        [
            [[1.230214194416, 0.976997405342], [0.976997405342, 1.230214194438]],
            [[0.900650833952, 0.594991415193], [0.594991415193, 0.900650834104]],
            [[0.878260867888, 0.550098305037], [0.550098305030, 0.878260867976]],
            [[0.718433867980, 0.399586801119], [0.399586801119, 0.718433867976]],
        ]
    )
    assert np.max(np.abs(np.asarray(exchange) - pyscf_reference)) < 1.5e-5


def test_private_gamma_slab_gdf_rhf_is_a3_invariant_and_matches_direct():
    """Production-shaped neutral SCF stays private and shares the slab gauge."""
    from vibeqc.periodic_rhf_gdf import _run_rhf_periodic_gamma_slab_gdf

    atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
    results = []
    systems = []
    bases = []
    options = []
    for padding in (15.0, 40.0):
        sysp = vq.slab_2d(
            [18.0, 0.0, 0.0],
            [0.0, 18.0, 0.0],
            atoms,
            normal_padding_bohr=padding,
        )
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = 12.0
        opts.lattice_opts.nuclear_cutoff_bohr = 15.0
        opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
        opts.damping = 0.3
        opts.max_iter = 60
        opts.use_diis = True
        result = _run_rhf_periodic_gamma_slab_gdf(
            sysp, basis, opts, ke_cutoff=30.0, progress=False
        )
        assert result.converged
        assert result.backend == "private-slab-truncated-gdf"
        results.append(result)
        systems.append(sysp)
        bases.append(basis)
        options.append(opts)

    assert abs(results[0].energy - results[1].energy) < 5e-8
    direct = vq.run_rhf_periodic_scf(
        systems[0],
        bases[0],
        vq.monkhorst_pack(systems[0], [1, 1, 1]),
        options[0],
        progress=False,
    )
    assert direct.converged
    assert abs(results[0].energy - direct.energy) < 6e-4
    # Out-of-process PySCF 2.13.1 RHF-GDF reference for the first system:
    # dimension=2, low_dim_ft_type='inf_vacuum', exxdiv='ewald', STO-3G,
    # def2-SVP-JKFIT, precision=1e-11. The individual Coulomb terms use a
    # different constant, but the neutral total is gauge invariant.
    assert abs(results[0].energy - (-1.1162240798985295)) < 1e-5


def test_public_multik_slab_gdf_rhf_is_compact_a3_invariant():
    """The public adapter preserves the externally matched KRHF gauge."""
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    energies = []
    for normal_length in (30.0, 60.0):
        sysp = vq.PeriodicSystem(
            2, np.diag([4.6, 4.6, normal_length]), atoms
        )
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = 18.0
        opts.lattice_opts.nuclear_cutoff_bohr = 30.0
        original_coulomb_method = opts.lattice_opts.coulomb_method
        opts.max_iter = 60
        opts.damping = 0.3
        opts.dynamic_damping = False
        opts.conv_tol_energy = 1e-9
        opts.conv_tol_grad = 1e-8
        result = run_krhf_periodic_gdf(
            sysp, basis, (2, 2, 1), opts, rsgdf_ke_cutoff=30.0
        )
        assert result.converged
        assert result.backend == "native-multik-slab-truncated-gdf-rhf"
        assert result.n_iter <= opts.max_iter
        assert result.guess_selection.effective == vq.InitialGuess.SAD
        assert opts.lattice_opts.coulomb_method == original_coulomb_method
        energies.append(result.energy)

    # Out-of-process PySCF 2.13.1 KRHF-GDF reference: dimension=2,
    # low_dim_ft_type='inf_vacuum', exxdiv='ewald', STO-3G,
    # def2-SVP-JKFIT, precision=1e-11, Gamma-centered (2,2,1).
    assert abs(energies[0] - (-0.9175852704211138)) < 1e-5
    assert abs(energies[0] - energies[1]) < 1e-7

    with pytest.raises(ValueError, match="positive integer"):
        run_krhf_periodic_gdf(
            sysp, basis, (2.5, 2, 1), opts, rsgdf_ke_cutoff=30.0
        )
    with pytest.raises(ValueError, match=r"\(n1,n2,1\)"):
        run_krhf_periodic_gdf(
            sysp, basis, (2, 2, 2), opts, rsgdf_ke_cutoff=30.0
        )


def test_slab_gdf_occupations_use_one_global_fermi_level():
    """Slab GDF fills the mesh under ONE Fermi level, not per-k Aufbau.

    This is the contract ``60104fc01`` established for the 3D native
    multi-k GDF route (PERIODIC-K6-NONCONV); the 2D slab adapter was not
    covered by it and wrote ``2.0`` into the first ``n_occ`` slots at every
    k independently. On a band-overlapping system that mis-classifies the
    band edges: the hBN sto-3g slab reported a VBM 3.270 eV *above* its own
    "CBM" purely because both were read off a per-k occupation index.

    On a gapped cell the two conventions agree exactly, so this pins both
    halves: the occupations are bit-identical to the legacy pattern here,
    AND they are what one global chemical potential produces.
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf
    from vibeqc.smearing.apply import _global_aufbau_with_mu

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    opts.max_iter = 60
    opts.damping = 0.3
    opts.dynamic_damping = False
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-8

    result = run_krhf_periodic_gdf(
        sysp, basis, (2, 2, 1), opts, rsgdf_ke_cutoff=30.0
    )
    assert result.converged

    expected_occ, expected_mu = _global_aufbau_with_mu(
        [np.asarray(np.real(e), dtype=float) for e in result.mo_energies],
        np.asarray(result.kpoint_weights, dtype=float),
        float(sysp.n_electrons()),
        occ_value=2.0,
    )
    for got, want in zip(result.occupations, expected_occ):
        assert np.array_equal(np.asarray(got, dtype=float), want)
    assert result.fermi_level == pytest.approx(expected_mu, abs=1e-12)

    # This cell is gapped, so the global fill must still be the plain
    # integer pattern -- the fix is numerically inert away from band
    # overlap, exactly as 60104fc01 was on the 3D route.
    for occ in result.occupations:
        assert list(np.asarray(occ, dtype=float)) == [2.0, 0.0]

    # The weighted particle count is conserved.
    n_elec = sum(
        float(w) * float(np.sum(occ))
        for w, occ in zip(result.kpoint_weights, result.occupations)
    )
    assert n_elec == pytest.approx(float(sysp.n_electrons()), abs=1e-12)


def test_slab_gdf_scf_density_is_built_from_the_global_filling():
    """The SCF density itself uses the global occupations, not ``[:n_occ]``.

    Occupying ``coeff[:, :n_occ]`` per k builds the density from the wrong
    subspace whenever bands cross between k points, and the resulting
    selection can flip between adjacent iterations -- a classic SCF
    oscillation mechanism (CLAUDE.md section 7). Pin that the converged
    per-k density is exactly ``C diag(occ) C^dagger`` for the *reported*
    global occupations.
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    opts.max_iter = 60
    opts.damping = 0.0
    opts.dynamic_damping = False
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-8

    result = run_krhf_periodic_gdf(
        sysp, basis, (2, 2, 1), opts, rsgdf_ke_cutoff=30.0
    )
    assert result.converged

    for coeff, occ, density in zip(
        result.mo_coeffs, result.occupations, result.density
    ):
        coeff = np.asarray(coeff)
        rebuilt = (
            coeff * np.asarray(occ, dtype=float)[None, :]
        ) @ coeff.conj().T
        rebuilt = 0.5 * (rebuilt + rebuilt.conj().T)
        assert np.allclose(np.asarray(density), rebuilt, atol=1e-9)


def test_slab_gdf_fails_closed_on_smearing_rather_than_guessing():
    """Smearing is not implemented on this route and must say so.

    Recorded because it settles the triage lane's discriminating probe:
    the ``smearing_method="fermi-dirac"`` variant of the hBN reproducer
    cannot run at all here, so the inverted band ordering arose with zero
    smearing involvement and the occupation *classification* is its primary
    cause, independent of any smearing default.
    """
    from vibeqc.periodic_rhf_gdf import _run_krhf_periodic_slab_gdf
    from vibeqc._vibeqc_core import CoulombMethod

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = CoulombMethod.SLAB_EWALD_2D
    opts.smearing_temperature = 0.01

    with pytest.raises(
        NotImplementedError, match="does not yet support smearing"
    ):
        _run_krhf_periodic_slab_gdf(sysp, basis, (2, 2, 1), opts)


def test_slab_energy_identity_from_parts():
    """Rebuild the slab KRHF-GDF total from parts: gradient rung-1 gate.

    The two-gauge slab composition (bare Parry/de Leeuw-Perram 2D-Ewald
    V_ne + e_nuc, signed truncated-Coulomb fit J/K with the BvK probe-charge
    exxdiv) closes only in the neutral total. Before any slab force term is
    differentiated, this pins that the driver's reported energy IS the value
    of E = sum_k w_k Tr[D(k) (T+V_ne)(k)] + 1/2 sum_k w_k Tr[D(k) (J(k) -
    1/2 K(k))] + e_nuc with every part produced by the same builders the
    driver uses (the dense-core lesson: an energy identity broken at the
    base geometry means a gradient would differentiate the wrong function).
    """
    from vibeqc._vibeqc_core import monkhorst_pack as _mp_native
    from vibeqc.aux_basis import (
        _build_lpq_bloch_slab_truncated,
        _contract_slab_gdf_multik_exchange,
        _slab_probe_charge_madelung_for_kmesh,
        default_aux_for,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic_k_gdf import (
        _clone_slab_lattice_options,
        run_krhf_periodic_gdf,
    )
    from vibeqc.periodic_v_ne_slab import (
        build_v_ne_slab_ewald_2d_k_cache,
        compute_v_ne_slab_ewald_2d_k_matrix,
    )

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    opts.max_iter = 60
    opts.damping = 0.3
    opts.dynamic_damping = False
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-8

    mesh = (2, 2, 1)
    result = run_krhf_periodic_gdf(
        sysp, basis, mesh, opts, rsgdf_ke_cutoff=30.0
    )
    assert result.converged

    # The public adapter runs the private driver under a cloned lat_opts with
    # the slab gauge forced on; reproduce that exact object.
    lat_opts = _clone_slab_lattice_options(opts.lattice_opts)

    kmesh = _mp_native(sysp, list(mesh), [0, 0, 0], False)
    kpoints = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = int(kpoints.shape[0])
    np.testing.assert_allclose(
        kpoints, np.asarray(result.kpoints_cart), rtol=0.0, atol=0.0
    )

    # One-electron parts: the driver's own lattice builders + Bloch sums.
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    T_lat = vq.compute_kinetic_lattice(basis, sysp, lat_opts)
    v_ne_cache = build_v_ne_slab_ewald_2d_k_cache(
        basis, sysp, lat_opts, alpha=0.0
    )
    overlaps = []
    hcores = []
    for kpoint in kpoints:
        overlap = np.asarray(vq.bloch_sum(S_lat, kpoint), dtype=complex)
        kinetic = np.asarray(vq.bloch_sum(T_lat, kpoint), dtype=complex)
        nuclear = compute_v_ne_slab_ewald_2d_k_matrix(
            basis, sysp, lat_opts, kpoint, alpha=0.0, cache=v_ne_cache
        )
        overlaps.append(0.5 * (overlap + overlap.conj().T))
        hcore = kinetic + nuclear
        hcores.append(0.5 * (hcore + hcore.conj().T))
    for rebuilt_h, driver_h in zip(hcores, result.hcore):
        assert np.max(np.abs(rebuilt_h - np.asarray(driver_h))) < 1e-12
    for rebuilt_s, driver_s in zip(overlaps, result.overlap):
        assert np.max(np.abs(rebuilt_s - np.asarray(driver_s))) < 1e-12

    # Two-electron parts: the signed truncated-Coulomb Bloch fits at the
    # driver's exact settings, contracted at the converged density.
    molecule = sysp.unit_cell_molecule()
    raw_aux = make_aux_basis_set(molecule, aux_name=default_aux_for(basis.name))
    modrho_aux = make_modrho_aux_basis(raw_aux, molecule)
    fits = {}
    for i, k_bra in enumerate(kpoints):
        for j, k_ket in enumerate(kpoints):
            fits[(i, j)] = _build_lpq_bloch_slab_truncated(
                sysp,
                basis,
                modrho_aux,
                k_bra,
                k_ket,
                ke_cutoff=30.0,
                lat_opts=lat_opts,
                linear_dep_thr=1e-9,
            )
    xi_bvk = _slab_probe_charge_madelung_for_kmesh(sysp, mesh)
    densities = [np.asarray(D, dtype=complex) for D in result.density]

    # The driver's J closure algebra, verbatim.
    rho = np.zeros(fits[(0, 0)].factors.shape[0], dtype=complex)
    for j in range(n_k):
        fit_jj = fits[(j, j)]
        rho += float(weights[j]) * fit_jj.metric_signs * np.einsum(
            "Pmn,nm->P", fit_jj.factors, densities[j], optimize=True
        )
    coulomb = []
    for i in range(n_k):
        block = np.einsum(
            "P,Pmn->mn", rho, fits[(i, i)].factors, optimize=True
        )
        coulomb.append(0.5 * (block + block.conj().T))
    exchange = _contract_slab_gdf_multik_exchange(
        fits, densities, overlaps, weights, bvk_probe_charge_madelung=xi_bvk
    )

    e_nuc = float(vq.nuclear_repulsion_per_cell(sysp, lat_opts))
    e_core = 0.0
    e_coulomb = 0.0
    e_exchange = 0.0
    for w, D, H, J, K in zip(weights, densities, hcores, coulomb, exchange):
        e_core += float(w) * float(np.real(np.trace(D @ H)))
        e_coulomb += 0.5 * float(w) * float(np.real(np.trace(D @ J)))
        e_exchange += -0.25 * float(w) * float(
            np.real(np.trace(D @ np.asarray(K)))
        )

    # Component pins against the driver-exposed fields (the bisection
    # handles if the total identity ever regresses)...
    assert abs(e_coulomb - result.e_coulomb) < 1e-10
    assert abs(e_exchange - result.e_hf_exchange) < 1e-10
    assert abs(e_nuc - result.e_nuclear) < 1e-10
    # ... and the rung-1 gate itself.
    rebuilt = e_core + e_coulomb + e_exchange + e_nuc
    assert abs(rebuilt - result.energy) < 1e-9, (
        f"slab energy identity broken: rebuilt {rebuilt!r} vs driver "
        f"{result.energy!r}"
    )


def test_slab_v_ne_fixed_density_gradient_matches_fd():
    """Rung-3 gate: analytic fixed-density slab V_ne derivative vs FD.

    Freezes the converged D(k) of the compact (2,2,1) H2 slab fixture and
    central-differences E = sum_k w_k Re Tr[D(k) V_ne(k)] by rebuilding the
    SCF's OWN V_ne cache/matrices at +-h per displaced geometry
    (``build_v_ne_slab_ewald_2d_k_cache`` at the driver's ``alpha=0.0`` ->
    0.4 resolution). Gates every Cartesian component INCLUDING the slab
    normal (the z-force is the slab-specific part: the z-resolved v_long
    F-structure and the v_g0 erf chain rule).

    Alpha gates: the total bare-block V_ne is alpha-invariant, so the total
    fixed-D derivative must be too (loose gate — mesh-convergence-limited);
    additionally the derivative at a non-default alpha must match FD of the
    energy REBUILT at that same alpha (each alpha's V_ne is a different but
    self-consistent function).

    Translation: with the AOs riding on the atoms, Tr[D V_ne] depends on
    relative positions only, so the fixed-D V_ne translation sum vanishes
    on its own (nuclear and AO sides cancel exactly for v_short/v_long and
    to quadrature accuracy for v_g0); the combined nn + V_ne sum is pinned
    too. Newton (equal-and-opposite on the 2-atom fixture) is implied.
    """
    from vibeqc._vibeqc_core import monkhorst_pack as _mp_native
    from vibeqc.periodic_k_gdf import (
        _clone_slab_lattice_options,
        run_krhf_periodic_gdf,
    )
    from vibeqc.periodic_v_ne_slab import (
        build_v_ne_slab_ewald_2d_k_cache,
        compute_v_ne_slab_ewald_2d_k_matrix,
    )
    from vibeqc.periodic_v_ne_slab_gradient import (
        compute_v_ne_slab_ewald_2d_gradient,
    )

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    lattice = np.diag([4.6, 4.6, 30.0])
    sysp = vq.PeriodicSystem(2, lattice, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    opts.max_iter = 60
    opts.damping = 0.3
    opts.dynamic_damping = False
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-8
    mesh = (2, 2, 1)
    result = run_krhf_periodic_gdf(sysp, basis, mesh, opts, rsgdf_ke_cutoff=30.0)
    assert result.converged
    lat_opts = _clone_slab_lattice_options(opts.lattice_opts)

    kmesh = _mp_native(sysp, list(mesh), [0, 0, 0], False)
    kpts = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    densities = [np.asarray(D, dtype=complex) for D in result.density]

    def e_vne(atom_list, alpha):
        """The SCF's own V_ne energy at frozen D: cache + per-k matrices."""
        s = vq.PeriodicSystem(2, lattice, atom_list)
        b = vq.BasisSet(s.unit_cell_molecule(), basis.name)
        cache = build_v_ne_slab_ewald_2d_k_cache(b, s, lat_opts, alpha=alpha)
        e = 0.0
        for w, k, D in zip(weights, kpts, densities):
            V = compute_v_ne_slab_ewald_2d_k_matrix(
                b, s, lat_opts, k, alpha=alpha, cache=cache
            )
            e += float(w) * float(np.real(np.trace(D @ V)))
        return e

    def fd_component(atom_idx, axis, alpha, h=1.0e-5):
        energies = []
        for sign in (1.0, -1.0):
            displaced = []
            for idx, atom in enumerate(atoms):
                xyz = np.asarray(atom.xyz, dtype=float).copy()
                if idx == atom_idx:
                    xyz[axis] += sign * h
                displaced.append(vq.Atom(int(atom.Z), xyz.tolist()))
            energies.append(e_vne(displaced, alpha))
        return (energies[0] - energies[1]) / (2.0 * h)

    grad = compute_v_ne_slab_ewald_2d_gradient(
        basis, sysp, lat_opts, densities, weights, kpts, alpha=0.0
    )
    assert grad.shape == (2, 3)

    # ---- gate 1: FD at the SCF's alpha (0.0 -> 0.4), every component.
    # Measured on this fixture: in-plane <= 1.1e-10, normal <= 3.9e-11.
    for atom_idx in range(2):
        for axis in range(3):
            fd_val = fd_component(atom_idx, axis, alpha=0.0)
            label = "normal" if axis == 2 else "in-plane"
            assert abs(grad[atom_idx, axis] - fd_val) < 1.0e-7, (
                f"slab V_ne gradient mismatch ({label}) at atom {atom_idx} "
                f"axis {axis}: analytic={grad[atom_idx, axis]:.12f}, "
                f"FD={fd_val:.12f}"
            )

    # The driver passes alpha=0.0; the cache resolves it to the forced slab
    # default 0.4. The derivative must resolve identically (same code path;
    # tolerance covers OpenMP reduction-order jitter in the C++ kernels,
    # not a formula difference).
    grad_04 = compute_v_ne_slab_ewald_2d_gradient(
        basis, sysp, lat_opts, densities, weights, kpts, alpha=0.4
    )
    np.testing.assert_allclose(grad, grad_04, rtol=0.0, atol=1.0e-13)

    # ---- gate 2: alpha-consistency. At alpha=0.6 the split is different
    # but self-consistent: the derivative must match FD of the energy
    # rebuilt at THAT alpha (measured 3.6e-11 / 5.4e-11 here) ...
    grad_06 = compute_v_ne_slab_ewald_2d_gradient(
        basis, sysp, lat_opts, densities, weights, kpts, alpha=0.6
    )
    for atom_idx, axis in ((0, 0), (0, 2)):
        fd_val = fd_component(atom_idx, axis, alpha=0.6)
        assert abs(grad_06[atom_idx, axis] - fd_val) < 1.0e-7
    # ... and because the TOTAL bare-block V_ne is alpha-invariant (the
    # rigorous Ewald-split gate of the value module), the total fixed-D
    # derivative is alpha-invariant as well, to mesh convergence
    # (measured 1.7e-15 / 4.3e-15 on this fixture).
    grad_03 = compute_v_ne_slab_ewald_2d_gradient(
        basis, sysp, lat_opts, densities, weights, kpts, alpha=0.3
    )
    assert np.max(np.abs(grad_06 - grad)) < 1.0e-9
    assert np.max(np.abs(grad_03 - grad)) < 1.0e-9

    # ---- gate 3: translation. Fixed-D Tr[D V_ne] depends on relative
    # positions only => the V_ne term's own translation sum vanishes
    # (measured ~3e-16); adding the rung-2 nn gradient keeps it zero.
    assert np.max(np.abs(grad.sum(axis=0))) < 1.0e-12
    g_nn = np.asarray(
        vq.nuclear_repulsion_slab_ewald_2d_gradient(sysp, lat_opts)
    )
    assert np.max(np.abs((grad + g_nn).sum(axis=0))) < 1.0e-12

    # ---- fail-loud guards ------------------------------------------------
    bulk = vq.PeriodicSystem(3, lattice, atoms)
    with pytest.raises(ValueError, match="dim == 2"):
        compute_v_ne_slab_ewald_2d_gradient(
            basis, bulk, lat_opts, densities, weights, kpts
        )
    plain_opts = vq.PeriodicRHFOptions().lattice_opts
    with pytest.raises(ValueError, match="SLAB_EWALD_2D"):
        compute_v_ne_slab_ewald_2d_gradient(
            basis, sysp, plain_opts, densities, weights, kpts
        )
    broken = [D.copy() for D in densities]
    broken[0] = broken[0] + np.array([[0.0, 1.0e-3], [0.0, 0.0]])
    with pytest.raises(ValueError, match="Hermitian"):
        compute_v_ne_slab_ewald_2d_gradient(
            basis, sysp, lat_opts, broken, weights, kpts
        )
    with pytest.raises(ValueError, match="length"):
        compute_v_ne_slab_ewald_2d_gradient(
            basis, sysp, lat_opts, densities[:-1], weights, kpts
        )


def test_signed_truncated_inverse_frechet_matches_dense_fd():
    """Rung-4 unit gate: the SIGNED truncated-inverse Fréchet response.

    The slab metric is INDEFINITE (finite negative K(0) zero mode), and
    its signed factor contract collapses onto f(M) = U_keep
    diag(1/lambda) U_keep^H with the |lambda| > thr keep criterion
    (sign/|lambda| = 1/lambda for both signs). Before any slab J/K
    assembly consumes the divided-difference table, verify it against a
    dense central difference of f(M) itself, on a matrix with kept AND
    dropped modes of BOTH signs under a random Hermitian perturbation —
    including the indefinite-specific entries: the opposite-sign
    kept/kept block (-1/(l_a l_b) > 0, POSITIVE, unlike any
    definite-metric response) and the kept/dropped rotation of a
    NEGATIVE kept mode against dropped modes of either sign.
    """
    from vibeqc.periodic_gdf_gradient import (
        _signed_truncated_inverse_frechet,
    )

    rng = np.random.default_rng(7)
    lam = np.array([2.0, 1.0, -1.5, 0.4, -0.3, 3.0e-5, -2.0e-5])
    thr = 1.0e-3
    n = lam.size
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(Z)  # Haar-ish unitary; any unitary works
    M = (Q * lam[None, :]) @ Q.conj().T
    M = 0.5 * (M + M.conj().T)
    dM = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    dM = 0.05 * (dM + dM.conj().T)  # small direction: the FD truncation
    # error is cubic in |dM| at fixed h while the compared derivative is
    # linear, so this buys two orders of gate margin for free.

    def f_of(mat):
        vals, vecs = np.linalg.eigh(mat)
        keep = np.abs(vals) > thr
        return (
            vecs[:, keep] * (1.0 / vals[keep])[None, :]
        ) @ vecs[:, keep].conj().T

    eigvals, U = np.linalg.eigh(M)
    keep = np.abs(eigvals) > thr
    # The scenario the docstring promises: kept and dropped modes of
    # both signs (dropped stay < thr under the h*dM perturbation).
    assert int(np.count_nonzero(keep & (eigvals > 0))) == 3
    assert int(np.count_nonzero(keep & (eigvals < 0))) == 2
    assert int(np.count_nonzero(~keep & (eigvals > 0))) == 1
    assert int(np.count_nonzero(~keep & (eigvals < 0))) == 1

    inv_eig, table = _signed_truncated_inverse_frechet(
        eigvals, keep, spectral_tol=1e-12, where="unit test"
    )
    np.testing.assert_array_equal(inv_eig[keep], 1.0 / eigvals[keep])
    assert np.all(inv_eig[~keep] == 0.0)
    # Opposite-sign kept pair: POSITIVE response entry.
    kept_idx = np.where(keep)[0]
    pos = kept_idx[eigvals[kept_idx] > 0][0]
    neg = kept_idx[eigvals[kept_idx] < 0][0]
    assert float(np.real(table[pos, neg])) > 0.0

    df_analytic = U @ (table * (U.conj().T @ dM @ U)) @ U.conj().T
    # h = 1e-5 balances the eigh-roundoff floor (~eps ||f|| / h) against
    # the (tiny, |dM|^3-scaled) truncation term; measured 1.2e-10 on
    # this seed. At h = 1e-6 the roundoff floor alone reaches ~1e-9.
    h = 1.0e-5
    df_fd = (f_of(M + h * dM) - f_of(M - h * dM)) / (2.0 * h)
    err = float(np.max(np.abs(df_analytic - df_fd)))
    assert err <= 1.0e-9, f"signed Fréchet vs dense FD: {err:.3e}"


def test_slab_gdf_gradient_cache_bit_consistent_with_scf_fit():
    """Rung-4 cache gate: the slab gradient cache mirrors the SCF's
    per-(i, j) ``_build_lpq_bloch_slab_truncated`` fits bit-for-bit.

    For EVERY (k_i, k_j) pair of the compact (2,2,1) fixture,
    reconstructs the signed factors from the cache's per-q eigensystem
    + per-pair T exactly as ``_factor_slab_truncated_gdf_metric`` does
    and demands ``np.array_equal`` against a direct SCF fit build —
    the M6-rung-9 lesson: any fit mismatch is Fréchet-amplified near
    threshold, so closeness gates are not enough. Also pins the
    slab-specific cache content: the q = 0 mesh retains the EXACT zero
    point with the finite negative Sundararaman-Arias weight
    K(0)/V = -pi L^2 / (2 V), the q = 0 metric is indefinite (one
    negative sign in the signature), and the q != 0 meshes contain no
    zero mode. Guard coverage: the generalized Bloch kernels reject a
    lone g_mesh/kernel_weights and an external-weights tail request.
    """
    from vibeqc._vibeqc_core import monkhorst_pack as _mp_native
    from vibeqc.aux_basis import (
        _build_lpq_bloch_slab_truncated,
        _rsgdf_weighted_2c_metric_gradient_bloch,
        _rsgdf_weighted_3c_tensor_gradient_bloch,
        default_aux_for,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic_gdf_gradient import _build_slab_gdf_gradient_cache
    from vibeqc.periodic_k_gdf import _clone_slab_lattice_options

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    lattice = np.diag([4.6, 4.6, 30.0])
    sysp = vq.PeriodicSystem(2, lattice, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    lat_opts = _clone_slab_lattice_options(opts.lattice_opts)
    kmesh = _mp_native(sysp, [2, 2, 1], [0, 0, 0], False)
    kpts = np.asarray(kmesh.kpoints, dtype=float)
    mol = sysp.unit_cell_molecule()
    raw_aux = make_aux_basis_set(mol, aux_name=default_aux_for(basis.name))
    modrho = make_modrho_aux_basis(raw_aux, mol)

    cache = _build_slab_gdf_gradient_cache(
        sysp, basis, modrho, kpts,
        ke_cutoff=30.0, lat_opts=lat_opts, linear_dep_thr=1e-9,
    )
    assert cache.n_aux == modrho.nbasis
    assert cache.n_orb == basis.nbasis

    covered = sorted(p for g in cache.groups.values() for p in g.pairs)
    assert covered == [(i, j) for i in range(4) for j in range(4)]
    assert len(cache.groups) == 4  # q=0 + three BZ-boundary transfers

    volume = float(abs(np.linalg.det(lattice)))
    for group in cache.groups.values():
        mesh_norms = np.linalg.norm(group.g_mesh, axis=1)
        assert group.kernel_weights.shape == (group.g_mesh.shape[0],)
        if float(np.linalg.norm(group.q)) < 1e-12:
            # The exact zero point rides in the mesh with the finite
            # NEGATIVE truncated-kernel gauge weight — the indefinite
            # metric's origin.
            zero_rows = np.where(mesh_norms == 0.0)[0]
            assert zero_rows.size == 1
            assert group.kernel_weights[zero_rows[0]] == (
                -0.5 * np.pi * 30.0 * 30.0 / volume
            )
            assert float(group.eigvals.min()) < 0.0
            assert np.any(group.signs == np.int8(-1))
        else:
            assert np.all(mesh_norms > 0.0)
            assert np.all(group.kernel_weights > 0.0)

    for group in cache.groups.values():
        keep = group.keep_mask
        scale = np.sqrt(np.abs(group.eigvals[keep]))[:, None]
        for pair_pos, (i, j) in enumerate(group.pairs):
            fit = _build_lpq_bloch_slab_truncated(
                sysp, basis, modrho, kpts[i], kpts[j],
                ke_cutoff=30.0, lat_opts=lat_opts, linear_dep_thr=1e-9,
            )
            assert np.array_equal(group.eigvals, fit.metric_eigenvalues)
            assert np.array_equal(group.signs, fit.metric_signs)
            T_flat = group.T_list[pair_pos].reshape(cache.n_aux, -1)
            factors = (
                (group.eigvecs[:, keep].conj().T @ T_flat) / scale
            ).reshape(group.n_fit, cache.n_orb, cache.n_orb)
            assert np.array_equal(factors, fit.factors)
            np.testing.assert_allclose(
                group.q, fit.q_cart, rtol=0.0, atol=0.0
            )

    # Kernel guard coverage for the new external-weights envelope.
    q0 = next(
        g for g in cache.groups.values()
        if float(np.linalg.norm(g.q)) < 1e-12
    )
    w2c = np.zeros((cache.n_aux, cache.n_aux), dtype=np.complex128)
    with pytest.raises(ValueError, match="supplied together"):
        _rsgdf_weighted_2c_metric_gradient_bloch(
            modrho, sysp, ke_cutoff=30.0, q_cart=q0.q, weight=w2c,
            g_mesh=q0.g_mesh,
        )
    with pytest.raises(ValueError, match="tail"):
        _rsgdf_weighted_2c_metric_gradient_bloch(
            modrho, sysp, ke_cutoff=30.0, q_cart=q0.q, weight=w2c,
            tail_ke_cutoff=60.0,
            g_mesh=q0.g_mesh, kernel_weights=q0.kernel_weights,
        )
    w3c = np.zeros(
        (cache.n_aux, cache.n_orb, cache.n_orb), dtype=np.complex128
    )
    with pytest.raises(ValueError, match="supplied together"):
        _rsgdf_weighted_3c_tensor_gradient_bloch(
            modrho, basis, sysp, ke_cutoff=30.0, weight=w3c,
            k_ket=kpts[0], q_cart=q0.q, lat_opts=lat_opts,
            kernel_weights=q0.kernel_weights,
        )
    with pytest.raises(ValueError, match="tail"):
        _rsgdf_weighted_3c_tensor_gradient_bloch(
            modrho, basis, sysp, ke_cutoff=30.0, weight=w3c,
            k_ket=kpts[0], q_cart=q0.q, lat_opts=lat_opts,
            tail_ke_cutoff=60.0,
            g_mesh=q0.g_mesh, kernel_weights=q0.kernel_weights,
        )


def _slab_fits_from_cache(cache):
    """Reconstruct the driver's per-(i, j) ``_SlabGdfBlochFit`` blocks
    from the gradient cache (bit-identical to the SCF fits per the
    cache gate), with the probe constant zeroed: the FD objectives
    below isolate the FITTED J/K terms the rung-4 assemblies
    differentiate (dxi/dR = 0; the xi S D S overlap response is the
    composition rung's term)."""
    from vibeqc.aux_basis import _SlabGdfBlochFit

    fits = {}
    for group in cache.groups.values():
        keep = group.keep_mask
        scale = np.sqrt(np.abs(group.eigvals[keep]))[:, None]
        for pair_pos, (i, j) in enumerate(group.pairs):
            T_flat = group.T_list[pair_pos].reshape(cache.n_aux, -1)
            factors = (
                (group.eigvecs[:, keep].conj().T @ T_flat) / scale
            ).reshape(group.n_fit, cache.n_orb, cache.n_orb)
            fits[(i, j)] = _SlabGdfBlochFit(
                factors=factors,
                metric_signs=group.signs,
                metric_eigenvalues=group.eigvals,
                q_cart=group.q,
                includes_finite_coulomb_zero_mode=True,
                primitive_exchange_probe_charge_madelung=0.0,
            )
    return fits


def _slab_ej_ek_from_cache(cache, densities, overlaps, weights):
    """(E_J, E_K^fit) through the DRIVER'S OWN contraction algebra on
    cache-reconstructed fits: the J closure of
    ``_run_krhf_periodic_slab_gdf`` verbatim and
    ``_contract_slab_gdf_multik_exchange`` at xi = 0 — exactly the
    unfactorized objectives the rung-4 assemblies differentiate."""
    from vibeqc.aux_basis import _contract_slab_gdf_multik_exchange

    fits = _slab_fits_from_cache(cache)
    n_k = len(densities)
    rho = np.zeros(fits[(0, 0)].factors.shape[0], dtype=complex)
    for j in range(n_k):
        fit_jj = fits[(j, j)]
        rho += float(weights[j]) * fit_jj.metric_signs * np.einsum(
            "Pmn,nm->P", fit_jj.factors, densities[j], optimize=True
        )
    e_j = 0.0
    for i in range(n_k):
        block = np.einsum(
            "P,Pmn->mn", rho, fits[(i, i)].factors, optimize=True
        )
        block = 0.5 * (block + block.conj().T)
        e_j += 0.5 * float(weights[i]) * float(
            np.real(np.trace(densities[i] @ block))
        )
    exchange = _contract_slab_gdf_multik_exchange(
        fits, densities, overlaps, weights, bvk_probe_charge_madelung=0.0
    )
    e_k = sum(
        -0.25 * float(weights[i]) * float(
            np.real(np.trace(densities[i] @ np.asarray(exchange[i])))
        )
        for i in range(n_k)
    )
    return e_j, e_k


def test_slab_j_and_k_gradient_fixed_density_vs_fd():
    """Rung-4 gate: analytic fixed-density slab GDF J and K fit
    derivatives vs FD, every Cartesian component INCLUDING the normal.

    Freezes the converged D(k) of the compact (2,2,1) H2 slab fixture;
    the FD side rebuilds system + basis + aux + gradient cache per
    displaced geometry (the fits are bit-identical to SCF fits per the
    cache gate) and re-evaluates E_J and E_K^fit through the driver's
    own contraction algebra at that geometry. The analytic side is the
    signed-Fréchet weight algebra of ``_compute_j_gradient_slab_gdf`` /
    ``_compute_k_gradient_slab_gdf`` — the INDEFINITE-metric response
    is live here (the q = 0 signature carries one negative sign from
    the finite K(0) zero mode). Measured on this fixture at h = 1e-4:
    J in-plane <= 6.7e-11, J normal <= 9.0e-11; K in-plane <= 1.1e-10,
    K normal <= 1.2e-10 — gate 1e-7 per component.

    Also pinned: the cache objectives reproduce the driver's OWN
    energy components at the converged density BITWISE (E_J vs
    e_coulomb 0.0; E_K^fit + xi-shift vs e_hf_exchange 0.0 — the
    cache bit-identity at work, and the split separating the fit term
    this rung differentiates from the probe-charge overlap response it
    defers), xi is atom-position independent (pure lattice functional,
    dxi/dR = 0 exactly), and Newton's third law on both fit terms
    (every phase depends on centre differences only under rigid
    translation; measured <= 4.6e-16).
    """
    from vibeqc._vibeqc_core import monkhorst_pack as _mp_native
    from vibeqc.aux_basis import (
        _slab_probe_charge_madelung_for_kmesh,
        default_aux_for,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic_gdf_gradient import (
        _build_slab_gdf_gradient_cache,
        _compute_j_gradient_slab_gdf,
        _compute_k_gradient_slab_gdf,
    )
    from vibeqc.periodic_k_gdf import (
        _clone_slab_lattice_options,
        run_krhf_periodic_gdf,
    )

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    lattice = np.diag([4.6, 4.6, 30.0])
    sysp = vq.PeriodicSystem(2, lattice, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    opts.max_iter = 60
    opts.damping = 0.3
    opts.dynamic_damping = False
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-8
    mesh = (2, 2, 1)
    result = run_krhf_periodic_gdf(
        sysp, basis, mesh, opts, rsgdf_ke_cutoff=30.0
    )
    assert result.converged
    lat_opts = _clone_slab_lattice_options(opts.lattice_opts)
    kmesh = _mp_native(sysp, list(mesh), [0, 0, 0], False)
    kpts = np.asarray(kmesh.kpoints, dtype=float)
    weights = np.asarray(kmesh.weights, dtype=float)
    densities = [np.asarray(D, dtype=complex) for D in result.density]
    overlaps = [np.asarray(S, dtype=complex) for S in result.overlap]

    def cache_for(atom_list):
        s = vq.PeriodicSystem(2, lattice, atom_list)
        b = vq.BasisSet(s.unit_cell_molecule(), basis.name)
        mol = s.unit_cell_molecule()
        raw = make_aux_basis_set(mol, aux_name=default_aux_for(b.name))
        modrho = make_modrho_aux_basis(raw, mol)
        return s, b, _build_slab_gdf_gradient_cache(
            s, b, modrho, kpts,
            ke_cutoff=30.0, lat_opts=lat_opts, linear_dep_thr=1e-9,
        )

    _, _, cache = cache_for(atoms)
    e_j0, e_k0 = _slab_ej_ek_from_cache(cache, densities, overlaps, weights)
    # The cache objectives ARE the driver's energy components at the
    # converged density: E_J vs e_coulomb directly; the exchange
    # separates as e_hf_exchange = E_K^fit + xi-shift with the BvK
    # probe constant (rung-1 identity, fit/probe split).
    assert abs(e_j0 - float(result.e_coulomb)) < 1e-9
    xi = _slab_probe_charge_madelung_for_kmesh(sysp, mesh)
    e_shift = sum(
        -0.25 * float(weights[i]) * xi * float(
            np.real(np.trace(
                densities[i] @ overlaps[i] @ densities[i] @ overlaps[i]
            ))
        )
        for i in range(len(densities))
    )
    assert abs(e_k0 + e_shift - float(result.e_hf_exchange)) < 1e-9
    # dxi/dR = 0, exactly: the probe constant is a pure lattice
    # functional (it never sees atom positions).
    moved = [vq.Atom(1, [0.9, 0.1, 0.75]), vq.Atom(1, [2.0, 1.7, -0.35])]
    assert _slab_probe_charge_madelung_for_kmesh(
        vq.PeriodicSystem(2, lattice, moved), mesh
    ) == xi

    grad_j = _compute_j_gradient_slab_gdf(
        sysp, basis, densities, weights, cache
    )
    grad_k = _compute_k_gradient_slab_gdf(
        sysp, basis, densities, weights, cache
    )
    assert grad_j.shape == (2, 3)
    assert grad_k.shape == (2, 3)
    # Fixed-D fit terms are invariant under rigid translation (every
    # phase depends on centre differences only): Newton's third law.
    assert np.max(np.abs(grad_j.sum(axis=0))) < 1e-12
    assert np.max(np.abs(grad_k.sum(axis=0))) < 1e-12

    h = 1.0e-4
    for atom_idx in range(2):
        for axis in range(3):
            vals = {}
            for sign in (+1.0, -1.0):
                displaced = []
                for idx, atom in enumerate(atoms):
                    xyz = np.asarray(atom.xyz, dtype=float).copy()
                    if idx == atom_idx:
                        xyz[axis] += sign * h
                    displaced.append(vq.Atom(int(atom.Z), xyz.tolist()))
                _, _, c_d = cache_for(displaced)
                vals[sign] = _slab_ej_ek_from_cache(
                    c_d, densities, overlaps, weights
                )
            fd_j = (vals[+1.0][0] - vals[-1.0][0]) / (2.0 * h)
            fd_k = (vals[+1.0][1] - vals[-1.0][1]) / (2.0 * h)
            label = "normal" if axis == 2 else "in-plane"
            assert abs(grad_j[atom_idx, axis] - fd_j) <= 1e-7, (
                f"slab J gradient mismatch ({label}) atom {atom_idx} "
                f"axis {axis}: analytic={grad_j[atom_idx, axis]:.12f} "
                f"FD={fd_j:.12f}"
            )
            assert abs(grad_k[atom_idx, axis] - fd_k) <= 1e-7, (
                f"slab K gradient mismatch ({label}) atom {atom_idx} "
                f"axis {axis}: analytic={grad_k[atom_idx, axis]:.12f} "
                f"FD={fd_k:.12f}"
            )


def _compact_slab_rhf_opts():
    """The compact-fixture RHF options at gradient-gate convergence
    (tightened vs the energy tests so the FD of the driver total is not
    convergence-noise limited at h = 2e-4)."""
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    opts.max_iter = 60
    opts.damping = 0.3
    opts.dynamic_damping = False
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-9
    return opts


def _displaced_compact_atoms(atoms, atom_idx, axis, sign, h):
    displaced = []
    for idx, atom in enumerate(atoms):
        xyz = np.asarray(atom.xyz, dtype=float).copy()
        if idx == atom_idx:
            xyz[axis] += sign * h
        displaced.append(vq.Atom(int(atom.Z), xyz.tolist()))
    return displaced


def test_slab_krhf_full_scf_gradient_vs_fd():
    """§ 6 rung 5 — THE slab composition gate: the public
    ``run_krhf_periodic_gdf(compute_gradient=True)`` gradient (bare
    2D-Ewald nn + kinetic fold + overlap-W fold + slab V_ne + signed
    DF-J/K + BvK probe-charge W-shift, all FD-gated per term in the
    rung 2-4 tests) matches a central difference of the DRIVER's
    converged total energy on the compact (2,2,1) fixture.

    The rung-1 identity is bit-exact, so the FD surface IS the function
    the assembler differentiates. Measured at h = 2e-4: atom-0 x
    1.9e-10, atom-0 z (the slab normal) 3.2e-11, atom-1 y 2.2e-10
    Ha/bohr — gate 1e-6 (the 3D full-SCF anchors sit at the same
    1e-8-class level). Newton residual measured 1.0e-15. A Γ (1,1,1)
    sanity arm pins the mesh reduction on the same route (measured
    2.1e-10 on the normal). If this gate ever fails, bisect with the
    per-term fixed-D gates above — do NOT paper over (CLAUDE.md § 7).
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    lattice = np.diag([4.6, 4.6, 30.0])
    h = 2e-4

    def run(atom_list, mesh, grad=False):
        sysp = vq.PeriodicSystem(2, lattice, atom_list)
        basis = vq.BasisSet(sysp.unit_cell_molecule(), basis_name)
        r = run_krhf_periodic_gdf(
            sysp,
            basis,
            mesh,
            _compact_slab_rhf_opts(),
            rsgdf_ke_cutoff=30.0,
            compute_gradient=grad,
        )
        assert r.converged
        return r

    basis_name = "sto-3g"
    r0 = run(atoms, (2, 2, 1), grad=True)
    grad = np.asarray(r0.gradient)
    assert grad.shape == (2, 3)
    assert np.max(np.abs(grad.sum(axis=0))) < 1e-12

    for atom_idx, axis in ((0, 0), (0, 2), (1, 1)):
        fd = (
            run(_displaced_compact_atoms(atoms, atom_idx, axis, +1.0, h),
                (2, 2, 1)).energy
            - run(_displaced_compact_atoms(atoms, atom_idx, axis, -1.0, h),
                  (2, 2, 1)).energy
        ) / (2.0 * h)
        label = "normal" if axis == 2 else "in-plane"
        assert abs(grad[atom_idx, axis]) > 1e-3  # non-trivial component
        assert abs(grad[atom_idx, axis] - fd) <= 1e-6, (
            f"slab KRHF full-SCF gradient mismatch ({label}) atom "
            f"{atom_idx} axis {axis}: analytic={grad[atom_idx, axis]:.12f} "
            f"FD={fd:.12f} diff={abs(grad[atom_idx, axis] - fd):.3e}"
        )

    # Γ-reduction sanity: the same route at (1,1,1) (q = 0 cache only).
    rg = run(atoms, (1, 1, 1), grad=True)
    grad_g = np.asarray(rg.gradient)
    assert np.max(np.abs(grad_g.sum(axis=0))) < 1e-12
    fd_g = (
        run(_displaced_compact_atoms(atoms, 0, 2, +1.0, h), (1, 1, 1)).energy
        - run(_displaced_compact_atoms(atoms, 0, 2, -1.0, h), (1, 1, 1)).energy
    ) / (2.0 * h)
    assert abs(grad_g[0, 2] - fd_g) <= 1e-6, (
        f"Gamma slab gradient mismatch: analytic={grad_g[0, 2]:.12f} "
        f"FD={fd_g:.12f} diff={abs(grad_g[0, 2] - fd_g):.3e}"
    )


def test_slab_krks_full_scf_gradient_vs_fd():
    """§ 6 rung 5 KS arm: ``run_krks_periodic_gdf(compute_gradient=
    True)`` on the compact (2,2,1) fixture vs a central difference of
    the driver total. PBE exercises the a_x = 0 no-K branch + the GGA
    XC Pulay of the folded slab density on the periodic-Becke
    quadrature; PBE0 adds the a_x-scaled signed DF-K + the BvK
    probe-charge W-shift under XC.

    Gate 2e-4 per component — deliberately looser than the 1e-6 KRHF
    gate, and the reason is pinned mechanically here, not assumed: the
    XC Pulay kernel (``xc_lattice_gradient_contribution``, no
    grid-motion terms) matches FD of the SCF's own
    ``build_xc_periodic`` energy AT FIXED GRID to 9e-12 (asserted
    below at 1e-9), so the full-SCF residual is entirely the moving
    periodic-Becke quadrature. On this compact cell (4.6-bohr in-plane
    images inside the partition reach) that residual is the
    ``becke_image_radius_bohr`` truncation's geometry derivative:
    measured full-SCF |analytic - FD| at h = 2e-4 — PBE 1.2e-5
    (in-plane) / 8.0e-5 (normal), PBE0 8.6e-6 / 6.0e-5 — and the
    isolated XC term's residual falls 8.0e-5 -> 6.3e-6 -> 1.4e-6 as
    the image radius grows 10 -> 15 -> 20 bohr at fixed grid density
    (grid-density-independent: doubling radial/angular points moves it
    < 1e-7). The 3D KRKS gates sit at 1e-8 only because that fixture's
    images are outside the 10-bohr partition reach entirely. Same
    class as the Γ KS ladder's documented 5e-3 quadrature-limited
    gates; a genuine composition bug would not shrink with the image
    radius and IS caught by the fixed-grid identity.
    """
    from vibeqc._vibeqc_core import (
        Functional,
        GridOptions,
        build_xc_periodic,
        xc_lattice_gradient_contribution,
    )
    from vibeqc._vibeqc_core import monkhorst_pack as _mp_native
    from vibeqc.periodic_grid import build_periodic_becke_grid
    from vibeqc.periodic_k_density import (
        real_space_density_from_per_k_density,
    )
    from vibeqc.periodic_k_gdf import (
        _clone_slab_lattice_options,
        run_krks_periodic_gdf,
    )

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    lattice = np.diag([4.6, 4.6, 30.0])
    mesh = (2, 2, 1)
    h = 2e-4

    def make_opts(functional):
        opts = vq.PeriodicKSOptions()
        opts.functional = functional
        opts.lattice_opts.cutoff_bohr = 18.0
        opts.lattice_opts.nuclear_cutoff_bohr = 30.0
        opts.max_iter = 80
        opts.damping = 0.3
        opts.use_diis = False
        opts.dynamic_damping = False
        opts.conv_tol_energy = 1e-11
        opts.conv_tol_grad = 1e-9
        return opts

    def run(atom_list, functional, grad=False):
        sysp = vq.PeriodicSystem(2, lattice, atom_list)
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        r = run_krks_periodic_gdf(
            sysp,
            basis,
            mesh,
            make_opts(functional),
            functional=functional,
            rsgdf_ke_cutoff=30.0,
            compute_gradient=grad,
        )
        assert r.converged
        return sysp, basis, r

    for functional, components in (
        ("PBE", ((0, 0), (0, 2))),
        ("PBE0", ((0, 2),)),
    ):
        sysp0, basis0, r0 = run(atoms, functional, grad=True)
        grad = np.asarray(r0.gradient)
        assert grad.shape == (2, 3)
        # Newton at quadrature accuracy (the XC grid-motion term is
        # only translationally consistent, like the 3D KS gates).
        assert np.max(np.abs(grad.sum(axis=0))) < 1e-7
        for atom_idx, axis in components:
            fd = (
                run(_displaced_compact_atoms(atoms, atom_idx, axis, +1.0, h),
                    functional)[2].energy
                - run(_displaced_compact_atoms(atoms, atom_idx, axis, -1.0, h),
                      functional)[2].energy
            ) / (2.0 * h)
            label = "normal" if axis == 2 else "in-plane"
            assert abs(grad[atom_idx, axis] - fd) <= 2e-4, (
                f"slab KRKS {functional} full-SCF gradient mismatch "
                f"({label}) atom {atom_idx} axis {axis}: "
                f"analytic={grad[atom_idx, axis]:.12f} FD={fd:.12f} "
                f"diff={abs(grad[atom_idx, axis] - fd):.3e}"
            )

        if functional != "PBE":
            continue
        # ---- fixed-grid XC identity: the composed XC Pulay term is
        # EXACT for the quadrature the SCF evaluated (measured 5e-12 /
        # 9e-12) — this is the arm that catches a genuine XC
        # composition bug, independent of the grid-motion floor above.
        opts = make_opts(functional)
        lat_opts = _clone_slab_lattice_options(opts.lattice_opts)
        kmesh_bloch = _mp_native(sysp0, list(mesh), [0, 0, 0], False)
        densities = [np.asarray(D, dtype=complex) for D in r0.density]
        func = Functional(functional, 1)
        grid0 = build_periodic_becke_grid(
            sysp0,
            grid_options=getattr(opts, "grid", None) or GridOptions(),
            image_radius_bohr=float(
                getattr(opts, "becke_image_radius_bohr", 0.0)
            ),
        )

        def e_xc_fixed_grid(atom_list):
            s = vq.PeriodicSystem(2, lattice, atom_list)
            b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
            S_lat = vq.compute_overlap_lattice(b, s, lat_opts)
            D_real = real_space_density_from_per_k_density(
                densities, kmesh_bloch, list(S_lat.cells)
            )
            return float(
                build_xc_periodic(b, s, grid0, func, D_real, lat_opts).e_xc
            )

        S_lat0 = vq.compute_overlap_lattice(basis0, sysp0, lat_opts)
        D_real0 = real_space_density_from_per_k_density(
            densities, kmesh_bloch, list(S_lat0.cells)
        )
        g_xc = np.asarray(
            xc_lattice_gradient_contribution(
                basis0, sysp0, grid0, func, D_real0, lat_opts
            ),
            dtype=float,
        )
        h_xc = 1e-5
        for atom_idx, axis in ((0, 0), (0, 2)):
            fd_xc = (
                e_xc_fixed_grid(
                    _displaced_compact_atoms(atoms, atom_idx, axis, +1.0, h_xc)
                )
                - e_xc_fixed_grid(
                    _displaced_compact_atoms(atoms, atom_idx, axis, -1.0, h_xc)
                )
            ) / (2.0 * h_xc)
            assert abs(g_xc[atom_idx, axis] - fd_xc) < 1e-9, (
                f"fixed-grid slab XC Pulay mismatch atom {atom_idx} axis "
                f"{axis}: analytic={g_xc[atom_idx, axis]:.12f} "
                f"FD={fd_xc:.12f}"
            )


@pytest.mark.parametrize(
    ("functional", "pyscf_reference"),
    [
        ("PBE", -0.9964813071419485),
        ("PBE0", -0.9872777530690655),
    ],
)
def test_public_multik_slab_gdf_rks_matches_pyscf_and_is_a3_invariant(
    functional,
    pyscf_reference,
):
    """Finite-torus XC composes with signed J/K on the compact slab.

    External references are PySCF 2.13.1 KRKS-GDF with ``dimension=2``,
    ``low_dim_ft_type='inf_vacuum'``, ``exxdiv='ewald'``, STO-3G,
    def2-SVP-JKFIT, precision 1e-11, and the Gamma-centered (2,2,1) mesh.
    The PBE0 pin also exercises the BvK probe-charge exchange correction;
    the direct slab route deliberately uses a different real-space exchange
    convention and is therefore not the hybrid oracle.
    """
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    energies = []
    for normal_length in (30.0, 60.0):
        sysp = vq.PeriodicSystem(
            2, np.diag([4.6, 4.6, normal_length]), atoms
        )
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        opts = vq.PeriodicKSOptions()
        opts.functional = functional
        opts.lattice_opts.cutoff_bohr = 18.0
        opts.lattice_opts.nuclear_cutoff_bohr = 30.0
        opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
        opts.max_iter = 60
        opts.damping = 0.3
        opts.use_diis = False
        opts.dynamic_damping = False
        opts.conv_tol_energy = 1e-9
        opts.conv_tol_grad = 1e-8
        result = run_krks_periodic_gdf(
            sysp,
            basis,
            (2, 2, 1),
            opts,
            functional=functional,
            rsgdf_ke_cutoff=30.0,
        )
        assert result.converged
        assert result.backend == "native-multik-slab-truncated-gdf-rks"
        assert result.functional == functional
        assert result.n_iter == 18
        assert np.isfinite(result.e_xc)
        energies.append(result.energy)

    assert abs(energies[0] - pyscf_reference) < 3e-4
    assert abs(energies[0] - energies[1]) < 1e-7


def test_public_multik_slab_gdf_rks_rejects_range_separation():
    """The fitted K is full-range only, so HSE-like routes stay closed."""
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    sysp, basis = _h2_slab()
    opts = _ks_opts()
    opts.functional = "HSE06"
    with pytest.raises(NotImplementedError, match="range-separated"):
        run_krks_periodic_gdf(
            sysp,
            basis,
            (1, 1, 1),
            opts,
            functional="HSE06",
            rsgdf_ke_cutoff=30.0,
        )


def test_public_multik_slab_gdf_rks_default_accelerator_converges():
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    atoms = [vq.Atom(1, [0.4, 0.3, 0.55]), vq.Atom(1, [2.1, 1.8, -0.55])]
    sysp = vq.PeriodicSystem(2, np.diag([4.6, 4.6, 30.0]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "PBE"
    opts.max_iter = 60
    opts.lattice_opts.cutoff_bohr = 18.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    result = run_krks_periodic_gdf(
        sysp,
        basis,
        (2, 2, 1),
        opts,
        functional="PBE",
        rsgdf_ke_cutoff=30.0,
    )
    assert result.converged
    assert result.n_iter == 4
    assert abs(result.energy - (-0.9964813071419485)) < 3e-4


def test_public_slab_gdf_rejects_custom_mesh_and_smearing():
    from vibeqc.kpoints import KPoints
    from vibeqc.periodic_k_gdf import run_krks_periodic_gdf

    sysp, basis = _h2_slab()
    opts = _ks_opts()
    custom_mesh = KPoints.monkhorst_pack(sysp, (1, 1, 1), symmetry=False)
    with pytest.raises(NotImplementedError, match="tuple mesh"):
        run_krks_periodic_gdf(
            sysp,
            basis,
            custom_mesh,
            opts,
            functional="PBE",
            rsgdf_ke_cutoff=30.0,
        )

    opts.smearing_temperature = 1e-3
    with pytest.raises(NotImplementedError, match="smearing"):
        run_krks_periodic_gdf(
            sysp,
            basis,
            (1, 1, 1),
            opts,
            functional="PBE",
            rsgdf_ke_cutoff=30.0,
        )


# ---------------------------------------------------------------------------
# 4. The Sec. 7 gates: a3-invariance and k-mesh-independent E_nuclear
# ---------------------------------------------------------------------------


def test_slab_total_is_a3_invariant():
    """The synthesized a3 is bookkeeping; the SCF total must not depend on it.

    This is the property the old dim=3-with-vacuum convention could not offer:
    there the total drifts with the vacuum gap as M_z^2 / V.
    """
    atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
    energies = []
    for pad in (15.0, 40.0):
        sysp = vq.slab_2d(
            [18.0, 0.0, 0.0], [0.0, 18.0, 0.0], atoms, normal_padding_bohr=pad
        )
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        km = vq.monkhorst_pack(sysp, [1, 1, 1])
        energies.append(
            vq.run_rks_periodic_scf(sysp, basis, km, _ks_opts(), progress=False).energy
        )
    # |a3| differs by 50 bohr between the two builds.
    assert abs(energies[0] - energies[1]) < 1e-9, energies


def test_slab_e_nuclear_is_k_mesh_independent():
    """E_nuclear is a purely geometric 2D-Ewald lattice sum: k-independent.

    This is the direct regression for the graphene symptom (321.39 Ha at
    k=(1,1,1) vs 557.97 Ha at k=(8,8,1) under the old AUTO->GDF routing).
    """
    sysp, basis = _h2_slab()
    e_nuc = []
    for mesh in [(1, 1, 1), (2, 2, 1)]:
        km = vq.monkhorst_pack(sysp, list(mesh))
        r = vq.run_rks_periodic_scf(sysp, basis, km, _ks_opts(), progress=False)
        e_nuc.append(float(r.e_nuclear))
    assert abs(e_nuc[0] - e_nuc[1]) < 1e-10, e_nuc


# ---------------------------------------------------------------------------
# 4. End-to-end through the public runner
# ---------------------------------------------------------------------------


def test_run_periodic_job_auto_dim2_matches_direct_slab_driver(tmp_path):
    """`run_periodic_job(jk_method='auto')` on a slab == the SLAB_EWALD_2D driver."""
    sysp, basis = _h2_slab()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    direct = vq.run_rks_periodic_scf(sysp, basis, km, _ks_opts(), progress=False)
    routed = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="PBE",
        jk_method="auto",
        kpoints=(1, 1, 1),
        output=str(tmp_path / "slab"),
    )
    assert routed.converged
    assert abs(routed.energy - direct.energy) < 1e-9


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 2, 1)])
def test_run_periodic_job_auto_dim2_uks_gamma_and_multi_k(tmp_path, mesh):
    """AUTO routes open-shell KS slabs to SLAB_EWALD_2D at Gamma AND multi-k.

    The runner dispatches UKS to the slab drivers directly (there is no unified
    `run_uks_periodic_scf`), so both arms need coverage. On this closed-shell
    singlet H2 layer the UKS total must reproduce the RKS one.
    """
    sysp, basis = _h2_slab()
    stem = tmp_path / f"uks{mesh[0]}"
    routed = vq.run_periodic_job(
        sysp,
        basis,
        method="UKS",
        functional="PBE",
        jk_method="auto",
        kpoints=mesh,
        output=str(stem),
        output_qvf=False,
    )
    assert routed.converged
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    rks = vq.run_rks_periodic_scf(sysp, basis, km, _ks_opts(), progress=False)
    assert abs(routed.energy - rks.energy) < 1e-7
    sidecars = [
        stem.with_suffix(".molden"),
        stem.parent / f"{stem.name}.population.txt",
        stem.parent / f"{stem.name}.population.json",
    ]
    if mesh == (1, 1, 1):
        assert all(path.is_file() for path in sidecars)
    else:
        assert all(not path.exists() for path in sidecars)


# The two sidecars refuse for different reasons, so each parametrisation
# pins its own clause instead of sharing one pattern loose enough to cover
# both. Molden writes the Gamma block itself, so its refusal is a statement
# about the mesh. Population is a route-capability rule: BIPOLE and
# chi-CCM-B carry a genuine full-k analysis, and every other route falls
# back to the single-Gamma restriction (see D118 in
# docs/aiccm2026dev_b_decisions.md).
_MULTIK_SIDECAR_REFUSALS = {
    "write_molden_file": r"k-mesh must contain an exact Gamma point",
    "write_population_file": r"require an exact single-Gamma result",
}


@pytest.mark.parametrize("option", list(_MULTIK_SIDECAR_REFUSALS))
def test_run_periodic_job_multik_sidecar_request_fails_before_scf(tmp_path, option):
    """A guaranteed Gamma-only sidecar cannot be promised on a non-Gamma mesh."""
    sysp, basis = _h2_slab()
    kwargs = {
        "write_molden_file": False,
        "write_population_file": False,
        option: True,
    }
    with pytest.raises(
        NotImplementedError,
        match=_MULTIK_SIDECAR_REFUSALS[option],
    ):
        vq.run_periodic_job(
            sysp,
            basis,
            method="UKS",
            functional="PBE",
            jk_method="auto",
            kpoints=(2, 2, 1),
            output=str(tmp_path / option),
            output_qvf=False,
            **kwargs,
        )


def test_run_periodic_job_auto_dim2_uks_multik_qvf_artifacts(tmp_path):
    """An incomplete lattice density is refused before grid certification.

    The archive omits that uncertified density while retaining valid,
    independent total and projected DOS sections.
    """
    import json
    import warnings
    import zipfile

    from vibeqc.output.formats.qvf import validate_qvf

    sysp, basis = _h2_slab()
    stem = tmp_path / "uks221-qvf"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vq.run_periodic_job(
            sysp,
            basis,
            method="UKS",
            functional="PBE",
            jk_method="auto",
            kpoints=(2, 2, 1),
            output=stem,
            output_qvf=True,
            dos_kmesh=(1, 1, 1),
            density_spacing_bohr=6.0,
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
            record_hostname=False,
        )

    assert result.converged
    optional_failures = [
        str(item.message)
        for item in caught
        if "vibe-qc output [optional_artifact]" in str(item.message)
    ]
    assert optional_failures == []
    density_failures = [
        str(item.message)
        for item in caught
        if "vibe-qc output [compatibility_fallback]" in str(item.message)
        and "density_grid" in str(item.message)
    ]
    assert len(density_failures) == 1
    assert "density cell list does not cover the BvK torus" in density_failures[0]
    assert "no cell in residue class" in density_failures[0]

    qvf_path = stem.with_suffix(".qvf")
    report = validate_qvf(qvf_path)
    assert report["valid"], report["errors"]

    with zipfile.ZipFile(qvf_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    kinds = [section["kind"] for section in manifest["sections"]]
    assert "volume.density" not in kinds
    for expected in ("dos.total", "dos.projected"):
        assert kinds.count(expected) == 1


def test_run_periodic_job_explicit_gdf_on_slab_runs_closed_shell_rks(tmp_path):
    sysp, basis = _h2_slab()
    stem = tmp_path / "slab_gdf"
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="PBE",
        jk_method="gdf",
        rsgdf_ke_cutoff=30.0,
        output=str(stem),
        output_qvf=False,
        citations=True,
        progress=False,
        record_hostname=False,
    )
    assert result.converged
    assert result.backend == "native-multik-slab-truncated-gdf-rks"
    assert stem.with_suffix(".out").is_file()
    assert stem.with_suffix(".system").is_file()
    references = stem.with_suffix(".references").read_text(encoding="utf-8")
    assert "Spencer" in references
    assert "Parry" in references


def test_run_periodic_job_slab_gdf_optimize_relaxes_on_slab_objective(
    monkeypatch, tmp_path
):
    """§ 6 rung 5 runner wiring: ``optimize=True`` on a slab GDF job
    re-runs the SAME slab driver with ``compute_gradient=True`` (the
    GDF optimizer-objective capture now covers the slab dispatch) —
    the SCF-objective-identity pin, mirroring
    ``test_multik_gdf_optimize_dispatches_to_multik_gradient_driver``.
    The relaxation tolerance is loose: this pins the wiring, not a
    relaxation minimum (the FD gates above own correctness)."""
    import vibeqc.periodic_runner as periodic_runner
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf as real_krhf

    calls: list[dict] = []

    def counting_krhf(system, basis, kmesh, opts, **kwargs):
        calls.append(
            {
                "compute_gradient": bool(
                    kwargs.get("compute_gradient", False)
                ),
                "gdf_method": kwargs.get("gdf_method"),
                "kmesh": tuple(kmesh),
                "dim": int(system.dim),
            }
        )
        return real_krhf(system, basis, kmesh, opts, **kwargs)

    monkeypatch.setattr(
        periodic_runner, "run_krhf_periodic_gdf", counting_krhf
    )

    sysp, basis = _h2_slab()
    r = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        jk_method="gdf",
        rsgdf_ke_cutoff=30.0,
        optimize=True,
        optimize_max_iter=10,
        optimize_conv_tol_grad=1e-2,
        output=str(tmp_path / "slab-gdf-opt"),
        output_qvf=False,
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )
    assert r.converged

    # First call is the plain SCF; every optimizer evaluation re-runs
    # the identical slab driver with the analytic gradient on.
    assert calls[0]["compute_gradient"] is False
    opt_calls = calls[1:]
    assert opt_calls, "optimizer never re-ran the captured slab driver"
    assert all(c["compute_gradient"] for c in opt_calls)
    assert all(c["dim"] == 2 for c in opt_calls)
    assert all(c["kmesh"] == (1, 1, 1) for c in opt_calls)
    assert all(c["gdf_method"] == "rsgdf" for c in opt_calls)

    out_text = (tmp_path / "slab-gdf-opt.out").read_text()
    assert "force objective     = GDF analytic gradient (rsgdf)" in out_text


def test_run_periodic_job_slab_gdf_optimize_cell_stays_fail_closed(tmp_path):
    """No slab GDF analytic stress exists; variable-cell relaxation on
    the slab GDF route stays closed with a named reason."""
    sysp, basis = _h2_slab()
    with pytest.raises(NotImplementedError, match="analytic stress"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            optimize=True,
            optimize_cell=True,
            output=str(tmp_path / "slab_gdf_cell"),
        )


def test_run_periodic_job_explicit_bipole_on_slab_fails_closed(tmp_path):
    """Regression for BUG-PER-003: dim=2 BIPOLE must not return a bulk energy."""
    sysp, basis = _h2_slab()
    with pytest.raises(NotImplementedError, match="bulk.*dim=3"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="bipole",
            kpoints=(1, 1, 1),
            output=str(tmp_path / "slab_bipole"),
        )
