"""Tests for the multi-k periodic RHF/RKS GDF SCF loop.

These tests deliberately avoid importing any external QC package
(CLAUDE.md § 10). They pin the vibe-qc side of the multi-k GDF
surface:

  - Γ-only delegation routes from KRHF/KRKS to the gamma driver
    and matches it to machine precision.
  - Non-Γ meshes actually run the multi-k SCF loop and converge
    on a small vacuum-padded test problem (H₂ in a 12 Å cubic box).
  - All Bravais lattices route through the same code path (cubic /
    hexagonal / triclinic exercised).
  - Per-k Fock matrices come out Hermitian; per-k MOs and densities
    have the right shape.

External-program parity (PySCF KRHF / CRYSTAL14 KRHF via vq) lives
in the out-of-process runners under ``examples/regression/core/`` —
not here.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_k_gdf as kgdf
from vibeqc._vibeqc_core import Functional, compute_overlap_lattice
from vibeqc.guess_read import resolve_periodic_read_density_k_closed
from vibeqc.periodic_k_gdf import (
    PeriodicKRHFGDFResult,
    PeriodicKRKSGDFResult,
    _build_k_from_lpq_cache,
    _gamma_kmesh_info,
    _madelung_for_kmesh,
    run_krhf_periodic_gdf,
    run_krks_periodic_gdf,
)

ANGSTROM_TO_BOHR = 1.8897261339213


def test_krhf_gdf_rejects_rhf_options_with_xc_before_setup(monkeypatch):
    from vibeqc import periodic_fit_source

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible XC options reached numerical setup")

    monkeypatch.setattr(periodic_fit_source, "resolve_fit_source", forbidden)
    with pytest.raises(TypeError, match="PeriodicKSOptions"):
        run_krhf_periodic_gdf(
            SimpleNamespace(dim=3), None, (1, 1, 1), vq.PeriodicRHFOptions(),
            functional="pbe", progress=False,
        )


@pytest.mark.parametrize("option_kind,functional", [
    ("rhf", None), ("ks", "pbe"), ("ks", None),
    ("default", "pbe"), ("default", None),
])
def test_krhf_gdf_xc_option_guard_preserves_valid_requests(
    monkeypatch, option_kind, functional,
):
    from vibeqc import periodic_fit_source

    class SetupReached(Exception):
        pass

    def stop(*args, **kwargs):
        raise SetupReached

    monkeypatch.setattr(periodic_fit_source, "resolve_fit_source", stop)
    options = {"rhf": vq.PeriodicRHFOptions, "ks": vq.PeriodicKSOptions,
               "default": lambda: None}[option_kind]()
    if option_kind == "ks":
        options.functional = "pbe"
    with pytest.raises(SetupReached):
        run_krhf_periodic_gdf(
            SimpleNamespace(dim=3), None, (1, 1, 1), options,
            functional=functional, progress=False,
        )


def test_krhf_gdf_xc_option_guard_preserves_fitted_xc_admission():
    class FittedXCReached(Exception):
        pass

    def require_functional(name):
        assert name == "pbe"
        raise FittedXCReached

    with pytest.raises(FittedXCReached):
        run_krhf_periodic_gdf(
            None, None, options=vq.PeriodicRHFOptions(), functional="pbe",
            fitted_support=object(),
            _chi_xc=SimpleNamespace(require_functional=require_functional),
            progress=False,
        )


def test_multik_pyscf_debug_example_matches_current_driver_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-oracle wrapper must call the live GDF API without stale kwargs."""
    example = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "debug"
        / "gdf_multik_pyscf_parity.py"
    )
    namespace = runpy.run_path(str(example))
    expected = SimpleNamespace(energy=-1.0, converged=True, n_iter=2)
    captured: dict[str, object] = {}

    def strict_driver(
        system,
        basis,
        *,
        kmesh,
        options,
        aux_basis,
        use_compcell,
        compcell_eta,
        progress,
    ):
        captured.update(
            system=system,
            basis=basis,
            kmesh=kmesh,
            options=options,
            aux_basis=aux_basis,
            use_compcell=use_compcell,
            compcell_eta=compcell_eta,
            progress=progress,
        )
        return expected

    monkeypatch.setattr(vq, "run_krhf_periodic_gdf", strict_driver)
    system = object()
    basis = object()
    result, elapsed = namespace["vibeqc_compcell"](
        system, basis, (2, 1, 1)
    )

    assert result is expected
    assert elapsed >= 0.0
    assert captured["system"] is system
    assert captured["basis"] is basis
    assert captured["kmesh"] == (2, 1, 1)
    assert captured["aux_basis"] == "def2-svp-jk"
    assert captured["use_compcell"] is True
    assert captured["compcell_eta"] == 0.25
    assert captured["progress"] is False


# =====================================================================
#                            Test builders
# =====================================================================


def _h2_cubic_box(box_bohr: float = 12.0):
    """H₂ molecule centred in a cubic Bravais cell of side ``box_bohr``.

    Vacuum-padded enough for the closed-shell minimum sto-3g basis;
    the cell is just a Bravais carrier for the multi-k tests.
    """
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * float(box_bohr),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _h2_hexagonal_box(a_bohr: float = 8.0, c_bohr: float = 12.0):
    """H₂ in a hexagonal (a=b, γ=120°) cell — Bravais coverage test."""
    a = float(a_bohr)
    c = float(c_bohr)
    lat = np.array(
        [
            [a, -0.5 * a, 0.0],
            [0.0, 0.5 * a * np.sqrt(3.0), 0.0],
            [0.0, 0.0, c],
        ]
    ).T
    system = vq.PeriodicSystem(
        3,
        lat,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _h2_triclinic_box():
    """H₂ in a triclinic Bravais cell — worst-case Bravais coverage."""
    lat = np.array(
        [
            [10.0, 0.0, 0.0],
            [2.0, 11.0, 0.0],
            [1.5, 2.5, 12.0],
        ]
    ).T
    system = vq.PeriodicSystem(
        3,
        lat,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _rhf_opts(max_iter: int = 30) -> vq.PeriodicRHFOptions:
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.max_iter = int(max_iter)
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 14.0
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    return opts


def test_skew_unequal_mesh_madelung_matches_probe_charge_backend():
    """Both exxdiv implementations use A.diag(mesh), with A column-wise."""
    from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
    from vibeqc.madelung import madelung_constant_for_cell

    lattice = np.array(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    mesh = (2, 3, 1)
    expected = 0.138352993811598
    xi_gdf = _madelung_for_kmesh(system, mesh)
    xi_direct = probe_charge_madelung_supercell(system, mesh)
    assert xi_gdf == pytest.approx(expected, abs=1e-12)
    assert xi_direct == pytest.approx(expected, abs=1e-12)
    assert xi_gdf == pytest.approx(xi_direct, abs=1e-12)

    supercell = vq.PeriodicSystem(
        3,
        lattice @ np.diag(mesh),
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    for eta in (0.1, 0.3, 0.7):
        assert madelung_constant_for_cell(
            supercell,
            eta=eta,
        ) == pytest.approx(expected, abs=1e-12)
    for alpha in (0.1, 0.3, 0.7):
        assert probe_charge_madelung_supercell(
            system,
            mesh,
            alpha=alpha,
        ) == pytest.approx(expected, abs=1e-12)


def test_gdf_exchange_contraction_uses_supplied_k_weights():
    """Symmetry-reduced k meshes must weight exchange like Coulomb."""

    lpq_cache = {
        (0, 0): np.ones((1, 1, 1), dtype=complex),
        (0, 1): np.ones((1, 1, 1), dtype=complex),
        (1, 0): np.ones((1, 1, 1), dtype=complex),
        (1, 1): np.ones((1, 1, 1), dtype=complex),
    }
    densities = [
        np.array([[2.0]], dtype=complex),
        np.array([[10.0]], dtype=complex),
    ]
    weights = np.array([0.25, 0.75])

    K_k = _build_k_from_lpq_cache(lpq_cache, densities, weights, nbasis=1)

    expected_weighted_average = 0.25 * 2.0 + 0.75 * 10.0
    assert K_k[0][0, 0].real == pytest.approx(expected_weighted_average)
    assert K_k[1][0, 0].real == pytest.approx(expected_weighted_average)
    assert K_k[0][0, 0].real != pytest.approx(0.5 * (2.0 + 10.0))


# =====================================================================
#      Factored (occupied-index) exchange == the density contraction
# =====================================================================
#
# Since 2026-08-02 `_build_k_from_lpq_cache` contracts the cderi against
# signed Gram factors of the density instead of the density itself,
# cutting each (k_i, k_j) pair from 2.naux.nbf^3 to 2.naux.nbf^2.n_occ.
# It is the same operator reassociated, so the historical dense
# contraction is retained as `_k_from_densities_dense` and pinned here.
# Without these, a bug in the factorisation would only show up as a
# quiet energy drift in the driver-level pins.


def _random_cderi_cache(rng, n_k, naux, nbf):
    return {
        (i, j): rng.standard_normal((naux, nbf, nbf))
        + 1j * rng.standard_normal((naux, nbf, nbf))
        for i in range(n_k)
        for j in range(n_k)
    }


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_exchange_auxiliary_panels_match_independent_contraction(noncontiguous):
    from vibeqc.periodic_k_gdf import (
        _k_from_densities_dense, _k_from_signed_factors, _signed_gram_factors,
    )

    rng = np.random.default_rng(142)
    nbf, naux = 4, 31
    factors = rng.normal(size=(naux, nbf, nbf)) + 1j * rng.normal(size=(naux, nbf, nbf))
    if noncontiguous:
        factors = factors.transpose(0, 2, 1)
    matrix = rng.normal(size=(nbf, nbf)) + 1j * rng.normal(size=(nbf, nbf))
    density = matrix + matrix.conj().T  # exercise both signed Gram blocks
    expected = sum(f @ density @ f.conj().T for f in factors)
    cache = {(0, 0): factors}
    # Fits at most a handful of auxiliary functions, forcing many panels.
    cap = 4096
    factored = _k_from_signed_factors(
        cache, [_signed_gram_factors(density)], np.ones(1), [0],
        nbasis=nbf, workspace_byte_cap=cap,
    )
    dense = _k_from_densities_dense(
        cache, [density], np.ones(1), [0], nbasis=nbf, workspace_byte_cap=cap,
    )
    np.testing.assert_allclose(factored[0], expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(dense[0], expected, rtol=0, atol=2e-12)
    from vibeqc.pbc_gdf import _build_k_from_lpq
    # The Gamma driver represents a real density/potential, including when
    # its auxiliary coordinates are complex (for example MDF factors).
    gamma_density = density.real
    gamma_expected = sum(f @ gamma_density @ f.conj().T for f in factors).real
    gamma = _build_k_from_lpq(factors, gamma_density, workspace_byte_cap=cap)
    np.testing.assert_allclose(gamma, gamma_expected, rtol=0, atol=2e-12)
    with pytest.raises(MemoryError, match="one auxiliary panel"):
        _k_from_densities_dense(
            cache, [density], np.ones(1), [0], nbasis=nbf, workspace_byte_cap=1,
        )


def test_factored_exchange_matches_the_dense_contraction():
    """An aufbau-shaped (low-rank PSD) density: the production case."""
    from vibeqc.periodic_k_gdf import _k_from_densities_dense

    rng = np.random.default_rng(20260802)
    n_k, naux, nbf, n_occ = 3, 10, 8, 3
    cache = _random_cderi_cache(rng, n_k, naux, nbf)
    W = [
        rng.standard_normal((nbf, n_occ))
        + 1j * rng.standard_normal((nbf, n_occ))
        for _ in range(n_k)
    ]
    dens = [w @ w.conj().T for w in W]
    weights = np.full(n_k, 1.0 / n_k)

    ref = _k_from_densities_dense(
        cache, dens, weights, range(n_k), nbasis=nbf
    )
    got = _build_k_from_lpq_cache(cache, dens, weights, nbasis=nbf)
    scale = max(float(np.max(np.abs(K))) for K in ref)
    for a, b in zip(ref, got):
        assert np.max(np.abs(a - b)) < 1e-12 * scale


def test_factored_exchange_is_exact_for_an_indefinite_density():
    """The signed split, not a projection onto the positive part.

    A caller may legitimately pass something that is Hermitian but not
    positive semi-definite (a density *difference*, a rough restart
    density). Dropping the negative eigenvalues would be a silent
    approximation, so they are carried as a second factor with s = -1.
    """
    from vibeqc.periodic_k_gdf import _k_from_densities_dense

    rng = np.random.default_rng(4)
    n_k, naux, nbf = 2, 6, 5
    cache = _random_cderi_cache(rng, n_k, naux, nbf)
    dens = []
    for _ in range(n_k):
        A = rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal(
            (nbf, nbf)
        )
        H = A + A.conj().T  # Hermitian, indefinite
        dens.append(H)
    assert min(
        float(np.min(np.linalg.eigvalsh(D))) for D in dens
    ) < 0.0, "fixture must actually be indefinite"
    weights = np.array([0.4, 0.6])

    ref = _k_from_densities_dense(
        cache, dens, weights, range(n_k), nbasis=nbf
    )
    got = _build_k_from_lpq_cache(cache, dens, weights, nbasis=nbf)
    scale = max(float(np.max(np.abs(K))) for K in ref)
    for a, b in zip(ref, got):
        assert np.max(np.abs(a - b)) < 1e-12 * scale


def test_non_hermitian_density_falls_back_to_the_dense_contraction():
    """``eigh`` reads one triangle, so a non-Hermitian input must not
    reach it: the wrapper falls back verbatim rather than silently
    symmetrising."""
    from vibeqc.periodic_k_gdf import (
        _k_from_densities_dense,
        _signed_gram_factors,
    )

    rng = np.random.default_rng(11)
    n_k, naux, nbf = 2, 5, 4
    cache = _random_cderi_cache(rng, n_k, naux, nbf)
    dens = [
        rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))
        for _ in range(n_k)
    ]
    assert _signed_gram_factors(dens[0]) is None
    weights = np.full(n_k, 0.5)

    ref = _k_from_densities_dense(
        cache, dens, weights, range(n_k), nbasis=nbf
    )
    got = _build_k_from_lpq_cache(cache, dens, weights, nbasis=nbf)
    for a, b in zip(ref, got):
        assert np.array_equal(a, b), "fallback must be the dense path itself"


def test_gram_factors_collapse_the_rank_to_the_occupied_count():
    """The saving is real: an aufbau density factors to n_occ columns,
    not nbf. If this regressed, the contraction would silently go back
    to the nbf^3 cost while still being correct."""
    from vibeqc.periodic_k_gdf import _signed_gram_factors

    rng = np.random.default_rng(7)
    nbf, n_occ = 20, 4
    C = rng.standard_normal((nbf, n_occ)) + 1j * rng.standard_normal(
        (nbf, n_occ)
    )
    D = 2.0 * (C @ C.conj().T)
    facs = _signed_gram_factors(D)
    assert len(facs) == 1, "an aufbau density has no negative block"
    sign, W = facs[0]
    assert sign == 1.0
    assert W.shape == (nbf, n_occ)
    assert np.max(np.abs(W @ W.conj().T - D)) < 1e-10 * float(
        np.max(np.abs(D))
    )


def test_gram_factors_of_a_zero_density_are_empty():
    from vibeqc.periodic_k_gdf import _signed_gram_factors

    assert _signed_gram_factors(np.zeros((4, 4), dtype=complex)) == []


# =====================================================================
#                     Γ delegation: tuple and BlochKMesh
# =====================================================================


@pytest.mark.parametrize("gdf_method", ["compcell", "mdf"])
@pytest.mark.parametrize("controls", [
    {"fock_mixing": 0.2}, {"level_shift": 0.2},
    {"smearing_temperature": 0.01},
])
def test_gamma_knobs_on_dense_core_compcell_mdf_never_reach_the_legacy_driver(
    monkeypatch, gdf_method, controls,
):
    """A Gamma knob on compcell/MDF runs the k-mesh engine at Nk=1 now, not
    the legacy driver (#400). On this dense-core cell both reach their Bloch
    pair builder (stubbed here; the image sum of #172 costs minutes on MgO,
    and MDF's dense-core gate was lifted with it)."""
    h = 7.958 / 2.0
    system = vq.PeriodicSystem(
        3, np.array([[0, h, h], [h, 0, h], [h, h, 0]]),
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [h, h, h])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    for name, value in controls.items():
        setattr(opts, name, value)

    def forbidden(*args, **kwargs):
        pytest.fail("dense-core legacy SCF must not start")

    class ReachedPairBuilder(Exception):
        pass

    def pair_builder(*args, **kwargs):
        raise ReachedPairBuilder

    monkeypatch.setattr(kgdf, "run_rhf_periodic_gamma_gdf", forbidden)
    monkeypatch.setattr(kgdf, "build_lpq_bloch_compcell", pair_builder)
    monkeypatch.setattr(kgdf, "build_lpq_bloch_mdf", pair_builder)
    with pytest.raises(ReachedPairBuilder):
        run_krhf_periodic_gdf(system, basis, (1, 1, 1), opts,
                             gdf_method=gdf_method, progress=False)


def test_gamma_kmesh_info_recognises_tuple_111():
    system, _ = _h2_cubic_box()
    assert _gamma_kmesh_info(system, (1, 1, 1)) is not None


def test_gamma_kmesh_info_recognises_blochkmesh_111():
    system, _ = _h2_cubic_box()
    bm = vq.monkhorst_pack(system, [1, 1, 1])
    assert _gamma_kmesh_info(system, bm) is not None


def test_gamma_kmesh_info_rejects_222():
    system, _ = _h2_cubic_box()
    assert _gamma_kmesh_info(system, (2, 2, 2)) is None


@pytest.mark.parametrize("consumer", ["counts", "gamma", "points", "rohf"])
@pytest.mark.parametrize("mesh", [
    (1.9, 1, 1), (2.0, 1, 1), (True, 1, 1), (np.bool_(True), 1, 1),
    ("2", 1, 1), "211", b"211", (2+0j, 1, 1), ((2,), 1, 1),
    np.array([2., 1., 1.]), np.array([True, True, True]),
])
def test_integer_mesh_counts_gdf_consumers_reject_before_native(
    consumer, mesh, monkeypatch,
):
    from vibeqc.periodic_rohf_gdf import _mesh_tuple_for_system as rohf_counts

    def forbidden(*args, **kwargs):
        pytest.fail("invalid GDF mesh reached native grid construction")

    monkeypatch.setattr(kgdf, "_mp_native", forbidden)
    consumers = {"counts": kgdf._mesh_tuple_for_system,
                 "gamma": kgdf._gamma_kmesh_info,
                 "points": kgdf._kmesh_to_kpoints_weights,
                 "rohf": rohf_counts}
    with pytest.raises(ValueError, match="must contain integers"):
        consumers[consumer](SimpleNamespace(dim=3), mesh)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("full", [False, True])
def test_integer_mesh_counts_gdf_preserve_padding_and_pinning(dim, full):
    system = vq.PeriodicSystem(dim, np.eye(3)*8.0, [vq.Atom(2, [0., 0., 0.])])
    mesh = np.array([2]*dim + ([7]*(3-dim) if full else []), dtype=np.uint32)
    expected = (2,)*dim + (1,)*(3-dim)
    assert kgdf._mesh_tuple_for_system(system, mesh) == expected
    points, weights = kgdf._kmesh_to_kpoints_weights(system, mesh)
    assert len(points) == len(weights) == 2**dim
    assert weights == pytest.approx(np.full(2**dim, 1/2**dim))


@pytest.mark.parametrize("kind", ["python", "native", "unstructured", "invalid"])
def test_integer_mesh_counts_gdf_preserve_typed_metadata(kind):
    from dataclasses import replace
    system = vq.PeriodicSystem(3, np.eye(3)*8.0, [vq.Atom(2, [0., 0., 0.])])
    mesh = vq.KPoints.gamma_centred(system, (2, 1, 1))
    if kind == "native":
        mesh = mesh.to_bloch_kmesh()
    elif kind == "unstructured":
        mesh = replace(mesh, mesh=None)
        with pytest.raises(ValueError, match="structured k-mesh"):
            kgdf._mesh_tuple_for_system(system, mesh)
        return
    elif kind == "invalid":
        mesh = replace(mesh, mesh=(2.9, 1, 1))
        with pytest.raises(ValueError, match="must contain integers"):
            kgdf._mesh_tuple_for_system(system, mesh)
        return
    assert kgdf._mesh_tuple_for_system(system, mesh) == (2, 1, 1)


@pytest.mark.parametrize("consumer", ["slab", "ibz_expand", "ibz_native"])
@pytest.mark.parametrize("mesh", [
    (1.9, 1, 1), (2.0, 1, 1), (True, 1, 1), (np.bool_(True), 1, 1),
    ("2", 1, 1), (2+0j, 1, 1), ((2,), 1, 1),
    np.array([2., 1., 1.]), np.array([True, True, True]),
])
def test_integer_mesh_counts_gdf_slab_and_ibz_refuse_coercion(consumer, mesh, monkeypatch):
    from vibeqc import _vibeqc_core as core

    def forbidden(*args, **kwargs):
        pytest.fail("invalid count reached IBZ native grid construction")

    monkeypatch.setattr(kgdf, "_mp_native", forbidden)
    monkeypatch.setattr(core, "monkhorst_pack", forbidden)
    system = SimpleNamespace(dim=3, symmetry=SimpleNamespace(operations=[]))
    metadata = SimpleNamespace(mesh=mesh, ir_mapping=[0], kpoints=[[0., 0., 0.]])
    with pytest.raises(ValueError, match="must contain integers"):
        if consumer == "slab":
            _call_slab_count_preflight(mesh)
        elif consumer == "ibz_expand":
            kgdf._expand_ibz_kmesh_to_full_bz(system, metadata)
        else:
            kgdf._resolve_ibz_native_state(
                system, None, metadata, None, None, exchange_producer="sr")


def _call_slab_count_preflight(mesh):
    import inspect
    # Supply every required keyword, but stop at the existing auxiliary
    # culling guard immediately after count validation. No SCF setup runs.
    kwargs = {name: None for name, arg in inspect.signature(
        kgdf._run_closed_shell_slab_gdf).parameters.items()
        if arg.kind == inspect.Parameter.KEYWORD_ONLY
        and arg.default is inspect.Parameter.empty}
    kwargs["aux_drop_eta"] = 1.0
    return kgdf._run_closed_shell_slab_gdf(None, None, mesh, None, **kwargs)


@pytest.mark.parametrize("mesh", [(2, 3), (2, 3, 1), np.array([2, 3], dtype=np.uint32)])
def test_integer_mesh_counts_gdf_slab_accepts_integer_active_or_full(mesh):
    with pytest.raises(NotImplementedError, match="auxiliary primitive culling"):
        _call_slab_count_preflight(mesh)


@pytest.mark.parametrize("mesh", [(0, 1), (-1, 1), (1,), (1, 1, 1, 1), (2, 2, 2)])
def test_integer_mesh_counts_gdf_slab_preserves_range_and_inactive_guards(mesh):
    with pytest.raises(ValueError, match="slab GDF requires"):
        _call_slab_count_preflight(mesh)


def test_krhf_gdf_gamma_via_tuple_matches_pbc_gdf_rhf():
    """``run_krhf_periodic_gdf(kmesh=(1,1,1))`` HF on the default dim=3
    SR/LR fit is the Nk=1 mesh of the multi-k engine and evaluates the
    same Hamiltonian as the exact-Gamma ``run_pbc_gdf_rhf``
    (exxdiv='ewald'). So the Nk=1 limit carries the same exxdiv='ewald'
    convention as the (2,1,1)+ multi-k path, NOT the legacy
    molecular-limit gamma driver.

    From 2026-06-15 until the SR/LR promotion (f6815d9) this Γ mesh
    delegated to ``run_pbc_gdf_rhf``. Since then the general engine owns
    the Γ limit of the SR/LR route, as the ``run_krhf_periodic_gdf``
    docstring says, and the two engines must still agree. Reconciles the
    default-vs-explicit and (1,1,1)-vs-(2,1,1) ~5.8 mHa Γ-GDF discrepancy.
    """
    system, basis = _h2_cubic_box()
    opts = _rhf_opts()
    r_via_k = run_krhf_periodic_gdf(system, basis, (1, 1, 1), opts)
    r_pbc = vq.run_pbc_gdf_rhf(
        system, basis, opts, gdf_method="rsgdf", exxdiv="ewald", progress=False
    )
    assert r_via_k.converged
    assert r_pbc.converged
    # The k-mesh engine ran at Nk=1; it did not hand the mesh on.
    assert str(r_via_k.backend) == "native-multi-k-gdf-gdf-rhf"
    # Two engines, one Hamiltonian.
    assert abs(r_via_k.energy - r_pbc.energy) < 1e-10, (
        f"Γ-routed KRHF energy {r_via_k.energy} differs from "
        f"run_pbc_gdf_rhf {r_pbc.energy} by {abs(r_via_k.energy - r_pbc.energy)}"
    )
    # QC468 retires public admission of the compact-cell legacy bridge.
    # Keep the supported two-engine parity above; the former >4 mHa
    # legacy separation is historical evidence, not a supported reference.
    with pytest.raises(NotImplementedError, match="molecular-limit exchange"):
        vq.run_rhf_periodic_gamma_gdf(system, basis, opts)


def test_krhf_gdf_gamma_via_blochkmesh_matches_tuple():
    """A Γ-only ``BlochKMesh`` takes the route a ``(1, 1, 1)`` tuple takes.

    This used to pin a ``gamma`` backend tag, i.e. delegation to the
    exact-Gamma driver. On the default dim=3 SR/LR fit neither spelling
    delegates since f6815d9; what must hold is that the two spellings of
    one mesh select one engine and one Hamiltonian.
    """
    system, basis = _h2_cubic_box()
    opts = _rhf_opts()
    bm = vq.monkhorst_pack(system, [1, 1, 1])
    r = run_krhf_periodic_gdf(system, basis, bm, opts)
    r_tuple = run_krhf_periodic_gdf(system, basis, (1, 1, 1), opts)
    assert r.converged and r_tuple.converged
    assert len(r.density) == 1
    assert str(r.backend) == str(r_tuple.backend) == "native-multi-k-gdf-gdf-rhf"
    assert r.energy == pytest.approx(r_tuple.energy, abs=1e-12)


@pytest.mark.parametrize(
    ("kmesh", "functional", "n_pairs"),
    [((1, 1, 1), None, 1), ((2, 1, 1), None, 4), ((2, 1, 1), "pbe", 2)],
    ids=["gamma-nk1-hf", "multik-hf", "multik-pure-dft"],
)
def test_result_counts_the_fit_pairs_the_scf_contracted(
    monkeypatch, kmesh, functional, n_pairs
):
    """``gdf_cderi_pairs`` is read off the fit the driver built.

    A consumer that needs to know whether the SCF built a density fit, or
    how many ``(k_i, k_j)`` tensors it held, reads this field; the CCM
    neutral-Bloch control once re-derived it from a copy of the
    ``use_compcell`` promotion rules, and the copy reported no fit for pure
    DFT on a 3-D k-mesh although the bulk SR/LR promotion builds its ``n_k``
    diagonal pairs. Exact exchange needs all ``n_k**2`` pairs; a Γ mesh on
    the SR/LR source runs the k-mesh loop at Nk=1 and fits its one pair.
    The spy counts what the SR/LR source (the 3-D default here) returned.
    """
    system, basis = _h2_cubic_box()
    built = []
    real_builder = kgdf._build_scf_range_separated_lpq_cache

    def _spy(*args, **kwargs):
        cache = real_builder(*args, **kwargs)
        built.append(len(cache))
        return cache

    monkeypatch.setattr(kgdf, "_build_scf_range_separated_lpq_cache", _spy)
    if functional is None:
        r = run_krhf_periodic_gdf(system, basis, kmesh, _rhf_opts())
    else:
        opts = vq.PeriodicKSOptions()
        opts.use_diis = True
        opts.max_iter = 30
        opts.conv_tol_energy = 1e-8
        opts.lattice_opts.cutoff_bohr = 14.0
        opts.lattice_opts.nuclear_cutoff_bohr = 16.0
        r = run_krks_periodic_gdf(
            system, basis, kmesh, opts, functional=functional
        )

    assert r.converged
    assert built == [n_pairs]
    assert r.gdf_cderi_pairs == n_pairs


def test_gamma_delegate_result_counts_its_single_fit():
    """The Γ delegate fits one Γ tensor, and the adapter says so.

    Off the SR/LR source a Γ mesh on a 3-D cell is handed to the exact-Γ
    compensated-cell driver, and the k-mesh result is adapted from the one
    that driver returns. (A chain's Γ mesh used to reach the legacy Γ driver,
    which now refuses dim=1; tests/test_periodic_lowd_guards.py pins that.)
    """
    system, basis = _h2_cubic_box()
    kwargs = {"gdf_method": "compcell"}
    r = run_krhf_periodic_gdf(system, basis, (1, 1, 1), _rhf_opts(), **kwargs)

    assert r.converged
    assert r.backend == "native-gamma-gdf-via-k-gdf"
    assert r.gdf_cderi_pairs == 1


class _StopAtFitBuild(Exception):
    """Raised by the fit-build spy once it has read the driver's flag."""


def _spy_gradient_envelope_and_fit_build(monkeypatch):
    """Record the ``use_compcell`` the envelope is given and the fit build uses.

    The envelope spy records ``use_compcell_effective`` and the real
    envelope's verdict without raising, so the driver runs on to the fit
    build. There the aux-basis spy reads the driver's own ``use_compcell``
    and stops the run before any cderi or SCF work.
    """
    seen = {"envelope": [], "verdict": [], "fit": []}
    real_envelope = kgdf._reject_unsupported_multik_gradient

    def envelope(entry, **kwargs):
        seen["envelope"].append(kwargs["use_compcell_effective"])
        try:
            real_envelope(entry, **kwargs)
            seen["verdict"].append(None)
        except NotImplementedError as exc:
            seen["verdict"].append(str(exc))

    def aux_basis(*args, **kwargs):
        caller = sys._getframe(1)
        assert caller.f_code.co_name == "run_krhf_periodic_gdf"
        seen["fit"].append(bool(caller.f_locals["use_compcell"]))
        raise _StopAtFitBuild

    monkeypatch.setattr(kgdf, "_reject_unsupported_multik_gradient", envelope)
    monkeypatch.setattr(kgdf, "make_aux_basis_set", aux_basis)
    return seen


def _envelope_probe_opts(functional):
    if functional is None:
        return _rhf_opts()
    opts = vq.PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 14.0
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    return opts


@pytest.mark.parametrize("use_compcell", [False, True], ids=["default", "explicit"])
@pytest.mark.parametrize("gdf_method", ["rsgdf", "compcell", "mdf"])
@pytest.mark.parametrize("functional", [None, "pbe", "pbe0"], ids=["hf", "pbe", "pbe0"])
def test_gradient_envelope_sees_the_fit_the_driver_builds(
    monkeypatch, functional, gdf_method, use_compcell
):
    """The gradient envelope is told whether the fit is built by the driver.

    The envelope runs before the SCF and refuses a gradient when the driver
    will not build the cached-Lpq fit. It used to re-derive that as
    ``bulk_sr or use_compcell or alpha > 0``, a partial copy of the promotion
    rules: on 29f3089 it said False where the driver builds the fit for pure
    DFT with ``fit_source='classic'`` or on dim=1. Earlier refusals caught
    both, so no gradient was refused for the wrong reason, but any promotion
    rule added to the driver alone would split the two. This grid is every
    3-D multi-k input that reaches both the envelope and the fit build.
    """
    system, basis = _h2_cubic_box()
    seen = _spy_gradient_envelope_and_fit_build(monkeypatch)
    with pytest.raises(_StopAtFitBuild):
        run_krhf_periodic_gdf(
            system,
            basis,
            (2, 1, 1),
            _envelope_probe_opts(functional),
            functional=functional,
            gdf_method=gdf_method,
            use_compcell=use_compcell,
            compute_gradient=True,
            progress=False,
        )

    assert len(seen["fit"]) == 1
    assert seen["envelope"] == seen["fit"]


def test_gradient_envelope_sees_the_dim1_pure_ks_promotion(monkeypatch):
    """A dim=1 pure-KS gradient is refused as a wire, not for a missing fit.

    The driver resolves this run to the cached-Lpq fit like every other
    closed-shell k-mesh run (it has no unfitted Hartree since #415; the
    dim < 3 pure-KS rule that used to promote it named the EWALD_3D J). The
    envelope's old copy of the rules was handed
    ``use_compcell_effective=False`` here; only the dim guard, which runs
    first, kept the wrong reason from surfacing. The driver then refuses
    dim=1 in the Coulomb gauge, before the fit build.
    """
    system = vq.PeriodicSystem(
        1,
        np.diag([6.0, 15.0, 15.0]),
        [vq.Atom(1, [0.0, 7.5, 7.5]), vq.Atom(1, [1.4, 7.5, 7.5])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    seen = _spy_gradient_envelope_and_fit_build(monkeypatch)
    with pytest.raises(NotImplementedError, match="no consistent Coulomb gauge"):
        run_krhf_periodic_gdf(
            system,
            basis,
            (3, 1, 1),
            _envelope_probe_opts("pbe"),
            functional="pbe",
            compute_gradient=True,
            progress=False,
        )

    assert seen["envelope"] == [True]
    assert "3D periodic systems only" in seen["verdict"][0]
    assert seen["fit"] == []


# =====================================================================
#               Multi-k SCF — H₂ in cubic vacuum-padded cell
# =====================================================================


# NOTE: RHF multi-k auto-routes to the exxdiv-corrected compcell GDF path
# (per-(k_i,k_j) Lpq, O(N_k²)) — see the routing block in periodic_k_gdf.py
# and the 2026-06-04 maintainer decision. These SCF-loop mechanics tests
# (convergence / Hermiticity / shapes / Bravais coverage) therefore use a
# small (2,1,1) mesh to keep the Lpq build cheap; a larger mesh on the
# correct path is exercised by the @slow test below + the parity guard.
def test_krhf_gdf_multik_h2_cubic_converges():
    """Smallest non-Γ smoke test: H₂ / sto-3g / [2, 1, 1] on cubic."""
    system, basis = _h2_cubic_box()
    opts = _rhf_opts()
    r = run_krhf_periodic_gdf(system, basis, (2, 1, 1), opts)
    assert r.converged, (
        f"H₂ KRHF [2,1,1] did not converge: "
        f"n_iter={r.n_iter}, grad={r.scf_trace[-1].grad_norm if r.scf_trace else None}"
    )
    assert len(r.density) == 2
    assert r.kpoints_cart.shape == (2, 3)
    assert np.isclose(float(np.sum(r.kpoint_weights)), 1.0, atol=1e-9)
    # Each per-k Fock must be Hermitian.
    for i, Fk in enumerate(r.fock):
        Fk_arr = np.asarray(Fk)
        assert np.allclose(Fk_arr, Fk_arr.conj().T, atol=1e-8), (
            f"F(k={i}) is not Hermitian"
        )


def test_krhf_gdf_multik_h2_density_matrices_are_hermitian():
    system, basis = _h2_cubic_box()
    opts = _rhf_opts()
    r = run_krhf_periodic_gdf(system, basis, (2, 1, 1), opts)
    for i, Dk in enumerate(r.density):
        D = np.asarray(Dk)
        assert np.allclose(D, D.conj().T, atol=1e-8), f"D(k={i}) is not Hermitian"


def test_krhf_gdf_multik_dft_plus_u_shifts_energy():
    """Closed-shell multi-k RHF/GDF supports the same Dudarev +U Fock hook as
    the GPW/BIPOLE routes."""
    system, basis = _h2_cubic_box()
    sites = [vq.HubbardSite(atom_index=0, l=0, U_ev=2.7)]

    baseline = run_krhf_periodic_gdf(system, basis, (2, 1, 1), _rhf_opts())
    with_u = run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        _rhf_opts(),
        dft_plus_u_sites=sites,
    )

    assert baseline.converged
    assert with_u.converged
    assert baseline.e_dft_plus_u == 0.0
    assert with_u.e_dft_plus_u > 0.0
    assert with_u.energy != pytest.approx(baseline.energy, abs=1e-8)
    assert with_u.energy == pytest.approx(
        with_u.e_electronic + with_u.e_nuclear + with_u.e_dft_plus_u,
        abs=1e-9,
    )


def test_krhf_gdf_multik_h2_per_k_mo_shapes():
    system, basis = _h2_cubic_box()
    opts = _rhf_opts()
    r = run_krhf_periodic_gdf(system, basis, (2, 1, 1), opts)
    nbf = basis.nbasis
    for i in range(len(r.density)):
        assert np.asarray(r.mo_energies[i]).ndim == 1
        assert np.asarray(r.mo_coeffs[i]).shape[0] == nbf
        # n_kept <= nbf (canonical orth may drop directions)
        assert np.asarray(r.mo_coeffs[i]).shape[1] <= nbf


@pytest.mark.slow
def test_krhf_gdf_222_h2_cubic_converges_compcell():
    """[2, 2, 2] mesh = 8 k-points. RHF auto-routes to the compcell GDF path
    (per-(k_i,k_j) Lpq, O(N_k²)), so this larger-mesh convergence check is
    marked slow. Exercises the per-k DIIS + Lpq cache on the correct
    (exxdiv-corrected) path.
    """
    system, basis = _h2_cubic_box()
    opts = _rhf_opts(max_iter=40)
    r = run_krhf_periodic_gdf(system, basis, (2, 2, 2), opts)
    assert r.converged
    assert len(r.density) == 8


@pytest.mark.slow
def test_krhf_gdf_multik_scf_descends_into_convergence():
    """Multi-k SCF must not converge above a visited iterate (E↔F canary).

    Every RHF iterate energy here is a determinant energy evaluated with
    the same E↔F pairing the SCF diagonalises, so a converged energy
    ABOVE the minimum visited iterate means the reported energy and the
    Fock operator disagree — an E↔F inconsistency, never a convergence-
    aid problem (CLAUDE.md § 7).

    Symptom regression for the 2026-07-09 finding (qc-input-library
    01706, the original investigation): on tight cells with
    non-negligible inter-cell AO-pair overlap the SCF energy rose into
    convergence — +3.8 mHa above the minimum visited on exactly this
    c-diamond/sto-3g (2,1,1) fixture — because the Γ pair-FT mirror
    corrupted the (k ≠ Γ, k = Γ) exchange fits (fixed in f8c213e8).
    Vacuum-padded fixtures (the H₂ boxes above) never showed the
    symptom; the tight diamond cell is load-bearing.
    """
    a_half = 0.5 * 3.567 * ANGSTROM_TO_BOHR  # diamond a_cube = 3.567 Å
    lat = np.array(
        [
            [0.0, a_half, a_half],
            [a_half, 0.0, a_half],
            [a_half, a_half, 0.0],
        ]
    )
    system = vq.PeriodicSystem(
        3,
        lat,
        [vq.Atom(6, [0.0, 0.0, 0.0]), vq.Atom(6, [0.5 * a_half] * 3)],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 120
    opts.conv_tol_energy = 1e-7
    r = run_krhf_periodic_gdf(system, basis, (2, 1, 1), opts)
    assert r.converged
    energies = [it.energy for it in r.scf_trace]
    # The converged energy is the minimum visited (was +3.8e-3 pre-fix).
    assert r.energy - min(energies) <= 1e-9, (
        f"SCF converged {r.energy - min(energies):+.3e} Ha above the "
        "minimum visited iterate — E↔F inconsistency"
    )
    # The tail descends monotonically (tolerance far below the µHa-to-
    # mHa symptom scale, far above BLAS-order noise).
    for e_prev, e_next in zip(energies[-4:], energies[-3:]):
        assert e_next <= e_prev + 5e-9, (
            f"rising SCF tail: {e_prev:.10f} -> {e_next:.10f}"
        )
    # Pin the post-f8c213e8, post-3e5bbd8a, post-D88 fixed point. D88's
    # column-vector BvK repair changes the (2,1,1) Madelung seam because this
    # primitive FCC matrix does not commute with diag(mesh). The resulting
    # +0.228526309751 Ha shift is analytically accounted for by six occupied
    # bands times the old-minus-new positive Madelung constant.
    assert r.energy == pytest.approx(-74.4119904741, abs=2e-6)


def test_resolve_multik_read_density_from_complex_coefficients():
    """Per-k READ reconstructs Hermitian density blocks from complex Bloch MOs."""
    coeffs = [
        np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex),
        np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                  [1.0j / np.sqrt(2.0), -1.0j / np.sqrt(2.0)]], dtype=complex),
    ]
    occs = [np.array([2.0, 0.0]), np.array([1.5, 0.5])]
    src = SimpleNamespace(mo_coeffs=coeffs, occupations=occs)

    density_k = resolve_periodic_read_density_k_closed(
        read_from=src,
        expected_n_k=2,
        n_basis=2,
    )

    assert len(density_k) == 2
    for D in density_k:
        np.testing.assert_allclose(D, D.conj().T, atol=1e-14)
    np.testing.assert_allclose(density_k[0], [[2.0, 0.0], [0.0, 0.0]])
    expected = (coeffs[1] * occs[1][None, :]) @ coeffs[1].conj().T
    np.testing.assert_allclose(density_k[1], expected)


def test_resolve_multik_read_density_from_qvf_bloch_payload(tmp_path):
    """QVF all-k Bloch payloads reconstruct the same D(k) blocks as memory."""
    from vibeqc.output.formats.qvf import qvf_bloch_wf_data, write_qvf
    from vibeqc.output.plan import OutputPlan

    system, basis = _h2_cubic_box()
    coeffs = [
        np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex),
        np.array([[1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
                  [1.0j / np.sqrt(2.0), -1.0j / np.sqrt(2.0)]], dtype=complex),
    ]
    occs = [np.array([2.0, 0.0]), np.array([1.5, 0.5])]
    src = SimpleNamespace(
        mo_coeffs=coeffs,
        occupations=occs,
        mo_energies=[np.array([-0.5, 0.1]), np.array([-0.4, 0.2])],
    )
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "prior",
        method="rhf",
        basis="sto-3g",
        functional=None,
        job_kind="periodic_scf",
    )
    qvf = write_qvf(
        tmp_path / "prior",
        plan,
        system=system,
        result=SimpleNamespace(converged=True, energy=-1.0, n_iter=1),
        method="rhf",
        basis="sto-3g",
        bloch_wf_data=qvf_bloch_wf_data(
            src,
            basis,
            system.unit_cell_molecule(),
            k_points=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        ),
    )

    density_k = resolve_periodic_read_density_k_closed(
        read_path=str(qvf),
        expected_n_k=2,
        n_basis=basis.nbasis,
    )

    assert len(density_k) == 2
    np.testing.assert_allclose(density_k[0], [[2.0, 0.0], [0.0, 0.0]])
    expected = (coeffs[1] * occs[1][None, :]) @ coeffs[1].conj().T
    np.testing.assert_allclose(density_k[1], expected)


def test_krhf_gdf_multik_read_restarts_from_inmemory_result():
    """A converged multi-k GDF result can seed a second run with per-k D(k)."""
    system, basis = _h2_cubic_box()
    base = run_krhf_periodic_gdf(system, basis, (2, 1, 1), _rhf_opts())
    assert base.converged

    density_k = resolve_periodic_read_density_k_closed(
        read_from=base,
        expected_n_k=2,
        n_basis=basis.nbasis,
    )
    restarted = run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        _rhf_opts(max_iter=4),
        initial_density_k=density_k,
    )

    assert restarted.converged
    assert restarted.n_iter <= 3
    assert restarted.energy == pytest.approx(base.energy, abs=1e-9)


def test_run_periodic_job_gdf_multik_read_from_inmemory_result(tmp_path):
    """The public READ surface accepts in-memory multi-k GDF results."""
    system, basis = _h2_cubic_box()
    common = dict(
        system=system,
        basis=basis,
        method="RHF",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        aux_basis="def2-svp-jk",
        max_iter=30,
        conv_tol_energy=1e-8,
        write_molden_file=False,
        citations=False,
        progress=False,
    )

    base = vq.run_periodic_job(
        output=str(tmp_path / "base"),
        initial_guess="HCORE",
        **common,
    )
    assert base.converged

    restarted = vq.run_periodic_job(
        output=str(tmp_path / "restart"),
        initial_guess="read",
        read_from=base,
        max_iter=4,
        **{k: v for k, v in common.items() if k != "max_iter"},
    )

    assert restarted.converged
    assert restarted.n_iter <= 3
    assert restarted.energy == pytest.approx(base.energy, abs=1e-9)


def test_run_periodic_job_gdf_multik_read_from_qvf_result(tmp_path):
    """The public READ surface accepts current QVF all-k restart archives."""
    system, basis = _h2_cubic_box()
    common = dict(
        system=system,
        basis=basis,
        method="RHF",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        aux_basis="def2-svp-jk",
        max_iter=30,
        conv_tol_energy=1e-8,
        write_molden_file=False,
        citations=False,
        progress=False,
        dos_kmesh=(1, 1, 1),
    )

    base = vq.run_periodic_job(
        output=str(tmp_path / "base"),
        output_qvf=True,
        initial_guess="HCORE",
        **common,
    )
    assert base.converged
    qvf = tmp_path / "base.qvf"
    assert qvf.is_file()

    restarted = vq.run_periodic_job(
        output=str(tmp_path / "restart"),
        output_qvf=False,
        initial_guess="read",
        read_from=qvf,
        max_iter=4,
        **{k: v for k, v in common.items() if k != "max_iter"},
    )

    assert restarted.converged
    assert restarted.n_iter <= 3
    assert restarted.energy == pytest.approx(base.energy, abs=1e-9)


# =====================================================================
#                          Bravais coverage
# =====================================================================


def test_krhf_gdf_multik_hexagonal_converges():
    """Hexagonal (a=b, γ=120°) cell — generic-Bravais coverage."""
    system, basis = _h2_hexagonal_box()
    opts = _rhf_opts()
    r = run_krhf_periodic_gdf(system, basis, (2, 1, 1), opts)
    assert r.converged


def test_krhf_gdf_multik_triclinic_converges():
    """Triclinic cell — no special crystal symmetry."""
    system, basis = _h2_triclinic_box()
    opts = _rhf_opts()
    r = run_krhf_periodic_gdf(system, basis, (2, 1, 1), opts)
    assert r.converged


# =====================================================================
#                              KRKS (DFT)
# =====================================================================


@pytest.mark.parametrize("kmesh", [(1, 1, 1), (2, 1, 1)], ids=["gamma", "k211"])
@pytest.mark.parametrize("functional", ["pbe", "b3lyp"])
def test_krks_gdf_returns_the_krks_result_class(functional, kmesh):
    """``run_krks_periodic_gdf`` returns the class it is annotated with.

    Only the exact-Gamma adapter used to build a ``PeriodicKRKSGDFResult``;
    the k-mesh loop returned ``PeriodicKRHFGDFResult`` for KS runs too.
    Since f6815d9 a dim=3 Γ mesh on the SR/LR fit runs that loop at Nk=1
    rather than delegating, so the Γ KS result lost its class. Pin the
    class, and the engine that produced it, on Γ and on a true k-mesh.
    """
    system, basis = _h2_cubic_box()
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.use_diis = True
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 14.0
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    r = run_krks_periodic_gdf(
        system,
        basis,
        kmesh,
        opts,
        functional=functional,
        progress=False,
    )
    assert isinstance(r, PeriodicKRKSGDFResult)
    assert str(r.backend) == "native-multi-k-gdf-gdf-rks"
    assert r.converged
    assert r.functional and r.functional.lower() == functional.lower()


def test_krhf_gdf_hf_result_is_not_the_krks_class():
    """The KS class switch must not leak onto plain HF runs."""
    system, basis = _h2_cubic_box()
    r = run_krhf_periodic_gdf(system, basis, (2, 1, 1), _rhf_opts())
    assert type(r) is PeriodicKRHFGDFResult


def test_krks_gdf_gamma_default_diis_aliases_have_identical_results():
    """Default-DIIS spellings cannot change the Gamma Hamiltonian route."""
    system, basis = _h2_cubic_box()
    results = []
    for alias in (None, "", "none", "diis"):
        opts = vq.PeriodicKSOptions()
        opts.functional = "pbe0"
        opts.use_diis = True
        opts.max_iter = 30
        opts.conv_tol_energy = 1e-8
        opts.lattice_opts.cutoff_bohr = 14.0
        opts.lattice_opts.nuclear_cutoff_bohr = 16.0
        results.append(
            run_krks_periodic_gdf(
                system,
                basis,
                (1, 1, 1),
                opts,
                functional="pbe0",
                density_mixer=alias,
                rsgdf_ke_cutoff=60.0,
                progress=False,
            )
        )

    reference = results[0]
    assert reference.converged
    for result in results[1:]:
        assert type(result) is type(reference)
        assert result.converged
        assert result.backend == reference.backend
        assert result.energy == pytest.approx(reference.energy, abs=1e-12)
        assert np.allclose(result.density[0], reference.density[0], atol=1e-12)


def test_multik_rks_xc_uses_real_space_density_blocks(monkeypatch):
    """The multi-k RKS XC build must not collapse D(k) to one Γ block."""

    system, basis = _h2_cubic_box()
    opts = _rhf_opts(max_iter=2)
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    kpoints_cart, _weights = kgdf._kmesh_to_kpoints_weights(system, kmesh)
    cells = kgdf._direct_cells(system, opts.lattice_opts.cutoff_bohr)
    captured = {}

    def fake_build_xc_periodic(
        _basis,
        _system,
        _grid,
        _func,
        density_set,
        lat_opts,
        density_domain,
    ):
        captured["density_domain"] = density_domain
        nonhome = []
        for idx, cell in enumerate(density_set.cells):
            cell_index = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
            if cell_index != (0, 0, 0):
                nonhome.append(float(np.linalg.norm(density_set.blocks[idx])))
        captured["max_nonhome_density_norm"] = max(nonhome, default=0.0)

        vxc_set = compute_overlap_lattice(basis, system, lat_opts)
        eye = np.eye(basis.nbasis)
        for idx, cell in enumerate(vxc_set.cells):
            cell_index = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
            scale = 1.0 if cell_index == (0, 0, 0) else 0.25
            vxc_set.set_block(idx, scale * eye)
        return SimpleNamespace(e_xc=0.125, V_xc=vxc_set)

    monkeypatch.setattr(kgdf, "build_xc_periodic", fake_build_xc_periodic)

    nbf = basis.nbasis
    density_k = [
        np.eye(nbf, dtype=complex),
        np.diag(np.linspace(0.25, 1.25, nbf)).astype(complex),
    ]
    e_xc, vxc_k = kgdf._build_xc_k_from_density(
        basis=basis,
        system=system,
        grid=None,
        func=Functional("pbe", 1),
        density_k=density_k,
        kmesh_bloch=kmesh,
        cells=cells,
        kpoints_cart=kpoints_cart,
        lat_opts=opts.lattice_opts,
    )

    assert e_xc == pytest.approx(0.125)
    assert captured["density_domain"] == vq.PeriodicXCDensityDomain.AUTO
    assert captured["max_nonhome_density_norm"] > 1.0e-12
    assert len(vxc_k) == len(kpoints_cart)
    assert all(v.shape == (basis.nbasis, basis.nbasis) for v in vxc_k)
    assert all(np.allclose(v, v.conj().T) for v in vxc_k)


# =====================================================================
#   Multi-k HF/hybrid auto-routes to the exxdiv-corrected compcell path
#
# The exxdiv Madelung exchange-divergence K-shift lives only on the
# use_compcell=True branch; the use_compcell=False Ewald-3D-K builder omits
# it, so HF/hybrid multi-k on that path is wrong by the Madelung shift
# (catastrophic +17 Ha on sparse meshes). The driver routes HF/hybrid to
# use_compcell=True + exxdiv='ewald' (the PySCF-µHa path). [Maintainer
# decision 2026-06-04.] Pure DFT no longer keeps the Ewald-3D J either: on
# the default dim=3 rsgdf route the SR/LR fit promotes it too (#284).
# =====================================================================
def test_multik_exxdiv_validation_on_use_compcell_false():
    """On the use_compcell=False path, HF/hybrid is auto-routed to
    the exxdiv-corrected compcell path (commit 500711b6). The inline
    exxdiv kwarg was removed; the driver validates internally. A pure
    DFT functional on use_compcell=False is promoted to the fitted
    SR/LR Hartree of the default dim=3 rsgdf route. Both converge."""
    system, basis = _h2_cubic_box()
    opts = _rhf_opts(max_iter=30)

    # HF on use_compcell=False → auto-routed to use_compcell=True + exxdiv='ewald'
    r = run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        opts,
        aux_basis="def2-svp-jk",
        use_compcell=False,
        progress=False,
    )
    assert r.converged

    # Pure DFT on use_compcell=False → promoted to the fitted SR/LR J.
    r_dft = run_krks_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        _rhf_opts(max_iter=30),
        functional="pbe",
        aux_basis="def2-svp-jk",
        progress=False,
    )
    assert r_dft.converged


#: The EWALD_3D J/K entry points the k-GDF driver's former
#: ``use_compcell=False`` branch called: the dense AO-pair-FT J cache, the
#: bounded J stream from D(k), and the real-space Fock builder. The branch
#: was unreachable and was removed (#415); the builders stay in
#: ``vibeqc.periodic_fock_multi_k`` for the Ewald-3D drivers.
_MULTIK_EWALD_3D_J_ENTRY_POINTS = (
    "make_ewald_3d_lattice_j_cache",
    "build_periodic_j_ewald3d_k_from_k_density",
    "build_periodic_fock_ewald3d_k",
)


def test_multik_gdf_driver_has_no_ewald_3d_j():
    """The k-GDF driver imports no EWALD_3D J builder (#415).

    Every closed-shell k-mesh run is promoted to the cached fit: the SR/LR
    source on dim=3 rsgdf, the exxdiv rule for HF/hybrids, and the
    explicit-gdf_method rule for pure DFT on mdf/compcell. That left the
    ``use_compcell=False`` EWALD_3D J/K branch unreachable, so it was
    deleted. Importing one of these names here again is the first step of
    reopening it.
    """
    present = [n for n in _MULTIK_EWALD_3D_J_ENTRY_POINTS if hasattr(kgdf, n)]
    assert not present, present


@pytest.mark.parametrize("functional", [None, "pbe"])
def test_multik_gdf_hartree_is_the_fitted_one(monkeypatch, functional):
    """Every closed-shell multi-k GDF run contracts J from the cached fit.

    HF/hybrids promote ``use_compcell=False`` for the exxdiv shift, and
    pure DFT on the default dim=3 rsgdf route for the SR/LR fit (#284).
    Either way the SCF builds the SR/LR fit exactly once and iterates on
    it. This replaces three tests that pinned the pure-DFT Ewald-J cache,
    its bounded stream and the grid-backend dispatch inside this driver:
    the SR/LR promotion had made that branch unreachable (the cache and
    stream tests were red on main, the grid test passed without reaching
    the code it named), and #415 removed it. The builders keep their own
    coverage in tests/test_ewald_j_cache.py and in the Ewald-3D drivers.
    """
    fit_calls: list[int] = []
    real_fit = kgdf._build_scf_range_separated_lpq_cache

    def counting_fit(*args, **kwargs):
        fit_calls.append(1)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(
        kgdf, "_build_scf_range_separated_lpq_cache", counting_fit
    )
    system, basis = _h2_cubic_box()
    kwargs = dict(use_compcell=False, progress=False)
    if functional is None:
        result = run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), _rhf_opts(max_iter=2), **kwargs
        )
    else:
        result = run_krks_periodic_gdf(
            system, basis, (2, 1, 1), _rhf_opts(max_iter=2),
            functional=functional, **kwargs,
        )
    assert result.n_iter > 0
    assert fit_calls == [1], "the SCF did not build the SR/LR fit once"


@pytest.mark.slow
def test_multik_hf_default_path_routes_to_correct_energy():
    """Multi-k HF with the DEFAULT use_compcell=False (the run_periodic_job
    path) auto-routes to the exxdiv-corrected compcell GDF path and lands on
    the PySCF KRHF.density_fit()/exxdiv='ewald' energy — NOT the
    divergence-uncorrected Ewald-3D-K value (which is +17 Ha for this mesh).
    Regression guard for the 2026-06-04 routing fix."""
    system, basis = _h2_cubic_box()
    opts = _rhf_opts(max_iter=30)
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    # No use_compcell / exxdiv passed → driver defaults (False / 'none').
    r = run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        opts,
        aux_basis="def2-svp-jk",
    )
    assert r.converged
    # Published target: E_HF = -1.12013988 Ha for H₂/STO-3G/12-bohr at
    # kmesh (2,1,1) (Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147,
    # 164119 (2017), doi:10.1063/1.4998644, Table II row 3) — the PySCF
    # KRHF.density_fit() exxdiv='ewald' value. The 2026-06-09 merge-drop
    # (b4a6faba) regressed this pin to -2.80467 Ha (a double-counted
    # Madelung shift on a non-routed path); restored 2026-06-10.
    assert r.energy == pytest.approx(-1.12013988, abs=5e-3)
    # Definitely not the divergence-uncorrected +17 Ha garbage.
    assert r.energy < 0.0


def test_ibz_reduced_kmesh_expands_to_full_bz():
    """A symmetry-reduced (IBZ) k-mesh is expanded to its full BZ, not
    computed on the irreducible wedge and not rejected.

    HF exchange needs the k_j sum over the FULL BZ: weighting the
    irreducible representatives is not equivalent to summing their orbits
    (the exchange integrand needs symmetry-unfolded L/D per orbit member,
    and weighting instead measured +1.389 Ha on MgO primitive FCC /
    STO-3G (2,2,2) vs its 2-point reduction -- 2026-07-17
    production-k-sampling audit). The driver therefore rebuilds the full
    Monkhorst-Pack mesh from the reduction's own metadata before any
    integral is built. That makes IBZ input correct; it does NOT make it
    cheaper -- every full-mesh point is still built and diagonalised.

    Checked at mesh normalisation so no SCF runs.
    """
    h = 3.979
    lat = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [h, h, h])]
    )
    vq.attach_symmetry(system, symprec=1e-6)
    kp_red = vq.KPoints.monkhorst_pack(system, (2, 2, 2), symmetry=True)
    weights = np.asarray(kp_red.weights, dtype=float).reshape(-1)
    assert weights.size < 8  # genuinely reduced
    assert not np.allclose(weights, 1.0 / weights.size)  # non-uniform

    kpts, w = kgdf._kmesh_to_kpoints_weights(system, kp_red)
    assert kpts.shape[0] == 8, kpts.shape
    # The expanded mesh is a plain full-BZ quadrature again.
    np.testing.assert_allclose(w, np.full(8, 1.0 / 8.0), atol=1e-12)


def _lih_fcc():
    a = 7.72
    lat = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [0.5 * a, 0, 0])]
    )
    vq.attach_symmetry(system, symprec=1e-6)
    return system


def test_ibz_reduced_kmesh_reproduces_the_explicit_full_mesh():
    """The expansion must land on exactly the mesh the user would have
    written by hand -- otherwise it is a different calculation wearing the
    same name. Γ-centred reduction against the Γ-centred tuple mesh."""
    system = _lih_fcc()
    kp_red = vq.KPoints.monkhorst_pack(
        system, (2, 2, 2), shift=(0, 0, 0), symmetry=True
    )
    assert np.asarray(kp_red.weights).reshape(-1).size < 8

    from_ibz, w_ibz = kgdf._kmesh_to_kpoints_weights(system, kp_red)
    from_tuple, w_tuple = kgdf._kmesh_to_kpoints_weights(system, (2, 2, 2))
    np.testing.assert_allclose(from_ibz, from_tuple, atol=1e-12)
    np.testing.assert_allclose(w_ibz, w_tuple, atol=1e-12)


def test_ibz_expansion_rebuilds_the_shifted_parent_not_gamma():
    """A reduction of the classical half-step-shifted MP mesh must expand
    to the SHIFTED parent, not the Γ-centred one.

    The native ``BlochKMesh`` calls the shift flags ``is_shift``; the
    Python ``KPoints`` dataclass calls them ``shift``. The first expansion
    cut read only ``is_shift``, so a shifted ``KPoints`` reduction fell
    back to shift (0,0,0) and silently rebuilt the Γ-centred mesh — a
    different BZ sampling (~44 mHa on MgO (2,2,2); see the mesh-convention
    caveat in KPoints.monkhorst_pack). The original expansion verification
    compared exactly such a shifted LiH reduction against the Γ-centred
    tuple mesh, so the bug masked itself.
    """
    system = _lih_fcc()
    kp_red = vq.KPoints.monkhorst_pack(system, (2, 2, 2), symmetry=True)
    assert kp_red.shift == (1, 1, 1)  # classical MP default for even mesh
    assert np.asarray(kp_red.weights).reshape(-1).size < 8
    kp_full = vq.KPoints.monkhorst_pack(system, (2, 2, 2), symmetry=False)
    assert kp_full.shift == (1, 1, 1)

    from_ibz, w_ibz = kgdf._kmesh_to_kpoints_weights(system, kp_red)
    from_full, w_full = kgdf._kmesh_to_kpoints_weights(system, kp_full)
    np.testing.assert_allclose(from_ibz, from_full, atol=1e-12)
    np.testing.assert_allclose(w_ibz, w_full, atol=1e-12)

    # And it is genuinely a different sampling from the Γ-centred tuple.
    from_tuple, _ = kgdf._kmesh_to_kpoints_weights(system, (2, 2, 2))
    assert not np.allclose(from_ibz, from_tuple, atol=1e-6)


def test_symmetry_requested_but_unreduced_mesh_is_accepted():
    """use_symmetry=True on a mesh whose points all sit in distinct orbits
    (identity ir_mapping) reduces nothing: the "wedge" already IS the full
    quadrature. The first expansion cut refused it (full_n == n points
    looked like unexpandable metadata); it must simply run."""
    system, _ = _h2_cubic_box()
    vq.attach_symmetry(system)
    km = vq.monkhorst_pack(system, [2, 1, 1], use_symmetry=True)
    ir = np.asarray(km.ir_mapping, dtype=np.int64).reshape(-1)
    assert ir.size == 2 and np.array_equal(ir, np.arange(2))  # unreduced

    assert kgdf._expand_ibz_kmesh_to_full_bz(system, km) is None
    kpts, w = kgdf._kmesh_to_kpoints_weights(system, km)
    assert kpts.shape[0] == 2
    np.testing.assert_allclose(w, np.full(2, 0.5), atol=1e-12)


def test_unexpandable_ibz_kmesh_still_fails_closed():
    """A reduction whose declared parent mesh cannot account for it must
    be refused, never expanded against fabricated metadata.

    This is the ``to_bloch_kmesh`` metadata-loss shape -- an explicit
    KPoints list reports the C++ default mesh (1, 1, 1), the same gap
    that produced the 78 mHa exxdiv bug. Expanding on it would silently
    run a different Brillouin zone than the caller asked for.
    """
    import dataclasses

    a = 7.72
    lat = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [0.5 * a, 0, 0])]
    )
    vq.attach_symmetry(system, symprec=1e-6)
    kp_red = vq.KPoints.monkhorst_pack(system, (2, 2, 2), symmetry=True)
    orphaned = dataclasses.replace(kp_red, mesh=(1, 1, 1))

    with pytest.raises(NotImplementedError, match="parent mesh is unknown"):
        kgdf._kmesh_to_kpoints_weights(system, orphaned)


@pytest.mark.slow
def test_ibz_reduced_kmesh_dft_matches_full_mesh():
    """KRKS (hybrid DFT) on an IBZ mesh must run and reproduce the full
    mesh -- the DFT/XC twin of test_ibz_reduced_kmesh_expands_to_full_bz.

    The first expansion cut (375b6363d) expanded only the k-point arrays
    inside _kmesh_to_kpoints_weights while the driver's ``kmesh_bloch``
    stayed the caller's wedge, so every KRKS/KUKS IBZ job died at the
    first XC build: real_space_density_from_kpoints_fractional got 8
    per-k densities against the 6-point wedge ("size mismatch across k
    inputs" on H2/STO-3G (2,2,2)). Its KRHF verification never noticed
    because the pure-HF loop folds no real-space density. The drivers now
    adopt the expanded mesh for everything derived from the kmesh, so
    this pins the k-list consistency through an actual XC SCF.
    """
    system, basis = _h2_cubic_box()
    vq.attach_symmetry(system)
    k_ibz = vq.monkhorst_pack(system, [2, 2, 2], use_symmetry=True)
    n_wedge = np.asarray(k_ibz.kpoints).reshape(-1, 3).shape[0]
    assert n_wedge < 8  # genuinely a wedge, or this test pins nothing
    k_full = vq.monkhorst_pack(system, [2, 2, 2], use_symmetry=False)

    r_ibz = run_krks_periodic_gdf(
        system, basis, k_ibz, functional="pbe0", progress=False
    )
    r_full = run_krks_periodic_gdf(
        system, basis, k_full, functional="pbe0", progress=False
    )
    assert r_ibz.converged and r_full.converged
    # The result must carry the full expanded mesh, not the wedge.
    assert r_ibz.kpoints_cart.shape[0] == 8
    assert r_ibz.energy == pytest.approx(r_full.energy, abs=1e-10)


class _FitBuildSentinel(Exception):
    pass


def _sentinel_fit_builders(monkeypatch, *names):
    """Replace the named fit builders; return the names in call order."""
    reached = []

    def _boom(name):
        def _raise(*a, **k):
            reached.append(name)
            raise _FitBuildSentinel(name)

        return _raise

    for name in names:
        monkeypatch.setattr(kgdf, name, _boom(name))
    return reached


def test_multik_dense_core_parity_hold_warns_on_the_classic_fit(monkeypatch):
    """Dense-core cells on the classic multi-k fit must WARN about the
    absolute-energy parity hold (P01 class) instead of silently returning
    the un-tailed value.

    Production-k-sampling audit (2026-07-17): MgO/STO-3G multi-k at the
    untailed default of the time is -0.503 Ha ((1,1,2)) / -0.505 Ha
    ((2,2,2)) vs live PySCF on identical meshes, while
    rsgdf_tail_ke_cutoff = 1.1*10*zeta_max closes (1,1,2) to +0.16 mHa.
    Re-measured 2026-09-29 on fit_source='classic' at (1,1,2):
    -271.6134118 Ha vs PySCF 2.14 GDF -271.1102721 (-503.1 mHa). Since
    f6815d9 the classic fit is reached only through an explicit
    fit_source='classic'; the default SR/LR fit is not held (next test).
    Sentinels replace the fit builders so no SCF runs; the warning must
    fire before the build.
    """
    # The hook path calls the per-pair builder; the classic route goes
    # through the shared-q batched cache build.
    reached = _sentinel_fit_builders(
        monkeypatch, "build_lpq_bloch_native_fft", "_build_rsgdf_lpq_cache_shared_q"
    )
    system, basis = _mgo_dense_core_system()

    with pytest.warns(UserWarning, match="parity-held"):
        with pytest.raises(_FitBuildSentinel):
            run_krhf_periodic_gdf(
                system, basis, kmesh=(1, 1, 2), fit_source="classic",
                progress=False,
            )
    assert reached == ["_build_rsgdf_lpq_cache_shared_q"]

    # A parity-sized tail lifts the hold: no warning (sentinel still fires).
    reached.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(_FitBuildSentinel):
            run_krhf_periodic_gdf(
                system,
                basis,
                kmesh=(1, 1, 2),
                fit_source="classic",
                rsgdf_tail_ke_cutoff=3300.0,
                progress=False,
            )
    assert reached == ["_build_rsgdf_lpq_cache_shared_q"]


def test_multik_dense_core_default_sr_fit_is_not_parity_held(monkeypatch):
    """The default dim=3 SR/LR fit carries no parity hold on the P01 cell.

    The hold marks the classic fit's missing high-|G| tail. The SR/LR fit
    has no tail to miss (``rsgdf_tail_ke_cutoff`` is obsolete there), and
    on this MgO/STO-3G cell it lands on PySCF 2.14 GDF: +2.9 uHa at Gamma
    and +3.2 uHa at (1,1,2) (2026-09-29), not the -0.5 Ha of the
    P01 class. f6815d9 skipped the hold on this route; the test that pinned
    the warning on the default route predated it.
    """
    reached = _sentinel_fit_builders(
        monkeypatch, "_build_scf_range_separated_lpq_cache"
    )
    system, basis = _mgo_dense_core_system()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_FitBuildSentinel):
            run_krhf_periodic_gdf(
                system, basis, kmesh=(1, 1, 2), progress=False
            )
    assert reached == ["_build_scf_range_separated_lpq_cache"]
    assert not [w for w in caught if "parity-held" in str(w.message)]


def _mgo_dense_core_system():
    """MgO primitive FCC / STO-3G -- the IID 146 incident cell (P01 class)."""
    h = 3.979
    lat = np.array([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]])
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [h, h, h])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


class _FitSentinel(Exception):
    """A fit builder was reached; ``args[0]`` names it, ``args[1]`` its kwargs."""


def _sentinel_multik_fit(monkeypatch):
    """Stop the k-GDF driver at whichever fit it builds.

    The SR/LR source, the classic shared-q source and the classic per-pair
    builder raise :class:`_FitSentinel` naming themselves, so no SCF runs.
    """

    def fit(name):
        def boom(*args, **kwargs):
            raise _FitSentinel(name, kwargs)

        return boom

    for name in (
        "_build_scf_range_separated_lpq_cache",
        "_build_rsgdf_lpq_cache_shared_q",
        "build_lpq_bloch_native_fft",
    ):
        monkeypatch.setattr(kgdf, name, fit(name))


def test_multik_pure_ks_consumes_rsgdf_tail_ke_cutoff(monkeypatch):
    """IID 146: multi-k PURE-DFT KRKS on dim=3 accepted
    ``rsgdf_tail_ke_cutoff``, echoed it in the .out, and silently ignored
    it -- the EWALD_3D real-space J it then ran builds no cderi, so the
    knob had nothing to act on. Measured: MgO/STO-3G (2,2,2) KRKS-LDA
    tailed (3291.6) vs untailed BYTE-IDENTICAL at -271.531535367637 (15
    it, both runs), while the same knob on the same cell moves multi-k
    KRHF by +0.505030889141 Ha.

    Neither source that can meet the knob ignores it silently now (pinned
    without an SCF):

    * the default SR/LR source (``fit_source='auto'`` on dim=3 rsgdf)
      has no reciprocal tail, so it retires the knob with a
      ``DeprecationWarning`` and builds its fit without one;
    * the classic source (``fit_source='classic'``) builds its fit
      carrying the caller's tail.

    Until 2026-09-29 this test still asserted the pre-SR contract: the
    tailed default reached the classic shared-q build, and the untailed
    default carried an IID 518 auto-sized tail. The SR/LR promotion
    (#284) retired both, so it was red on main: nothing stopped the SR/LR
    build, and the MgO SCF ran instead of raising. The route does not
    depend on the cell, so it is pinned on the cheap sparse H2 box.
    """
    _sentinel_multik_fit(monkeypatch)
    system, basis = _h2_cubic_box()

    with pytest.warns(DeprecationWarning, match="obsolete for the SR/LR fit"):
        with pytest.raises(_FitSentinel) as sr:
            run_krks_periodic_gdf(
                system, basis, (1, 1, 2), functional="lda",
                rsgdf_tail_ke_cutoff=3300.0, progress=False,
            )
    assert sr.value.args[0] == "_build_scf_range_separated_lpq_cache"

    with pytest.raises(_FitSentinel) as classic:
        run_krks_periodic_gdf(
            system, basis, (1, 1, 2), functional="lda",
            fit_source="classic", rsgdf_tail_ke_cutoff=3300.0,
            progress=False,
        )
    name, kwargs = classic.value.args
    assert name == "_build_rsgdf_lpq_cache_shared_q"
    assert kwargs.get("tail_ke_cutoff") == 3300.0


@pytest.mark.parametrize("cell", ["dense_core_mgo", "sparse_h2"])
def test_multik_pure_ks_default_builds_the_untailed_sr_fit(monkeypatch, cell):
    """The untailed default pure-DFT dim=3 multi-k route builds the SR/LR
    fit on dense-core and sparse cells alike.

    IID 518 once had this route resolve an auto-sized rsgdf tail on the
    dense-core class: left on the EWALD_3D J, MgO/STO-3G (2,2,2)
    KRKS-LDA sat 1032.5 mHa below PySCF and CRYSTAL 23. A sparse H2 cell,
    where the classifier resolves no tail, was the negative control that
    kept the EWALD_3D J. The SR/LR source (#284) replaced both routes: it
    never uses an exponent-scaled reciprocal tail, so there is none to
    resolve, and the dense-core parity-hold classifier does not run on
    it. This replaces ``test_multik_pure_ks_auto_tails_dense_core`` and
    ``test_multik_pure_ks_sparse_cell_keeps_ewald3d``, both red on main.
    """
    _sentinel_multik_fit(monkeypatch)
    if cell == "dense_core_mgo":
        system, basis = _mgo_dense_core_system()
    else:
        system, basis = _h2_cubic_box()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_FitSentinel) as reached:
            run_krks_periodic_gdf(
                system, basis, (1, 1, 2), functional="lda", progress=False
            )
    assert reached.value.args[0] == "_build_scf_range_separated_lpq_cache"
    assert not [w for w in caught if "parity-held" in str(w.message)]
    assert not [w for w in caught if "rsgdf_tail_ke_cutoff" in str(w.message)]


def test_multik_tail_with_non_rsgdf_gdf_method_fails_closed():
    """The high-|G| tail completion exists only on the rsgdf cderi
    builders; any other multi-k gdf_method silently ignored it (the same
    L67 silent-no-op family as IID 146). Fail closed like
    run_kuhf_periodic_gdf does."""
    system, basis = _h2_cubic_box()
    with pytest.raises(NotImplementedError, match="rsgdf"):
        run_krhf_periodic_gdf(
            system,
            basis,
            (2, 1, 1),
            gdf_method="mdf",
            rsgdf_tail_ke_cutoff=100.0,
            progress=False,
        )


class _InfoRecorder:
    """Progress sink that keeps the driver's ``info`` lines."""

    def __init__(self):
        from vibeqc.progress import resolve_progress

        self._inner = resolve_progress(False)
        self.messages = []

    def info(self, message, *args, **kwargs):
        self.messages.append(str(message))
        return self._inner.info(message, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.parametrize("driver", ["krks", "kuks"])
@pytest.mark.parametrize(
    ("gdf_method", "builder"),
    [("mdf", "build_lpq_bloch_mdf"), ("compcell", "build_lpq_bloch_compcell")],
)
def test_multik_pure_ks_honours_explicit_gdf_method(
    monkeypatch, driver, gdf_method, builder
):
    """A pure-DFT dim=3 multi-k job with an explicit non-default
    ``gdf_method`` must build the fit it asked for. The closed-shell
    driver kept the default ``use_compcell=False`` on the EWALD_3D J,
    which builds no cderi, so ``gdf_method='mdf'`` / ``'compcell'`` was
    accepted and silently ignored (the L67 family of IID 146). Found
    while lifting the MDF dense-core gate (#172): MgO/def2-SVP RKS-PBE
    (4,4,4) on mdf never reached the MDF builder.

    Pinned without an SCF: the requested builder raises a sentinel. (The
    EWALD_3D J this used to fall back to was removed with #415.) The
    open-shell driver always builds the cached fit; its rows guard that it
    stays so.
    """

    class _LpqSentinel(Exception):
        pass

    def _lpq_boom(*a, **k):
        raise _LpqSentinel()

    monkeypatch.setattr(kgdf, builder, _lpq_boom)
    run = (
        run_krks_periodic_gdf
        if driver == "krks"
        else kgdf.run_kuks_periodic_gdf
    )
    system, basis = _h2_cubic_box(6.0)
    plog = _InfoRecorder()
    with pytest.raises(_LpqSentinel):
        run(
            system,
            basis,
            (2, 1, 1),
            functional="pbe",
            gdf_method=gdf_method,
            progress=plog,
        )
    if driver == "krks":
        assert any(
            f"gdf_method={gdf_method!r}" in m and "cached-Lpq" in m
            for m in plog.messages
        ), f"promotion not logged; info lines were {plog.messages!r}"


# IID 146 anchors, MgO primitive FCC / STO-3G, kmesh (2,2,2), LDA
# (vibe-qc 'lda' = LDA_X + LDA_C_VWN = PySCF xc='lda,vwn'):
#
# * External: PySCF 2.13.1 KRKS(GDF) out-of-process reference
#   (regpass-2026-08-19-p01p13 registered row, prec 1e-10; prec
#   sensitivity 5.6e-8).
# * Untailed multi-k KRKS (EWALD_3D J route): -271.531535367637 --
#   measured identically with AND without the tail knob pre-fix (the
#   byte-identical no-op this fix removes); stable across versions
#   (archived v0.15.21 leg drift +1.47e-7).
# * Post-fix tailed at the parity size (rsgdf_tail_ke_cutoff =
#   1.1*10*zeta_max = 3291.6114): -270.502658896013 (15 it), residual
#   -3.64 mHa vs PySCF -- the -1.0325 Ha dense-core offset was
#   essentially all tail. (That full-size run costs ~38 min at the
#   conftest's single-OpenMP-thread test env, so this gate runs the
#   cheap rung below instead.)
# * Post-fix tailed at the CHEAP tail this test runs (800):
#   -270.545807077311 (8 it, ~383 s single-threaded), residual
#   -46.8 mHa -- most of the closure at ~8x less cderi-build cost
#   (mid-ladder point; Gamma twin g1x-tail-800: -21.5 mHa).
E_PYSCF_KRKS_LDA_MGO_222 = -270.4990190962243
E_UNTAILED_KRKS_LDA_MGO_222 = -271.531535367637


@pytest.mark.slow
def test_multik_krks_tail_closes_dense_core_offset():
    """IID 146 energy-level regression: a tailed multi-k KRKS on the
    dense-core MgO cell must land near the external PySCF value, not at
    the untailed EWALD_3D fixed point. Pre-fix the tailed run returned
    the untailed value byte-identically (knob unconsumed), 1.0325 Ha
    below PySCF. This runs the cheap tail=800 rung (measured
    -270.545807077311, 8 it, ~383 s single-threaded): gate at 0.1 Ha vs
    PySCF (2.1x the measured -46.8 mHa residual, 10x below the pre-fix
    offset) plus a 0.5 Ha minimum shift off the untailed fixed point
    (measured +0.9857; pre-fix exactly 0.0). The untailed leg is not
    rerun here (its EWALD_3D route costs ~80 min on this cell); its
    value is pinned above from the incident lane's repeated
    measurement."""
    system, basis = _mgo_dense_core_system()
    r = run_krks_periodic_gdf(
        system,
        basis,
        (2, 2, 2),
        functional="lda",
        rsgdf_tail_ke_cutoff=800.0,
        progress=False,
    )
    assert r.converged
    # The knob moved the energy off the untailed fixed point by the
    # J-tail scale (pre-fix: exactly 0.0).
    assert abs(float(r.energy) - E_UNTAILED_KRKS_LDA_MGO_222) > 0.5
    # ... and onto the external reference.
    assert float(r.energy) == pytest.approx(
        E_PYSCF_KRKS_LDA_MGO_222, abs=0.1
    )


@pytest.mark.slow
def test_multik_krks_default_auto_tail_closes_dense_core_offset():
    """IID 518 energy-level regression, on the DEFAULT route: no knob.

    This is the P13 deck's own configuration -- ``run_periodic_job`` passes
    no ``rsgdf_tail_ke_cutoff`` for plain multi-k GDF RKS, so pre-fix this
    landed on the untailed EWALD_3D fixed point -271.531535367637, which is
    -1033.2 mHa below CRYSTAL 23 SVWN SHRINK-2 (-270.49836634642) and
    -1032.5 mHa below PySCF -- two independent references that agree with
    each other to 0.65 mHa.

    Measured post-fix on this cell at defaults: -270.502804735242 Ha
    (converged, 459 s at 8 OpenMP threads -- the untailed EWALD_3D route it
    replaces costs ~4848 s, so the fix is also ~8x faster). Residual
    -3.79 mHa vs the pinned PySCF row and -4.44 mHa vs CRYSTAL: that
    residual is a SEPARATE open defect (IID 526), ~23x the documented
    ~0.16 mHa shared real-space-truncation floor, and it is what remains of
    IID 518 after the tail is accounted for. Do not tighten this gate onto
    the residual: that is IID 526's contract, not this test's.

    Gated at 0.05 Ha vs PySCF (13x the measured residual, 20x below the
    pre-fix offset) plus a 0.5 Ha minimum shift off the untailed fixed
    point, which pre-fix was exactly 0.0.
    """
    system, basis = _mgo_dense_core_system()
    r = run_krks_periodic_gdf(
        system, basis, (2, 2, 2), functional="lda", progress=False
    )
    assert r.converged
    # Pre-fix this difference was exactly 0.0: the default route WAS the
    # untailed fixed point.
    assert abs(float(r.energy) - E_UNTAILED_KRKS_LDA_MGO_222) > 0.5
    assert float(r.energy) == pytest.approx(
        E_PYSCF_KRKS_LDA_MGO_222, abs=0.05
    )
    # The auto-sized tail lifts the hold, so the result must no longer
    # carry the +PARITY_HELD provenance tag. A corrected number still
    # tagged held would be a worse artifact than the honest untailed one.
    assert "PARITY_HELD" not in str(getattr(r, "backend", ""))


# --- IBZ-native exchange (increment 1) ---------------------------------
# Exchange is the one n_k^2 term: K(k_i) needs the ket sum over the WHOLE
# zone for every bra. The symmetry saving is on the BRA index only --
# reducing the ket sum as well is what measured +1.389 Ha on MgO. These
# pin that _build_k_ibz_native evaluates exactly the IBZ rows of the
# full-BZ builder, and that the real symmetry transport feeds it.


def _ibz_native_synthetic(n_k, nbasis, n_fit, seed=5):
    rng = np.random.default_rng(seed)
    cache = {}
    for i in range(n_k):
        for j in range(n_k):
            cache[(i, j)] = (
                rng.normal(size=(n_fit, nbasis, nbasis))
                + 1j * rng.normal(size=(n_fit, nbasis, nbasis))
            )
    dens = []
    for _ in range(n_k):
        a = rng.normal(size=(nbasis, nbasis)) + 1j * rng.normal(
            size=(nbasis, nbasis)
        )
        dens.append(a + a.conj().T)          # Hermitian, as a density is
    return cache, dens


@pytest.mark.parametrize("rows", [[0], [0, 2], [1, 3], [0, 1, 2, 3]])
def test_ibz_native_exchange_matches_full_bz_rows(rows):
    """The IBZ-native builder reproduces exactly the corresponding rows
    of the full-BZ builder, for any choice of bra subset."""
    n_k, nbf, n_fit = 4, 5, 7
    cache, dens = _ibz_native_synthetic(n_k, nbf, n_fit)
    w = np.full(n_k, 1.0 / n_k)

    full = _build_k_from_lpq_cache(cache, dens, w, nbasis=nbf)
    native = kgdf._build_k_ibz_native(cache, dens, w, rows, nbasis=nbf)

    assert len(native) == len(rows)
    for out, i in zip(native, rows):
        np.testing.assert_allclose(out, full[i], atol=1e-12, rtol=1e-12)


def test_ibz_native_exchange_rejects_bad_inputs():
    n_k, nbf, n_fit = 3, 4, 5
    cache, dens = _ibz_native_synthetic(n_k, nbf, n_fit)
    w = np.full(n_k, 1.0 / n_k)
    with pytest.raises(ValueError, match="weights_full"):
        kgdf._build_k_ibz_native(cache, dens, w[:2], [0], nbasis=nbf)
    with pytest.raises(IndexError):
        kgdf._build_k_ibz_native(cache, dens, w, [0, n_k], nbasis=nbf)


def test_ibz_native_exchange_with_real_symmetry_unfolded_densities():
    """End-to-end shape of the increment: take a converged full-BZ SCF,
    keep only the wedge densities, unfold them with the real transport,
    and confirm the IBZ-native exchange reproduces the full-BZ exchange
    at the wedge points.

    This is the step that would silently break if the ket sum were
    reduced to the wedge as well, or if the transport were dropped.
    """
    from vibeqc._vibeqc_core import monkhorst_pack as _mp
    from vibeqc.periodic_k_symmetry import (
        expand_k_matrices_to_full,
        star_operations,
    )

    a = 7.72
    lat = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [0.5 * a, 0, 0])]
    )
    vq.attach_symmetry(system, symprec=1e-6)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    kfull = _mp(system, [2, 2, 2], [0, 0, 0], False)
    kibz = _mp(system, [2, 2, 2], [0, 0, 0], True)
    star = star_operations(system, kibz, kfull)
    kf = [np.asarray(k, dtype=float) for k in kfull.kpoints]
    rows = [
        int(np.argmin([np.linalg.norm(k - np.asarray(kk)) for k in kf]))
        for kk in kibz.kpoints
    ]
    assert len(rows) < len(kf)          # genuinely reduced

    n_k, nbf, n_fit = len(kf), 6, 9
    cache, dens_full = _ibz_native_synthetic(n_k, nbf, n_fit, seed=11)
    # Replace the densities by ones that are genuinely symmetry-related:
    # unfold the wedge subset through the real transport.
    dens_unfolded = expand_k_matrices_to_full(
        [dens_full[i] for i in rows], star, system, basis, kfull
    )
    w = np.full(n_k, 1.0 / n_k)

    full = _build_k_from_lpq_cache(cache, dens_unfolded, w, nbasis=nbf)
    native = kgdf._build_k_ibz_native(
        cache, dens_unfolded, w, rows, nbasis=nbf
    )
    for out, i in zip(native, rows):
        np.testing.assert_allclose(out, full[i], atol=1e-12, rtol=1e-12)


# --- IBZ-native driver flag (increment 2) ------------------------------


def _ibz_native_system():
    a = 7.72
    lat = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [0.5 * a, 0, 0])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _ibz_native_opts(cutoff=20.0):
    o = vq.PeriodicRHFOptions()
    o.max_iter = 40
    o.conv_tol_energy = 1e-9
    o.lattice_opts.cutoff_bohr = cutoff
    return o


@pytest.mark.slow
def test_ibz_native_reproduces_full_bz_energy():
    """The opt-in IBZ-native path must land on the full-BZ answer.

    Exchange is built at the wedge and symmetry-transported; everything
    else stays full-mesh, so the result shape is unchanged too. The
    agreement is limited by the cell list, not by the symmetry: at
    ``cutoff_bohr = 20`` the transport residual on K is ~8e-6 and the
    energies agree to ~1e-7 Ha.
    """
    system, basis = _ibz_native_system()
    vq.attach_symmetry(system, symprec=1e-6)
    common = dict(
        aux_basis="def2-svp-jk", rsgdf_ke_cutoff=80.0, progress=False
    )
    # Cutoff 26: the regime where the symmetry gate is sharp (symmetric
    # and broken densities are ~6000x apart there, ~3x apart at 20).
    full = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 2, 2),
        options=_ibz_native_opts(cutoff=26.0), **common
    )
    native = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 2, 2),
        options=_ibz_native_opts(cutoff=26.0), ibz_native=True, **common
    )
    assert full.converged and native.converged
    assert abs(full.energy - native.energy) < 1e-6, (
        full.energy, native.energy
    )
    # Result shape is untouched: consumers still see the full mesh.
    assert len(native.density) == len(full.density)
    assert len(native.mo_energies) == len(full.mo_energies)
    assert np.asarray(native.kpoints_cart).shape == np.asarray(
        full.kpoints_cart
    ).shape


def test_ibz_native_requires_attached_symmetry():
    """Without symmetry there is no wedge -- refuse, do not silently
    fall back to full-BZ, which would make the flag look effective."""
    system, basis = _ibz_native_system()          # no attach_symmetry
    with pytest.raises(NotImplementedError, match="attach_symmetry"):
        run_krhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=_ibz_native_opts(),
            aux_basis="def2-svp-jk", ibz_native=True, progress=False,
        )


def test_ibz_native_refuses_without_exact_exchange():
    """The flag reduces the HF-exchange n_k^2 term; on a pure functional
    there is nothing to reduce, so asking for it is a mistake worth
    reporting rather than a silent no-op."""
    system, basis = _ibz_native_system()
    vq.attach_symmetry(system, symprec=1e-6)
    with pytest.raises(NotImplementedError, match="exact exchange"):
        run_krks_periodic_gdf(
            system, basis, kmesh=(2, 2, 2),
            options=vq.PeriodicKSOptions(), functional="pbe",
            aux_basis="def2-svp-jk", ibz_native=True, progress=False,
        )


# --- IBZ-native on the open-shell driver -------------------------------


@pytest.mark.slow
def test_ibz_native_open_shell_m1_matches_full_bz():
    """UHF at multiplicity 1 has a symmetric density, so the wedge
    transport is exact and the energy matches the full-BZ run."""
    from vibeqc.periodic_k_gdf import run_kuhf_periodic_gdf

    system, basis = _ibz_native_system()
    vq.attach_symmetry(system, symprec=1e-6)
    common = dict(
        aux_basis="def2-svp-jk", rsgdf_ke_cutoff=80.0, progress=False
    )
    full = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 2, 2),
        options=_ibz_native_opts(cutoff=26.0), **common
    )
    native = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 2, 2),
        options=_ibz_native_opts(cutoff=26.0), ibz_native=True, **common
    )
    assert full.converged and native.converged
    assert abs(full.energy - native.energy) < 1e-9
    assert len(native.density_alpha) == len(full.density_alpha)
    assert len(native.density_beta) == len(full.density_beta)


def test_ibz_native_refuses_a_symmetry_broken_state(monkeypatch):
    """The guard must also reject a nonsymmetric density after the initial guess.

    A spontaneous triplet symmetry break depends on the guess and integral
    accuracy. Instead rotate natural orbitals at one nonrepresentative k point
    after the first alpha/beta checks. Keep the physical metric occupations,
    Hermiticity and electron count, and call the actual production guard.
    """
    import vibeqc.periodic_k_gdf as kgdf

    system, basis = _ibz_native_system()
    system.multiplicity = 3
    vq.attach_symmetry(system, symprec=1e-6)
    original = kgdf._require_ibz_symmetric_state
    calls = 0
    witnessed = []

    def inject_after_initial_guess(matrices, overlaps, rows, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # alpha and beta passed on the first iteration
            index = next(i for i in range(len(matrices)) if i not in rows)
            values, vectors = np.linalg.eigh(overlaps[index])
            assert np.min(values) > 0
            x = vectors / np.sqrt(values)[None, :]
            sx = overlaps[index] @ x
            occupations, natural = np.linalg.eigh(sx.conj().T @ matrices[index] @ sx)
            angle = 0.08
            rotation = np.eye(len(occupations))
            rotation[0, 0] = rotation[-1, -1] = np.cos(angle)
            rotation[0, -1] = -np.sin(angle)
            rotation[-1, 0] = np.sin(angle)
            c = x @ natural @ rotation
            changed = (c * occupations[None, :]) @ c.conj().T
            np.testing.assert_allclose(
                np.trace(changed @ overlaps[index]),
                np.trace(matrices[index] @ overlaps[index]), atol=1e-12, rtol=0,
            )
            np.testing.assert_allclose(changed, changed.conj().T, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                np.linalg.eigvalsh(sx.conj().T @ changed @ sx), occupations,
                atol=1e-12, rtol=0,
            )
            # The driver passes views of its accepted density arrays. Update
            # that state in place so the real pre-exchange guard sees it.
            matrices[index][...] = changed
            observed, floor = kgdf._ibz_symmetry_consistency(
                matrices, overlaps, rows, *args,
            )
            assert observed > 1e-3
            assert floor < 1e-7
            witnessed.append((observed, floor))
        return original(matrices, overlaps, rows, *args, **kwargs)

    monkeypatch.setattr(kgdf, "_require_ibz_symmetric_state", inject_after_initial_guess)
    options = _ibz_native_opts(cutoff=26.0)
    options.initial_guess = vq.InitialGuess.HCORE
    with pytest.raises(NotImplementedError, match="symmetry"):
        kgdf.run_kuhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=options,
            aux_basis="def2-svp-jk", ibz_native=True, progress=False,
        )
    assert calls == 3
    assert len(witnessed) == 1


def test_ibz_native_refuses_cosx_exchange():
    """COSX builds K in real space from its own bridge and never touches
    the per-(k_i,k_j) cderi cache the flag reduces."""
    system, basis = _ibz_native_system()
    vq.attach_symmetry(system, symprec=1e-6)
    with pytest.raises(NotImplementedError, match="COSX|k_exchange"):
        run_krhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=_ibz_native_opts(),
            aux_basis="def2-svp-jk", k_exchange="cosx",
            use_compcell=True,   # COSX's own precondition, so the
            ibz_native=True,     # ibz_native guard is what fires
            progress=False,
        )


def test_ibz_native_symmetry_gate_calibrates_against_the_overlap():
    """The gate compares the density's transport residual to a floor
    measured on the overlap, not to zero.

    A truncated Bloch sum is not exactly symmetric either -- the
    per-atom lattice shifts of shift-bearing operations clip differently
    at the cell-list boundary -- so an absolute comparison would refuse
    every calculation at a loose cutoff. The overlap carries that same
    truncation asymmetry and no state information, which makes it the
    right reference.
    """
    from vibeqc._vibeqc_core import monkhorst_pack as _mp
    from vibeqc.periodic_k_symmetry import star_operations

    system, basis = _ibz_native_system()
    vq.attach_symmetry(system, symprec=1e-6)
    kfull = _mp(system, [2, 2, 2], [0, 0, 0], False)
    kibz = _mp(system, [2, 2, 2], [0, 0, 0], True)
    star = star_operations(system, kibz, kfull)
    rows, _s, _k = kgdf._resolve_ibz_native_state(
        system,
        np.asarray([np.asarray(k, dtype=float) for k in kfull.kpoints]),
        (2, 2, 2),
        np.full(len(list(kfull.kpoints)), 1.0 / len(list(kfull.kpoints))),
        None,
        exchange_producer="sr",
    )

    from vibeqc._vibeqc_core import (
        LatticeSumOptions, bloch_sum, compute_overlap_lattice,
    )

    o = LatticeSumOptions()
    o.cutoff_bohr = 26.0
    S_lat = compute_overlap_lattice(basis, system, o)
    S_k = [
        np.asarray(bloch_sum(S_lat, np.asarray(k, dtype=float)))
        for k in kfull.kpoints
    ]

    # The overlap against itself is the floor, so it must pass its own gate.
    observed, floor = kgdf._ibz_symmetry_consistency(
        S_k, S_k, rows, star, system, basis, kfull
    )
    assert observed == pytest.approx(floor)
    kgdf._require_ibz_symmetric_state(
        S_k, S_k, rows, star, system, basis, kfull, label="overlap"
    )

    # A deliberately broken state must be refused.
    broken = [S.copy() for S in S_k]
    broken[1] = broken[1] * 1.01
    with pytest.raises(NotImplementedError, match="does not respect"):
        kgdf._require_ibz_symmetric_state(
            broken, S_k, rows, star, system, basis, kfull, label="test"
        )


def test_ibz_native_admits_non_symmorphic_groups(monkeypatch):
    """A glide/screw cell passes the wedge gate on the SR/LR source (#452).

    The wedge transport applies a per-atom lattice-shift Bloch phase
    ``exp(i k.L_a)`` with the image shift defined by
    ``x_perm[a] = R x_a + tau + L_a``, so a fractional translation ``tau``
    is carried by ``L_a`` and no further ``exp(-i k.tau)`` exists in this
    AO-centred Bloch convention. Si diamond (Fd-3m, No. 227) therefore
    resolves a wedge whose star map uses glide-type operations, and the
    driver reaches the exchange build instead of refusing the group by
    name (the refusal this test used to pin). The numerical proof lives in
    ``tests/test_periodic_k_gdf_ibz_nonsymmorphic.py``.
    """
    bohr = 0.529177210903
    a_si = 5.431 / bohr
    lat = np.array(
        [[0, a_si / 2, a_si / 2], [a_si / 2, 0, a_si / 2], [a_si / 2, a_si / 2, 0]]
    ).T
    system = vq.PeriodicSystem(
        3, lat,
        [vq.Atom(14, [0.0, 0.0, 0.0]),
         vq.Atom(14, list(lat @ np.array([0.25, 0.25, 0.25])))],
    )
    vq.attach_symmetry(system)

    def _has_tau(op):
        return np.linalg.norm(
            (np.asarray(op.translation, dtype=float) + 0.5) % 1.0 - 0.5
        ) > 1e-8

    # Precondition: this really is the non-symmorphic case.
    assert sum(_has_tau(op) for op in system.symmetry.operations) == 24

    from vibeqc.periodic_k_gdf import _resolve_ibz_native_state
    from vibeqc._vibeqc_core import monkhorst_pack as _mp
    kmesh_full = _mp(system, [2, 2, 2], [0, 0, 0], False)
    kpts = np.array([np.asarray(k, dtype=float) for k in kmesh_full.kpoints])
    rows, star_map, _ = _resolve_ibz_native_state(
        system, kpts, (2, 2, 2), np.full(len(kpts), 1.0 / len(kpts)), None,
        exchange_producer="sr",
    )
    assert len(rows) == 3 and len(star_map.operations) == 48
    assert any(_has_tau(star_map.operations[o]) for _, o, _ in star_map.entries)

    # The driver no longer refuses the group: it reaches the SR/LR cderi
    # build, where a sentinel stops this test before any integral.
    class _BuildReached(Exception):
        pass

    def intercept(*args, **kwargs):
        raise _BuildReached

    for name in ("_build_rsgdf_lpq_cache_shared_q",
                 "_build_scf_range_separated_lpq_cache"):
        monkeypatch.setattr(kgdf, name, intercept)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 3
    with pytest.raises(_BuildReached):
        run_krhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=opts,
            aux_basis="def2-svp-jk", ibz_native=True, progress=False,
        )


def test_ibz_native_rows_stay_in_wedge_order():
    """The wedge rows must be returned in ``kmesh_ibz`` order, never
    sorted.

    ``star_map.entries`` carries ``i_rep`` as an index into
    ``kmesh_ibz.kpoints``, and ``expand_k_matrices_to_full`` looks up
    ``M_k_ibz[i_rep]``. Sorting the rows silently matches each wedge
    matrix to the wrong star.

    This is invisible on small meshes because their wedge order is
    already ascending -- (2,2,2) gives [0, 4, 6] and (3,3,3) gives
    [0, 9, 12, 21], both already sorted -- so it only appears once the
    order is not monotonic. (4,4,4) gives [0, 16, 32, 20, 36, 52, 40,
    57], which sorting permutes, and it cost 0.44 Ha before being
    caught by a production-scale run.
    """
    from vibeqc._vibeqc_core import monkhorst_pack as _mp

    system, basis = _ibz_native_system()
    vq.attach_symmetry(system, symprec=1e-6)
    lattice = np.asarray(system.lattice, dtype=float)

    def _frac(k):
        return (lattice.T @ np.asarray(k, dtype=float).reshape(3)) / (
            2.0 * np.pi
        )

    for mesh in ((2, 2, 2), (3, 3, 3), (4, 4, 4)):
        kfull = _mp(system, list(mesh), [0, 0, 0], False)
        kibz = _mp(system, list(mesh), [0, 0, 0], True)
        kpts = np.asarray(
            [np.asarray(k, dtype=float) for k in kfull.kpoints]
        )
        weights = np.full(len(kpts), 1.0 / len(kpts))
        rows, star_map, _kf = kgdf._resolve_ibz_native_state(
            system, kpts, mesh, weights, None, exchange_producer="sr"
        )
        # Every representative, in the wedge's own order.
        expected = []
        for kk in kibz.kpoints:
            f_ref = _frac(kk)
            d = [
                float(np.max(np.abs(((_frac(kpts[i]) - f_ref) + 0.5) % 1.0
                                    - 0.5)))
                for i in range(len(kpts))
            ]
            expected.append(int(np.argmin(d)))
        assert rows == expected, (mesh, rows, expected)
        # And the star map's rep indices address them positionally.
        for i_rep, _op, _trev in star_map.entries:
            assert 0 <= i_rep < len(rows), (mesh, i_rep, len(rows))

    # The (4,4,4) case must genuinely be non-monotonic, or this test
    # would pass even with the sorting bug present.
    kfull = _mp(system, [4, 4, 4], [0, 0, 0], False)
    kpts = np.asarray([np.asarray(k, dtype=float) for k in kfull.kpoints])
    rows, _sm, _kf = kgdf._resolve_ibz_native_state(
        system, kpts, (4, 4, 4),
        np.full(len(kpts), 1.0 / len(kpts)), None, exchange_producer="sr",
    )
    assert rows != sorted(rows), (
        "the (4,4,4) wedge order is monotonic here, so this test can no "
        "longer discriminate the sorting bug -- pick another mesh"
    )


# ---------------------------------------------------------------------
#   symmetry_reduce_k: the same reduction from run_periodic_job
# ---------------------------------------------------------------------
#
# The algorithm above worked, but `ibz_native` was a driver-only keyword:
# it had ZERO hits in periodic_runner.py, so the production entry point
# could not request it and the memory preflight -- which always charged
# n_k^2 -- aborted a reduced run at the full-mesh number. These pin the
# plumbing: the kwarg reaches the driver, the preconditions fail closed
# rather than silently running full-BZ, and the reduction is exact.


def _nacl_primitive():
    """NaCl rocksalt primitive cell, Fm-3m (No. 225) -- symmorphic, so the
    per-atom lattice-shift Bloch transport represents every operation."""
    a = 5.64 * 1.8897261254578281  # angstrom -> bohr
    lat = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    ).T
    return vq.PeriodicSystem(
        3,
        lat,
        [
            vq.Atom(11, [0.0, 0.0, 0.0]),
            vq.Atom(17, list(lat @ np.array([0.5, 0.5, 0.5]))),
        ],
    )


def _run_periodic_job_kwargs(tmp_path, stem, **extra):
    base = dict(
        method="RHF",
        jk_method="gdf",
        kpoints=(2, 2, 2),
        aux_basis="def2-svp-jk",
        max_iter=60,
        conv_tol_energy=1e-9,
        output=str(tmp_path / stem),
        progress=False,
        output_qvf=False,
        write_xyz_file=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
        citations=False,
    )
    base.update(extra)
    return base


def test_symmetry_reduce_k_resolves_the_nacl_wedge():
    """The runner's wedge count is what the preflight is charged at, so it
    must be the spglib reduction of the mesh actually being run."""
    from vibeqc.periodic_runner import _ibz_kpoint_count

    system = _nacl_primitive()
    assert _ibz_kpoint_count(system, (2, 2, 2)) is None, (
        "without an attached symmetry model there is no wedge to report"
    )
    vq.attach_symmetry(system, symprec=1e-4)
    assert system.symmetry.number == 225
    assert _ibz_kpoint_count(system, (2, 2, 2)) == 3
    assert _ibz_kpoint_count(system, (4, 4, 4)) == 8


def test_symmetry_reduce_k_requires_attached_symmetry(tmp_path):
    """Never silently ignore the flag: without symmetry there is no wedge,
    and a run that looks reduced while costing n_k^2 is the failure mode
    this plumbing exists to prevent."""
    from vibeqc.periodic_runner import run_periodic_job

    system = _nacl_primitive()          # no attach_symmetry, symmetry=False
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="attached symmetry model"):
        run_periodic_job(
            system,
            basis,
            **_run_periodic_job_kwargs(
                tmp_path, "nosym", symmetry_reduce_k=True
            ),
        )


def test_symmetry_reduce_k_refuses_non_gdf_and_gamma(tmp_path):
    """The reduction acts on the per-(k_i,k_j) cderi cache. Routes without
    one, and a Γ mesh with nothing to reduce, fail closed."""
    from vibeqc.periodic_runner import run_periodic_job

    system = _nacl_primitive()
    vq.attach_symmetry(system, symprec=1e-4)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="jk_method='gdf'"):
        run_periodic_job(
            system,
            basis,
            **_run_periodic_job_kwargs(
                tmp_path, "gpw", jk_method="gpw", symmetry_reduce_k=True
            ),
        )
    with pytest.raises(NotImplementedError, match="true multi-k"):
        run_periodic_job(
            system,
            basis,
            **_run_periodic_job_kwargs(
                tmp_path, "gamma", kpoints=(1, 1, 1), symmetry_reduce_k=True
            ),
        )


def test_symmetry_reduce_k_refuses_optimization(tmp_path):
    """The wedge-native SCF converges the k-transported exchange, whose
    gradient is not the captured single-point objective."""
    from vibeqc.periodic_runner import run_periodic_job

    system = _nacl_primitive()
    vq.attach_symmetry(system, symprec=1e-4)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="geometry optimization"):
        run_periodic_job(
            system,
            basis,
            **_run_periodic_job_kwargs(
                tmp_path, "opt", optimize=True, symmetry_reduce_k=True
            ),
        )


def test_symmetry_reduce_k_preflight_charges_the_wedge(monkeypatch):
    """The guard and the printed [memory] estimate must both reflect
    n_IBZ x n_k. Charging n_k^2 aborts a reduced run that fits -- the
    reason the reduction could not be demonstrated from the entry point."""
    from vibeqc.periodic_runner import (
        PeriodicJKMethod,
        _periodic_gdf_estimate,
    )
    from types import SimpleNamespace

    import scipy.linalg.lapack as lapack
    import vibeqc.aux_basis as auxiliary
    import vibeqc.periodic_k_gdf as gdf
    from vibeqc import _vibeqc_core as core
    from tests.periodic_memory_contract import (
        controlled_wedge_gdf_inventory, headline, wedge_cache_bytes,
    )

    # Declare planner/native inputs, retaining the real category arithmetic.
    # This makes exact bytes portable without estimating from reported totals.
    monkeypatch.setenv("VIBEQC_MEMORY_HEADROOM", "1.5")
    monkeypatch.setenv("VIBEQC_VNE_EWALD3D_FT_CHUNK_MIB", "256")
    monkeypatch.setattr(core, "get_num_threads", lambda: 2)
    monkeypatch.setattr(core, "gdf_short_range_workspace_bytes", lambda *args: 80 * 1024**2)
    monkeypatch.setattr(
        lapack, "get_lapack_funcs",
        lambda *args, **kwargs: (
            SimpleNamespace(int_dtype=SimpleNamespace(itemsize=4)),
            lambda *args, **kwargs: (32.0, 0, 64, 0),
        ),
    )
    monkeypatch.setattr(
        auxiliary, "_plan_range_separated_gdf_cutoffs",
        lambda *args, **kwargs: SimpleNamespace(ke_cutoff=2.0),
    )
    oneel_calls = []

    def controlled_oneel_admission(*args, **kwargs):
        oneel_calls.append(True)
        return SimpleNamespace(
            cutoff_bohr=4.0, nuclear_cutoff_bohr=3.0, pair_complete_1e=False,
        )

    # The preflight uses the unscreened admission domain, not the later
    # screened lattice domain. Keep this fixture on the actual sizing path.
    monkeypatch.setattr(gdf, "_oneel_admission_opts", controlled_oneel_admission)

    system = _nacl_primitive()
    vq.attach_symmetry(system, symprec=1e-4)
    basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
    common = dict(
        resolved_jk=PeriodicJKMethod.GDF,
        method_upper="RHF",
        functional=None,
        kpoints=(4, 4, 4),
        aux_basis="def2-svp-jk",
    )
    full = _periodic_gdf_estimate(system, basis, **common)
    reduced = _periodic_gdf_estimate(system, basis, n_ibz_kpoints=8, **common)
    assert len(oneel_calls) == 2, "controlled one-electron admission was bypassed"
    key = "GDF Lpq factor cache"
    assert basis.nbasis == 33
    assert auxiliary.make_aux_basis_set(
        system.unit_cell_molecule(), aux_name="def2-svp-jk",
    ).nbasis == 240
    assert full.estimate.by_category[key] == wedge_cache_bytes(
        33, 240, 64, exchange=True,
    )
    assert reduced.estimate.by_category[key] == wedge_cache_bytes(
        33, 240, 64, exchange=True, bras=8,
    )
    assert "GDF dense AO-pair FT bundle" not in full.estimate.by_category
    assert "GDF dense AO-pair FT bundle" not in reduced.estimate.by_category
    assert full.n_ibz_kpoints is None and reduced.n_ibz_kpoints == 8
    # The exchange wedge also retains the 56 other Hartree diagonals.
    assert full.estimate.by_category[key] / reduced.estimate.by_category[
        key
    ] == pytest.approx(64**2 / (8 * 64 + 56))
    # Charge every 3D allocation independently: equality alone would miss a
    # common omission. Only the exchange factor cache depends on the wedge.
    a = 5.64 * 1.8897261254578281
    for result, bras in ((full, None), (reduced, 8)):
        expected = controlled_wedge_gdf_inventory(33, 240, a, bras=bras)
        categories = result.estimate.by_category
        for category, byte_count in expected.items():
            assert category in categories, f"missing {category}"
            assert categories[category] == byte_count, f"allocation {category}"
        assert set(categories) == set(expected), "category inventory"
        assert result.estimate.total_bytes == headline(expected), "headline headroom"
    # A pure functional has no n_k^2 exchange term, so the wedge is ignored
    # rather than under-charging a diagonal-only cache.
    pure = dict(common, method_upper="RKS", functional="pbe")
    pure_full = _periodic_gdf_estimate(system, basis, **pure)
    pure_reduced = _periodic_gdf_estimate(system, basis, n_ibz_kpoints=8, **pure)
    assert len(oneel_calls) == 4, "pure-functional admission was bypassed"
    # This arm checks the diagonal cache and J-only workspace, not its XC sum.
    for result in (pure_full, pure_reduced):
        assert result.n_ibz_kpoints is None
        assert result.estimate.by_category[key] == wedge_cache_bytes(
            33, 240, 64, exchange=False,
        ), "pure diagonal cache"
        assert result.estimate.by_category["GDF contraction workspace"] == 1024**2
    assert pure_full.estimate.by_category[key] == pure_reduced.estimate.by_category[key]


@pytest.mark.slow
def test_symmetry_reduce_k_matches_full_bz_energy(tmp_path):
    """End-to-end through run_periodic_job: the space-group reduction is
    EXACT, so the converged energy must match the unreduced run. Anything
    else means the transport, not the k-mesh, changed the answer.

    NaCl primitive / STO-3G / (2,2,2): 3 of 8 k-points carry the exchange
    build. Measured 2026-08-14 at the runner's default 15-bohr lattice
    cutoff: -616.3438077935 (full) vs -616.3438073219 (reduced), i.e.
    4.7e-7 Ha -- the cell-list transport residual, not a symmetry error
    (the driver-level test at cutoff_bohr = 26 closes to ~1e-9).
    """
    from vibeqc.periodic_runner import run_periodic_job

    energies = {}
    for tag, reduce_k in (("full", False), ("reduced", True)):
        system = _nacl_primitive()
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        result = run_periodic_job(
            system,
            basis,
            **_run_periodic_job_kwargs(
                tmp_path,
                f"nacl_{tag}",
                symmetry="attach",
                symmetry_reduce_k=reduce_k,
            ),
        )
        assert result.converged, tag
        energies[tag] = float(result.energy)

    assert abs(energies["full"] - energies["reduced"]) < 1e-6, energies

    # The .out states the reduction it performed -- the printed number is
    # what makes the memory saving auditable, so pin it.
    reduced_out = (tmp_path / "nacl_reduced.out").read_text()
    assert "symmetry_reduce_k" in reduced_out
    assert "exchange bras 8 -> 3 (irreducible wedge)" in reduced_out
    assert "Lpq cderi pairs 64 -> 24" in reduced_out
    assert "symmetry_reduce_k" not in (tmp_path / "nacl_full.out").read_text()


# =====================================================================
#   Linear-dependence reporting (LINEAR-DEPENDENCE-STATE-UNREPORTED)
# =====================================================================
#
# Canonical orthogonalisation discards directions per k-point, and the
# density fit discards them again on its own threshold. Both numbers were
# computed and thrown away, so a periodic run could silently work in a
# smaller variational space than the basis implies with nothing in the
# .out to say so. That is the state whose absence made the MDF dense-core
# divergence take three sessions to attribute (it turned out to be
# gdf_linear_dep_threshold: 1e-9 gave -62874 Ha, 1e-3 gave -271.146
# against a -271.0499 reference).


def test_gdf_result_reports_linear_dependence_state():
    """The driver surfaces what it discarded, including when it is zero."""
    from vibeqc.linear_dependence import PeriodicLinearDependenceSummary

    system, basis = _h2_cubic_box()
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.max_iter = 50
    opts.conv_tol_energy = 1e-9
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    r = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=opts,
        aux_basis="def2-svp-jk", gdf_method="rsgdf",
        rsgdf_ke_cutoff=200.0, progress=False,
    )
    assert r.converged
    summary = r.linear_dependence
    assert isinstance(summary, PeriodicLinearDependenceSummary)
    assert summary.n_basis == basis.nbasis
    # One entry per k-point, each a real retained dimension.
    assert len(summary.n_kept_per_k) == r.kpoints_cart.shape[0]
    assert all(0 < n <= basis.nbasis for n in summary.n_kept_per_k)
    # A well-conditioned vacuum box discards nothing -- the negative
    # observation this block exists to make.
    assert summary.n_discarded_max == 0
    assert not summary.any_discarded
    assert summary.threshold > 0.0
    assert summary.min_overlap_eigenvalue > 0.0
    assert summary.condition_number > 1.0
    # Auxiliary fit side, which is the GDF-specific failure mode.
    assert summary.n_aux > 0
    assert 0 < summary.n_fit_kept <= summary.n_aux
    assert summary.aux_threshold > 0.0
    assert "threshold" in summary.one_line()


def test_open_shell_gdf_drivers_report_linear_dependence_state():
    """KUHF and KROHF surface it too, not just the closed-shell driver.

    The claim covered "the periodic GDF routes" plural; wiring only KRHF
    would have left the open-shell drivers computing `n_kept` and
    discarding it exactly as before.
    """
    from vibeqc.linear_dependence import PeriodicLinearDependenceSummary
    from vibeqc.periodic_k_gdf import run_kuhf_periodic_gdf
    from vibeqc.periodic_rohf_gdf import run_krohf_periodic_gdf

    system, basis = _h2_cubic_box()
    system.multiplicity = 3
    opts = vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.use_diis = True
    opts.max_iter = 50
    opts.conv_tol_energy = 1e-9
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    common = dict(
        kmesh=(2, 1, 1), options=opts, aux_basis="def2-svp-jk",
        gdf_method="rsgdf", rsgdf_ke_cutoff=200.0, progress=False,
    )
    for driver in (run_kuhf_periodic_gdf, run_krohf_periodic_gdf):
        r = driver(system, basis, **common)
        assert r.converged, driver.__name__
        summary = r.linear_dependence
        assert isinstance(summary, PeriodicLinearDependenceSummary), (
            driver.__name__
        )
        assert summary.n_basis == basis.nbasis
        assert len(summary.n_kept_per_k) == r.kpoints_cart.shape[0]
        assert summary.n_discarded_max == 0
        assert summary.min_overlap_eigenvalue > 0.0
        assert 0 < summary.n_fit_kept <= summary.n_aux


def test_linear_dependence_summary_counts_discarded_directions():
    """The retained dimension is per k and the discard count is the max."""
    from vibeqc.linear_dependence import (
        PeriodicLinearDependenceSummary,
        format_periodic_linear_dependence,
    )

    s = PeriodicLinearDependenceSummary(
        n_basis=14,
        n_kept_per_k=[14, 11, 13],
        threshold=1e-7,
        min_overlap_eigenvalue=3.2e-8,
        max_overlap_eigenvalue=31.0,
        n_aux=189,
        n_fit_kept=83,
        aux_threshold=1e-9,
    )
    assert s.n_discarded_max == 3  # 14 - min(14, 11, 13)
    assert s.any_discarded
    assert s.condition_number == pytest.approx(31.0 / 3.2e-8)
    text = format_periodic_linear_dependence(s)
    # The four quantities the bug report asked for must all be present.
    assert "retained after orthog." in text
    assert "directions discarded      = 3" in text
    assert "linear-dep threshold" in text
    assert "min eigenvalue of S(k)" in text
    # And the aux fit, which is what governs the DF 1/sqrt(lambda) blowup.
    assert "auxiliary fit retained    = 83" in text


def test_linear_dependence_summary_handles_a_singular_overlap():
    """A non-positive-definite S must not produce a bogus condition
    number: report it as infinite rather than a negative ratio."""
    from vibeqc.linear_dependence import PeriodicLinearDependenceSummary

    s = PeriodicLinearDependenceSummary(
        n_basis=4,
        n_kept_per_k=[3],
        threshold=1e-7,
        min_overlap_eigenvalue=-5.4e-3,
        max_overlap_eigenvalue=31.0,
    )
    assert s.condition_number == float("inf")
    assert s.n_discarded_max == 1


@pytest.mark.parametrize('threads,cap', [(1, 2**20), (4, 2**20), (64, 4096)])
def test_k_parallel_exchange_matches_full_weighted_operator(monkeypatch, threads, cap):
    from vibeqc.periodic_k_gdf import _k_from_signed_factors, _signed_gram_factors
    from vibeqc import _vibeqc_core as core

    monkeypatch.setattr(core, 'get_num_threads', lambda: threads)
    rng = np.random.default_rng(649)
    nk, nbf, naux = 4, 3, 37
    cache = _random_cderi_cache(rng, nk, naux, nbf)
    raw = rng.normal(size=(nk, nbf, nbf)) + 1j * rng.normal(size=(nk, nbf, nbf))
    densities = raw + raw.conj().transpose(0, 2, 1)
    weights = np.array([.1, .2, .3, .4])
    rows = [3, 0, 2]  # preserve requested wedge order
    actual = _k_from_signed_factors(
        cache, [_signed_gram_factors(d) for d in densities], weights, rows,
        nbasis=nbf, workspace_byte_cap=cap,
    )
    expected = [sum(weights[j] * sum(f @ densities[j] @ f.conj().T
                                   for f in cache[i, j]) for j in range(nk))
                for i in rows]
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=0)


def test_k_parallel_exchange_workspace_is_shared(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor as RealPool
    from vibeqc import _vibeqc_core as core
    import vibeqc.periodic_k_gdf as driver

    monkeypatch.setattr(core, 'get_num_threads', lambda: 64)
    rng = np.random.default_rng(664)
    nk, nbf, naux, cap = 4, 3, 19, 2**20
    cache = _random_cderi_cache(rng, nk, naux, nbf)
    calls, teams = [], []
    planner = driver._exchange_auxiliary_panel_rank

    def record_rank(a, n, per_aux, budget):
        calls.append((budget, 32*n*n + per_aux))
        return planner(a, n, per_aux, budget)

    def pool(*, max_workers):
        teams.append(max_workers)
        return RealPool(max_workers=max_workers)

    monkeypatch.setattr(driver, '_exchange_auxiliary_panel_rank', record_rank)
    monkeypatch.setattr('concurrent.futures.ThreadPoolExecutor', pool)
    driver._k_from_signed_factors(
        cache, [[(1., np.eye(nbf))] for _ in range(nk)], np.full(nk, 1/nk),
        range(nk), nbasis=nbf, workspace_byte_cap=cap,
    )
    assert teams == [nk]
    assert calls and all(minimum <= budget for budget, minimum in calls)
    assert max(budget for budget, _ in calls) * teams[0] <= cap


@pytest.mark.parametrize("method", ["rhf", "rks"])
@pytest.mark.parametrize("fit", ["compcell", "mdf"])
@pytest.mark.parametrize("explicit", [False, True])
def test_gamma_delegate_preserves_fit_controls(monkeypatch, method, fit, explicit):
    """Capture actual public delegation before the target performs any work."""
    import inspect
    from contextlib import nullcontext
    from types import SimpleNamespace
    from vibeqc import guess, pbc_gdf, periodic_k_gdf as driver

    options = vq.PeriodicRHFOptions() if method == "rhf" else vq.PeriodicKSOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    if method == "rks":
        options.functional = "pbe"
    system = SimpleNamespace(dim=3, multiplicity=1, n_electrons=lambda: 2,
                             unit_cell=[SimpleNamespace(Z=2)], unit_cell_molecule=lambda: None)
    basis = SimpleNamespace(nbasis=2)
    monkeypatch.setattr(driver, "_gdf_density_return_finalizer", lambda *a: lambda result: result)
    monkeypatch.setattr(driver, "_options_or_default", lambda opts, **k: opts)
    monkeypatch.setattr(driver, "_coerce_periodic_driver_guess", lambda *a, **k: vq.InitialGuess.HCORE)
    monkeypatch.setattr(driver, "periodic_guess_capabilities", lambda *a, **k: ())
    monkeypatch.setattr(guess, "guess_ecp_context", lambda *a, **k: None)
    monkeypatch.setattr(guess, "validate_guess_ecp", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_periodic_ecp_context", lambda *a: None)
    monkeypatch.setattr(driver, "Functional", lambda *a: SimpleNamespace(is_external=False))
    monkeypatch.setattr(driver, "_gamma_kmesh_info", lambda *a: SimpleNamespace())
    monkeypatch.setattr(driver, "_gamma_restart_options", lambda *a: nullcontext())
    monkeypatch.setattr(driver, "resolve_progress", lambda *a, **k: SimpleNamespace(info=lambda *a: None))
    controls = dict(compcell_eta=.7, apply_aft_correction=False,
                    aft_ft_convention="libcint", aft_precision=1e-7,
                    rcut_strategy="flat", rcut_precision=1e-6)
    signature = inspect.signature(pbc_gdf.run_pbc_gdf_rhf)
    captured = {}
    class Delegated(RuntimeError):
        pass
    def capture(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        captured.update(bound.arguments)
        raise Delegated
    monkeypatch.setattr(pbc_gdf, "run_pbc_gdf_rhf", capture)
    run = getattr(driver, f"run_k{method}_periodic_gdf")
    with pytest.raises(Delegated):
        run(system, basis, (1, 1, 1), options, gdf_method=fit,
            **(controls if explicit else {}))
    for name, value in controls.items():
        assert captured[name] == (value if explicit else signature.parameters[name].default)
    assert captured["gdf_method"] == fit
    assert captured["functional"] == ("pbe" if method == "rks" else None)
    assert captured["options"] is options
    assert captured["exxdiv"] == "ewald"
