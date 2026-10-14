"""Direct chi hybrid assembly contracts with deterministic integral doubles.

These source-isolated algebra tests do not import the native extension or run
SCF. Native support convergence and independent acceptance remain separate.
"""
from __future__ import annotations

import ast
from dataclasses import MISSING, dataclass, field, fields
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "python" / "vibeqc"


class _WithoutImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None

    def visit_Import(self, node):
        return None


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert {node.name for node in nodes} == set(names)
    module = _WithoutImports().visit(ast.Module(body=nodes, type_ignores=[]))
    module.body.insert(0, ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class _Lattice:
    def __init__(self, blocks, cells=None):
        self.blocks = [np.asarray(block).copy() for block in blocks]
        self.nbf = self.blocks[0].shape[0]
        self.cells = cells or [SimpleNamespace(index=(0, 0, 0), r_cart=np.zeros(3))]

    def set_block(self, index, block):
        self.blocks[index] = np.asarray(block).copy()


def _contract(density, cells, blocks, **kwargs):
    return float(sum(np.einsum("ij,ji->", d, f).real
                     for d, f in zip(density.blocks, blocks)))


@pytest.fixture
def fock(monkeypatch):
    module = ModuleType("_chi_direct_fock_contract")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = module.__dict__
    ns.update(np=np, math=math, dataclass=dataclass, field=field,
              SimpleNamespace=SimpleNamespace,
              core=SimpleNamespace(periodic_jk_coulomb_selection=True),
              _vibeqc_core=SimpleNamespace())
    _definitions(_SOURCE / "pbc_bipole_fock.py", {
        "BipoleFockContext", "BipoleScreenedExchangeExecution",
        "BipoleChiScreenedExchangeExecution", "BipoleRestrictedFockBuild",
        "BipoleUnrestrictedFockBuild", "split_k_density_list",
        "_screened_exchange_lattice", "_screened_k_blocks_for_cells",
        "_same_lattice_density", "_reuse_paired_spin_jk", "_sr_workspace_for", "_can_pair_sr",
        "_add_screened_exchange_restricted", "_add_screened_exchange_unrestricted",
        "build_bipole_restricted_fock", "build_bipole_unrestricted_fock",
    }, ns)
    _definitions(_SOURCE / "bipole_fock_ewald.py", {"exchange_q0_gauge_constant"}, ns)
    ns.update(
        _cell_key=lambda cell: tuple(cell.index),
        home_cell_block=lambda d: d.blocks[0].copy(),
        _lattice_contract_blocks=_contract,
        _lattice_contract=lambda d, f, **k: _contract(d, f.cells, f.blocks),
        make_lattice_matrix_set=lambda n, cells, blocks: _Lattice(blocks, cells),
        _make_lms_beta=lambda n, cells, blocks: _Lattice(blocks, cells),
        compute_rho_hat_from_k_density=lambda *a: None,
        compute_J_long_range_real_space_blocks=lambda *a, **k: [np.zeros((2, 2))],
        compute_K_long_range_gamma=lambda cache, d: 2.0 * d,
        compute_K_long_range_mesh=lambda cache, ks, w, channels: [
            [2.0 * d for d in ds] for ds in channels],
    )
    calls = []
    def integrals(basis, system, opts, density, omega, extent, **kwargs):
        calls.append((float(omega), density, kwargs))
        j = 3.0 * density.blocks[0] if kwargs.get("compute_coulomb", True) else np.zeros_like(density.blocks[0])
        return SimpleNamespace(J=_Lattice([j], density.cells),
                               K=_Lattice([(1.0 + omega) * density.blocks[0]], density.cells))
    ns["_sr_image_padded_jk"] = integrals
    module.integral_calls = calls
    return module


def _context(fock, n_k=1):
    kwargs = {item.name: None for item in fields(fock.BipoleFockContext)
              if item.default is MISSING and item.default_factory is MISSING}
    # Required context fields; optional infrastructure stays disabled.
    kwargs.update(basis=SimpleNamespace(nbasis=2),
        system=SimpleNamespace(lattice=np.eye(3)),
        lat_opts_2e=SimpleNamespace(pair_complete_1e=False),
        use_ewald_j_split=True, exchange_split_active=True, n_k=n_k,
        omega_used=.4, ewald_precision=1e-10, ewald_cell_volume=50., n_elec=2,
        xi_madelung=.31, j_lr_cache=object(), x_lr_cache=object(),
        incremental_jk=None, rep_cell_indices=None, fock_sym_map=None,
        mp_config=SimpleNamespace(enabled=False),
        s_lat=_Lattice([np.eye(2)]), s_k_list=[np.eye(2)] * n_k,
        k_points=np.array([[2*np.pi*i/n_k, 0, 0] for i in range(n_k)]),
        weights=np.full(n_k, 1./n_k), k_points_full=[], weights_full=np.ones(n_k)/n_k,
        ir_mapping=np.array([], dtype=int), bvk_mesh=(n_k, 1, 1), n_occ=1,
        plog=None, sr_image_extent=8.0)
    return fock.BipoleFockContext(**kwargs)


def _assembly(c_full, c_sr):
    return SimpleNamespace(c_full=c_full, c_sr=c_sr, omega_screen=.2,
                           is_screened=c_sr != 0.)


def _build(fock, spin, ctx, density, exx, **kwargs):
    if spin == "rks":
        return fock.build_bipole_restricted_fock(ctx, density,
            coeffs_for_rho=None, alpha_hf=exx.c_full,
            exchange_assembly=exx, **kwargs)
    return fock.build_bipole_unrestricted_fock(ctx, density, density,
        n_alpha=1, n_beta=1, coeffs_alpha_for_rho=None, coeffs_beta_for_rho=None,
        alpha_hf=exx.c_full, exchange_assembly=exx, **kwargs)


@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("c_full,c_sr", [(0., .25), (.25, 0.), (.65, -.46), (1., -1.)])
def test_direct_kernel_arms_close_fock_energy_and_full_only_madelung(fock, spin, c_full, c_sr):
    ctx = _context(fock)
    density = _Lattice([np.array([[1.1, .13], [.13, .4]])])
    exx = _assembly(c_full, c_sr)
    result = _build(fock, spin, ctx, density, exx, _chi_exchange=True,
                    use_incremental=False)
    d = density.blocks[0]
    gauge = ctx.xi_madelung - np.pi/(ctx.omega_used**2 * ctx.ewald_cell_volume)
    full_k = (1. + ctx.omega_used + 2. + gauge) * d
    screened_k = (1. + exx.omega_screen) * d
    k = c_full * full_k + c_sr * screened_k
    bg = -np.pi*ctx.n_elec/(ctx.omega_used**2 * ctx.ewald_cell_volume)
    factor = .5 if spin == "rks" else 1.
    j = (3. if spin == "rks" else 6.) * d + bg * np.eye(2)
    if spin == "rks":
        actual = result.f2e_real.blocks[0].astype(complex)
        if result.k_corr_per_k is not None:
            actual -= .5 * result.k_corr_per_k[0]
        expected_energy = -.25 * np.trace(d @ k)
    else:
        actual = result.f2e_alpha_real.blocks[0].astype(complex)
        if result.k_corr_alpha_per_k is not None:
            actual -= result.k_corr_alpha_per_k[0]
        expected_energy = -np.trace(d @ k)
    np.testing.assert_allclose(actual, j - factor*k, atol=1e-13)
    assert result.e_exchange == pytest.approx(expected_energy)
    assert sum(result.e_hf_exchange_components.values()) == result.e_exchange
    sr_factor = -.25 if spin == "rks" else -1.
    assert result.e_hf_exchange_components["erfc"] == pytest.approx(
        sr_factor*c_sr*np.trace(d@screened_k))
    assert result.e_hf_exchange_components["full_zero"] == pytest.approx(
        sr_factor*c_full*ctx.xi_madelung*np.trace(d@d))
    finite_factor = -.25 if spin == "rks" else -1.
    assert (result.e_exchange_finite_size or 0.) == pytest.approx(
        finite_factor*c_full*ctx.xi_madelung*np.trace(d@d))
    expected_kernels = {ctx.omega_used} | ({exx.omega_screen} if c_sr else set())
    assert {omega for omega, _, _ in fock.integral_calls} == expected_kernels
    if c_sr:
        assert result.screened_exchange_execution.c_sr == c_sr
        assert result.screened_exchange_execution.schema == "vibeqc.chi.screened-exchange-execution/v1"
    else:
        assert result.screened_exchange_execution is None


@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_mixed_exchange_stays_closed_for_legacy_callers(fock, spin):
    with pytest.raises(NotImplementedError, match="nonzero full-range"):
        _build(fock, spin, _context(fock), _Lattice([np.eye(2)]), _assembly(1., -1.))
    assert fock.integral_calls == []
    with pytest.raises(ValueError, match="finite and positive"):
        fock.BipoleScreenedExchangeExecution(-1., .2)


@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_exact_rebuild_reseeds_base_and_rebuilds_screened_arm(fock, spin):
    ctx = _context(fock)
    seen = []
    def seed(*args):
        seen.append(args)
    accumulator = SimpleNamespace(seed=seed)
    kwargs = dict(_chi_exchange=True, use_incremental=False, reseed_incremental=True)
    if spin == "rks":
        ctx.incremental_jk = accumulator
    else:
        kwargs.update(incremental_jk_alpha=accumulator, incremental_jk_beta=accumulator)
    _build(fock, spin, ctx, _Lattice([np.eye(2)]), _assembly(.65, -.46), **kwargs)
    assert len(seen) == (1 if spin == "rks" else 2)
    # The equal spin densities share the screened arm within this rebuild.
    assert sum(omega == .2 for omega, _, _ in fock.integral_calls) == 1
    assert all(kwargs.get("compute_coulomb", True) is False
               for omega, _, kwargs in fock.integral_calls if omega == .2)


def test_custom_unrestricted_accumulators_keep_the_build_callback_contract(fock):
    # Imports are stripped by this algebra fixture. Supply a sentinel for the
    # native accumulator class; custom objects must use their own build method.
    fock.IncrementalJK = type("NativeAccumulator", (), {})
    seen = []
    def build(density, callback):
        seen.append(density)
        result = callback(density)
        return result.J, result.K
    density = _Lattice([np.eye(2)])
    result = _build(fock, "uks", _context(fock), density, _assembly(.25, 0.),
        use_incremental=True,
        incremental_jk_alpha=SimpleNamespace(build=build),
        incremental_jk_beta=SimpleNamespace(build=build))
    assert seen == [density, density]
    assert len(fock.integral_calls) == 1
    np.testing.assert_array_equal(result.f2e_alpha_real.blocks, result.f2e_beta_real.blocks)


@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_complex_character_density_drives_full_exchange_and_zero_mode(fock, spin):
    ctx = _context(fock, n_k=3)
    ds = np.array([np.diag([1., .3]), [[.8, .2j], [-.2j, .5]],
                   [[.8, -.2j], [.2j, .5]]])
    density = _Lattice([np.diag([.9, .4])])
    # This selected support has no complete torus; a reconstruction would be
    # both unavailable and inconsistent with the retained character density.
    def forbidden(*args):
        raise AssertionError("must retain the actual complex character density")
    fock.bvk_torus_density_matrices = forbidden
    supplied = ds if spin == "rks" else np.array([ds, ds])
    exx = _assembly(.65, -.46)
    result = _build(fock, spin, ctx, density, exx,
                   _chi_exchange=True, _chi_density_k=supplied)
    gauge = ctx.xi_madelung - np.pi/(ctx.omega_used**2*ctx.ewald_cell_volume*3)
    corrections = result.k_corr_per_k if spin == "rks" else result.k_corr_alpha_per_k
    np.testing.assert_allclose(corrections, exx.c_full*(2.+gauge)*ds)
    factor = -.25 if spin == "rks" else -1.
    expected = factor*exx.c_full*ctx.xi_madelung*sum(np.trace(d@d).real for d in ds)/3
    assert result.e_exchange_finite_size == pytest.approx(expected)


@pytest.mark.parametrize("invalid", ["shape", "finite", "hermitian", "time_reversal"])
def test_retained_density_rejects_inconsistent_direct_state(fock, invalid):
    ctx = _context(fock, n_k=3)
    ds = np.tile(np.eye(2), (3, 1, 1)).astype(complex)
    if invalid == "shape":
        ds = ds[:2]
    elif invalid == "finite":
        ds[0, 0, 0] = np.nan
    elif invalid == "hermitian":
        ds[1, 0, 1] = .2j
    else:
        ds[1, 0, 1], ds[1, 1, 0] = .2j, -.2j
    with pytest.raises((ValueError, NotImplementedError)):
        fock.split_k_density_list(ctx, object(), retained_density_k=ds)


@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_driver_selects_chi_assembly_and_passes_authoritative_density(fock, spin):
    tree = ast.parse((_SOURCE / f"pbc_bipole_{spin}.py").read_text())
    driver = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == f"run_pbc_bipole_{spin}")
    resolve = next(node for node in driver.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "exx"
                           for target in node.targets))
    legacy = _assembly(.25, 0.)
    chi = _assembly(.65, -.46)
    ns = dict(_chi_xc=SimpleNamespace(exchange_assembly=chi), func=object(),
              resolve_periodic_exchange=lambda *a, **k: legacy)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[resolve], type_ignores=[])),
                 "<direct assembly selection>", "exec"), ns)
    assert ns["exx"] is chi
    ns["_chi_xc"] = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[resolve], type_ignores=[])),
                 "<legacy assembly selection>", "exec"), ns)
    assert ns["exx"] is legacy
    nested = next(node for node in driver.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_build_fock_for_density")
    call = next(node for node in nested.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "fb" for t in node.targets))
    ds = np.tile(np.eye(2), (3, 1, 1))
    recorded = []
    ns.update(_chi_xc=object(), _fock_ctx=object(), density=object(),
              D_alpha=object(), D_beta=object(), coeffs_for_rho=None,
              coeffs_alpha_for_rho=None, coeffs_beta_for_rho=None,
              n_alpha=1, n_beta=1, incremental_jk_alpha=None,
              incremental_jk_beta=None, incremental_jk_total=None,
              alpha_hf=.65, exx=chi, patom_hf_like=False,
              xc_density_k=ds, use_incremental=False, reseed_incremental=True,
              build_bipole_restricted_fock=lambda *a, **k: recorded.append(k),
              build_bipole_unrestricted_fock=lambda *a, **k: recorded.append(k))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[call], type_ignores=[])),
                 "<authoritative hybrid density>", "exec"), ns)
    assert recorded[0]["_chi_exchange"] is True
    assert recorded[0]["_chi_density_k"] is ds
    assert recorded[0]["exchange_assembly"] is chi
    assert recorded[0]["use_incremental"] is False
    assert recorded[0]["reseed_incremental"] is True


def test_unrestricted_exchange_has_no_cross_spin_contraction(fock):
    ctx = _context(fock)
    a = np.array([[.9, .14], [.14, .3]])
    b = np.array([[.4, -.11], [-.11, .6]])
    exx = _assembly(.65, -.46)
    def build(da, db):
        return fock.build_bipole_unrestricted_fock(ctx, _Lattice([da]), _Lattice([db]),
            n_alpha=1, n_beta=1, coeffs_alpha_for_rho=None, coeffs_beta_for_rho=None,
            alpha_hf=exx.c_full, exchange_assembly=exx, _chi_exchange=True,
            use_incremental=False)
    result = build(a, b)
    gauge = ctx.xi_madelung - np.pi/(ctx.omega_used**2 * ctx.ewald_cell_volume)
    coefficient = exx.c_full*(1.+ctx.omega_used+2.+gauge)+exx.c_sr*(1.+exx.omega_screen)
    assert result.e_exchange == pytest.approx(-.5*coefficient*(np.trace(a@a)+np.trace(b@b)))
    da = np.array([[.2, -.08], [-.08, -.1]])
    db = np.array([[-.15, .06], [.06, .12]])
    step = 1e-5
    numerical = (build(a+step*da, b+step*db).e_exchange
                 - build(a-step*da, b-step*db).e_exchange)/(2*step)
    analytic = -coefficient*(np.trace(da@a)+np.trace(db@b))
    assert numerical == pytest.approx(analytic, abs=2e-11)


def test_screened_kernel_masks_only_private_pair_density(fock):
    ctx = _context(fock)
    ctx.fock_sym_map = SimpleNamespace(pair_resolved=True)
    ctx.rep_cell_indices = [0]
    density = _Lattice([np.eye(2)])
    selected = _Lattice([np.diag([1., 0.])])
    seen = []
    fock._pair_resolved_density_for_sr = lambda *a: selected
    def reduced(*args, **kwargs):
        seen.append(args[3])
        return SimpleNamespace(K=selected)
    fock.build_jk_reduced_symmetrized = reduced
    assert fock._screened_exchange_lattice(ctx, density, .2, _chi_exchange=True) is selected
    assert seen == [selected]
    fock._screened_exchange_lattice(ctx, density, .2)
    assert seen == [selected, density]
    np.testing.assert_array_equal(density.blocks[0], np.eye(2))


@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("symmetry", [False, True])
def test_screened_fold_exports_each_vector_once_per_pass(fock, monkeypatch, spin, symmetry):
    class CountedLattice:
        def __init__(self, values):
            self.values = [v.copy() for v in values]
            self.cells = cells
            self.reads = 0

        @property
        def blocks(self):
            self.reads += 1
            return [v.copy() for v in self.values]

        def set_block(self, i, block):
            self.values[i] = block.copy()

    cells = [SimpleNamespace(index=(i, 0, 0), r_cart=np.array([i, 0., 0.])) for i in range(3)]
    values = [np.array([[i + 2., .4], [.2, i + 3.]]) for i in range(3)]
    k_blocks = [np.array([[i + .1, .05], [.03, .2]]) for i in range(3)]
    alpha, beta = CountedLattice(values), CountedLattice(values)
    density = _Lattice([np.eye(2) for _ in cells], cells)
    base = SimpleNamespace(f2e_real=alpha, f2e_alpha_real=alpha, f2e_beta_real=beta,
                           e_exchange=None, e_exchange_finite_size=None)
    monkeypatch.setattr(fock, "_screened_exchange_lattice", lambda *a, **k: object())
    monkeypatch.setattr(fock, "_screened_k_blocks_for_cells", lambda *a: k_blocks)
    factor = .5 if spin == "rks" else 1.
    expected = [v - factor * k for v, k in zip(values, k_blocks)]
    symmetry_calls = []

    def project(blocks, mapping, **kwargs):
        # The second export must observe the just-written screened correction.
        for actual, want in zip(blocks, expected):
            np.testing.assert_array_equal(actual, want)
        blocks[:] = [.5 * (b + b.T) for b in blocks]
        symmetry_calls.append(None)

    monkeypatch.setattr(fock, "symmetrize_fock_blocks", project, raising=False)
    ctx = SimpleNamespace(fock_sym_map=object() if symmetry else None)
    if spin == "rks":
        fock._add_screened_exchange_restricted(ctx, density, base, _assembly(0., .25))
    else:
        fock._add_screened_exchange_unrestricted(ctx, density, density, base, _assembly(0., .25))
    used = [alpha] if spin == "rks" else [alpha, beta]
    for lattice in used:
        assert lattice.reads == 1 + int(symmetry)
        for actual, want in zip(lattice.values, expected):
            np.testing.assert_array_equal(actual, .5 * (want + want.T) if symmetry else want)
    assert len(symmetry_calls) == len(used) * int(symmetry)
