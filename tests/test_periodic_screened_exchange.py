"""Unit tests for the shared periodic CAM exchange resolution.

Pins the assembly table verified 2026-07-10 (the original investigation):

    functional | c_full (a+b) | c_sr (-b) | omega_screen
    -----------+--------------+-----------+-------------
    PBE0       | 0.25         | 0         | --
    HSE06      | 0.000        | +0.25     | 0.11
    wB97X      | 1.000        | (fails closed)
    CAM-B3LYP  | 0.650        | (fails closed)
    LC-wPBE    | 1.000        | (fails closed)
"""

import pytest

from vibeqc._vibeqc_core import Functional
from vibeqc.periodic_screened_exchange import (
    PeriodicExchangeAssembly,
    resolve_periodic_exchange,
)


def test_pure_functional_needs_no_exchange():
    asm = resolve_periodic_exchange(Functional("pbe", 1), where="test")
    assert asm == PeriodicExchangeAssembly(0.0, 0.0, 0.0)
    assert not asm.needs_exchange
    assert not asm.is_screened


def test_global_hybrid_pbe0_is_full_range_only():
    asm = resolve_periodic_exchange(Functional("pbe0", 1), where="test")
    assert asm.c_full == pytest.approx(0.25)
    assert asm.c_sr == 0.0
    assert asm.omega_screen == 0.0
    assert asm.needs_exchange
    assert not asm.is_screened


def test_hse06_is_pure_short_range_erfc():
    # HSE06: cam_alpha = 0.25, cam_beta = -0.25, rsh_omega = 0.11
    # (Krukau 2006) => K_HF = 0.25 * K_erfc(0.11), NO full-range arm.
    asm = resolve_periodic_exchange(Functional("hse06", 1), where="test")
    assert asm.c_full == pytest.approx(0.0, abs=1e-12)
    assert asm.c_sr == pytest.approx(0.25)
    assert asm.omega_screen == pytest.approx(0.11)
    assert asm.needs_exchange
    assert asm.is_screened


def test_none_functional_means_pure_hf():
    asm = resolve_periodic_exchange(None, where="test")
    assert asm.c_full == 1.0
    assert asm.c_sr == 0.0
    assert not asm.is_screened


@pytest.mark.parametrize("name", ["wb97x", "cam-b3lyp", "lc-wpbe"])
def test_long_range_heavy_rsh_fails_closed(name):
    func = Functional(name, 1)
    assert bool(func.is_range_separated)
    with pytest.raises(NotImplementedError, match="full-range exact-exchange arm"):
        resolve_periodic_exchange(func, where="test_caller")


def test_error_message_names_the_caller():
    with pytest.raises(NotImplementedError, match="run_some_driver:"):
        resolve_periodic_exchange(
            Functional("cam-b3lyp", 1), where="run_some_driver"
        )


def test_hse06_assembly_reproduces_erfc_identity():
    # The CAM convention K_op = a/r + b*erf(w r)/r rewritten via
    # erf = 1 - erfc must give c_full = a + b and c_sr = -b exactly.
    func = Functional("hse06", 1)
    asm = resolve_periodic_exchange(func, where="test")
    a, b = float(func.cam_alpha), float(func.cam_beta)
    assert asm.c_full == pytest.approx(a + b, abs=1e-12)
    assert asm.c_sr == pytest.approx(-b, abs=1e-12)


# ===================================================================
# SCF gates (H2 in a 20-bohr box / sto-3g -- molecular-limit regime
# where the Γ and multi-k conventions coincide).
# ===================================================================

import functools

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import PeriodicKSOptions
from vibeqc.periodic_screened_exchange import (
    build_exchange_gamma,
    reject_unscreened_range_separated,
)

_BOX = 20.0


def _h2_box(multiplicity=1):
    c = _BOX / 2.0
    sys3 = vq.PeriodicSystem(
        3,
        np.eye(3) * _BOX,
        [vq.Atom(1, [c - 0.35, c, c]), vq.Atom(1, [c + 0.35, c, c])],
        multiplicity=multiplicity,
    )
    basis = vq.make_basis(sys3.unit_cell_molecule(), "sto-3g")
    return sys3, basis


def _ks_opts(functional: str) -> PeriodicKSOptions:
    opts = PeriodicKSOptions()
    opts.functional = functional
    return opts


@functools.lru_cache(maxsize=None)
def _gamma_ewald(functional: str):
    from vibeqc.periodic_rks_ewald import run_rks_periodic_gamma_ewald3d

    sys3, basis = _h2_box()
    r = run_rks_periodic_gamma_ewald3d(
        sys3, basis, options=_ks_opts(functional), progress=False
    )
    assert r.converged
    return r.energy, r.e_hf_exchange


# A Gamma-only (1,1,1) run carries a Born-von-Karman-periodic density,
# D(g) = D(Gamma) on every lattice cell, so the screened exchange between
# the home-cell pair densities and their 20-bohr images is part of the
# periodic sum: erfc(0.11 r)/r is 9e-5 at r = 20 bohr, and on this fixture
# the first image shell contributes -1.66e-4 Ha to ``e_hf_exchange`` (the
# 28.3-bohr shell -1.6e-6, 34.6 bohr -6.8e-9, 40 bohr -3.6e-11, 44.7 bohr
# -1.1e-12; nothing beyond at 1e-15). The multi-k Ewald drivers sum the
# real-space K_erfc only out to ``lattice_opts.cutoff_bohr`` (15 bohr by
# default), which truncates every image shell here. The physical BIPOLE
# pair domain (#316) bounds exchange by the pair-interaction radius that
# ``sr_image_precision`` resolves (47.6 bohr at the 1e-6 production
# default), so the production route includes them (#501). A reference
# summed to 42 bohr holds exactly the shells the production domain keeps:
# pair centres sit within 0.7 bohr of the lattice points, so 42 bohr lies
# between the 40- and 44.7-bohr shells for both the index ball and the
# pair-interaction rule, and the omitted tail is below 2e-12.
_CONVERGED_K_CUTOFF_BOHR = 42.0


@functools.lru_cache(maxsize=None)
def _multi_k_ewald_result(
    functional: str,
    uks: bool = False,
    cutoff_bohr: float | None = None,
):
    from vibeqc.periodic_rks_multi_k_ewald import (
        run_rks_periodic_multi_k_ewald3d,
    )
    from vibeqc.periodic_uks_multi_k_ewald import (
        run_uks_periodic_multi_k_ewald3d,
    )

    sys3, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sys3, [1, 1, 1])
    fn = run_uks_periodic_multi_k_ewald3d if uks else run_rks_periodic_multi_k_ewald3d
    opts = _ks_opts(functional)
    if cutoff_bohr is not None:
        # The real-space K_erfc image sum of the Ewald drivers stops at
        # the lattice cutoff (see _CONVERGED_K_CUTOFF_BOHR).
        opts.lattice_opts.cutoff_bohr = float(cutoff_bohr)
    r = fn(sys3, basis, kmesh, opts, progress=False)
    assert r.converged
    return r


def _multi_k_ewald(functional: str, uks: bool = False):
    r = _multi_k_ewald_result(functional, uks)
    return float(getattr(r, "energy_per_cell", getattr(r, "energy", None)))


@functools.lru_cache(maxsize=None)
def _bipole_result(
    functional: str,
    uks: bool = False,
    sr_image_precision: float | None = 1e-6,
    eri_interaction_cutoff_bohr: float | None = None,
):
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    sys3, basis = _h2_box()
    kmesh = vq.monkhorst_pack(sys3, [1, 1, 1])
    fn = run_pbc_bipole_uks if uks else run_pbc_bipole_rks
    opts = _ks_opts(functional)
    if eri_interaction_cutoff_bohr is not None:
        # Explicit pair-interaction reach of the physical pair domain,
        # used with ``sr_image_precision=None`` to pin what the padding
        # changes (see _CONVERGED_K_CUTOFF_BOHR).
        opts.lattice_opts.eri_interaction_cutoff_bohr = float(
            eri_interaction_cutoff_bohr
        )
    r = fn(
        sys3,
        basis,
        kmesh,
        opts,
        sr_image_precision=sr_image_precision,
        progress=False,
    )
    assert r.converged
    return r


def _bipole(
    functional: str,
    uks: bool = False,
    sr_image_precision: float | None = 1e-6,
):
    r = _bipole_result(functional, uks, sr_image_precision)
    return float(r.energy)


def _screened_fock_ctx(**overrides):
    """A real ``BipoleFockContext`` carrying placeholder required fields.

    Built from the dataclass itself, so a field added to
    ``BipoleFockContext`` with a default arrives here carrying that
    production default. A hand-listed ``SimpleNamespace`` stub cannot:
    when ``perf(aiccm-b): farm direct output cells across MPI``
    (4ead07bcc) added ``output_cell_farming_task_kind`` /
    ``output_cell_farming_strategy``, the stub this replaced went stale
    and the screened-K traversal raised ``AttributeError`` on fields
    every real context already had.
    """
    import dataclasses

    from vibeqc.pbc_bipole_fock import BipoleFockContext

    placeholders = {
        f.name: None
        for f in dataclasses.fields(BipoleFockContext)
        if f.init
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    }
    return BipoleFockContext(**{**placeholders, **overrides})


@pytest.mark.parametrize("k_only", [False, True])
def test_bipole_screened_k_uses_padded_internal_domain(monkeypatch, k_only):
    """The dedicated HSE K traversal honors M5's resolved ket-image ball."""
    from types import SimpleNamespace

    import vibeqc.pbc_bipole_fock as bipole_fock
    from vibeqc import _vibeqc_core as core

    monkeypatch.setattr(core, "periodic_jk_coulomb_selection", k_only, raising=False)

    calls = []
    sentinel = object()

    def _padded(basis, system, opts, density, omega, extent, **kwargs):
        calls.append((basis, system, opts, density, omega, extent))
        assert kwargs.get("compute_coulomb", True) is (not k_only)
        return SimpleNamespace(K=sentinel)

    monkeypatch.setattr(bipole_fock, "_sr_image_padded_jk", _padded)
    monkeypatch.setattr(
        bipole_fock,
        "build_jk_2e_real_space",
        lambda *args, **kwargs: pytest.fail("unpadded K builder was used"),
    )
    ctx = _screened_fock_ctx(
        basis=object(),
        system=object(),
        lat_opts_2e=object(),
        rep_cell_indices=None,
        sr_image_extent=42.0,
        exact_zone_bohr=None,
    )
    density = object()
    assert (
        bipole_fock._screened_exchange_lattice(ctx, density, 0.11) is sentinel
    )
    assert calls == [
        (ctx.basis, ctx.system, ctx.lat_opts_2e, density, 0.11, 42.0)
    ]


@pytest.mark.parametrize("k_only", [False, True])
def test_paired_screened_k_without_padding_preserves_both_channels(monkeypatch, k_only):
    from types import SimpleNamespace
    import vibeqc.pbc_bipole_fock as fock
    from vibeqc import _vibeqc_core as core

    monkeypatch.setattr(core, "periodic_jk_coulomb_selection", k_only)
    cells = [object()]
    monkeypatch.setattr(fock, "_sr_internal_cells", lambda *args: cells)
    monkeypatch.setattr(fock, "_sr_workspace_for", lambda ctx: None)
    first, second = object(), object()
    outputs = (object(), object())
    def domains(*args, **kwargs):
        assert args[3] is first and args[4] is cells and args[5] == [0]
        assert kwargs["companion_density"] is second
        assert kwargs["compute_coulomb"] is False
        return tuple(SimpleNamespace(K=value) for value in outputs)
    monkeypatch.setattr(fock, "_build_jk_domains_output_cells_mpi", domains)
    monkeypatch.setattr(fock, "build_jk_2e_real_space",
        lambda *args: pytest.fail("paired exchange entered the single-density builder"))
    ctx = _screened_fock_ctx(lat_opts_2e=SimpleNamespace(pair_complete_1e=True))
    assert fock._screened_exchange_lattice(ctx, first, .11, companion_density=second) == outputs


def test_bipole_screened_k_forwards_the_declared_farming_contract(
    monkeypatch,
):
    """The screened-K traversal forwards the context's farming fields.

    A *production* screened context always carries the serial default
    (``task_kind=None``): ``run_pbc_bipole_rks`` raises
    ``NotImplementedError`` for ``farm_output_cells and exx.is_screened``
    because one single-phase execution record cannot attest both the
    base J and the separate screened-K traversal. The first case below
    pins exactly that -- the dedicated erfc-K build must not farm behind
    the driver's back.

    The second case pins the plumbing 4ead07bcc wired, which the future
    multi-phase schema will rely on. It is a statement about this
    helper, NOT a claim that a screened hybrid can be farmed today; the
    driver gate above is what makes that configuration unreachable.
    Nothing pinned either half when 4ead07bcc landed, so the only signal
    that this route touches those fields at all was the
    ``AttributeError`` it raised against a stale stub.
    """
    from types import SimpleNamespace

    import vibeqc.pbc_bipole_fock as bipole_fock

    seen = []

    def _padded(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(K=object())

    monkeypatch.setattr(bipole_fock, "_sr_image_padded_jk", _padded)
    base = dict(
        basis=object(),
        system=object(),
        lat_opts_2e=object(),
        rep_cell_indices=None,
        sr_image_extent=42.0,
        exact_zone_bohr=None,
    )

    bipole_fock._screened_exchange_lattice(
        _screened_fock_ctx(**base), object(), 0.11
    )
    assert seen[-1]["output_cell_farming_task_kind"] is None
    assert seen[-1]["output_cell_farming_strategy"] == "cyclic"

    bipole_fock._screened_exchange_lattice(
        _screened_fock_ctx(
            **base,
            output_cell_farming_task_kind="direct-eri-output-cell",
            output_cell_farming_strategy="block",
        ),
        object(),
        0.11,
    )
    assert (
        seen[-1]["output_cell_farming_task_kind"]
        == "direct-eri-output-cell"
    )
    assert seen[-1]["output_cell_farming_strategy"] == "block"


def test_bipole_screened_execution_is_emitted_after_energy_contraction(
    monkeypatch,
):
    """A failed screened-K contraction must not leave a success marker."""
    from types import SimpleNamespace

    import vibeqc.pbc_bipole_fock as bipole_fock

    class _FockBlocks:
        def __init__(self):
            self.cells = [object()]
            self.blocks = [np.array([[2.0]])]

        def set_block(self, index, block):
            self.blocks[index] = np.asarray(block)

    base = bipole_fock.BipoleRestrictedFockBuild(
        f2e_real=_FockBlocks(),
        k_corr_per_k=None,
    )
    ctx = _screened_fock_ctx(fock_sym_map=None)
    exx = SimpleNamespace(c_sr=0.25, omega_screen=0.11)
    monkeypatch.setattr(
        bipole_fock,
        "_screened_exchange_lattice",
        lambda *args: object(),
    )
    monkeypatch.setattr(
        bipole_fock,
        "_screened_k_blocks_for_cells",
        lambda *args: [np.array([[1.0]])],
    )

    def _failed_contraction(*args, **kwargs):
        raise RuntimeError("contraction failed")

    monkeypatch.setattr(
        bipole_fock,
        "_lattice_contract_blocks",
        _failed_contraction,
    )
    with pytest.raises(RuntimeError, match="contraction failed"):
        bipole_fock._add_screened_exchange_restricted(
            ctx,
            object(),
            base,
            exx,
        )
    assert base.screened_exchange_execution is None

    monkeypatch.setattr(
        bipole_fock,
        "_lattice_contract_blocks",
        lambda *args, **kwargs: 4.0,
    )
    result = bipole_fock._add_screened_exchange_restricted(
        ctx,
        object(),
        base,
        exx,
    )
    execution = result.screened_exchange_execution
    assert execution is not None
    assert execution.schema == (
        "vibeqc.pbc-bipole.screened-exchange-execution/v1"
    )
    assert execution.assembly == "short-range-direct"
    assert execution.c_sr == pytest.approx(0.25)
    assert execution.omega_screen_bohr_inv == pytest.approx(0.11)


def test_bipole_screened_execution_is_frozen_and_rejects_invalid_values():
    from dataclasses import FrozenInstanceError

    from vibeqc.pbc_bipole_fock import BipoleScreenedExchangeExecution

    execution = BipoleScreenedExchangeExecution(
        c_sr=0.25,
        omega_screen_bohr_inv=0.11,
    )
    with pytest.raises(FrozenInstanceError):
        execution.c_sr = 0.5

    for c_sr, omega in [
        (True, 0.11),
        (0.0, 0.11),
        (-0.25, 0.11),
        (float("nan"), 0.11),
        (10**1000, 0.11),
        (0.25, 0.0),
        (0.25, float("inf")),
    ]:
        with pytest.raises(ValueError, match="finite and positive"):
            BipoleScreenedExchangeExecution(
                c_sr=c_sr,
                omega_screen_bohr_inv=omega,
            )


@pytest.mark.parametrize("change", [
    None, "density", "tiny_density", "nan", "order", "index", "coordinate",
    "nbf", "missing_cell", "missing_block",
])
def test_screened_exchange_spin_reuse_requires_exact_lattice_density(change):
    from copy import deepcopy
    from types import SimpleNamespace

    from vibeqc.pbc_bipole_fock import _same_lattice_density

    left = SimpleNamespace(
        nbf=1,
        cells=[SimpleNamespace(index=np.array([i, 0, 0]),
                               r_cart=np.array([4.0 * i, 0.0, 0.0]))
               for i in range(2)],
        blocks=[np.array([[0.5]]), np.array([[0.1]])],
    )
    right = deepcopy(left)
    if change == "density":
        right.blocks[1][0, 0] *= -1
    elif change == "tiny_density":
        right.blocks[1][0, 0] = np.nextafter(0.1, 1.0)
    elif change == "nan":
        left.blocks[0][0, 0] = right.blocks[0][0, 0] = np.nan
    elif change == "order":
        right.cells.reverse()
        right.blocks.reverse()
    elif change == "index":
        right.cells[1].index[0] += 1
    elif change == "coordinate":
        right.cells[1].r_cart[0] = np.nextafter(4.0, 5.0)
    elif change == "nbf":
        right.nbf = 2
    elif change == "missing_cell":
        right.cells.pop()
    elif change == "missing_block":
        right.blocks.pop()
    assert _same_lattice_density(left, right) is (change is None)


@pytest.mark.parametrize("history", ["equal", "different", "reset"])
@pytest.mark.parametrize("with_exchange", [False, True])
def test_joint_incremental_build_keeps_independent_sources_and_resets(history, with_exchange):
    from types import SimpleNamespace
    from vibeqc.bipole_fock_ewald import IncrementalJK
    from vibeqc._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set

    system, _ = _h2_box()
    cells = list(direct_lattice_cells(system, .1))
    def lattice(value):
        return make_lattice_matrix_set(1, cells, [np.array([[value]])])
    def raw(density):
        value = density.blocks[0][0, 0]
        return SimpleNamespace(J=lattice(3 * value), K=lattice(2 * value) if with_exchange else None)
    first, second = IncrementalJK(), IncrementalJK()
    first.seed(lattice(.25), lattice(.75), lattice(.5) if with_exchange else None)
    previous = .125 if history == "different" else .25
    second.seed(lattice(previous), lattice(3 * previous), lattice(2 * previous) if with_exchange else None)
    if history == "reset":
        second._since_reset = second.reset_every
    calls = []
    def paired(left, right):
        calls.append((left.blocks[0][0, 0], right.blocks[0][0, 0]))
        return raw(left), raw(right)
    results = IncrementalJK.build_pair(first, second, lattice(.5), lattice(.75), paired)
    assert calls == [(.25, .75 if history == "reset" else .75 - previous)]
    for (j, k), value in zip(results, [.5, .75]):
        np.testing.assert_array_equal(j.blocks[0], [[3 * value]])
        if with_exchange:
            np.testing.assert_array_equal(k.blocks[0], [[2 * value]])
        else:
            assert k is None
    assert first._since_reset == 1
    assert second._since_reset == (0 if history == "reset" else 1)
    results[0][0].set_block(0, np.array([[99.]]))
    np.testing.assert_array_equal(first._J_acc[0], [[1.5]])
    np.testing.assert_array_equal(results[1][0].blocks[0], [[2.25]])


def test_joint_incremental_kernel_failure_preserves_both_histories():
    from vibeqc.bipole_fock_ewald import IncrementalJK
    from vibeqc._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set

    system, _ = _h2_box()
    cells = list(direct_lattice_cells(system, .1))
    density = make_lattice_matrix_set(1, cells, [np.array([[.5]])])
    first, second = IncrementalJK(), IncrementalJK()
    first.seed(density, density, density)
    def fail(*args):
        raise RuntimeError("kernel failed")
    with pytest.raises(RuntimeError, match="kernel failed"):
        IncrementalJK.build_pair(first, second, density, density, fail)
    assert first._since_reset == 0
    np.testing.assert_array_equal(first._J_acc[0], [[.5]])
    assert second._D_prev is None


@pytest.mark.parametrize("history", ["equal", "different", "reset"])
def test_paired_spin_jk_reuse_preserves_incremental_histories(history):
    from types import SimpleNamespace
    from vibeqc.bipole_fock_ewald import IncrementalJK
    from vibeqc.pbc_bipole_fock import _reuse_paired_spin_jk
    from vibeqc._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set

    system, _ = _h2_box()
    cells = list(direct_lattice_cells(system, 0.1))

    def lattice(value):
        return make_lattice_matrix_set(1, cells, [np.array([[value]])])

    calls = []

    def raw(density):
        value = float(density.blocks[0][0, 0])
        calls.append(value)
        return SimpleNamespace(J=lattice(3 * value), K=lattice(2 * value))

    alpha, beta = IncrementalJK(), IncrementalJK()
    alpha.seed(lattice(0.4), lattice(1.2), lattice(0.8))
    b_prev = 0.2 if history == "different" else 0.4
    beta.seed(lattice(b_prev), lattice(3 * b_prev), lattice(2 * b_prev))
    if history == "reset":
        beta._since_reset = beta.reset_every
    paired = _reuse_paired_spin_jk(raw)
    ja, ka = alpha.build(lattice(0.5), paired)
    jb, kb = beta.build(lattice(0.5), paired)
    assert len(calls) == (1 if history == "equal" else 2)
    for j, k in [(ja, ka), (jb, kb)]:
        np.testing.assert_allclose(j.blocks[0], [[1.5]], atol=1e-15, rtol=0)
        np.testing.assert_allclose(k.blocks[0], [[1.0]], atol=1e-15, rtol=0)
    # Returned matrices and accumulator histories remain independent.
    ja.set_block(0, np.array([[99.0]]))
    np.testing.assert_allclose(jb.blocks[0], [[1.5]], atol=1e-15, rtol=0)
    np.testing.assert_allclose(alpha._J_acc[0], [[1.5]], atol=1e-15, rtol=0)
    assert alpha._since_reset == 1
    assert beta._since_reset == (0 if history == "reset" else 1)


def test_paired_spin_jk_reuse_copies_results_and_expires_between_builds():
    from types import SimpleNamespace
    from vibeqc.pbc_bipole_fock import _reuse_paired_spin_jk
    from vibeqc._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set

    system, _ = _h2_box()
    cells = list(direct_lattice_cells(system, 0.1))
    density = make_lattice_matrix_set(1, cells, [np.array([[0.5]])])
    calls = []

    def raw(d):
        calls.append(d)
        return SimpleNamespace(
            J=make_lattice_matrix_set(1, cells, [np.array([[1.5]])]),
            K=make_lattice_matrix_set(1, cells, [np.array([[1.0]])]),
        )

    for expected_calls in [1, 2]:
        paired = _reuse_paired_spin_jk(raw)
        alpha, beta = paired(density), paired(density)
        assert len(calls) == expected_calls
        alpha.J.set_block(0, np.array([[99.0]]))
        alpha.K.set_block(0, np.array([[88.0]]))
        np.testing.assert_array_equal(beta.J.blocks[0], [[1.5]])
        np.testing.assert_array_equal(beta.K.blocks[0], [[1.0]])


@pytest.mark.parametrize("method", ["hf", "pbe0", "hse06"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
def test_joint_spin_integrals_preserve_lithium_doublet_scf(monkeypatch, method, mesh):
    import vibeqc.pbc_bipole_fock as fock
    from vibeqc import _vibeqc_core as core
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    # Three electrons: both spin channels are occupied and unequal.
    system = vq.PeriodicSystem(3, np.eye(3) * 20., [vq.Atom(3, [10., 10., 10.])], multiplicity=2)
    basis = vq.make_basis(system.unit_cell_molecule(), "sto-3g")
    evaluations, shared = [], []
    single_native = core.build_jk_2e_real_space_domains
    pair_native = core.build_jk_2e_real_space_domains_pair
    def single(*args, **kwargs):
        result = single_native(*args, **kwargs)
        evaluations.append(result.integral_evaluations)
        return result
    def pair(*args, **kwargs):
        result = pair_native(*args, **kwargs)
        shared.append(result[0].integral_evaluations)
        evaluations.append(result[0].integral_evaluations)
        return result
    monkeypatch.setattr(core, "build_jk_2e_real_space_domains", single)
    monkeypatch.setattr(core, "build_jk_2e_real_space_domains_pair", pair)
    def run():
        driver = run_pbc_bipole_uhf if method == "hf" else run_pbc_bipole_uks
        options = vq.PeriodicRHFOptions() if method == "hf" else _ks_opts(method)
        result = driver(system, basis, vq.monkhorst_pack(system, mesh), options,
            sr_image_precision=1e-6, progress=False)
        assert result.converged
        return result
    actual = run()
    shared_total = sum(evaluations)
    assert shared and sum(shared) > 0
    evaluations.clear()
    monkeypatch.setattr(fock, "_can_pair_sr", lambda *args: False)
    expected = run()
    assert 0 < shared_total < sum(evaluations)
    assert actual.energy == expected.energy
    assert actual.n_iter == expected.n_iter
    for spin in ["alpha", "beta"]:
        for left, right in zip(getattr(actual, f"fock_{spin}"), getattr(expected, f"fock_{spin}")):
            np.testing.assert_array_equal(left, right)
    if method != "hf":
        assert actual.e_hf_exchange == expected.e_hf_exchange
        assert actual.e_xc == expected.e_xc


@pytest.mark.parametrize("method", ["hf", "pbe0"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
@pytest.mark.parametrize("multiplicity", [1, 3])
@pytest.mark.parametrize("incremental", [False, True])
def test_bipole_paired_spin_jk_matches_separate_scf(
    monkeypatch, method, mesh, multiplicity, incremental
):
    import vibeqc.pbc_bipole_fock as fock
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    original = fock._sr_image_padded_jk
    same_density = fock._same_lattice_density
    traversals, matches = [], []

    def tracked(*args, **kwargs):
        traversals.append(None)
        return original(*args, **kwargs)

    def equality(*args):
        match = same_density(*args)
        matches.append(match)
        return match

    monkeypatch.setattr(fock, "_sr_image_padded_jk", tracked)
    monkeypatch.setattr(fock, "_same_lattice_density", equality)
    system, basis = _h2_box(multiplicity=multiplicity)
    driver = run_pbc_bipole_uhf if method == "hf" else run_pbc_bipole_uks

    def run():
        opts = vq.PeriodicRHFOptions() if method == "hf" else _ks_opts(method)
        result = driver(
            system, basis, vq.monkhorst_pack(system, mesh), opts,
            sr_image_precision=1e-6, use_incremental_fock=incremental,
            progress=False,
        )
        assert result.converged
        return result

    reused = run()
    reused_count = len(traversals)
    traversals.clear()
    monkeypatch.setattr(fock, "_same_lattice_density", lambda *args: False)
    monkeypatch.setattr(fock, "_can_pair_sr", lambda *args: False)
    separate = run()
    assert matches and len(traversals) > 0
    assert 2 * reused_count == len(traversals)
    if multiplicity == 1:
        assert any(matches)
    # An open-shell iteration can have equal (including zero) deltas even
    # when its absolute densities differ. The callback inputs decide reuse.
    assert reused.n_iter == separate.n_iter
    assert reused.energy == separate.energy
    for spin in ["alpha", "beta"]:
        for actual, expected in zip(getattr(reused, f"fock_{spin}"),
                                    getattr(separate, f"fock_{spin}")):
            np.testing.assert_array_equal(actual, expected)
    if method != "hf":
        assert reused.e_hf_exchange == separate.e_hf_exchange
        assert reused.e_xc == separate.e_xc


class TestGammaEwaldHse06:
    def test_hse06_is_not_pbe0_and_not_pbe(self):
        e_pbe, _ = _gamma_ewald("pbe")
        e_pbe0, _ = _gamma_ewald("pbe0")
        e_hse, _ = _gamma_ewald("hse06")
        # The old behaviour silently ran HSE06 as PBE0 (identical
        # energy). The screened kernel must move it.
        assert abs(e_hse - e_pbe0) > 1e-5
        assert abs(e_hse - e_pbe) > 1e-3

    def test_screened_exchange_energy_is_smaller_in_magnitude(self):
        # erfc(w r)/r < 1/r pointwise => |E_x[HSE06]| < |E_x[PBE0]|
        # at equal 25% coefficient.
        _, k_pbe0 = _gamma_ewald("pbe0")
        _, k_hse = _gamma_ewald("hse06")
        assert k_pbe0 < 0.0 and k_hse < 0.0
        assert abs(k_hse) < abs(k_pbe0)

    def test_erfc_kernel_omega_limits(self):
        # Kernel-level convention gate: omega -> 0 recovers the
        # full-range K; large omega kills the kernel.
        from vibeqc._vibeqc_core import LatticeSumOptions

        sys3, basis = _h2_box()
        lat_opts = LatticeSumOptions()
        n = basis.nbasis
        rng = np.random.default_rng(7)
        d = rng.standard_normal((n, n))
        D = 0.5 * (d + d.T)
        K_full = build_exchange_gamma(
            basis, sys3, lat_opts, D, PeriodicExchangeAssembly(0.25, 0.0, 0.0)
        )
        K_tiny_omega = build_exchange_gamma(
            basis, sys3, lat_opts, D, PeriodicExchangeAssembly(0.0, 0.25, 1e-7)
        )
        K_huge_omega = build_exchange_gamma(
            basis, sys3, lat_opts, D, PeriodicExchangeAssembly(0.0, 0.25, 100.0)
        )
        assert np.linalg.norm(K_tiny_omega - K_full) < 1e-5 * np.linalg.norm(
            K_full
        )
        assert np.linalg.norm(K_huge_omega) < 0.05 * np.linalg.norm(K_full)


class TestCrossDriverHse06:
    """All four ewald drivers and BIPOLE agree on HSE06 (F4-style gate)."""

    def test_multi_k_gamma_point_matches_gamma_driver(self):
        e_gamma, _ = _gamma_ewald("hse06")
        assert _multi_k_ewald("hse06") == pytest.approx(e_gamma, abs=1e-8)

    def test_uks_matches_rks_on_closed_shell(self):
        from vibeqc.periodic_uks_ewald import run_uks_periodic_gamma_ewald3d

        sys3, basis = _h2_box()
        r = run_uks_periodic_gamma_ewald3d(
            sys3, basis, options=_ks_opts("hse06"), progress=False
        )
        assert r.converged
        e_gamma, _ = _gamma_ewald("hse06")
        assert r.energy == pytest.approx(e_gamma, abs=1e-7)
        assert _multi_k_ewald("hse06", uks=True) == pytest.approx(
            e_gamma, abs=1e-7
        )

    def test_bipole_production_matches_multi_k_ewald(self):
        # Compare the independently wired production routes. The multi-k
        # Ewald reference is summed past the Gamma-only image shells (see
        # _CONVERGED_K_CUTOFF_BOHR); at its default 15-bohr cutoff it
        # truncates them all and sits 1.7e-4 Ha above the production
        # screened exchange (#501). The production domain is resolved to
        # ``sr_image_precision=1e-6`` and Schwarz-screened, so the gate is
        # its declared precision rather than the bit level (measured
        # agreement 1e-15; the screened 44.7-bohr shell is 1.1e-12).
        bipole = _bipole_result("hse06")
        reference = _multi_k_ewald_result(
            "hse06", cutoff_bohr=_CONVERGED_K_CUTOFF_BOHR
        )
        assert bipole.e_hf_exchange == pytest.approx(
            reference.e_hf_exchange, abs=1e-10
        )
        assert bipole.e_xc == pytest.approx(reference.e_xc, abs=1e-10)
        assert bipole.energy == pytest.approx(reference.energy, abs=1e-8)

    def test_bipole_physical_exchange_reach_is_the_pair_interaction_radius(
        self,
    ):
        # The only thing the padding changes in K is the pair-interaction
        # reach: the unpadded physical build with that reach set
        # explicitly reproduces the index-ball Ewald reference at the same
        # cutoff bit for bit (both unscreened), and the production route
        # to its precision.
        production = _bipole_result("hse06")
        same_reach = _bipole_result(
            "hse06",
            sr_image_precision=None,
            eri_interaction_cutoff_bohr=_CONVERGED_K_CUTOFF_BOHR,
        )
        reference = _multi_k_ewald_result(
            "hse06", cutoff_bohr=_CONVERGED_K_CUTOFF_BOHR
        )
        assert same_reach.sr_image_extent_bohr is None
        assert same_reach.e_hf_exchange == pytest.approx(
            reference.e_hf_exchange, abs=1e-12
        )
        assert same_reach.e_hf_exchange == pytest.approx(
            production.e_hf_exchange, abs=1e-10
        )

    def test_bipole_production_hse06_domain_is_precision_converged(self):
        production = _bipole_result("hse06")
        tight = _bipole_result("hse06", sr_image_precision=1e-9)
        historical = _bipole_result("hse06", sr_image_precision=None)
        reference = _multi_k_ewald_result(
            "hse06", cutoff_bohr=_CONVERGED_K_CUTOFF_BOHR
        )
        truncated = _multi_k_ewald_result("hse06")

        assert production.sr_image_extent_bohr is not None
        assert tight.sr_image_extent_bohr > production.sr_image_extent_bohr
        assert production.energy == pytest.approx(tight.energy, abs=1e-10)
        assert production.e_hf_exchange == pytest.approx(
            tight.e_hf_exchange, abs=1e-10
        )
        # The ``sr_image_precision=None`` opt-out is a historical truncation
        # diagnostic, not a production parity target: its J domain is
        # unpadded, and its K keeps the pair-interaction reach at the
        # 15-bohr electronic cutoff, exactly the Ewald reference's default
        # truncation.
        assert abs(production.energy - historical.energy) > 5e-5
        assert historical.e_hf_exchange == pytest.approx(
            truncated.e_hf_exchange, abs=1e-12
        )
        # What the padding adds to K is the image-shell exchange the Ewald
        # reference gains once its cutoff passes the shells (-1.7e-4 Ha).
        assert (
            production.e_hf_exchange - historical.e_hf_exchange
        ) == pytest.approx(
            reference.e_hf_exchange - truncated.e_hf_exchange, abs=1e-10
        )
        assert production.e_hf_exchange == pytest.approx(
            reference.e_hf_exchange, abs=1e-10
        )

    def test_bipole_uks_matches_bipole_rks(self):
        assert _bipole("hse06", uks=True) == pytest.approx(
            _bipole("hse06"), abs=1e-6
        )

    @pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
    @pytest.mark.parametrize("uks,multiplicity", [(False, 1), (True, 1), (True, 3)])
    def test_bipole_hse06_reuses_both_kernel_preparations(
        self, monkeypatch, mesh, uks, multiplicity
    ):
        import vibeqc.pbc_bipole_fock as fock
        from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
        from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

        workspaces = {}
        original = fock._sr_workspace_for

        def tracked(ctx):
            workspace = original(ctx)
            assert workspace is not None
            workspaces[id(workspace)] = workspace
            return workspace

        monkeypatch.setattr(fock, "_sr_workspace_for", tracked)
        system, basis = _h2_box(multiplicity=multiplicity)
        driver = run_pbc_bipole_uks if uks else run_pbc_bipole_rks
        result = driver(
            system, basis, vq.monkhorst_pack(system, mesh), _ks_opts("hse06"),
            sr_image_precision=1e-6, progress=False,
        )
        assert result.converged
        assert result.n_iter > 1
        assert len(workspaces) == 1
        workspace = next(iter(workspaces.values()))
        # Ewald J and HSE K have different omega values. They must retain
        # separate bounds over shared geometry throughout all SCF iterations,
        # including both spin channels and the final-density build.
        assert workspace.preparation_count == 2
        assert workspace.geometry_preparation_count == 1

    @pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
    @pytest.mark.parametrize("multiplicity", [1, 3])
    def test_bipole_hse06_spin_reuse_matches_separate_builds(
        self, monkeypatch, mesh, multiplicity
    ):
        import vibeqc.pbc_bipole_fock as fock
        from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

        original = fock._screened_exchange_lattice
        same_density = fock._same_lattice_density
        traversals = []
        reused_spins = []

        def tracked(*args, **kwargs):
            traversals.append(None)
            return original(*args, **kwargs)

        def tracked_equality(*args):
            equal = same_density(*args)
            reused_spins.append(equal)
            return equal

        monkeypatch.setattr(fock, "_screened_exchange_lattice", tracked)
        monkeypatch.setattr(fock, "_same_lattice_density", tracked_equality)
        system, basis = _h2_box(multiplicity=multiplicity)

        def run():
            result = run_pbc_bipole_uks(
                system, basis, vq.monkhorst_pack(system, mesh), _ks_opts("hse06"),
                sr_image_precision=1e-6, progress=False,
            )
            assert result.converged
            return result

        reused = run()
        reused_count = len(traversals)
        traversals.clear()
        # Force the original two-traversal path as a numerical oracle.
        monkeypatch.setattr(fock, "_same_lattice_density", lambda *args: False)
        monkeypatch.setattr(fock, "_can_pair_sr", lambda *args: False)
        separate = run()
        assert len(traversals) > 0
        # Even a singlet may have slightly unequal spin densities in an
        # iteration. Reuse only the exactly equal builds, never round them.
        assert 2 * reused_count == len(traversals)
        assert bool(sum(reused_spins)) is (multiplicity == 1)
        assert reused.n_iter == separate.n_iter
        assert reused.energy == separate.energy
        assert reused.e_hf_exchange == separate.e_hf_exchange
        assert reused.e_xc == separate.e_xc

    @pytest.mark.parametrize("uks,multiplicity", [(False, 1), (True, 1), (True, 3)])
    @pytest.mark.parametrize("mesh,precision", [((1, 1, 1), 1e-6),
                                               ((2, 1, 1), 1e-6),
                                               ((2, 1, 1), None)])
    def test_bipole_hse06_exchange_only_matches_combined_scf(
        self, monkeypatch, uks, multiplicity, mesh, precision
    ):
        from vibeqc import _vibeqc_core as core
        from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
        from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

        assert core.periodic_jk_coulomb_selection
        system, basis = _h2_box(multiplicity=multiplicity)
        driver = run_pbc_bipole_uks if uks else run_pbc_bipole_rks

        def run():
            result = driver(
                system, basis, vq.monkhorst_pack(system, mesh), _ks_opts("hse06"),
                sr_image_precision=precision, progress=False,
            )
            assert result.converged
            return result

        selected = run()
        monkeypatch.setattr(core, "periodic_jk_coulomb_selection", False)
        combined = run()
        assert selected.n_iter == combined.n_iter
        assert selected.energy == combined.energy
        assert selected.e_hf_exchange == combined.e_hf_exchange
        assert selected.e_xc == combined.e_xc

    @pytest.mark.parametrize("uks", [False, True])
    def test_bipole_hse06_reports_direct_screened_execution(self, uks):
        execution = _bipole_result(
            "hse06",
            uks=uks,
        ).screened_exchange_execution
        assert execution is not None
        assert execution.schema == (
            "vibeqc.pbc-bipole.screened-exchange-execution/v1"
        )
        assert execution.assembly == "short-range-direct"
        assert execution.c_sr == pytest.approx(0.25)
        assert execution.omega_screen_bohr_inv == pytest.approx(0.11)

    @pytest.mark.parametrize("functional", ["pbe", "pbe0"])
    @pytest.mark.parametrize("uks", [False, True])
    def test_bipole_unscreened_routes_report_no_screened_execution(
        self,
        functional,
        uks,
    ):
        assert (
            _bipole_result(
                functional,
                uks=uks,
            ).screened_exchange_execution
            is None
        )

    def test_bipole_hse06_is_not_pbe0(self):
        assert abs(_bipole("hse06") - _bipole("pbe0")) > 1e-5


class TestSlabRunnerHse06:
    """§7 gates on the dim=2 slab through run_periodic_job AUTO."""

    @staticmethod
    def _run_slab(tmp_path, functional, min_a3=30.0, mesh=(1, 1, 1)):
        atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
        sysp = vq.slab_2d(
            [18.0, 0.0, 0.0], [0.0, 18.0, 0.0], atoms, min_a3_bohr=min_a3
        )
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        r = vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional=functional,
            jk_method="auto",
            kpoints=mesh,
            output=str(tmp_path / f"slab_{functional}_{min_a3:g}"),
        )
        assert r.converged
        return r

    def test_slab_hse06_differs_from_pbe0(self, tmp_path):
        e_hse = self._run_slab(tmp_path, "hse06").energy
        e_pbe0 = self._run_slab(tmp_path, "pbe0").energy
        assert abs(e_hse - e_pbe0) > 1e-5

    def test_slab_hse06_total_is_a3_invariant(self, tmp_path):
        # The synthesized a3 is bookkeeping only -- §7 gate.
        e_30 = self._run_slab(tmp_path, "hse06", min_a3=30.0).energy
        e_80 = self._run_slab(tmp_path, "hse06", min_a3=80.0).energy
        assert e_80 == pytest.approx(e_30, abs=1e-8)

    def test_slab_hse06_citations_reach_bibtex(self, tmp_path):
        self._run_slab(tmp_path, "hse06")
        bib = (tmp_path / "slab_hse06_30.bibtex").read_text()
        assert "heyd_scuseria_ernzerhof_hse_2003" in bib
        assert "krukau_hse06_2006" in bib

    def test_slab_hse06_matches_vacuum_converged_3d_reference(self, tmp_path):
        # In-house §7 reference (the original investigation defaults):
        # the vacuum-free dim=2 total must equal the 3D-with-vacuum
        # EWALD_3D total once the vacuum is converged (isolated layer).
        from vibeqc._vibeqc_core import CoulombMethod
        from vibeqc.periodic_rks_ewald import run_rks_periodic_gamma_ewald3d

        e_slab = self._run_slab(tmp_path, "hse06").energy
        atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
        sys3 = vq.PeriodicSystem(
            3, np.diag([18.0, 18.0, 45.0]), atoms
        )
        basis = vq.BasisSet(sys3.unit_cell_molecule(), "sto-3g")
        opts = _ks_opts("hse06")
        opts.lattice_opts.coulomb_method = CoulombMethod.EWALD_3D
        r3 = run_rks_periodic_gamma_ewald3d(
            sys3, basis, options=opts, progress=False
        )
        assert r3.converged
        assert e_slab == pytest.approx(r3.energy, abs=2e-6)


class TestFailClosedRoutes:
    """Routes without a screened K refuse RS functionals loudly."""

    def test_guard_admits_none_and_pure_and_global(self):
        reject_unscreened_range_separated(None, where="t")
        reject_unscreened_range_separated(Functional("pbe", 1), where="t")
        reject_unscreened_range_separated(Functional("pbe0", 1), where="t")

    @pytest.mark.parametrize("name", ["hse06", "wb97x", "cam-b3lyp"])
    def test_guard_rejects_all_rs(self, name):
        with pytest.raises(NotImplementedError, match="full-range only"):
            reject_unscreened_range_separated(
                Functional(name, 1), where="test_route"
            )

    def test_multi_k_gdf_fails_closed_on_hse06(self):
        from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

        sys3, basis = _h2_box()
        with pytest.raises(NotImplementedError, match="range-separated"):
            run_krhf_periodic_gdf(
                sys3, basis, kmesh=(2, 1, 1), functional="hse06"
            )

    def test_multi_k_gdf_uks_fails_closed_on_hse06(self):
        from vibeqc.periodic_k_gdf import run_kuhf_periodic_gdf

        sys3, basis = _h2_box()
        with pytest.raises(NotImplementedError, match="range-separated"):
            run_kuhf_periodic_gdf(
                sys3, basis, kmesh=(2, 1, 1), functional="hse06"
            )

    def test_gamma_gdf_uks_fails_closed_on_hse06(self):
        from vibeqc.pbc_gdf import run_pbc_gdf_uks

        sys3, basis = _h2_box()
        with pytest.raises(NotImplementedError, match="range-separated"):
            run_pbc_gdf_uks(sys3, basis, functional="hse06")

    def test_legacy_gamma_gdf_fails_closed_on_hse06(self):
        from vibeqc.periodic_rhf_gdf import run_rhf_periodic_gamma_gdf

        sys3, basis = _h2_box()
        with pytest.raises(NotImplementedError, match="range-separated"):
            run_rhf_periodic_gamma_gdf(sys3, basis, functional="hse06")

    def test_gpw_fails_closed_on_hse06(self):
        from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw

        sys3, basis = _h2_box()
        with pytest.raises(NotImplementedError, match="range-separated"):
            run_periodic_rhf_gpw(sys3, basis, functional="hse06")

    def test_lr_heavy_rsh_fails_closed_on_ewald_route(self):
        from vibeqc.periodic_rks_ewald import run_rks_periodic_gamma_ewald3d

        sys3, basis = _h2_box()
        with pytest.raises(
            NotImplementedError, match="full-range exact-exchange arm"
        ):
            run_rks_periodic_gamma_ewald3d(
                sys3, basis, options=_ks_opts("cam-b3lyp"), progress=False
            )
