"""M4a/M5: the SR erfc ket-image pad and precision-derived default.

The 2026-07-12 split-gap triage (the original investigation)
proved the (c_lam, c_sig) ket-image sum must reach the *smeared* erfc
kernel range, not just the output cutoff: at an 8-bohr ball MgO/STO-3G
loses -515 mHa of J_SR. `sr_image_extent_bohr` is the explicit M4a
absolute-radius oracle. M5 derives the production radius from
`sr_image_precision=1e-8` and enables charge-pair Schwarz screening;
only the internal image ball grows and output blocks stay on the cutoff
cell list. `sr_image_precision=None` reproduces the historical domain.

Correctness contract pinned here: for any density on the cutoff
template, the padded build's home-cell J/K blocks are IDENTICAL to a
plain build at the extent cutoff (same internal quartet set for the
home output cell). Together with the Poisson-anchored wide-ball pin in
test_pbc_bipole_multik_ewald_split.py::
test_j_sr_traversal_ball_converges_diffuse_home_element (same code
path), this anchors the pad to the reciprocal-space truth values.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    PeriodicRHFOptions,
    build_jk_2e_real_space,
    compute_overlap_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
from vibeqc.pbc_bipole_common import (
    bipole_sr_image_extent,
    resolve_bipole_sr_image_extent,
)
from vibeqc.pbc_bipole_fock import (
    _build_jk_domains_output_cells_mpi,
    _sr_image_padded_jk,
)

ANG2BOHR = 1.0 / 0.529177210903


def test_active_sr_screening_public_descriptions_name_charge_pair_bound():
    """Public help must describe the active bound, not its removed predecessor."""
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[1] / "python" / "vibeqc"
    for filename, function in (
        ("pbc_bipole.py", "run_pbc_bipole_rhf"),
        ("bipole_multipole.py", "screened_multipole_interaction_tensor"),
        ("pbc_bipole_common.py", "resolve_bipole_sr_image_extent"),
    ):
        module = ast.parse((package / filename).read_text(encoding="utf-8"))
        node = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == function
        )
        description = " ".join(ast.get_docstring(node).split())
        assert "charge-pair Schwarz" in description, function
        assert "QQR" not in description, function


def _mgo():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lat(cutoff: float) -> LatticeSumOptions:
    opts = LatticeSumOptions()
    opts.cutoff_bohr = float(cutoff)
    opts.nuclear_cutoff_bohr = float(cutoff)
    return opts


def _home_density(basis, system, cutoff: float):
    """Deterministic symmetric home-cell-only density: D(0) = S(0)."""
    opts = _lat(cutoff)
    S_lat = compute_overlap_lattice(basis, system, opts)
    cells = list(direct_lattice_cells(system, float(cutoff)))
    nbf = int(basis.nbasis)
    blocks = [
        np.asarray(S_lat.blocks[0], dtype=float)
        if c == 0
        else np.zeros((nbf, nbf))
        for c in range(len(cells))
    ]
    return make_lattice_matrix_set(nbf, cells, blocks)


def test_padded_home_block_equals_wide_ball_build():
    """cutoff 8 + extent 16 home J/K == plain cutoff-16 home J/K.

    The padded traversal runs the identical internal quartet set for the
    home output cell as a plain build whose cutoff IS the extent, so the
    home blocks must agree to machine precision — this is the exactness
    contract of the M4a pad. (Would FAIL before M4a: the unpadded
    cutoff-8 home J[Mg 3s, Mg 3s] is percent-level short of the wide
    value — the −515 mHa split defect.)
    """
    system, basis = _mgo()
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V_cell))
    D_h0 = _home_density(basis, system, cutoff=8.0)

    padded = _sr_image_padded_jk(basis, system, _lat(8.0), D_h0, omega, 16.0)
    wide = build_jk_2e_real_space(basis, system, _lat(16.0), D_h0, omega)

    J_pad0 = np.asarray(padded.J.blocks[0], dtype=float)
    K_pad0 = np.asarray(padded.K.blocks[0], dtype=float)
    J_wide0 = np.asarray(wide.J.blocks[0], dtype=float)
    K_wide0 = np.asarray(wide.K.blocks[0], dtype=float)
    assert np.max(np.abs(J_pad0 - J_wide0)) < 1e-12
    assert np.max(np.abs(K_pad0 - K_wide0)) < 1e-12

    # And the pad genuinely recovers SR mass vs the unpadded build:
    unpadded = build_jk_2e_real_space(basis, system, _lat(8.0), D_h0, omega)
    J_un0 = np.asarray(unpadded.J.blocks[0], dtype=float)
    # AO 5 = Mg 3s: the diffuse pair the triage anchored (1.6055 -> 1.8529
    # on the Hcore density; the direction and percent scale hold for any
    # symmetric home density with diffuse content).
    assert J_pad0[5, 5] > J_un0[5, 5] * 1.01

    # Output template unchanged: padded blocks live on the cutoff list.
    assert len(padded.J.cells) == len(D_h0.cells)


def test_output_cells_beyond_home_also_match_wide_build():
    """Every output cell of the padded build matches the wide build."""
    system, basis = _mgo()
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V_cell))
    D_h0 = _home_density(basis, system, cutoff=8.0)

    padded = _sr_image_padded_jk(basis, system, _lat(8.0), D_h0, omega, 14.0)
    wide = build_jk_2e_real_space(basis, system, _lat(14.0), D_h0, omega)
    wide_by_key = {
        tuple(int(x) for x in np.asarray(c.index).reshape(3)): np.asarray(
            wide.J.blocks[i], dtype=float
        )
        for i, c in enumerate(wide.J.cells)
    }
    worst = 0.0
    for i, c in enumerate(padded.J.cells):
        key = tuple(int(x) for x in np.asarray(c.index).reshape(3))
        worst = max(
            worst,
            float(
                np.max(
                    np.abs(np.asarray(padded.J.blocks[i], dtype=float) - wide_by_key[key])
                )
            ),
        )
    assert worst < 1e-12


@pytest.mark.parametrize("compute_exchange", [False, True])
@pytest.mark.parametrize("compute_coulomb", [False, True])
def test_chi_output_cell_farming_serial_is_bit_identical(compute_exchange, compute_coulomb):
    """The χ scheduling seam changes no serial direct-ERI arithmetic."""
    system, basis = _mgo()
    volume = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(volume))
    density = _home_density(basis, system, cutoff=8.0)
    kwargs = {"compute_exchange": compute_exchange, "compute_coulomb": compute_coulomb}
    from vibeqc._vibeqc_core import PeriodicJKWorkspace
    workspace = PeriodicJKWorkspace()
    reference = _sr_image_padded_jk(
        basis, system, _lat(8.0), density, omega, 14.0, **kwargs
    )
    farmed = _sr_image_padded_jk(
        basis,
        system,
        _lat(8.0),
        density,
        omega,
        14.0,
        output_cell_farming_task_kind="chi-direct-output-cell",
        output_cell_farming_strategy="cyclic",
        workspace=workspace,
        **kwargs,
    )
    for got, want in zip(farmed.J.blocks, reference.J.blocks):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    if compute_exchange:
        for got, want in zip(farmed.K.blocks, reference.K.blocks):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    else:
        assert farmed.K is None
    executed = farmed.output_cell_farming_execution
    assert executed.task_kind == "chi-direct-output-cell"
    assert executed.strategy == "cyclic"
    assert executed.world_size == 1
    assert executed.active is False
    assert executed.global_task_count == len(farmed.J.cells)
    assert executed.local_task_counts == (len(farmed.J.cells),)
    assert executed.complete_internal_translation_sum is True
    assert workspace.preparation_count == 1


def test_chi_output_cell_farming_empty_rank_skips_native(
    monkeypatch,
):
    """An empty MPI assignment must not hit native [] == all semantics."""
    from types import SimpleNamespace

    import vibeqc.mpi as mpi_module
    import vibeqc.pbc_bipole_fock as fock_module
    import vibeqc._vibeqc_core as core

    cells = [
        SimpleNamespace(index=np.asarray((0, 0, 0), dtype=int)),
        SimpleNamespace(index=np.asarray((1, 0, 0), dtype=int)),
    ]

    class EmptyRankComm:
        def allgather(self, local):
            schema, n_tasks, strategy, fingerprint, _, _, _ = local
            return [
                (
                    schema,
                    n_tasks,
                    strategy,
                    fingerprint,
                    0,
                    (0,),
                    [(np.asarray([[10.0]]), np.asarray([[20.0]]))],
                ),
                (
                    schema,
                    n_tasks,
                    strategy,
                    fingerprint,
                    1,
                    (1,),
                    [(np.asarray([[11.0]]), np.asarray([[21.0]]))],
                ),
                local,
            ]

    world = vq.MPIWorld(comm=None, rank=2, size=3, active=True)
    world.comm = EmptyRankComm()
    monkeypatch.setattr(mpi_module, "_mpi_world", world)

    def native_must_not_run(*_args, **_kwargs):
        raise AssertionError("empty rank entered native all-cell traversal")

    monkeypatch.setattr(
        core, "build_jk_2e_real_space_domains", native_must_not_run
    )
    monkeypatch.setattr(
        fock_module,
        "make_lattice_matrix_set",
        lambda _nbf, used_cells, blocks: SimpleNamespace(
            cells=list(used_cells), blocks=list(blocks)
        ),
    )
    workspace = core.PeriodicJKWorkspace()
    got = _build_jk_domains_output_cells_mpi(
        SimpleNamespace(nbasis=1),
        object(),
        object(),
        object(),
        cells,
        [0, 1],
        omega=0.3,
        task_kind="chi-direct-output-cell",
        workspace=workspace,
    )
    assert [float(block[0, 0]) for block in got.J.blocks] == [10.0, 11.0]
    assert [float(block[0, 0]) for block in got.K.blocks] == [20.0, 21.0]
    assert got.output_cell_farming_execution.local_task_count == 0
    assert got.output_cell_farming_execution.local_task_counts == (1, 1, 0)
    assert workspace.preparation_count == 0


class _CountedNativeBlocks:
    """Model the native vector export, including its copy semantics."""

    def __init__(self, blocks):
        self.values = blocks
        self.reads = 0

    @property
    def blocks(self):
        self.reads += 1
        return [block.copy() for block in self.values]


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("exchange", [False, True])
@pytest.mark.parametrize("zone", [None, 2.0])
def test_padded_projection_exports_each_native_vector_once(monkeypatch, physical, exchange, zone):
    from types import SimpleNamespace
    import vibeqc.pbc_bipole_fock as fock

    system = vq.PeriodicSystem(3, np.eye(3) * 4., [vq.Atom(2, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = _lat(4.1)
    opts.pair_complete_1e = physical
    exported = []
    wanted = {}

    def native(_basis, _system, _opts, density, cells, indices, masks, **kwargs):
        blocks = [np.zeros((1, 1)) for _ in cells]
        for i in indices:
            blocks[i][0, 0] = i + 1.
            wanted[tuple(cells[i].index)] = i + 1.
        j, k = _CountedNativeBlocks(blocks), _CountedNativeBlocks([-b for b in blocks])
        exported.extend([j, k])
        return SimpleNamespace(J=j, K=k if exchange else None)

    monkeypatch.setattr(fock, "_build_jk_domains_output_cells_mpi", native)
    result = fock._sr_image_padded_jk(
        basis, system, opts, object(), .4, 6.,
        compute_exchange=exchange, exact_zone_bohr=zone,
    )
    assert exported[0].reads == 1
    assert exported[1].reads == int(exchange)
    for cell, block in zip(result.J.cells, result.J.blocks):
        np.testing.assert_array_equal(block, [[wanted.get(tuple(cell.index), 0.)]])
    if exchange:
        for j, k in zip(result.J.blocks, result.K.blocks):
            np.testing.assert_array_equal(k, -j)
    else:
        assert result.K is None


@pytest.mark.parametrize("exchange", [False, True])
def test_output_farming_exports_each_owned_vector_once(monkeypatch, exchange):
    from types import SimpleNamespace
    import vibeqc.mpi as mpi
    import vibeqc.pbc_bipole_fock as fock
    import vibeqc._vibeqc_core as core

    cells = [SimpleNamespace(index=(i, 0, 0)) for i in range(5)]
    requested = [4, 1, 3, 0, 2]
    j = _CountedNativeBlocks([np.array([[i + 1.]]) for i in range(5)])
    k = _CountedNativeBlocks([np.array([[-i - 1.]]) for i in range(5)])
    ordered = [(j.values[i], k.values[i] if exchange else None) for i in requested]

    def gather(local):
        for got, pos in zip(local, [0, 2, 4]):
            np.testing.assert_array_equal(got[0], ordered[pos][0])
            if exchange:
                np.testing.assert_array_equal(got[1], ordered[pos][1])
        return ordered

    partition = SimpleNamespace(size=2, rank=0, strategy="cyclic", n_tasks=5,
        n_local=3, local_indices=[0, 2, 4], task_counts=(3, 2),
        task_fingerprint="test", allgather_ordered=gather)
    monkeypatch.setattr(mpi.LatticeOutputPartition, "create", lambda *a, **k: partition)
    monkeypatch.setattr(core, "build_jk_2e_real_space_domains",
                        lambda *a, **kw: SimpleNamespace(J=j, K=k if exchange else None))
    monkeypatch.setattr(fock, "make_lattice_matrix_set",
                        lambda n, c, b: SimpleNamespace(cells=c, blocks=b))
    result = fock._build_jk_domains_output_cells_mpi(
        SimpleNamespace(nbasis=1), object(), object(), object(), cells, requested,
        omega=.4, compute_exchange=exchange, task_kind="chi-direct-output-cell",
    )
    assert (j.reads, k.reads) == (1, int(exchange))
    np.testing.assert_array_equal(np.asarray(result.J.blocks).ravel(), np.arange(1., 6.))
    if exchange:
        np.testing.assert_array_equal(np.asarray(result.K.blocks).ravel(), -np.arange(1., 6.))
    else:
        assert result.K is None


@pytest.mark.parametrize("exchange", [False, True])
def test_paired_output_empty_worklist_never_enters_native(monkeypatch, exchange):
    from types import SimpleNamespace
    import vibeqc.pbc_bipole_fock as fock
    import vibeqc._vibeqc_core as core

    def unexpected(*args, **kwargs):
        pytest.fail("empty paired worklist reached native all-cells semantics")
    monkeypatch.setattr(core, "build_jk_2e_real_space_domains_pair", unexpected)
    monkeypatch.setattr(fock, "make_lattice_matrix_set",
        lambda n, c, b: SimpleNamespace(cells=c, blocks=[np.array(x, copy=True) for x in b]))
    cells = [SimpleNamespace(index=(i, 0, 0)) for i in range(3)]
    args = (SimpleNamespace(nbasis=1), object(), object(), object(), cells, [])
    results = fock._build_jk_domains_output_cells_mpi(
        *args, omega=.4, companion_density=object(), compute_exchange=exchange,
    )
    for result in results:
        np.testing.assert_array_equal(result.J.blocks, np.zeros((3, 1, 1)))
        if exchange:
            np.testing.assert_array_equal(result.K.blocks, np.zeros((3, 1, 1)))
        else:
            assert result.K is None
    results[0].J.blocks[0][0, 0] = 1.
    assert results[1].J.blocks[0][0, 0] == 0.
    with pytest.raises(ValueError, match="serial output-cell ownership"):
        fock._build_jk_domains_output_cells_mpi(
            *args, omega=.4, companion_density=object(), task_kind="test-farming",
        )


def test_extent_formula_properties():
    system, basis = _mgo()
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V_cell))
    loose = bipole_sr_image_extent(basis, system, omega, precision=1e-4)
    tight = bipole_sr_image_extent(basis, system, omega, precision=1e-10)
    assert tight > loose > 0.0
    # Consistent with the triage's empirical ~16-bohr kernel range at
    # eps ~ 1e-6 (this bound adds the 6.89-bohr max atom offset).
    mid = bipole_sr_image_extent(basis, system, omega, precision=1e-6)
    assert 15.0 < mid < 30.0
    with pytest.raises(ValueError, match="omega"):
        bipole_sr_image_extent(basis, system, 0.0)
    with pytest.raises(ValueError, match="precision"):
        bipole_sr_image_extent(basis, system, omega, precision=2.0)


def test_m5_precision_resolver_enables_padded_screened_domain():
    """M5 default: precision -> absolute radius + screening on both options."""
    system, basis = _mgo()
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V_cell))
    user = _lat(8.0)
    derived = _lat(8.0)

    got = resolve_bipole_sr_image_extent(
        basis,
        system,
        user,
        derived,
        omega,
        use_ewald_j_split=True,
        sr_image_precision=1e-6,
        sr_image_extent_bohr=None,
    )
    expected = 8.0 + bipole_sr_image_extent(
        basis, system, omega, precision=1e-6
    )
    assert got == pytest.approx(expected, abs=1e-12)
    assert user.sr_range_screening is True
    assert derived.sr_range_screening is True


def test_physical_interaction_radius_is_independent_of_atom_image_labels():
    from scipy.special import erfcinv

    lattice = np.diag([6., 8., 9.])
    radii = []
    for label in [0, 1000]:
        positions = [[.1, .2, .3], [1.4+6*label, .3, .1]]
        system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, p) for p in positions])
        basis = vq.BasisSet(system.unit_cell_molecule(), [
            vq.ShellInfo(i, 0, False, [a], [1.], p)
            for i, (a, p) in enumerate(zip([.7, 1.1], positions))
        ], "physical-radius-oracle", False)
        user, derived = _lat(5.1), _lat(5.1)
        user.pair_complete_1e = derived.pair_complete_1e = True
        radii.append(resolve_bipole_sr_image_extent(
            basis, system, user, derived, .4, use_ewald_j_split=True,
            sr_image_precision=1e-8, sr_image_extent_bohr=None,
        ))
    # Each weighted Gaussian product center is at most R_pair/2 from its
    # midpoint. The remaining reach is the slowest radial erfc decay.
    expected = 5.1+erfcinv(1e-8)*np.sqrt(1/.7+1/.4**2)
    assert radii == pytest.approx([expected, expected], rel=0, abs=2e-14)


def test_m5_precision_none_restores_historical_domain():
    system, basis = _mgo()
    user = _lat(8.0)
    derived = _lat(8.0)
    got = resolve_bipole_sr_image_extent(
        basis,
        system,
        user,
        derived,
        0.5,
        use_ewald_j_split=True,
        sr_image_precision=None,
        sr_image_extent_bohr=None,
    )
    assert got is None
    assert user.sr_range_screening is False
    assert derived.sr_range_screening is False


def test_explicit_extent_overrides_precision_without_forcing_screening():
    system, basis = _mgo()
    user = _lat(8.0)
    derived = _lat(8.0)
    got = resolve_bipole_sr_image_extent(
        basis,
        system,
        user,
        derived,
        0.5,
        use_ewald_j_split=True,
        sr_image_precision=1e-6,
        sr_image_extent_bohr=16.0,
    )
    assert got == 16.0
    assert user.sr_range_screening is False
    assert derived.sr_range_screening is False


@pytest.mark.parametrize("extent", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("physical_pairs", [False, True])
def test_nonfinite_sr_extent_is_rejected_before_domain_build(
    extent, physical_pairs, monkeypatch,
):
    import vibeqc.pbc_bipole_fock as fock

    opts = _lat(8.0)
    opts.pair_complete_1e = physical_pairs

    def unexpected_domain(*args, **kwargs):
        pytest.fail("invalid SR extent reached domain construction")

    monkeypatch.setattr(fock, "_sr_padded_lat_opts", unexpected_domain)
    with pytest.raises(ValueError, match="finite"):
        fock._sr_image_padded_jk(
            None, None, opts, None, .5, extent,
        )
    # The driver resolver must reject the same invalid radius even when it
    # overrides precision; no native domain or basis access is needed.
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="finite"):
        resolve_bipole_sr_image_extent(
            None, SimpleNamespace(dim=3), opts, opts, .5,
            use_ewald_j_split=True, sr_image_precision=None,
            sr_image_extent_bohr=extent,
        )


@pytest.mark.parametrize("omega", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_sr_omega_is_rejected_before_basis_access(omega):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="omega.*finite"):
        bipole_sr_image_extent(None, None, omega)
    opts = _lat(8.0)
    with pytest.raises(RuntimeError, match="omega"):
        resolve_bipole_sr_image_extent(
            None, SimpleNamespace(dim=3), opts, opts, omega,
            use_ewald_j_split=True, sr_image_precision=1e-8,
            sr_image_extent_bohr=None,
        )


def test_precision_resolver_skips_non_sr_routes_and_validates_input():
    system, basis = _mgo()
    user = _lat(8.0)
    derived = _lat(8.0)
    assert resolve_bipole_sr_image_extent(
        basis,
        system,
        user,
        derived,
        None,
        use_ewald_j_split=False,
        sr_image_precision=1e-6,
        sr_image_extent_bohr=None,
    ) is None
    assert resolve_bipole_sr_image_extent(
        basis,
        system,
        user,
        derived,
        0.5,
        use_ewald_j_split=True,
        sr_image_precision=1e-6,
        sr_image_extent_bohr=None,
        erfc_sr_build_active=False,
    ) is None
    assert resolve_bipole_sr_image_extent(
        basis,
        system,
        user,
        derived,
        0.5,
        use_ewald_j_split=True,
        sr_image_precision=1e-6,
        sr_image_extent_bohr=16.0,
        erfc_sr_build_active=False,
    ) is None
    with pytest.raises(ValueError, match="sr_image_precision"):
        resolve_bipole_sr_image_extent(
            basis,
            system,
            user,
            derived,
            0.5,
            use_ewald_j_split=True,
            sr_image_precision=1.0,
            sr_image_extent_bohr=None,
        )


def test_extent_below_cutoff_raises():
    system, basis = _mgo()
    V_cell = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V_cell))
    D_h0 = _home_density(basis, system, cutoff=8.0)
    with pytest.raises(ValueError, match="exceed the electronic cutoff"):
        _sr_image_padded_jk(basis, system, _lat(8.0), D_h0, omega, 6.0)


def _h2_cell():
    box = 8.0
    c = box / 2
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


@pytest.mark.parametrize("exchange_split", [True, False])
def test_driver_defaults_to_precision_padded_sr_domain(exchange_split):
    """End-to-end M5 default: padded SR mass moves E vs the legacy opt-out."""
    # The physical pair default (#316) refuses the legacy gauge
    # (exchange_split=False); that row replays the historical operator
    # explicitly (#503).
    system, basis = _h2_cell()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)

    def _opts():
        opts = PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = 6.0
        opts.max_iter = 1
        opts.use_diis = False
        return opts

    r_plain = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _opts(),
        use_exchange_ewald_split=exchange_split,
        historical_domain=not exchange_split,
        sr_image_precision=None,
        progress=False,
    )
    r_pad = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _opts(),
        use_exchange_ewald_split=exchange_split,
        historical_domain=not exchange_split,
        progress=False,
    )
    assert np.isfinite(float(r_pad.energy))
    assert r_plain.sr_image_extent_bohr is None
    assert r_pad.sr_image_extent_bohr is not None
    assert r_pad.sr_image_extent_bohr > 6.0
    # The recovered J_SR mass is repulsive: the padded first-iteration
    # total must not be BELOW the unpadded one, and on this diffuse H2
    # cell it moves by a measurable amount.
    assert float(r_pad.energy) >= float(r_plain.energy)
    assert abs(float(r_pad.energy) - float(r_plain.energy)) > 1e-6


def test_extent_composes_with_sym3b_reduce():
    """M4b: sr_image_extent + use_fock_symmetry_reduce compose (the
    padded internal ball runs through the M1 full-domain binding; this
    raised NotImplementedError before M4b). Enforce == reduce stays
    exact under the pad, and the pad recovers repulsive J_SR mass."""
    # The physical pair default (#316) refuses the legacy SYM3b projector;
    # this test pins the historical composition, so it replays it explicitly
    # (#503).
    system, basis = _h2_cell()
    vq.attach_symmetry(system)
    kmesh = vq.monkhorst_pack(system, [1, 1, 1], use_symmetry=False)

    def _opts():
        opts = PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = 6.0
        opts.max_iter = 1
        opts.use_diis = False
        return opts

    e_red = float(
        vq.run_pbc_bipole_rhf(
            system, basis, kmesh, _opts(),
            sr_image_extent_bohr=14.0,
            use_fock_symmetry_reduce=True,
            historical_domain=True,
            progress=False,
        ).energy
    )
    auto = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _opts(),
        sr_image_extent_bohr=14.0,
        historical_domain=True,
        progress=False,
    )
    e_auto = float(auto.energy)
    e_enf = float(
        vq.run_pbc_bipole_rhf(
            system, basis, kmesh, _opts(),
            sr_image_extent_bohr=14.0,
            use_fock_symmetry=True,
            use_fock_symmetry_reduce=False,
            historical_domain=True,
            progress=False,
        ).energy
    )
    e_nopad = float(
        vq.run_pbc_bipole_rhf(
            system, basis, kmesh, _opts(),
            use_fock_symmetry_reduce=True,
            sr_image_precision=None,
            historical_domain=True,
            progress=False,
        ).energy
    )
    historical_auto = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        _opts(),
        sr_image_precision=None,
        historical_domain=True,
        progress=False,
    )
    assert np.isfinite(e_red)
    assert auto.pair_resolved_fock_domain is True
    assert historical_auto.pair_resolved_fock_domain is False
    assert abs(e_auto - e_red) < 1e-12
    assert abs(e_red - e_enf) < 1e-12
    # The recovered J_SR mass is repulsive.
    assert e_red >= e_nopad
    assert abs(e_red - e_nopad) > 1e-8
