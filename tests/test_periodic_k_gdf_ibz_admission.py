"""Source admission for the IBZ-native multi-k GDF exchange (``ibz_native``).

``run_periodic_job(symmetry_reduce_k=True)`` builds K(k) at the irreducible
wedge only and transports it to the other star members. That is exact only
for operations mapping the *executed* exchange source onto itself. These
tests build the SR/LR and classic cderi caches of one explicit finite
approximation and compare transported against directly built exchange on a
nonsymmetric complex trial density: the same finite Hamiltonian, no SCF.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.periodic_k_gdf as kgdf
from vibeqc import _vibeqc_core as core
from vibeqc.aux_basis import (
    _build_lpq_range_separated_shared_q,
    build_lpq_bloch_native_fft,
    make_modrho_aux_basis,
)
from vibeqc.periodic_k_gdf import _build_k_from_lpq_cache
from vibeqc.periodic_k_symmetry import (
    _ao_source_atom_index,
    _op_atom_shifts_cart,
    expand_k_matrices_to_full,
    star_operations,
)
from vibeqc.symmetry_scf import build_ao_permutation_cache


def _lih_fcc():
    """Rock-salt LiH: Li at the origin, H displaced by half a cube edge."""
    a = 7.72
    lattice = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]).T
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(3, [0., 0., 0.]), vq.Atom(1, [0.5 * a, 0., 0.])]
    )
    vq.attach_symmetry(system, symprec=1e-6)
    return system


def _mesh(system, mesh):
    kmesh = core.monkhorst_pack(system, list(mesh), [0, 0, 0], False)
    kpts = np.array([np.asarray(k, dtype=float) for k in kmesh.kpoints])
    return kpts, np.full(len(kpts), 1.0 / len(kpts))


# --- Admission of operations ------------------------------------------------


def test_admission_follows_the_exchange_source():
    """SR/LR admits every symmorphic operation; classic only shift-constant ones.

    The expected classic set is derived independently of the atom-mapping
    helpers: with Li fixed at the origin, an operation shifts both atoms by
    the same lattice vector iff its integer rotation fixes H's fractional
    coordinate exactly, not merely modulo a lattice vector.
    """
    system = _lih_fcc()
    operations = [op for op in system.symmetry.operations
                  if np.allclose(np.asarray(op.translation, dtype=float), 0.0)]
    assert len(operations) == 48
    h_frac = np.linalg.solve(np.asarray(system.lattice, dtype=float),
                             np.asarray(system.unit_cell[1].xyz, dtype=float))
    expected = [i for i, op in enumerate(operations)
                if np.allclose(np.asarray(op.rotation, dtype=float) @ h_frac, h_frac)]

    sr = kgdf._admit_ibz_exchange_operations(system, "sr")
    assert (sr.support, sr.admitted, sr.refused) == (
        "pair-centred", tuple(range(48)), ())
    # The physical classic fit (#466) shares the SR/LR pair-centred domain.
    physical = kgdf._admit_ibz_exchange_operations(system, "classic-physical")
    assert (physical.support, physical.admitted, physical.refused) == (
        "pair-centred", tuple(range(48)), ())

    classic = kgdf._admit_ibz_exchange_operations(system, "classic")
    assert classic.support == "origin-centred"
    assert list(classic.admitted) == expected and len(expected) == 8
    assert {index for index, _ in classic.refused} == set(range(48)) - set(expected)
    # Refused operations move H by one or two half cube edges relative to Li.
    assert sorted({round(spread, 6) for _, spread in classic.refused}) == [3.86, 7.72]


@pytest.mark.parametrize("producer", [
    "compcell", "mdf", "private", "classic-screened",
])
def test_unqualified_exchange_sources_are_refused(producer):
    with pytest.raises(NotImplementedError, match="no qualified symmetry action"):
        kgdf._admit_ibz_exchange_operations(_lih_fcc(), producer)


def test_exchange_producer_mirrors_the_cache_dispatch():
    name = kgdf._ibz_exchange_producer
    assert name(gdf_method="rsgdf", bulk_sr=True, private_builder=None,
                fit_screen_threshold=1e-8) == "sr"
    assert name(gdf_method="rsgdf", bulk_sr=False, private_builder=None,
                fit_screen_threshold=0.0) == "classic"
    assert name(gdf_method="rsgdf", bulk_sr=False, private_builder=None,
                fit_screen_threshold=0.0,
                classic_support_policy="historical_index_ball") == "classic"
    assert name(gdf_method="rsgdf", bulk_sr=False, private_builder=None,
                fit_screen_threshold=0.0,
                classic_support_policy="physical_pairs") == "classic-physical"
    assert name(gdf_method="rsgdf", bulk_sr=False, private_builder=None,
                fit_screen_threshold=1e-8) == "classic-screened"
    assert name(gdf_method="compcell", bulk_sr=False, private_builder=None,
                fit_screen_threshold=0.0) == "compcell"
    assert name(gdf_method="rsgdf", bulk_sr=True, private_builder=object(),
                fit_screen_threshold=0.0) == "private"


@pytest.mark.parametrize("mesh", [(2, 2, 2), (3, 3, 3), (4, 4, 4)])
def test_classic_wedge_is_refused_and_sr_wedge_resolves(mesh):
    system = _lih_fcc()
    kpts, weights = _mesh(system, mesh)
    rows, star_map, _ = kgdf._resolve_ibz_native_state(
        system, kpts, mesh, weights, None, exchange_producer="sr")
    assert len(rows) < len(kpts)
    with pytest.raises(NotImplementedError, match="origin-centred") as refusal:
        kgdf._resolve_ibz_native_state(
            system, kpts, mesh, weights, None, exchange_producer="classic")
    # The refusal names the source, not the user's electronic state.
    assert "symmetry break" not in str(refusal.value)
    assert "8 of the 48" in str(refusal.value)
    # The physical classic fit resolves the same wedge as SR/LR (#466).
    physical, _, _ = kgdf._resolve_ibz_native_state(
        system, kpts, mesh, weights, None, exchange_producer="classic-physical")
    assert list(physical) == list(rows)


def test_star_operations_accepts_only_attached_operation_subsets():
    system = _lih_fcc()
    kfull = core.monkhorst_pack(system, [2, 2, 2], [0, 0, 0], False)
    kibz = core.monkhorst_pack(system, [2, 2, 2], [0, 0, 0], True)
    admitted = kgdf._admit_ibz_exchange_operations(system, "sr").operations
    subset = [admitted[0], admitted[2], admitted[4], admitted[6],
              admitted[16], admitted[32]]
    star = star_operations(system, kibz, kfull, operations=subset)
    assert len(star.operations) == len(subset)
    # A shift-constant subset of LiH cannot cover the (2,2,2) stars.
    classic = kgdf._admit_ibz_exchange_operations(system, "classic")
    with pytest.raises(ValueError, match="not in the star"):
        star_operations(system, kibz, kfull, operations=[
            classic.operations[i] for i in classic.admitted])
    # An operation of another crystal is not an admissible subset member.
    other = vq.PeriodicSystem(3, np.eye(3) * 5.0, [vq.Atom(2, [0., 0., 0.])])
    vq.attach_symmetry(other, symprec=1e-6)
    foreign = [op for op in other.symmetry.operations
               if not any(np.array_equal(np.asarray(op.rotation), np.asarray(o.rotation))
                          for o in admitted)]
    assert foreign
    with pytest.raises(ValueError, match="subset"):
        star_operations(system, kibz, kfull, operations=[foreign[0]])
    with pytest.raises(ValueError, match="subset"):
        star_operations(system, kibz, kfull, operations=[])


# --- Driver: refusal precedes every build; no cell-list warning -------------


def _driver_options(cutoff):
    options = vq.PeriodicRHFOptions()
    options.max_iter = 2
    options.lattice_opts.cutoff_bohr = cutoff
    return options


class _BuildReached(Exception):
    pass


def test_driver_refuses_classic_source_before_any_cderi_build(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an exchange cderi build ran before admission")

    for name in ("_build_rsgdf_lpq_cache_shared_q",
                 "_build_scf_range_separated_lpq_cache"):
        monkeypatch.setattr(kgdf, name, forbidden)
    system = _lih_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="origin-centred"):
        kgdf.run_krhf_periodic_gdf(
            system, basis, kmesh=(2, 2, 2), options=_driver_options(26.0),
            aux_basis="def2-svp-jk", fit_source="classic", ibz_native=True,
            classic_support_policy="historical_index_ball", progress=False,
        )


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
@pytest.mark.parametrize("controls, producer", [
    ({}, "sr"),
    ({"fit_source": "sr", "fit_screen_threshold": 1e-9}, "sr"),
    ({"fit_source": "classic"}, "classic-physical"),
    ({"fit_source": "classic", "classic_support_policy": "historical_index_ball"},
     "classic"),
    ({"fit_source": "classic", "fit_screen_threshold": 1e-9,
      "classic_support_policy": "historical_index_ball"}, "classic-screened"),
    ({"gdf_method": "compcell"}, "compcell"),
])
def test_driver_declares_the_executed_exchange_source(monkeypatch, driver,
                                                      controls, producer):
    """Both drivers name the cderi producer they will run, before any integral.

    The wrapper records the declaration and stops the run; a 14-bohr cell
    list no longer triggers a transport warning on the way (that residual
    belonged to operations the executed source does not map onto itself).
    """
    declared = []

    def intercept(*args, exchange_producer, **kwargs):
        declared.append(exchange_producer)
        raise _BuildReached

    monkeypatch.setattr(kgdf, "_resolve_ibz_native_state", intercept)
    system = _lih_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(_BuildReached):
            getattr(kgdf, driver)(
                system, basis, kmesh=(2, 2, 2), options=_driver_options(14.0),
                aux_basis="def2-svp-jk", ibz_native=True, progress=False,
                **controls,
            )
    assert declared == [producer]
    assert not [w for w in caught if "ibz_native" in str(w.message)]


# --- Numerical acceptance: transported vs directly built exchange ------------


def _cscl_fixture():
    """Two-atom Pm-3m cell whose inversion moves the body-centre atom a cell.

    Pure s/p/d shells with distinct exponents on the two species; 42 of the
    48 symmorphic operations shift the two atoms by different lattice vectors.
    """
    a = 5.0
    centers = [[0., 0., 0.], [a / 2, a / 2, a / 2]]
    system = vq.PeriodicSystem(
        3, np.eye(3) * a, [vq.Atom(3, centers[0]), vq.Atom(1, centers[1])])
    vq.attach_symmetry(system, symprec=1e-6)
    molecule = system.unit_cell_molecule()

    def basis(auxiliary):
        spec = ({0: [(0, .5), (1, .45), (2, .6)], 1: [(0, .4), (1, .35), (2, .5)]}
                if auxiliary else
                {0: [(0, .9), (1, .7), (2, .8)], 1: [(0, .6), (1, .5)]})
        shells = [core.ShellInfo(i, l, True, [e], [1.], centers[i])
                  for i in (0, 1) for l, e in spec[i]]
        return core.BasisSet(molecule, shells, "ibz-admission-fixture")

    return system, basis(False), basis(True)


@pytest.fixture(scope="module")
def exchange_sources():
    system, orbital, auxiliary = _cscl_fixture()
    kpts, weights = _mesh(system, (2, 2, 2))
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(system.lattice)).T
    frac = np.linalg.solve(reciprocal, kpts.T).T

    def index_of(k):
        d = frac - np.linalg.solve(reciprocal, k)
        (hit,) = np.flatnonzero(np.all(np.abs(d - np.rint(d)) < 1e-8, axis=1))
        return int(hit)

    # SR/LR: explicit pair/auxiliary image radii and reciprocal sphere; one
    # shared-q batch per momentum transfer, every bra of the mesh.
    sr = {}
    for m in range(len(kpts)):
        batch = _build_lpq_range_separated_shared_q(
            system, orbital, auxiliary, kpts, kpts[m], omega=.5,
            pair_cutoff=8., auxiliary_cutoff=9., ke_cutoff=6.,
            linear_dep_thr=1e-9, memory_byte_cap=256 << 20,
            native_workspace_byte_cap=32 << 20, image_candidate_cap=10**6,
            reciprocal_candidate_cap=10**6,
        )
        for i in range(len(kpts)):
            sr[(i, index_of(kpts[i] + kpts[m]))] = batch.factors[i]
    # Classic: the all-reciprocal fit over one origin-centred 8-bohr list
    # (the builder's historical default), and over physical AO pairs within
    # 8 bohr of each other (#466).
    options = core.LatticeSumOptions()
    options.cutoff_bohr = 8.0
    modrho = make_modrho_aux_basis(auxiliary, system.unit_cell_molecule())

    def classic_cache(**support):
        return {
            (i, j): np.asarray(build_lpq_bloch_native_fft(
                system, orbital, modrho, kpts[i], kpts[j], ke_cutoff=6.0,
                lat_opts=options, linear_dep_thr=1e-9, **support))
            for i in range(len(kpts)) for j in range(len(kpts))
        }

    return dict(system=system, orbital=orbital, kpts=kpts, weights=weights,
                index_of=index_of, frac=frac, reciprocal=reciprocal,
                caches={"sr": sr, "classic": classic_cache(),
                        "classic-physical": classic_cache(
                            support_policy="physical_pairs",
                            memory_byte_cap=4 << 30)})


def _relative(values, references):
    worst = max(float(np.max(np.abs(v - r))) for v, r in zip(values, references))
    return worst / max(float(np.max(np.abs(r))) for r in references)


def _group_action(src):
    """Return ``act(o, blocks)`` pushing per-k blocks through operation ``o``.

    ``g`` acts on each k block exactly as ``expand_k_matrices_to_full``
    transports K: real Wigner/atom permutation times the per-atom
    lattice-shift Bloch phase at the target momentum. ``o=None`` is scalar
    time reversal, ``M(-k) = M(k)*``.
    """
    system, orbital, kpts = src["system"], src["orbital"], src["kpts"]
    n_k = len(kpts)
    operations = kgdf._admit_ibz_exchange_operations(system, "sr").operations
    rotations = build_ao_permutation_cache(system, orbital, operations)
    shifts = _op_atom_shifts_cart(system, operations)
    ao_atom = _ao_source_atom_index(orbital)

    def act(o, blocks):
        out = [None] * n_k
        for j in range(n_k):
            if o is None:
                out[src["index_of"](-kpts[j])] = blocks[j].conj()
                continue
            w = np.asarray(operations[o].rotation, dtype=float)
            w_inv_t = np.rint(np.linalg.inv(w)).astype(int).T
            t = src["index_of"](src["reciprocal"] @ (w_inv_t @ src["frac"][j]))
            phase = np.exp(1j * (shifts[o] @ kpts[t]))[ao_atom]
            gamma = np.asarray(rotations[o], dtype=float) * phase[None, :]
            out[t] = gamma @ blocks[j] @ gamma.conj().T
        return out

    spreads = [float(np.max(np.abs(s - s[0]))) for s in shifts]
    return act, spreads


def _trial_density(src, seed):
    nbf = src["orbital"].nbasis
    rng = np.random.default_rng(seed)
    # Nonsymmetric, non-Hermitian complex blocks at every k.
    return [rng.normal(size=(nbf, nbf)) + 1j * rng.normal(size=(nbf, nbf))
            for _ in range(len(src["kpts"]))]


def _exchange(src, cache, density):
    return [np.asarray(x) for x in _build_k_from_lpq_cache(
        cache, density, src["weights"], nbasis=src["orbital"].nbasis)]


def _equivariance(src, cache):
    """Per operation: max relative |K[g.D] - g.K[D]| over the whole mesh."""
    act, spreads = _group_action(src)
    density = _trial_density(src, 2026)
    reference = _exchange(src, cache, density)
    results = [
        (spread, _relative(_exchange(src, cache, act(o, density)), act(o, reference)))
        for o, spread in enumerate(spreads)
    ]
    time_reversal = _relative(_exchange(src, cache, act(None, density)),
                              act(None, reference))
    return results, time_reversal


def test_sr_exchange_is_equivariant_under_every_admitted_operation(exchange_sources):
    results, time_reversal = _equivariance(
        exchange_sources, exchange_sources["caches"]["sr"])
    assert len(results) == 48
    assert sum(spread > 1e-8 for spread, _ in results) == 42
    # Measured: <= 3.6e-15 including the shift-varying and improper ones.
    assert max(residual for _, residual in results) < 1e-12
    assert time_reversal < 1e-12


def test_classic_exchange_is_equivariant_only_under_shift_constant_operations(
        exchange_sources):
    """The negative control for the admission criterion (vibe-qc #237)."""
    results, time_reversal = _equivariance(
        exchange_sources, exchange_sources["caches"]["classic"])
    constant = [r for spread, r in results if spread <= 1e-8]
    varying = [r for spread, r in results if spread > 1e-8]
    assert len(constant) == 6 and len(varying) == 42
    # Measured 2026-10-06: 1.8e-15 against 1.3e-2 .. 1.8e-2 at this 8-bohr list.
    assert max(constant) < 1e-12
    assert min(varying) > 1e-3
    assert time_reversal < 1e-12


def test_physical_classic_exchange_is_equivariant_under_every_operation(
        exchange_sources):
    """The classic fit on physical AO pairs is closed under all 48 (#466).

    Same cell, list radius and trial density as the negative control above;
    only the AO-pair support differs, so the 42 shift-varying operations
    that miss there must transport here like the SR/LR source.
    """
    results, time_reversal = _equivariance(
        exchange_sources, exchange_sources["caches"]["classic-physical"])
    assert len(results) == 48
    assert sum(spread > 1e-8 for spread, _ in results) == 42
    # Measured: <= 2.3e-15 on every operation (historical ball: 1.3e-2 ..
    # 1.8e-2 on the 42 shift-varying ones).
    assert max(residual for _, residual in results) < 1e-12
    assert time_reversal < 1e-12


@pytest.mark.parametrize("source", ["sr", "classic-physical"])
def test_pair_centred_wedge_exchange_transports_to_the_unreduced_build(
        exchange_sources, source):
    """The executed reduction: K at wedge bras, transported, equals full K,
    on the SR/LR source and on the physical classic fit (#466).

    The density is a symmetric state: a nonsymmetric complex trial density
    projected onto the invariant subspace by averaging over the 48
    operations and time reversal (so each representative is also invariant
    under its little group). The reference contracts the full-mesh cache.
    """
    src = exchange_sources
    system, orbital, kpts = src["system"], src["orbital"], src["kpts"]
    cache, nbf = src["caches"][source], orbital.nbasis
    act, spreads = _group_action(src)
    raw = _trial_density(src, 401)
    images = [act(o, raw) for o in range(len(spreads))]
    images += [act(None, image) for image in images]
    density = [sum(image[j] for image in images) / len(images)
               for j in range(len(kpts))]
    # The projection is invariant, and far from the raw density.
    assert all(_relative(act(o, density), density) < 1e-13
               for o in range(len(spreads)))
    assert _relative(density, raw) > 0.5

    rows, star_map, kfull = kgdf._resolve_ibz_native_state(
        system, kpts, (2, 2, 2), src["weights"], None, exchange_producer=source)
    assert len(rows) == 4
    full = _exchange(src, cache, density)
    reduced = kgdf._build_k_ibz_native(
        {key: value for key, value in cache.items() if key[0] in rows},
        density, src["weights"], rows, nbasis=nbf)
    transported = expand_k_matrices_to_full(reduced, star_map, system, orbital, kfull)
    assert _relative(transported, full) < 1e-12
    # Transporting the unprojected density's wedge rows is not the full build.
    unsymmetric = _exchange(src, cache, raw)
    assert _relative(expand_k_matrices_to_full(
        [unsymmetric[i] for i in rows], star_map, system, orbital, kfull),
        unsymmetric) > 1e-2


# --- Bind the executed quadrature, not only its representatives (#474) ---


@pytest.mark.parametrize("mesh,shift", [
    ((2, 2, 2), (0, 0, 0)), ((2, 2, 2), (1, 1, 1)),
    ((3, 3, 3), (1, 1, 1)), ((2, 2, 3), (0, 0, 1)),
])
@pytest.mark.parametrize("python_mesh", [False, True])
def test_ibz_quadrature_preserves_shift_and_reciprocal_wraps(mesh, shift, python_mesh):
    system, _, _ = _cscl_fixture()
    native = core.monkhorst_pack(system, list(mesh), list(shift), False)
    declared = (vq.KPoints.monkhorst_pack(system, mesh, shift=shift, symmetry=False)
                if python_mesh else native)
    points = np.asarray(native.kpoints)
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(system.lattice)).T
    wraps = np.random.default_rng(474).integers(-3, 4, size=points.shape)
    wrapped = points + wraps @ reciprocal.T
    rows, star, full = kgdf._resolve_ibz_native_state(
        system, wrapped, declared, np.asarray(native.weights), None,
        exchange_producer="sr")
    assert tuple(full.is_shift) == shift
    np.testing.assert_allclose(full.kpoints, points, atol=1e-14)
    frac = points @ np.asarray(system.lattice) / (2 * np.pi)
    # Check each transported relation with an independent reciprocal action.
    for j, (rep, op, trev) in enumerate(star.entries):
        rotation = np.asarray(star.operations[op].rotation)
        image = (-1 if trev else 1) * np.linalg.solve(rotation.T, frac[rows[rep]])
        np.testing.assert_allclose(image - frac[j], np.rint(image - frac[j]), atol=1e-12)


@pytest.mark.parametrize("defect", ["swap", "duplicate", "off-grid", "nan", "inf"])
def test_ibz_quadrature_rejects_corruption_outside_representatives(defect):
    system = _lih_fcc()
    points, weights = _mesh(system, (2, 2, 2))
    rows, _, _ = kgdf._resolve_ibz_native_state(
        system, points, (2, 2, 2), weights, None, exchange_producer="sr")
    a, b = [j for j in range(len(points)) if j not in rows][:2]
    changed = points.copy()
    if defect == "swap":
        changed[[a, b]] = changed[[b, a]]
    elif defect == "duplicate":
        changed[a] = changed[b]
    else:
        changed[a, 0] = {"off-grid": changed[a, 0] + .01,
                         "nan": np.nan, "inf": np.inf}[defect]
    np.testing.assert_array_equal(changed[rows], points[rows])
    with pytest.raises((NotImplementedError, ValueError), match="(every executed k point|finite k points)"):
        kgdf._resolve_ibz_native_state(
            system, changed, (2, 2, 2), weights, None, exchange_producer="sr")


@pytest.mark.parametrize("defect", ["short", "shape", "nan", "inf", "negative", "unnormalized", "near-uniform"])
def test_ibz_quadrature_rejects_invalid_weights(defect):
    system = _lih_fcc()
    points, weights = _mesh(system, (2, 2, 2))
    if defect == "short":
        weights = weights[:-1]
    elif defect == "shape":
        weights = weights[:, None]
    elif defect == "unnormalized":
        weights *= 2
    elif defect == "near-uniform":
        # Passed the former np.allclose(w, w[0]) default rtol=1e-5.
        weights[1] += 1e-7
        weights[2] -= 1e-7
    else:
        weights[1] = {"nan": np.nan, "inf": np.inf, "negative": -.125}[defect]
    with pytest.raises((NotImplementedError, ValueError), match="(weight|quadrature)"):
        kgdf._resolve_ibz_native_state(
            system, points, (2, 2, 2), weights, None, exchange_producer="sr")


@pytest.mark.parametrize("shift", [(1, 0), (1, 0, 2), (1, 0, .5), (True, 0, 0)])
def test_ibz_quadrature_rejects_invalid_shift_metadata(shift):
    from types import SimpleNamespace

    system = _lih_fcc()
    points, weights = _mesh(system, (2, 2, 2))
    with pytest.raises(ValueError, match="half-step"):
        kgdf._resolve_ibz_native_state(
            system, points, SimpleNamespace(mesh=(2, 2, 2), shift=shift),
            weights, None, exchange_producer="sr")


def test_ibz_mesh_aggregate_budget_exact_and_one_short(monkeypatch):
    from vibeqc.symmetry_shared import Budget

    system = _lih_fcc()
    points, weights = _mesh(system, (2, 2, 2))
    required_bytes, required_work = kgdf._ibz_mesh_reservation(8, 48, 2)
    kgdf._resolve_ibz_native_state(
        system, points, (2, 2, 2), weights, None, exchange_producer="sr",
        budget=Budget(required_bytes, required_work))

    class Unscannable:
        def __len__(self):
            return 8

        def __array__(self, *args, **kwargs):
            raise AssertionError("coordinate/weight scan preceded admission")

    def forbidden(*args, **kwargs):
        raise AssertionError("mesh factory or source mapping preceded admission")

    monkeypatch.setattr(core, "monkhorst_pack", forbidden)
    monkeypatch.setattr(kgdf, "_admit_ibz_exchange_operations", forbidden)
    for budget, error in ((Budget(required_bytes - 1, required_work), MemoryError),
                          (Budget(required_bytes, required_work - 1), ValueError)):
        with pytest.raises(error, match="budget exceeded"):
            kgdf._resolve_ibz_native_state(
                system, Unscannable(), (2, 2, 2), Unscannable(), None,
                exchange_producer="sr", budget=budget)


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
def test_ibz_driver_mesh_preflight_precedes_expansion(monkeypatch, driver):
    import vibeqc.memory as memory

    system = _lih_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def forbidden(*args, **kwargs):
        raise AssertionError("quadrature was materialized before resource admission")

    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 1)
    monkeypatch.setattr(kgdf, "_expand_ibz_kmesh_to_full_bz", forbidden)
    monkeypatch.setattr(kgdf, "_kmesh_to_kpoints_weights", forbidden)
    with pytest.raises(MemoryError, match="budget exceeded"):
        getattr(kgdf, driver)(system, basis, kmesh=(2, 2, 2),
                             ibz_native=True, progress=False)


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
def test_ibz_driver_refuses_wrong_full_mesh_before_integrals(monkeypatch, driver):
    system = _lih_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    declared = vq.KPoints.gamma_centred(system, (2, 2, 2))
    declared.kpoints_cart[[1, 2]] = declared.kpoints_cart[[2, 1]]

    def forbidden(*args, **kwargs):
        raise AssertionError("integral work preceded quadrature admission")

    monkeypatch.setattr(kgdf, "compute_overlap_lattice", forbidden)
    monkeypatch.setattr(kgdf, "_build_scf_range_separated_lpq_cache", forbidden)
    with pytest.raises(NotImplementedError, match="every executed k point"):
        getattr(kgdf, driver)(system, basis, kmesh=declared,
                             ibz_native=True, progress=False)


def test_shifted_sr_exchange_matches_full_finite_source():
    """A shifted mesh has an unshifted transfer grid; never use its k as q."""
    system, orbital, auxiliary = _cscl_fixture()
    declared = vq.KPoints.monkhorst_pack(system, (2, 2, 2), shift=(1, 1, 1))
    kpts, weights = np.asarray(declared.kpoints_cart), np.asarray(declared.weights)
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(system.lattice)).T
    frac = np.linalg.solve(reciprocal, kpts.T).T

    def index_of(k):
        delta = frac - np.linalg.solve(reciprocal, k)
        (j,) = np.flatnonzero(np.all(np.abs(delta - np.rint(delta)) < 1e-8, axis=1))
        return int(j)

    cache = {}
    for q in kpts - kpts[0]:
        batch = _build_lpq_range_separated_shared_q(
            system, orbital, auxiliary, kpts, q, omega=.5,
            pair_cutoff=8., auxiliary_cutoff=9., ke_cutoff=6.,
            linear_dep_thr=1e-9, memory_byte_cap=256 << 20,
            native_workspace_byte_cap=32 << 20, image_candidate_cap=10**6,
            reciprocal_candidate_cap=10**6,
        )
        for i, factor in enumerate(batch.factors):
            cache[(i, index_of(kpts[i] + q))] = factor
    src = dict(system=system, orbital=orbital, kpts=kpts, weights=weights,
               index_of=index_of, frac=frac, reciprocal=reciprocal)
    act, spreads = _group_action(src)
    raw = _trial_density(src, 474)
    raw_full = _exchange(src, cache, raw)
    for op in range(len(spreads)):
        assert _relative(_exchange(src, cache, act(op, raw)), act(op, raw_full)) < 1e-11
    assert _relative(_exchange(src, cache, act(None, raw)), act(None, raw_full)) < 1e-11

    # Construct an invariant test density only; no production state is repaired.
    images = [act(op, raw) for op in range(len(spreads))]
    images += [act(None, image) for image in images]
    density = [sum(image[j] for image in images) / len(images) for j in range(8)]
    rows, star, full = kgdf._resolve_ibz_native_state(
        system, kpts, declared, weights, None, exchange_producer="sr")
    assert len(rows) < len(kpts)
    class CountedCache(dict):
        def __init__(self, values):
            super().__init__(values)
            self.used = []

        def __getitem__(self, key):
            self.used.append(key)
            return super().__getitem__(key)

    wedge_cache = CountedCache({key: factor for key, factor in cache.items() if key[0] in rows})
    full_cache = CountedCache(cache)
    reduced = kgdf._build_k_ibz_native(
        wedge_cache,
        density, weights, rows, nbasis=orbital.nbasis)
    reference = _exchange(src, full_cache, density)
    assert len(wedge_cache.used) == len(rows) * 8 == len(set(wedge_cache.used))
    assert len(full_cache.used) == 64 == len(set(full_cache.used))
    assert _relative(expand_k_matrices_to_full(reduced, star, system, orbital, full), reference) < 1e-11
    assert _relative(expand_k_matrices_to_full(
        [raw_full[i] for i in rows], star, system, orbital, full), raw_full) > 1e-2


@pytest.mark.parametrize("python_mesh", [False, True])
def test_runner_preflight_counts_the_shifted_wedge(python_mesh):
    from vibeqc.periodic_runner import _ibz_kpoint_count

    system, _, _ = _cscl_fixture()
    native = core.monkhorst_pack(system, [2, 2, 2], [1, 1, 1], False)
    declared = (vq.KPoints.monkhorst_pack(system, (2, 2, 2), shift=(1, 1, 1))
                if python_mesh else native)
    count = _ibz_kpoint_count(system, declared)
    assert count == 1
    assert _ibz_kpoint_count(system, (2, 2, 2)) == 4
    rows, _, _ = kgdf._resolve_ibz_native_state(
        system, np.asarray(native.kpoints), declared, np.asarray(native.weights),
        None, exchange_producer="sr")
    assert count == len(rows)


def test_runner_mesh_reservation_precedes_native_factories(monkeypatch):
    from vibeqc.periodic_runner import _ibz_kpoint_count
    import vibeqc.memory as memory

    system = _lih_fcc()

    def forbidden(*args, **kwargs):
        raise AssertionError("runner materialized a mesh before admission")

    monkeypatch.setattr(memory, "available_memory_bytes", lambda: 1)
    monkeypatch.setattr(core, "monkhorst_pack", forbidden)
    with pytest.raises(MemoryError, match="budget exceeded"):
        _ibz_kpoint_count(system, (2, 2, 2))


@pytest.mark.parametrize("defect", ["short", "shape", "complex"])
def test_ibz_quadrature_checks_array_metadata_before_conversion(monkeypatch, defect):
    system = _lih_fcc()
    points, weights = _mesh(system, (2, 2, 2))
    if defect == "short":
        points = points[:-1]
    elif defect == "shape":
        points = points[:, None, :]
    else:
        points = points.astype(complex) + 1j

    def forbidden(*args, **kwargs):
        raise AssertionError("mesh materialized for malformed coordinates")

    monkeypatch.setattr(core, "monkhorst_pack", forbidden)
    with pytest.raises((ValueError, NotImplementedError), match="(count|shaped|real numeric)"):
        kgdf._resolve_ibz_native_state(
            system, points, (2, 2, 2), weights, None, exchange_producer="sr")


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
@pytest.mark.parametrize("defect", [
    "complex-points", "complex-weights", "point-shape", "weight-shape",
    "oversized-points", "oversized-mapping",
])
def test_ibz_driver_rejects_original_payload_before_normalization(monkeypatch, driver, defect):
    system = _lih_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    declared = vq.KPoints.gamma_centred(system, (2, 2, 2))
    if defect == "complex-points":
        declared.kpoints_cart = declared.kpoints_cart.astype(complex) + 1j
    elif defect == "complex-weights":
        declared.weights = declared.weights.astype(complex) + 1j
    elif defect == "point-shape":
        declared.kpoints_cart = declared.kpoints_cart[:, None, :]
    elif defect == "weight-shape":
        declared.weights = declared.weights[:, None]
    elif defect == "oversized-points":
        declared.kpoints_cart = np.tile(declared.kpoints_cart, (2, 1))
    else:
        declared.ir_mapping = np.arange(16)

    def forbidden(*args, **kwargs):
        raise AssertionError("original payload was coerced before metadata admission")

    monkeypatch.setattr(kgdf, "_expand_ibz_kmesh_to_full_bz", forbidden)
    monkeypatch.setattr(kgdf, "_kmesh_to_kpoints_weights", forbidden)
    with pytest.raises(ValueError, match="(real numeric|shaped|input point count|ir_mapping)"):
        getattr(kgdf, driver)(system, basis, kmesh=declared,
                             ibz_native=True, progress=False)


@pytest.mark.parametrize("python_mesh", [False, True])
def test_shifted_input_wedge_expands_to_the_same_admitted_parent(python_mesh):
    system = _lih_fcc()
    declared = (vq.KPoints.monkhorst_pack(
        system, (2, 2, 2), shift=(1, 1, 1), symmetry=True) if python_mesh else
        core.monkhorst_pack(system, [2, 2, 2], [1, 1, 1], True))
    assert len(declared) < 8
    kgdf._preflight_ibz_mesh(system, declared)
    full = kgdf._expand_ibz_kmesh_to_full_bz(system, declared)
    assert tuple(full.is_shift) == (1, 1, 1) and len(full) == 8
    rows, star, admitted = kgdf._resolve_ibz_native_state(
        system, np.asarray(full.kpoints), full, np.asarray(full.weights),
        None, exchange_producer="sr")
    assert len(rows) == len(declared) and len(star.entries) == 8
    assert tuple(admitted.is_shift) == (1, 1, 1)


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
@pytest.mark.parametrize("defect", [
    "nan-points", "inf-points", "off-grid", "duplicate", "nan-weights",
    "inf-weights", "negative-weights", "custom-weights", "unnormalized",
    "negative-map", "outside-map", "wrong-orbits", "float-map", "bool-map",
    "map-shape", "wrong-shift",
])
def test_ibz_input_reduction_refuses_before_expansion(monkeypatch, driver, defect):
    system = _lih_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    reduced = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=True)
    assert 1 < len(reduced) < 8
    if defect == "nan-points":
        reduced.kpoints_cart[0, 0] = np.nan
    elif defect == "inf-points":
        reduced.kpoints_cart[0, 0] = np.inf
    elif defect == "off-grid":
        reduced.kpoints_cart[0, 0] += .03
    elif defect == "duplicate":
        reduced.kpoints_cart[0] = reduced.kpoints_cart[1]
    elif defect == "nan-weights":
        reduced.weights[0] = np.nan
    elif defect == "inf-weights":
        reduced.weights[0] = np.inf
    elif defect == "negative-weights":
        reduced.weights[0] = -1
    elif defect == "custom-weights":
        reduced.weights[0] += 1e-7
        reduced.weights[1] -= 1e-7
    elif defect == "unnormalized":
        reduced.weights *= 1.00001
    elif defect == "negative-map":
        reduced.ir_mapping[0] = -1
    elif defect == "outside-map":
        reduced.ir_mapping[0] = len(reduced)
    elif defect == "wrong-orbits":
        reduced.ir_mapping[:] = 0
    elif defect == "float-map":
        reduced.ir_mapping = reduced.ir_mapping.astype(float)
    elif defect == "bool-map":
        reduced.ir_mapping = reduced.ir_mapping.astype(bool)
    elif defect == "map-shape":
        reduced.ir_mapping = reduced.ir_mapping[:, None]
    else:
        reduced.shift = (1, 1, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reduction reached expansion or integral work")

    monkeypatch.setattr(kgdf, "_expand_ibz_kmesh_to_full_bz", forbidden)
    monkeypatch.setattr(kgdf, "_oneel_lattice_opts", forbidden)
    with pytest.raises((ValueError, NotImplementedError), match="(input reduction|ir_mapping)"):
        getattr(kgdf, driver)(system, basis, kmesh=reduced,
                             ibz_native=True, progress=False)


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, 0, 1), (1, 1, 1)])
@pytest.mark.parametrize("kind", ["native", "python", "permuted-wrapped"])
def test_ibz_input_reduction_preserves_genuine_parent(shift, kind):
    system = _lih_fcc()
    reduced = vq.KPoints.monkhorst_pack(system, (2, 2, 2), shift=shift, symmetry=True)
    if kind == "native":
        reduced = reduced.to_bloch_kmesh()
    elif kind == "permuted-wrapped":
        order = np.arange(len(reduced))[::-1]
        inverse = np.argsort(order)
        reduced.kpoints_cart = reduced.kpoints_cart[order].copy()
        reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(system.lattice)).T
        wraps = np.random.default_rng(484).integers(-3, 4, reduced.kpoints_cart.shape)
        reduced.kpoints_cart += wraps @ reciprocal.T
        reduced.weights = reduced.weights[order]
        reduced.ir_mapping = inverse[reduced.ir_mapping]
    kgdf._validate_ibz_input_reduction(system, reduced)
    expanded = kgdf._expand_ibz_kmesh_to_full_bz(system, reduced)
    assert expanded is not None
    # Independent Cartesian coordinates from the last-axis-fastest grid.
    from itertools import product
    frac = (np.array(list(product(range(2), repeat=3))) + np.array(shift) / 2) / 2
    expected = 2 * np.pi * np.linalg.solve(np.asarray(system.lattice).T, frac.T).T
    np.testing.assert_allclose(expanded.kpoints, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(expanded.weights, np.full(8, 1/8))


def test_ibz_input_reduction_requires_consistent_relabeling():
    system = _lih_fcc()
    reduced = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=True)
    reduced.kpoints_cart = reduced.kpoints_cart[::-1].copy()
    reduced.weights = reduced.weights[::-1].copy()
    with pytest.raises(NotImplementedError, match="ir_mapping"):
        kgdf._validate_ibz_input_reduction(system, reduced)


def test_ibz_input_reduction_is_bound_to_current_lattice():
    system = _lih_fcc()
    reduced = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=True)
    other = vq.PeriodicSystem(3, np.asarray(system.lattice) * 1.03,
        [vq.Atom(a.Z, np.asarray(a.xyz) * 1.03) for a in system.unit_cell])
    vq.attach_symmetry(other)
    with pytest.raises(NotImplementedError, match="representative"):
        kgdf._validate_ibz_input_reduction(other, reduced)


def test_ibz_input_reduction_budget_exact_and_one_short(monkeypatch):
    from dataclasses import replace
    from vibeqc.symmetry_shared import Budget

    system = _lih_fcc()
    reduced = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=True)
    n_operations = len(system.symmetry.operations)
    # Independent expression for the unchanged aggregate reservation.
    byte_cap = 1024 * 8 + 512 * n_operations + 64 * n_operations * 2
    work_cap = 128 * n_operations * (8**2 + 2**2) + 256 * 8
    kgdf._validate_ibz_input_reduction(
        system, reduced, budget=Budget(byte_cap, work_cap))

    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("payload conversion before admission")

        @property
        def shape(self):
            raise AssertionError("payload metadata before admission")

    poisoned = replace(reduced, kpoints_cart=Unreadable(), weights=Unreadable(),
                       ir_mapping=Unreadable())

    def forbidden(*args, **kwargs):
        raise AssertionError("native reduction before resource admission")

    monkeypatch.setattr(core, "monkhorst_pack", forbidden)
    for caps, error in [((byte_cap - 1, work_cap), MemoryError),
                        ((byte_cap, work_cap - 1), ValueError)]:
        with pytest.raises(error, match="budget exceeded"):
            kgdf._validate_ibz_input_reduction(system, poisoned, budget=Budget(*caps))


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
@pytest.mark.parametrize("reduction", [False, True])
def test_ibz_input_reduction_reaches_same_driver_full_mesh(monkeypatch, driver, reduction):
    system = _lih_fcc()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    reduced = vq.KPoints.monkhorst_pack(system, (2, 2, 2), symmetry=True)
    expected = vq.KPoints.monkhorst_pack(system, (2, 2, 2), symmetry=False)

    class Reached(Exception):
        pass

    def observe(*args, **kwargs):
        np.testing.assert_array_equal(kwargs["k_points_cart"], expected.kpoints_cart)
        raise Reached

    monkeypatch.setattr(kgdf, "_oneel_lattice_opts", observe)
    with pytest.raises(Reached):
        getattr(kgdf, driver)(system, basis, kmesh=reduced,
                             ibz_native=reduction, progress=False)


def _singleton_system():
    # A triclinic metric has only inversion and identity. Each Gamma mesh
    # coordinate 0 or 1/2 is its own orbit, although grid order still differs.
    lattice = np.array([[5.3, .4, .2], [0., 6.1, .3], [0., 0., 7.2]])
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0., 0., 0.])])
    vq.attach_symmetry(system)
    return system


@pytest.mark.parametrize("driver", [
    "run_krhf_periodic_gdf", "run_kuhf_periodic_gdf",
    "run_krks_periodic_gdf", "run_kuks_periodic_gdf",
])
@pytest.mark.parametrize("mesh,shift", [
    ((2, 2, 2), (0, 0, 0)), ((2, 1, 2), (0, 1, 0)),
    ((1, 1, 1), (1, 1, 1)),
])
@pytest.mark.parametrize("kind", ["python", "native", "permuted-wrapped"])
def test_ibz_singletons_adopt_native_full_order(monkeypatch, driver, mesh, shift, kind):
    from itertools import product

    system = _singleton_system()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    reduced = vq.KPoints.monkhorst_pack(system, mesh, shift=shift, symmetry=True)
    n_full = int(np.prod(mesh))
    assert len(reduced) == n_full
    if kind == "native":
        reduced = reduced.to_bloch_kmesh()
    elif kind == "permuted-wrapped":
        order = np.arange(n_full)[::-1]
        wraps = np.random.default_rng(486).integers(-2, 3, (n_full, 3))
        reduced.kpoints_cart = reduced.kpoints_cart[order] + 2 * np.pi * np.linalg.solve(
            np.asarray(system.lattice).T, wraps.T).T
        reduced.weights = reduced.weights[order]
        reduced.ir_mapping = np.argsort(order)[reduced.ir_mapping]
    # Derive the native full grid independently, with the last axis fastest.
    frac = (np.array(list(product(*(range(n) for n in mesh)))) + np.array(shift) / 2) / mesh
    expected = 2 * np.pi * np.linalg.solve(np.asarray(system.lattice).T, frac.T).T
    original_expand = kgdf._expand_ibz_kmesh_to_full_bz
    original_resolve = kgdf._resolve_ibz_native_state
    seen = []

    def expand(system, normalized):
        # The general expander never receives an unaudited singleton mapping.
        assert len(normalized.ir_mapping) == 0
        return original_expand(system, normalized)

    def resolve(system, points, normalized, weights, *args, **kwargs):
        assert isinstance(normalized, core.BlochKMesh)
        assert tuple(normalized.mesh) == mesh and tuple(normalized.is_shift) == shift
        np.testing.assert_allclose(points, expected, rtol=0, atol=2e-15)
        np.testing.assert_array_equal(weights, np.full(n_full, 1 / n_full))
        result = original_resolve(system, points, normalized, weights, *args, **kwargs)
        assert sorted(result[0]) == list(range(n_full))
        seen.append(normalized)
        return result

    class Reached(Exception):
        pass

    def observe(*args, **kwargs):
        assert len(seen) == 1
        np.testing.assert_allclose(kwargs["k_points_cart"], expected, rtol=0, atol=2e-15)
        raise Reached

    monkeypatch.setattr(kgdf, "_expand_ibz_kmesh_to_full_bz", expand)
    monkeypatch.setattr(kgdf, "_resolve_ibz_native_state", resolve)
    monkeypatch.setattr(kgdf, "_oneel_lattice_opts", observe)
    kwargs = {"functional": "pbe0"} if "ks" in driver else {}
    with pytest.raises(Reached):
        getattr(kgdf, driver)(system, basis, kmesh=reduced,
                             ibz_native=True, progress=False, **kwargs)


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
@pytest.mark.parametrize("defect", [
    "nan-points", "off-grid", "duplicate", "custom-weights", "nan-weights",
    "negative-map", "wrong-map", "float-map", "relabel-only", "wrong-shift",
])
def test_ibz_singletons_refuse_before_full_factory(monkeypatch, driver, defect):
    system = _singleton_system()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    reduced = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=True)
    if defect == "nan-points":
        reduced.kpoints_cart[0, 0] = np.nan
    elif defect == "off-grid":
        reduced.kpoints_cart[0, 0] += .01
    elif defect == "duplicate":
        reduced.kpoints_cart[0] = reduced.kpoints_cart[1]
    elif defect == "custom-weights":
        reduced.weights[0] += .01
        reduced.weights[1] -= .01
    elif defect == "nan-weights":
        reduced.weights[0] = np.nan
    elif defect == "negative-map":
        reduced.ir_mapping[0] = -1
    elif defect == "wrong-map":
        reduced.ir_mapping[:] = 0
    elif defect == "float-map":
        reduced.ir_mapping = reduced.ir_mapping.astype(float)
    elif defect == "relabel-only":
        reduced.ir_mapping = reduced.ir_mapping[::-1].copy()
    else:
        reduced.shift = (1, 1, 1)
    original = core.monkhorst_pack

    def factory(system, mesh, shift, symmetry):
        assert symmetry, "invalid input reached canonical full-mesh allocation"
        return original(system, mesh, shift, symmetry)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid singleton reduction reached expansion or integrals")

    monkeypatch.setattr(core, "monkhorst_pack", factory)
    monkeypatch.setattr(kgdf, "_expand_ibz_kmesh_to_full_bz", forbidden)
    monkeypatch.setattr(kgdf, "_oneel_lattice_opts", forbidden)
    with pytest.raises((ValueError, NotImplementedError), match="(input reduction|ir_mapping)"):
        getattr(kgdf, driver)(system, basis, kmesh=reduced, ibz_native=True, progress=False)


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
@pytest.mark.parametrize("defect", ["reordered", "displaced"])
def test_ibz_singletons_do_not_normalize_unmapped_full_input(monkeypatch, driver, defect):
    system = _singleton_system()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    full = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=False)
    assert full.ir_mapping.size == 0
    if defect == "reordered":
        full.kpoints_cart = full.kpoints_cart[::-1].copy()
    else:
        full.kpoints_cart[0, 0] += .01
    assert kgdf._validate_ibz_input_reduction(system, full) is None

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid unmapped quadrature reached integrals")

    monkeypatch.setattr(kgdf, "_oneel_lattice_opts", forbidden)
    with pytest.raises(NotImplementedError, match="every executed k point"):
        getattr(kgdf, driver)(system, basis, kmesh=full, ibz_native=True, progress=False)


def test_ibz_singletons_budget_precedes_both_mesh_factories(monkeypatch):
    from dataclasses import replace
    from vibeqc.symmetry_shared import Budget

    system = _singleton_system()
    reduced = vq.KPoints.gamma_centred(system, (2, 2, 2), symmetry=True)
    n_ops = len(system.symmetry.operations)
    byte_cap = 1024 * 8 + 512 * n_ops + 64 * n_ops
    work_cap = 128 * n_ops * (8**2 + 1) + 256 * 8
    original = core.monkhorst_pack
    calls = []

    def factory(system, mesh, shift, symmetry):
        calls.append(symmetry)
        return original(system, mesh, shift, symmetry)

    monkeypatch.setattr(core, "monkhorst_pack", factory)
    full = kgdf._validate_ibz_input_reduction(
        system, reduced, budget=Budget(byte_cap, work_cap))
    assert calls == [True, False] and len(full.ir_mapping) == 0

    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("payload read before budget admission")

        @property
        def shape(self):
            raise AssertionError("payload metadata before budget admission")

    poisoned = replace(reduced, kpoints_cart=Unreadable(), weights=Unreadable(),
                       ir_mapping=Unreadable())
    calls.clear()
    for caps, error in [((byte_cap - 1, work_cap), MemoryError),
                        ((byte_cap, work_cap - 1), ValueError)]:
        with pytest.raises(error, match="budget exceeded"):
            kgdf._validate_ibz_input_reduction(system, poisoned, budget=Budget(*caps))
    assert calls == []
