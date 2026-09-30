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
            progress=False,
        )


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
@pytest.mark.parametrize("controls, producer", [
    ({}, "sr"),
    ({"fit_source": "sr", "fit_screen_threshold": 1e-9}, "sr"),
    ({"fit_source": "classic"}, "classic"),
    ({"fit_source": "classic", "fit_screen_threshold": 1e-9}, "classic-screened"),
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
    # Classic: the all-reciprocal fit over one origin-centred 8-bohr list.
    options = core.LatticeSumOptions()
    options.cutoff_bohr = 8.0
    modrho = make_modrho_aux_basis(auxiliary, system.unit_cell_molecule())
    classic = {
        (i, j): np.asarray(build_lpq_bloch_native_fft(
            system, orbital, modrho, kpts[i], kpts[j], ke_cutoff=6.0,
            lat_opts=options, linear_dep_thr=1e-9))
        for i in range(len(kpts)) for j in range(len(kpts))
    }
    return dict(system=system, orbital=orbital, kpts=kpts, weights=weights,
                index_of=index_of, frac=frac, reciprocal=reciprocal,
                caches={"sr": sr, "classic": classic})


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
    # Measured: 2.0e-15 against 1.7e-2 .. 2.6e-2 at this 8-bohr list.
    assert max(constant) < 1e-12
    assert min(varying) > 1e-3
    assert time_reversal < 1e-12


def test_sr_wedge_exchange_transports_to_the_unreduced_build(exchange_sources):
    """The executed reduction: K at wedge bras, transported, equals full K.

    The density is a symmetric state: a nonsymmetric complex trial density
    projected onto the invariant subspace by averaging over the 48
    operations and time reversal (so each representative is also invariant
    under its little group). The reference contracts the full-mesh cache.
    """
    src = exchange_sources
    system, orbital, kpts = src["system"], src["orbital"], src["kpts"]
    cache, nbf = src["caches"]["sr"], orbital.nbasis
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
        system, kpts, (2, 2, 2), src["weights"], None, exchange_producer="sr")
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
