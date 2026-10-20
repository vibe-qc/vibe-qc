"""Non-symmorphic admission for the IBZ-native multi-k GDF exchange (#452).

``run_periodic_job(symmetry_reduce_k=True)`` transports K(k) from the wedge
through admitted operations. A glide or screw ``{R|tau}`` maps home-cell atom
``a`` to ``x_b = R x_a + tau + L_a`` and, in the AO-centred Bloch convention
``phi_{mu,k} = sum_T exp(i k.T) chi_mu(r - x_a - T)``, acts as

    g phi_{mu,k} = exp(i k'.L_a) sum_mu' D_{mu'mu}(R) phi_{mu',k'},   k' = R k,

so the fractional translation enters only through the atom-image shift
``L_a``; an explicit ``exp(-i k.tau)`` is the atom-centred convention's global
phase and cancels in operator transport. These tests prove that law on Bloch
overlaps, against the native Seitz transport, and on explicit finite SR/LR and
classic exchange sources of one finite approximation: transported against
directly built K on nonsymmetric non-Hermitian complex trial densities, for
every operation of diamond Fd-3m and hcp P6_3/mmc, proper and improper, glide
and screw, with and without time reversal. No SCF runs except in the slow
end-to-end energy comparison.
"""
from __future__ import annotations

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
from vibeqc.symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix
from vibeqc.symmetry_lattice import lattice_to_cartesian_rotation
from vibeqc.symmetry_scf import build_ao_permutation_cache


# --- Cells: one diamond-type and one hcp cell, pure s/p/d on both atoms -----


def _diamond():
    """Two carbon atoms in the primitive Fd-3m cell, origin on an atom."""
    a = 6.74
    lattice = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]).T
    frac = [[0., 0., 0.], [.25, .25, .25]]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(6, list(lattice @ f)) for f in frac])
    vq.attach_symmetry(system, symprec=1e-6)
    return system, {6: [(0, .9), (1, .7), (2, .8)]}, {6: [(0, .5), (1, .45), (2, .6)]}


def _hcp():
    """Two magnesium atoms in the P6_3/mmc cell (6_3 screw and c glide)."""
    a, c = 6.06, 1.624 * 6.06
    lattice = np.array([[a, 0., 0.], [-a / 2, a * np.sqrt(3) / 2, 0.], [0., 0., c]]).T
    frac = [[1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75]]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(12, list(lattice @ f)) for f in frac])
    vq.attach_symmetry(system, symprec=1e-6)
    return system, {12: [(0, .6), (1, .5), (2, .7)]}, {12: [(0, .4), (1, .35), (2, .5)]}


_CELLS = {
    "diamond": (_diamond, "Fd-3m", 227, 48, 24),
    "hcp": (_hcp, "P6_3/mmc", 194, 24, 12),
}


def _basis(system, spec, name):
    shells = [core.ShellInfo(i, l, True, [e], [1.], list(atom.xyz))
              for i, atom in enumerate(system.unit_cell) for l, e in spec[int(atom.Z)]]
    return core.BasisSet(system.unit_cell_molecule(), shells, name)


def _mesh(system, mesh):
    kmesh = core.monkhorst_pack(system, list(mesh), [0, 0, 0], False)
    kpts = np.array([np.asarray(k, dtype=float) for k in kmesh.kpoints])
    return kpts, np.full(len(kpts), 1.0 / len(kpts))


def _has_tau(op):
    return bool(np.any(np.abs(np.asarray(op.translation, dtype=float)) > 1e-8))


def _own_atom_action(system, op):
    """Independent atom map and image shifts for ``{W|tau}``: returns
    ``(perm, L_cart)`` with ``x_perm[a] = R x_a + tau + L_a`` found by a plain
    search over lattice images, not through the library's helpers."""
    A = np.asarray(system.lattice, dtype=float)
    R = A @ np.asarray(op.rotation, dtype=float) @ np.linalg.inv(A)
    t = A @ np.asarray(op.translation, dtype=float)
    X = np.array([list(atom.xyz) for atom in system.unit_cell])
    Z = [int(atom.Z) for atom in system.unit_cell]
    perm, shifts = [], []
    for a in range(len(X)):
        moved = R @ X[a] + t
        hits = []
        for b in range(len(X)):
            if Z[b] != Z[a]:
                continue
            f = np.linalg.solve(A, X[b] - moved)
            if np.all(np.abs(f - np.rint(f)) < 1e-6):
                hits.append((b, A @ np.rint(f)))
        assert len(hits) == 1, "operation must map every atom to one image"
        perm.append(hits[0][0])
        shifts.append(hits[0][1])
    return np.asarray(perm), np.asarray(shifts)


def _build_sources(system, orbital, auxiliary, kpts, *, radius):
    """Explicit finite SR/LR and classic caches for every bra/ket pair."""
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(system.lattice)).T
    frac = np.linalg.solve(reciprocal, kpts.T).T

    def index_of(k):
        d = frac - np.linalg.solve(reciprocal, k)
        (hit,) = np.flatnonzero(np.all(np.abs(d - np.rint(d)) < 1e-8, axis=1))
        return int(hit)

    sr = {}
    for m in range(len(kpts)):
        batch = _build_lpq_range_separated_shared_q(
            system, orbital, auxiliary, kpts, kpts[m], omega=.5,
            pair_cutoff=radius, auxiliary_cutoff=radius + 1., ke_cutoff=6.,
            linear_dep_thr=1e-9, memory_byte_cap=512 << 20,
            native_workspace_byte_cap=64 << 20, image_candidate_cap=10**6,
            reciprocal_candidate_cap=10**6,
        )
        for i in range(len(kpts)):
            sr[(i, index_of(kpts[i] + kpts[m]))] = batch.factors[i]
    options = core.LatticeSumOptions()
    options.cutoff_bohr = radius
    modrho = make_modrho_aux_basis(auxiliary, system.unit_cell_molecule())

    def classic_cache(**support):
        # No support policy: the historical origin-centred list.
        return {
            (i, j): np.asarray(build_lpq_bloch_native_fft(
                system, orbital, modrho, kpts[i], kpts[j], ke_cutoff=6.0,
                lat_opts=options, linear_dep_thr=1e-9, **support))
            for i in range(len(kpts)) for j in range(len(kpts))
        }

    return dict(index_of=index_of, frac=frac, reciprocal=reciprocal,
                caches={"sr": sr, "classic": classic_cache(),
                        "classic-physical": classic_cache(
                            support_policy="physical_pairs",
                            memory_byte_cap=4 << 30)})


def _make_cell(name, mesh):
    maker, symbol, number, n_ops, n_tau = _CELLS[name]
    system, orbital_spec, auxiliary_spec = maker()
    assert system.symmetry.international_symbol == symbol
    assert system.symmetry.number == number
    operations = list(system.symmetry.operations)
    assert len(operations) == n_ops
    assert sum(_has_tau(op) for op in operations) == n_tau
    orbital = _basis(system, orbital_spec, f"{name}-orbital")
    auxiliary = _basis(system, auxiliary_spec, f"{name}-auxiliary")
    kpts, weights = _mesh(system, mesh)
    src = _build_sources(system, orbital, auxiliary, kpts, radius=8.0)
    src.update(system=system, orbital=orbital, auxiliary=auxiliary, kpts=kpts,
               weights=weights, mesh=mesh, operations=operations, name=name)
    # Operations whose reciprocal action maps this mesh onto itself: every
    # one on the (2,2,2) meshes, a subgroup on the anisotropic (2,2,3) mesh
    # (the wedge, and so the state the SCF holds, is built from those).
    src["compatible"] = [
        o for o, op in enumerate(operations)
        if all(_on_mesh(src, np.rint(np.linalg.inv(
            np.asarray(op.rotation, dtype=float))).astype(int).T @ q)
               for q in src["frac"])
    ]
    return src


def _on_mesh(src, q):
    d = src["frac"] - q
    return bool(np.any(np.all(np.abs(d - np.rint(d)) < 1e-8, axis=1)))


@pytest.fixture(scope="module", params=sorted(_CELLS))
def cell(request):
    return _make_cell(request.param, (2, 2, 2))


@pytest.fixture(scope="module")
def diamond_223():
    """Diamond on a (2,2,3) mesh: the classic fit's equal-shift subset covers
    these stars through one admitted glide-type operation."""
    return _make_cell("diamond", (2, 2, 3))


# --- Group action on per-k blocks (all operations, time reversal) ----------


def _group_action(src):
    """``act(o, blocks)`` pushes per-k blocks through operation ``o`` exactly as
    ``expand_k_matrices_to_full`` transports K: real Wigner/atom permutation
    times the per-atom lattice-shift Bloch phase at the target momentum.
    ``o=None`` is scalar time reversal. ``drop_phase`` is the negative
    control that forgets ``L_a``. ``o`` must be mesh-compatible."""
    system, orbital, kpts = src["system"], src["orbital"], src["kpts"]
    operations = src["operations"]
    rotations = build_ao_permutation_cache(system, orbital, operations)
    shifts = _op_atom_shifts_cart(system, operations)
    ao_atom = _ao_source_atom_index(orbital)
    n_k = len(kpts)

    def act(o, blocks, *, drop_phase=False):
        out = [None] * n_k
        for j in range(n_k):
            if o is None:
                out[src["index_of"](-kpts[j])] = blocks[j].conj()
                continue
            w = np.asarray(operations[o].rotation, dtype=float)
            w_inv_t = np.rint(np.linalg.inv(w)).astype(int).T
            t = src["index_of"](src["reciprocal"] @ (w_inv_t @ src["frac"][j]))
            if drop_phase:
                phase = np.ones(len(ao_atom))
            else:
                phase = np.exp(1j * (shifts[o] @ kpts[t]))[ao_atom]
            gamma = np.asarray(rotations[o], dtype=float) * phase[None, :]
            out[t] = gamma @ blocks[j] @ gamma.conj().T
        return out

    spreads = [float(np.max(np.abs(s - s[0]))) for s in shifts]
    return act, spreads


def _trial_density(src, seed):
    nbf = src["orbital"].nbasis
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(nbf, nbf)) + 1j * rng.normal(size=(nbf, nbf))
            for _ in range(len(src["kpts"]))]


def _projected_density(src, seed):
    """A symmetric state: the trial density averaged over the mesh-compatible
    group and time reversal, so it is invariant under every operation the
    wedge can use, including each representative's little group."""
    act, spreads = _group_action(src)
    raw = _trial_density(src, seed)
    images = [act(o, raw) for o in src["compatible"]]
    images += [act(None, image) for image in images]
    return raw, [sum(image[j] for image in images) / len(images)
                 for j in range(len(src["kpts"]))]


def _exchange(src, cache, density):
    return [np.asarray(x) for x in _build_k_from_lpq_cache(
        cache, density, src["weights"], nbasis=src["orbital"].nbasis)]


def _relative(values, references):
    worst = max(float(np.max(np.abs(v - r))) for v, r in zip(values, references))
    return worst / max(float(np.max(np.abs(r))) for r in references)


# --- Admission ---------------------------------------------------------------


def test_every_space_group_operation_is_a_candidate(cell):
    """SR/LR admits all operations; classic admits those with equal image
    shifts, whether or not they carry a fractional translation. The expected
    classic set comes from an independent atom-image search."""
    system, operations = cell["system"], cell["operations"]
    sr = kgdf._admit_ibz_exchange_operations(system, "sr")
    assert sr.support == "pair-centred"
    assert len(sr.operations) == len(operations)
    assert sr.admitted == tuple(range(len(operations))) and sr.refused == ()
    physical = kgdf._admit_ibz_exchange_operations(system, "classic-physical")
    assert (physical.support, physical.admitted, physical.refused) == (
        "pair-centred", sr.admitted, ())

    classic = kgdf._admit_ibz_exchange_operations(system, "classic")
    expected = []
    for index, op in enumerate(operations):
        _, shifts = _own_atom_action(system, op)
        if np.max(np.abs(shifts - shifts[0])) < 1e-8:
            expected.append(index)
    assert classic.support == "origin-centred"
    assert list(classic.admitted) == expected
    assert {i for i, _ in classic.refused} == set(range(len(operations))) - set(expected)
    with_tau = sum(_has_tau(operations[i]) for i in classic.admitted)
    assert (len(expected), with_tau) == {"diamond": (12, 6), "hcp": (4, 0)}[cell["name"]]
    # The admitted and refused operations are the attached objects themselves.
    assert all(classic.operations[i] is operations[i] for i in classic.admitted)


def test_star_map_takes_admitted_glides_but_not_stripped_ones(cell):
    system, mesh = cell["system"], cell["mesh"]
    kfull = core.monkhorst_pack(system, list(mesh), [0, 0, 0], False)
    kibz = core.monkhorst_pack(system, list(mesh), [0, 0, 0], True)
    operations = cell["operations"]
    star = star_operations(system, kibz, kfull, operations=operations)
    used = {o for _, o, _ in star.entries}
    assert any(_has_tau(operations[o]) for o in used)
    # The same rotation without its translation is not an attached operation.
    glide = next(op for op in operations if _has_tau(op))
    stripped = core._make_periodic_ao_bloch_operation(
        np.ascontiguousarray(np.asarray(glide.rotation, dtype=np.int32)),
        np.zeros(3))
    with pytest.raises(ValueError, match="subset"):
        star_operations(system, kibz, kfull, operations=[stripped])
    # The default remains the symmorphic subgroup for other consumers.
    try:
        default = star_operations(system, kibz, kfull)
    except ValueError as refusal:
        assert "not in the star" in str(refusal)
    else:
        assert not any(_has_tau(op) for op in default.operations)


def test_uncovered_wedge_on_an_incompatible_mesh_is_refused_by_name():
    """SR/LR admits every operation, so a wedge the mesh-compatible
    operations cannot cover (diamond at (2,3,4)) is an ``ibz_native``
    refusal naming the mesh, not a bare star-map error (reviewer
    reservation on #452)."""
    system, _, _ = _diamond()
    mesh = (2, 3, 4)
    kpts, weights = _mesh(system, mesh)
    with pytest.raises(NotImplementedError, match="whole mesh onto itself") as refusal:
        kgdf._resolve_ibz_native_state(
            system, kpts, mesh, weights, None, exchange_producer="sr")
    message = str(refusal.value)
    assert "ibz_native=True" in message and "(2, 3, 4)" in message
    assert "non-symmorphic" not in message and "symmetry break" not in message
    # Classic on the same mesh is refused for its source first.
    with pytest.raises(NotImplementedError, match="origin-centred"):
        kgdf._resolve_ibz_native_state(
            system, kpts, mesh, weights, None, exchange_producer="classic")


# --- The Bloch law on overlaps and against the native Seitz transport -------


def _overlap_lattice(src, cutoff=36.0):
    options = core.LatticeSumOptions()
    options.cutoff_bohr = cutoff
    lms = core.compute_overlap_lattice(src["orbital"], src["system"], options)
    cells = np.array([np.asarray(c.r_cart, dtype=float) for c in lms.cells])
    blocks = np.array([np.asarray(b, dtype=float) for b in lms.blocks])
    return cells, blocks


def test_bloch_overlap_transports_through_glides_and_screws(cell):
    """``S(Rk) = G S(k) G^H`` with ``G = P(R) diag(exp(i k'.L_a))`` on off-mesh
    and on-mesh k, every operation and time reversal. Negative controls:
    the per-atom phases dropped (the atom-centred law transplanted into the
    AO-centred convention), image shifts rounded without tau, and a wrong
    operation. An explicit ``exp(-i k'.tau)`` on top is a gauge and passes."""
    system, orbital, operations = cell["system"], cell["orbital"], cell["operations"]
    A = np.asarray(system.lattice, dtype=float)
    cells, blocks = _overlap_lattice(cell)
    ao_atom = _ao_source_atom_index(orbital)
    X = np.array([list(atom.xyz) for atom in system.unit_cell])

    def S(k):
        return np.einsum("c,cij->ij", np.exp(1j * (cells @ k)), blocks)

    rng = np.random.default_rng(7)
    ks = [rng.normal(size=3) * 0.3 for _ in range(3)] + list(cell["kpts"][:3])
    S_k = [S(k) for k in ks]
    worst = {"law": 0., "tau_gauge": 0., "trev": 0.}
    dropped, without_tau, wrong = [], [], []
    for index, op in enumerate(operations):
        R = lattice_to_cartesian_rotation(np.asarray(op.rotation, dtype=int), A)
        t = A @ np.asarray(op.translation, dtype=float)
        ap = atom_permutation_under_op(system, R, t)
        L = np.asarray(ap.lattice_shift, dtype=float) @ A.T
        P = build_ao_permutation_matrix(orbital, R, ap)
        perm, own = _own_atom_action(system, op)
        assert np.array_equal(perm, ap.perm) and np.allclose(own, L, atol=1e-9)
        L_wrong = np.array([A @ np.rint(np.linalg.solve(A, X[ap.perm[a]] - R @ X[a]))
                            for a in range(len(X))])
        shift_matters = np.max(np.abs((L_wrong - L_wrong[0]) - (L - L[0]))) > 1e-8
        other = operations[(index + 7) % len(operations)]
        R_other = lattice_to_cartesian_rotation(np.asarray(other.rotation, dtype=int), A)
        res = dict.fromkeys(("law", "tau_gauge", "trev", "dropped", "without_tau", "wrong"), 0.)
        for k, Sk in zip(ks, S_k):
            kp = R @ k
            target = S(kp)
            scale = float(np.max(np.abs(target)))
            G = P * np.exp(1j * (L @ kp))[ao_atom][None, :]
            res["law"] = max(res["law"], np.max(np.abs(G @ Sk @ G.conj().T - target)) / scale)
            G_gauge = np.exp(-1j * (kp @ t)) * G
            res["tau_gauge"] = max(res["tau_gauge"], np.max(np.abs(G_gauge @ Sk @ G_gauge.conj().T - target)) / scale)
            res["trev"] = max(res["trev"], np.max(np.abs((G @ Sk @ G.conj().T).conj() - S(-kp))) / scale)
            G_dropped = np.exp(-1j * (kp @ t)) * P.astype(complex)
            res["dropped"] = max(res["dropped"], np.max(np.abs(G_dropped @ Sk @ G_dropped.conj().T - target)) / scale)
            G_without = P * np.exp(1j * (L_wrong @ kp))[ao_atom][None, :]
            res["without_tau"] = max(res["without_tau"], np.max(np.abs(G_without @ Sk @ G_without.conj().T - target)) / scale)
            other_target = S(R_other @ k)
            res["wrong"] = max(res["wrong"], np.max(np.abs(G @ Sk @ G.conj().T - other_target)) / np.max(np.abs(other_target)))
        for key in worst:
            worst[key] = max(worst[key], res[key])
        if np.max(np.abs(L - L[0])) > 1e-8:
            dropped.append(res["dropped"])
        if shift_matters:
            without_tau.append(res["without_tau"])
        wrong.append(res["wrong"])
    # Measured: diamond 6.7e-16, hcp 1.5e-15 (law, gauge and time reversal).
    assert worst["law"] < 1e-13 and worst["tau_gauge"] < 1e-13 and worst["trev"] < 1e-13
    # Measured: diamond 0.21 to 0.26 over 36 operations; hcp 2.2e-3 to 5.2e-3 over 20.
    assert len(dropped) == {"diamond": 36, "hcp": 20}[cell["name"]]
    assert min(dropped) > 1e-3
    assert min(wrong) > 1e-3
    if cell["name"] == "hcp":
        # With the origin off the atoms, dropping tau from the image shift
        # changes 12 relative shifts; measured 3.4e-3.
        assert len(without_tau) == 12 and min(without_tau) > 1e-3
    else:
        # With the origin on an atom the rounding happens to agree; the
        # phase controls above carry the negative evidence there.
        assert without_tau == []


def _transport_controls():
    options = core._PeriodicAOBlochTransportOptions()
    options.maximum_atom_mapping_residual_bohr = 1e-9
    options.maximum_basis_origin_residual_bohr = 1e-10
    options.maximum_rotation_orthogonality_residual = 1e-11
    options.maximum_polynomial_reconstruction_residual = 1e-10
    options.maximum_pure_rotation_unitarity_residual = 1e-10
    options.minimum_relative_lattice_volume = 1e-10
    inventory = core._PeriodicAOBlochTransportInventory()
    inventory.numerical_replicas = 1
    inventory.backend_margin_bytes_per_replica = 1 << 20
    caps = core._PeriodicAOBlochTransportCaps()
    for field in ("maximum_atoms", "maximum_shells", "maximum_contractions",
                  "maximum_basis_functions", "maximum_columns"):
        setattr(caps, field, 512)
    caps.maximum_basis_numeric_lanes = 10000
    for field in ("maximum_borrowed_numerical_bytes", "maximum_owned_numerical_bytes",
                  "maximum_control_storage_bytes", "maximum_per_replica_inventoried_bytes",
                  "maximum_node_inventoried_bytes"):
        setattr(caps, field, 16 << 20)
    caps.maximum_work_units = 10**9
    return options, inventory, caps


def test_python_transport_matches_the_native_seitz_action(cell):
    """The transport's Gamma equals the native ``periodic_ao_bloch_transport``
    action on the identity panel, which carries the full Seitz translation
    through the atom image and adds no separate global phase."""
    system, orbital, operations = cell["system"], cell["orbital"], cell["operations"]
    A = np.asarray(system.lattice, dtype=float)
    B = 2 * np.pi * np.linalg.inv(A).T
    grid = core._RegularKMesh(list(cell["mesh"]))
    ao_atom = _ao_source_atom_index(orbital)
    identity = np.ascontiguousarray(np.eye(orbital.nbasis, dtype=complex))
    controls = _transport_controls()
    worst = 0.
    for source in range(len(grid)):
        k = B @ np.asarray(grid.fractional_at(source), dtype=float)
        for trev in (False, True):
            for op in operations:
                R = lattice_to_cartesian_rotation(np.asarray(op.rotation, dtype=int), A)
                ap = atom_permutation_under_op(system, R, A @ np.asarray(op.translation, dtype=float))
                L = np.asarray(ap.lattice_shift, dtype=float) @ A.T
                P = build_ao_permutation_matrix(orbital, R, ap)
                result = core._apply_periodic_ao_bloch_operation(
                    orbital, system, op, grid, source, trev, identity, *controls)
                q_target = np.asarray(grid.fractional_at(result.memory.target_index), dtype=float)
                d = q_target - np.linalg.solve(B, (-1 if trev else 1) * (R @ k))
                assert np.all(np.abs(d - np.rint(d)) < 1e-9)
                gamma = P * np.exp(1j * (L @ (B @ q_target)))[ao_atom][None, :]
                worst = max(worst, float(np.max(np.abs(result.coefficients_copy() - gamma))))
    # Measured: diamond 7.4e-16, hcp 9.2e-16.
    assert worst < 1e-13


# --- Exchange equivariance on explicit finite sources -----------------------


def test_sr_exchange_is_equivariant_under_every_operation(cell):
    act, spreads = _group_action(cell)
    cache = cell["caches"]["sr"]
    density = _trial_density(cell, 2026)
    reference = _exchange(cell, cache, density)
    residuals = [_relative(_exchange(cell, cache, act(o, density)), act(o, reference))
                 for o in range(len(spreads))]
    n_tau = sum(_has_tau(op) for op in cell["operations"])
    assert len(residuals) == len(cell["operations"])
    assert sum(s > 1e-8 for s in spreads) == {"diamond": 36, "hcp": 20}[cell["name"]]
    # Measured: diamond <= 6.7e-15 (24 glide/inversion-type, 36 shift-varying),
    # hcp <= 2.0e-15 (12 screw/glide, 20 shift-varying).
    assert max(residuals) < 1e-12
    assert max(r for r, op in zip(residuals, cell["operations"]) if _has_tau(op)) < 1e-12
    assert n_tau == {"diamond": 24, "hcp": 12}[cell["name"]]
    assert _relative(_exchange(cell, cache, act(None, density)), act(None, reference)) < 1e-12
    # Negative controls: the per-atom phase dropped (measured >= 0.37 and
    # >= 3.9e-2), and one glide's K compared with another operation's image.
    dropped = [_relative(_exchange(cell, cache, act(o, density, drop_phase=True)),
                         act(o, reference, drop_phase=True))
               for o, s in enumerate(spreads) if s > 1e-8]
    assert min(dropped) > 1e-2
    glide = next(o for o, op in enumerate(cell["operations"]) if _has_tau(op))
    assert _relative(_exchange(cell, cache, act(glide, density)), act(0, reference)) > 1e-1


def test_classic_exchange_admits_only_equal_shift_operations(cell):
    """The origin-centred ball is closed under an operation iff all image
    shifts agree, tau included: equal-shift glides pass to roundoff, the
    others miss (the negative control for the admission criterion)."""
    act, spreads = _group_action(cell)
    cache = cell["caches"]["classic"]
    density = _trial_density(cell, 2026)
    reference = _exchange(cell, cache, density)
    residuals = [_relative(_exchange(cell, cache, act(o, density)), act(o, reference))
                 for o in range(len(spreads))]
    admission = kgdf._admit_ibz_exchange_operations(cell["system"], "classic")
    equal = [residuals[o] for o in admission.admitted]
    varying = [residuals[o] for o, _ in admission.refused]
    assert all(spreads[o] <= 1e-8 for o in admission.admitted)
    assert all(spreads[o] > 1e-8 for o, _ in admission.refused)
    # Measured: diamond 12 equal (6 with tau) <= 6.9e-15, 36 others 1.6e-4 to
    # 1.9e-4; hcp 4 equal <= 2.0e-15, 20 others 7.5e-8 to 1.2e-3.
    assert max(equal) < 1e-12
    assert min(varying) > {"diamond": 1e-5, "hcp": 1e-9}[cell["name"]]
    assert _relative(_exchange(cell, cache, act(None, density)), act(None, reference)) < 1e-12


def test_physical_classic_exchange_admits_every_operation(cell):
    """On physical AO pairs (#466) the classic fit is closed under every
    operation, glides and screws included, like the SR/LR source."""
    act, spreads = _group_action(cell)
    cache = cell["caches"]["classic-physical"]
    density = _trial_density(cell, 2026)
    reference = _exchange(cell, cache, density)
    residuals = [_relative(_exchange(cell, cache, act(o, density)), act(o, reference))
                 for o in range(len(spreads))]
    admission = kgdf._admit_ibz_exchange_operations(cell["system"], "classic-physical")
    assert admission.admitted == tuple(range(len(spreads)))
    assert any(_has_tau(cell["operations"][o]) and spreads[o] > 1e-8
               for o in range(len(spreads)))
    # Measured: diamond <= 8.5e-15 over 48 (24 with tau), hcp <= 2.1e-15 over 24.
    assert max(residuals) < 1e-12
    assert _relative(_exchange(cell, cache, act(None, density)), act(None, reference)) < 1e-12


# --- Stabilizers and the executed reduction --------------------------------


def _little_group(src, j):
    """Operations (with the time-reversal flag) fixing full-mesh point j."""
    stabilizer = []
    for o, op in enumerate(src["operations"]):
        w_inv_t = np.rint(np.linalg.inv(np.asarray(op.rotation, dtype=float))).astype(int).T
        q = w_inv_t @ src["frac"][j]
        for trev, sign in ((False, 1), (True, -1)):
            d = sign * q - src["frac"][j]
            if np.all(np.abs(d - np.rint(d)) < 1e-8):
                stabilizer.append((o, trev))
    return stabilizer


def _check_reduction(src, producer, *, projected, raw, tolerance=1e-12):
    system, orbital, kpts = src["system"], src["orbital"], src["kpts"]
    cache = src["caches"][producer]
    rows, star_map, kfull = kgdf._resolve_ibz_native_state(
        system, kpts, src["mesh"], src["weights"], None, exchange_producer=producer)
    assert len(rows) < len(kpts)
    used = {o for _, o, _ in star_map.entries}
    assert any(_has_tau(star_map.operations[o]) for o in used)
    # Every transporting operation maps the whole mesh onto itself.
    assert all(src["operations"].index(star_map.operations[o]) in src["compatible"]
               for o in used)
    full = _exchange(src, cache, projected)
    reduced = kgdf._build_k_ibz_native(
        {key: value for key, value in cache.items() if key[0] in rows},
        projected, src["weights"], rows, nbasis=orbital.nbasis)
    transported = expand_k_matrices_to_full(reduced, star_map, system, orbital, kfull)
    assert _relative(transported, full) < tolerance
    # Transporting the unprojected density's wedge rows is not the full build.
    unsymmetric = _exchange(src, cache, raw)
    assert _relative(expand_k_matrices_to_full(
        [unsymmetric[i] for i in rows], star_map, system, orbital, kfull),
        unsymmetric) > 1e-2
    return rows, star_map, full


def test_stabilizers_fix_the_wedge_exchange_and_reduction_matches_full_build(cell):
    """Projected symmetric density: every little-group member, glides and
    screws included, fixes K at its representative; K built at the wedge
    bras and transported through the star map equals the full-mesh build."""
    act, spreads = _group_action(cell)
    assert cell["compatible"] == list(range(len(spreads)))
    raw, projected = _projected_density(cell, 401)
    assert all(_relative(act(o, projected), projected) < 1e-13 for o in range(len(spreads)))
    assert _relative(projected, raw) > 0.5
    rows, star_map, full = _check_reduction(cell, "sr", projected=projected, raw=raw)
    checked = 0
    for i in rows:
        for o, trev in _little_group(cell, i):
            image = act(o, full)
            if trev:
                image = act(None, image)
            assert _relative([image[i]], [full[i]]) < 1e-12
            checked += 1
    assert checked > len(rows)
    # Stabilizers with a fractional translation were exercised.
    assert any(_has_tau(cell["operations"][o]) for i in rows for o, _ in _little_group(cell, i))


def test_classic_reduction_runs_through_an_admitted_glide(diamond_223):
    """At (2,2,3) the diamond classic fit's 12 equal-shift operations cover the
    stars, through one glide-type operation; the reduction is exact there and
    refused at (2,2,2) and (3,3,3), where other operations would be needed.

    The anisotropic mesh also pins the star map's whole-mesh rule: an
    operation that sends one representative onto a mesh point without
    permuting the mesh is not used (measured 0.34 on the SR reduction when
    the first such match was taken), and a star map built from incompatible
    operations alone is refused.
    """
    src = diamond_223
    system, mesh = src["system"], src["mesh"]
    # Only a subgroup maps a (2,2,3) mesh onto itself.
    assert 1 < len(src["compatible"]) < len(src["operations"])
    kfull = core.monkhorst_pack(system, list(mesh), [0, 0, 0], False)
    kibz = core.monkhorst_pack(system, list(mesh), [0, 0, 0], True)
    incompatible = [op for o, op in enumerate(src["operations"]) if o not in src["compatible"]]
    with pytest.raises(ValueError, match="mesh-compatible"):
        star_operations(system, kibz, kfull, operations=incompatible)
    star = star_operations(system, kibz, kfull, operations=src["operations"])
    assert {src["operations"].index(star.operations[o]) for _, o, _ in star.entries} <= set(src["compatible"])
    raw, projected = _projected_density(src, 402)
    rows, star_map, _ = _check_reduction(src, "classic", projected=projected, raw=raw)
    used = {o for _, o, _ in star_map.entries}
    assert len(used) == 2 and sum(_has_tau(star_map.operations[o]) for o in used) == 1
    _check_reduction(src, "sr", projected=projected, raw=raw)
    for mesh in ((2, 2, 2), (3, 3, 3)):
        kpts, weights = _mesh(src["system"], mesh)
        with pytest.raises(NotImplementedError, match="12 of the 48") as refusal:
            kgdf._resolve_ibz_native_state(
                src["system"], kpts, mesh, weights, None, exchange_producer="classic")
        assert "non-symmorphic" not in str(refusal.value)
        assert "symmetry break" not in str(refusal.value)


# --- Driver: refusals precede every integral; glides pass the gate ---------


class _BuildReached(Exception):
    pass


def _driver_cell():
    """Si-free diamond carbon in STO-3G: the driver-level non-symmorphic cell."""
    system, _, _ = _diamond()
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _driver_options(cutoff=14.0):
    options = vq.PeriodicRHFOptions()
    options.max_iter = 2
    options.lattice_opts.cutoff_bohr = cutoff
    return options


@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
def test_driver_refuses_classic_and_unqualified_sources_before_any_build(monkeypatch, driver):
    def forbidden(*args, **kwargs):
        raise AssertionError("an exchange cderi build ran before admission")

    for name in ("_build_rsgdf_lpq_cache_shared_q", "_build_scf_range_separated_lpq_cache"):
        monkeypatch.setattr(kgdf, name, forbidden)
    system, basis = _driver_cell()
    run = getattr(kgdf, driver)
    with pytest.raises(NotImplementedError, match="12 of the 48") as refusal:
        run(system, basis, kmesh=(2, 2, 2), options=_driver_options(),
            aux_basis="def2-svp-jk", fit_source="classic", ibz_native=True,
            classic_support_policy="historical_index_ball", progress=False)
    assert "origin-centred" in str(refusal.value)
    assert "non-symmorphic" not in str(refusal.value)
    with pytest.raises(NotImplementedError, match="no qualified symmetry action"):
        run(system, basis, kmesh=(2, 2, 2), options=_driver_options(),
            aux_basis="def2-svp-jk", gdf_method="compcell", ibz_native=True, progress=False)


@pytest.mark.parametrize("controls", [{}, {"fit_source": "classic"}],
                         ids=["sr", "classic-physical"])
@pytest.mark.parametrize("driver", ["run_krhf_periodic_gdf", "run_kuhf_periodic_gdf"])
def test_driver_passes_a_non_symmorphic_group_to_the_pair_centred_build(
        monkeypatch, driver, controls):
    """Positive control: on the SR/LR source and on the physical classic fit
    (the classic default, #466) the glide group is admitted and the run
    reaches the exchange cderi build (stopped there by a sentinel)."""
    declared = []

    def intercept(*args, **kwargs):
        declared.append(True)
        raise _BuildReached

    for name in ("_build_rsgdf_lpq_cache_shared_q", "_build_scf_range_separated_lpq_cache"):
        monkeypatch.setattr(kgdf, name, intercept)
    system, basis = _driver_cell()
    with pytest.raises(_BuildReached):
        getattr(kgdf, driver)(
            system, basis, kmesh=(2, 2, 2), options=_driver_options(),
            aux_basis="def2-svp-jk", ibz_native=True, progress=False, **controls)
    assert declared == [True]


# --- End to end: the wedge-native SCF lands on the full-BZ energy ----------


def _he_cell(name):
    """Two helium atoms on the diamond or hcp sites: the cheapest cells whose
    space groups carry glides and screws (STO-3G, one s function per atom)."""
    if name == "diamond":
        a = 6.0
        lattice = 0.5 * a * np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]).T
        frac = [[0., 0., 0.], [.25, .25, .25]]
    else:
        a, c = 5.6, 1.633 * 5.6
        lattice = np.array([[a, 0., 0.], [-a / 2, a * np.sqrt(3) / 2, 0.], [0., 0., c]]).T
        frac = [[1 / 3, 2 / 3, .25], [2 / 3, 1 / 3, .75]]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, list(lattice @ f)) for f in frac])
    vq.attach_symmetry(system, symprec=1e-6)
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["diamond", "hcp"])
def test_ibz_native_reproduces_the_full_bz_energy_on_a_non_symmorphic_cell(name):
    """SR/LR source, (2,2,2), 12-bohr cell list: the wedge SCF, whose star map
    transports through a glide (diamond) or the 6_3 screw/glide (hcp),
    converges to the full-mesh energy. Measured: both agree to every printed
    digit (dE = 0.0 Ha at 1e-12), 43 s against 57 s and 32 s against 42 s on
    a shared host, which is a parity statement and not a speedup claim."""
    system, basis = _he_cell(name)
    assert system.symmetry.number == {"diamond": 227, "hcp": 194}[name]

    def options():
        o = vq.PeriodicRHFOptions()
        o.max_iter = 40
        o.conv_tol_energy = 1e-9
        o.lattice_opts.cutoff_bohr = 12.0
        return o

    common = dict(kmesh=(2, 2, 2), aux_basis="def2-svp-jk", rsgdf_ke_cutoff=80.0,
                  progress=False)
    full = kgdf.run_krhf_periodic_gdf(system, basis, options=options(), **common)
    native = kgdf.run_krhf_periodic_gdf(system, basis, options=options(),
                                        ibz_native=True, **common)
    assert full.converged and native.converged
    assert abs(full.energy - native.energy) < 1e-8, (full.energy, native.energy)
    assert len(native.density) == len(full.density)
