"""Actual finite-source Gaussian RHF, independently contracted tiny He cells.

The onsite AO-image cutoff and finite reciprocal/Ewald cuts are deliberate
test Hamiltonians, not converged bulk He or production DLPNO benchmarks.
No external QC runtime, target-system job, damping, DIIS or shift is used.
"""

from __future__ import annotations

import gc
import hashlib
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _system
from tests.test_periodic_gaussian_source_context import (
    _exact_caps as _basis_caps, _make as _context, _options as _source_options,
)
from tests.test_periodic_gaussian_one_electron import (
    _normalized_basis, _options as _one_options, _live as _one_live, _caps as _one_caps,
)
from tests.test_periodic_gaussian_nuclear import (
    _options as _nuclear_options, _live as _nuclear_live, _caps as _nuclear_caps,
)
from tests.test_periodic_gaussian_fock import (
    _bundle as _fock_bundle, _caps as _fock_caps, _q_data, _physical_tile,
)
from tests.test_periodic_gaussian_metric import _live
from tests.test_bounded_periodic_rhf_solver import _options as _scf_options


_CAP_TO_PLAN = {
    "maximum_owned_numeric_bytes": "owned_numeric_upper_bound",
    "maximum_per_replica_inventoried_bytes": "per_replica_inventoried_bytes",
    "maximum_node_inventoried_bytes": "node_inventoried_bytes",
    "maximum_state_numeric_bytes": "state_bytes_upper_bound",
    "maximum_one_electron_panel_calls": "one_electron_panel_calls",
    "maximum_progress_callbacks": "progress_callback_upper_bound",
    "maximum_work_units": "work_units_upper_bound",
}


def _bundle(mesh=(1, 1, 1), *, iterations=48, pair_block=4):
    # Libint normalizes both primitive s functions explicitly. No fetched
    # chemical basis data or orbital-energy labels enter this fixture.
    ao = _normalized_basis([(0, (0, 0, 0), [1.8], [1.0], True),
                            (0, (0, 0, 0), [0.35], [1.0], True)])
    auxiliary = _normalized_basis([(0, (0, 0, 0), [0.6], [1.0], True),
                                   (0, (0, 0, 0), [1.3], [1.0], True)])
    system = _system(np.eye(3)*8.0)
    system.unit_cell = [core.Atom(2, [0.0, 0.0, 0.0])]
    system.charge, system.multiplicity = 0, 1
    source_options = _source_options()
    source_options.reciprocal_energy_cutoff = 2.0
    source_options.ao_pair_image_cutoff_bohr = 2.0
    context = _context(system=system, ao=ao, auxiliary=auxiliary,
                       mesh=core._RegularKMesh(list(mesh)), options=source_options)
    b = SimpleNamespace(ao=ao, auxiliary=auxiliary, system=system, context=context,
                        mask=np.zeros((context.n_kpoints, 1), dtype=np.uint8), live=_live())
    # Reuse only explicit bounded Fock controls, not its old physical inputs.
    b.config = _fock_bundle(mesh=mesh).config
    metric, tile = b.config.metric, b.config.tile
    metric.basis_verification_caps = tile.basis_verification_caps = _basis_caps(context)
    b.config.metric, b.config.tile = metric, tile
    options = core._PeriodicGaussianRHFOptions()
    options.one_electron_pair_block = pair_block
    options.one_electron = _one_options(b)
    options.nuclear = _nuclear_options(b, alpha=0.6, rcut=3.0, gcut=2.0)
    options.scf = _scf_options(maximum_iterations=iterations,
        maximum_diis_history=0, minimum_band_gap_hartree=0.01,
        density_closure_absolute_tolerance=2e-13, density_closure_relative_tolerance=2e-13,
        commutator_tolerance=1e-11, energy_change_tolerance=2e-13)
    b.options = options
    caps = core._PeriodicGaussianRHFCaps()
    one = _one_caps()
    one.maximum_owned_numeric_bytes = 65536
    one.maximum_per_replica_inventoried_bytes = 2**24
    one.maximum_node_inventoried_bytes = 2**26
    nuclear = _nuclear_caps()
    nuclear.maximum_owned_numeric_bytes = 2**20
    nuclear.maximum_per_replica_inventoried_bytes = 2**24
    nuclear.maximum_node_inventoried_bytes = 2**26
    nuclear.maximum_atom_count = 1
    nuclear.maximum_reciprocal_candidates = 65536
    caps.one_electron, caps.nuclear, caps.fock = one, nuclear, _fock_caps()
    scf = core._BoundedPeriodicRHFCaps()
    scf.maximum_owned_numerical_bytes = 2**20
    scf.maximum_total_numerical_bytes = 2**22
    scf.maximum_work_units = 10**18
    scf.maximum_fock_calls = iterations
    scf.other_live_numerical_bytes = 0
    caps.scf = scf
    caps.maximum_owned_numeric_bytes = 2**22
    caps.maximum_per_replica_inventoried_bytes = 2**24
    caps.maximum_node_inventoried_bytes = 2**26
    caps.maximum_state_numeric_bytes = 2**20
    caps.maximum_one_electron_panel_calls = 256
    caps.maximum_progress_callbacks = 65536
    caps.maximum_work_units = 10**18
    b.caps = caps
    return b


def _plan(b, *, mask=None, caps=None, system=None):
    return core._plan_periodic_gaussian_rhf(b.context, b.system if system is None else system,
        b.mask if mask is None else mask, b.options, b.config, b.live, b.caps if caps is None else caps)


def _run(b, *, progress=None, mask=None, caps=None, ao=None, auxiliary=None, system=None):
    return core._run_periodic_gaussian_rhf(b.context, b.ao if ao is None else ao,
        b.auxiliary if auxiliary is None else auxiliary, b.system if system is None else system,
        b.mask if mask is None else mask, b.options, b.config, b.live,
        b.caps if caps is None else caps, progress)


def _hermitian(a):
    # Match the documented bounded structural-projection budget, after
    # independently checking that native raw operators meet that budget.
    assert np.max(np.abs(a-a.conj().swapaxes(-1, -2))) < 2e-11
    return 0.5*a+0.5*a.conj().swapaxes(-1, -2)


def _dense_hamiltonian(b):
    """Native actual S/T/V; independently contracted dense actual B(q,k).

    Dense storage is test-only, at most 3*3*2*2*2 complex factor entries.
    No call to either the native Fock builder or the native RHF iterator.
    """
    nk, n = b.context.n_kpoints, b.ao.nbasis
    overlap, hcore = [], []
    for k in range(nk):
        st = core._build_periodic_gaussian_one_electron_panel(b.context, b.ao, b.auxiliary,
            b.system, k, 0, n*n, b.options.one_electron, _one_live(), b.caps.one_electron)
        v = core._build_periodic_gaussian_nuclear_panel(b.context, b.ao, b.auxiliary,
            b.system, k, 0, n*n, b.options.nuclear, _nuclear_live(), b.caps.nuclear)
        matrices = st.values_copy().reshape(2, n, n)
        overlap.append(matrices[0])
        hcore.append(matrices[1]+v.values_copy().reshape(4, n, n)[3])
    nuclear = core._build_periodic_gaussian_nuclear_ewald(b.context, b.system,
        b.options.nuclear, _nuclear_live(), b.caps.nuclear).energy
    factors = []
    for q in range(nk):
        source, _, whitener = _q_data(b, q)
        factors.append([_physical_tile(b, source, whitener, bra).matrix.reshape(-1, n, n)
                        for bra in range(nk)])
    s, h = _hermitian(np.array(overlap)), _hermitian(np.array(hcore))

    def response(density):
        j, exchange = np.zeros_like(density), np.zeros_like(density)
        z = sum(np.einsum("pij,ji->p", factors[0][k], density[k]) for k in range(nk))/nk
        for k in range(nk):
            j[k] = np.einsum("p,pji->ij", z, factors[0][k].conj())
        for q in range(nk):
            for bra in range(nk):
                ket = b.context.ket_index(bra, q)
                for matrix in factors[q][bra]:
                    exchange[ket] += matrix.conj().T@density[bra]@matrix/nk
        return j-0.5*exchange

    def fock(density):
        return _hermitian(h+response(density))

    def energy(density, f):
        return nuclear+np.einsum("kij,kji->", density, h+f).real/(2*nk)

    return SimpleNamespace(s=s, h=h, nuclear=nuclear, factors=factors,
                           response=response, fock=fock, energy=energy)


def _projector(coefficients):
    occupied = coefficients[:, :, :1]
    return 2*occupied@occupied.conj().swapaxes(1, 2)


def _numpy_rhf(hamiltonian):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.s)
    assert np.min(eigenvalues) > 1e-6
    x = eigenvectors/np.sqrt(eigenvalues)[:, None, :]

    def canonicalize(fock):
        eps, u = np.linalg.eigh(_hermitian(x.conj().swapaxes(1, 2)@fock@x))
        return eps, x@u

    _, c = canonicalize(hamiltonian.h)
    d = _projector(c)
    for iteration in range(100):
        f = hamiltonian.fock(d)
        eps, c = canonicalize(f)
        updated = _projector(c)
        if np.max(np.abs(updated-d)) < 2e-14:
            d = updated
            f = hamiltonian.fock(d)
            eps, c = canonicalize(f)
            return SimpleNamespace(density=d, fock=f, coefficients=c, energies=eps,
                                   energy=hamiltonian.energy(d, f), iterations=iteration+1)
        d = updated
    raise AssertionError("tiny independent undamped NumPy RHF did not converge")


def _input_identity(b):
    pieces = []
    def u64(n):
        pieces.append(struct.pack(">Q", int(n)%(2**64)))
    def text(value):
        encoded = value.encode()
        u64(len(encoded))
        pieces.append(encoded)
    def real(value):
        pieces.append(struct.pack(">d", 0.0 if value == 0 else value))
    text("vibeqc.periodic.gaussian-rhf.original-input")
    u64(1)
    text(b.context.source_context_identity_sha256)
    for value in (len(b.system.unit_cell), b.system.charge, b.system.multiplicity, 2):
        u64(value)
    for value in np.asarray(b.system.lattice).ravel():
        real(value)
    for atom in b.system.unit_cell:
        u64(atom.Z)
        for value in atom.xyz:
            real(value)
    u64(b.mask.size)
    pieces.append(b.mask.tobytes())
    return hashlib.sha256(b"".join(pieces)).hexdigest()


@pytest.fixture(scope="module", params=[(1, 1, 1), (2, 1, 1), (3, 1, 1)])
def converged_case(request):
    b = _bundle(request.param)
    h = _dense_hamiltonian(b)
    oracle = _numpy_rhf(h)
    r = _run(b)
    assert r.converged
    return b, h, oracle, r


def test_downstream_physical_input_recheck_uses_actual_hf_state_masks(converged_case):
    b, _, _, result = converged_case
    result.verify_physical_inputs(b.ao, b.auxiliary, b.system)
    changed = _system(np.asarray(b.system.lattice).copy())
    changed.unit_cell = [core.Atom(2, [.01, 0, 0])]
    with pytest.raises(ValueError, match="physical input content"):
        result.verify_physical_inputs(b.ao, b.auxiliary, changed)
    with pytest.raises((ValueError, RuntimeError), match="basis"):
        result.verify_physical_inputs(b.auxiliary, b.ao, b.system)


def test_actual_gaussian_rhf_matches_independent_generalized_numpy_oracle(converged_case):
    b, h, oracle, result = converged_case
    state = result.state
    c = np.array([state.coefficients(k) for k in range(state.n_kpoints)])
    d = _projector(c)
    f = np.array([state.fock(k) for k in range(state.n_kpoints)])
    eps = np.array([state.orbital_energies(k) for k in range(state.n_kpoints)])
    np.testing.assert_allclose(d, oracle.density, atol=4e-9, rtol=4e-9)
    np.testing.assert_allclose(f, h.fock(d), atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(f, oracle.fock, atol=4e-9, rtol=4e-9)
    np.testing.assert_allclose(eps, oracle.energies, atol=4e-9, rtol=4e-9)
    np.testing.assert_allclose(np.diagonal(h.s, axis1=1, axis2=2), 1, atol=3e-13)
    np.testing.assert_allclose(c.conj().swapaxes(1, 2)@h.s@c,
                               np.broadcast_to(np.eye(2), h.s.shape), atol=3e-11)
    np.testing.assert_allclose(f@c, h.s@(c*eps[:, None, :]), atol=3e-9, rtol=3e-9)
    np.testing.assert_allclose(np.einsum("kij,kji->k", d, h.s), 2, atol=3e-11)
    assert state.reference_energy_per_cell == pytest.approx(h.energy(d, f), abs=3e-11)
    assert state.reference_energy_per_cell == pytest.approx(oracle.energy, abs=3e-9)
    assert state.band_gap_hartree == pytest.approx(eps[:, 1].min()-eps[:, 0].max(), abs=3e-11)
    assert result.diagnostics.nuclear_energy_per_cell == h.nuclear
    assert result.diagnostics.completed_panel_calls == result.plan.one_electron_panel_calls == 2*len(d)
    assert result.diagnostics.completed_fock_calls == result.diagnostics.last_scf_snapshot.fock_calls+1
    assert abs(result.diagnostics.capture_energy_change) <= b.options.scf.energy_change_tolerance
    assert result.diagnostics.maximum_capture_fock_hermiticity_defect < 2e-11
    assert result.diagnostics.last_scf_snapshot.commutator_frobenius_rms <= b.options.scf.commutator_tolerance
    assert result.diagnostics.capture_commutator_frobenius_rms <= b.options.scf.commutator_tolerance
    assert (result.diagnostics.maximum_capture_projected_eigen_relative_residual
            <= b.options.scf.eigen_relative_tolerance)
    eigenvalues, eigenvectors = np.linalg.eigh(h.s)
    x = eigenvectors/np.sqrt(eigenvalues)[:, None, :]
    commutator = x.conj().swapaxes(1, 2)@(f@d@h.s-h.s@d@f)@x
    assert result.diagnostics.capture_commutator_frobenius_rms == pytest.approx(
        np.linalg.norm(commutator)/np.sqrt(len(d)), abs=3e-13)
    assert result.original_input_identity_sha256 == _input_identity(b)


def test_capture_has_actual_native_state_and_explicit_frozen_masks(converged_case):
    b, _, _, r = converged_case
    state = r.state
    assert r.matched_finite_gaussian_hf_source and not r.infinite_source_accuracy_certified
    assert state.converged and state.electrons_per_cell == 2
    assert (state.n_frozen_core, state.n_correlated_occupied, state.n_virtual) == (0, 1, 1)
    np.testing.assert_array_equal(state.mesh, b.context.mesh)
    np.testing.assert_array_equal(state.is_shift, [0, 0, 0])
    for k in range(state.n_kpoints):
        assert state.weight(k) == 1/state.n_kpoints
        np.testing.assert_array_equal(state.frozen_core_mask(k), [0, 0])
        np.testing.assert_array_equal(state.correlated_occupied_mask(k), [1, 0])
        np.testing.assert_array_equal(state.virtual_mask(k), [0, 1])
        np.testing.assert_array_equal(state.occupations(k), [2, 0])
        np.testing.assert_array_equal(state.kpoint_cartesian(k), b.context.k_record(k).cartesian)
    for name in ("original_input_identity_sha256", "one_electron_source_identity_sha256",
                 "final_fock_source_identity_sha256", "reference_source_identity_sha256"):
        value = getattr(r, name)
        assert len(value) == 64 and set(value) <= set("0123456789abcdef")
    with pytest.raises((ValueError, RuntimeError), match="unfinished"):
        _ = r.unfinished
    with pytest.raises(TypeError):
        core._PeriodicGaussianRHFResult()


def test_state_owner_and_matrix_copies_survive_factory_and_input_lifetimes():
    b = _bundle()
    r = _run(b)
    state = r.state
    identity = state.numerical_payload_sha256
    expected = state.fock(0).copy()
    copy = state.fock(0)
    copy[:] = 100
    del r, b
    gc.collect()
    np.testing.assert_array_equal(state.fock(0), expected)
    assert state.numerical_payload_sha256 == identity


def test_one_evaluated_snapshot_does_not_forge_a_converged_state():
    b = _bundle(iterations=1)
    h = _dense_hamiltonian(b)
    r = _run(b)
    assert not r.converged and not r.matched_finite_gaussian_hf_source
    unfinished = r.unfinished
    assert unfinished.final_snapshot.status == core._BoundedPeriodicRHFStatus.ITERATION_LIMIT
    assert unfinished.final_snapshot.fock_calls == r.diagnostics.completed_fock_calls == 1
    d, f = unfinished.density_copy(), unfinished.fock_copy()
    np.testing.assert_allclose(f, h.fock(d), atol=3e-11)
    assert unfinished.final_snapshot.energy_per_cell == pytest.approx(h.energy(d, f), abs=3e-11)
    for attribute in ("state", "final_fock_source_identity_sha256", "reference_source_identity_sha256"):
        with pytest.raises((ValueError, RuntimeError), match="converged"):
            getattr(r, attribute)


def test_progress_snapshots_controls_and_reports_live_nested_source_and_scf_stages():
    b = _bundle(iterations=1)
    events = []
    def callback(event):
        events.append(event)
        if event.stage == core._PeriodicGaussianRHFStage.BEGIN:
            scf = b.options.scf
            scf.maximum_iterations = 0
            b.options.scf = scf
            b.config.ao_column_block = 0
            b.caps.maximum_work_units = 0
    r = _run(b, progress=callback)
    assert not r.converged
    stages = {event.stage for event in events}
    for stage in (core._PeriodicGaussianRHFStage.BEGIN, core._PeriodicGaussianRHFStage.NUCLEAR_ENERGY,
                  core._PeriodicGaussianRHFStage.ONE_ELECTRON, core._PeriodicGaussianRHFStage.TWO_ELECTRON,
                  core._PeriodicGaussianRHFStage.SCF, core._PeriodicGaussianRHFStage.FINISHED):
        assert stage in stages
    assert events[0].stage == core._PeriodicGaussianRHFStage.BEGIN
    assert events[-1].stage == core._PeriodicGaussianRHFStage.FINISHED
    assert events[-1].scf.fock_calls == 1
    assert r.diagnostics.completed_progress_callbacks == len(events) <= r.plan.progress_callback_upper_bound
    assert r.plan.fock.config.ao_column_block == 2


@pytest.mark.parametrize("when", ["begin", "two_electron", "scf"])
def test_callback_exceptions_cancel_without_partial_state(when):
    b = _bundle(iterations=1)
    stage = {"begin": core._PeriodicGaussianRHFStage.BEGIN,
             "two_electron": core._PeriodicGaussianRHFStage.TWO_ELECTRON,
             "scf": core._PeriodicGaussianRHFStage.SCF}[when]
    def callback(event):
        if event.stage == stage:
            raise RuntimeError("cancel-actual-rhf")
    with pytest.raises(RuntimeError, match="cancel-actual-rhf"):
        _run(b, progress=callback)


@pytest.mark.parametrize("what", ["mask", "system"])
def test_borrowed_physical_inputs_are_rechecked_after_callbacks(what):
    b = _bundle(iterations=1)
    def callback(event):
        if event.stage == core._PeriodicGaussianRHFStage.BEGIN:
            if what == "mask":
                b.mask[0, 0] = 2
            else:
                b.system.unit_cell = [core.Atom(2, [0.01, 0.0, 0.0])]
    with pytest.raises(ValueError, match="binary|input changed"):
        _run(b, progress=callback)


@pytest.mark.parametrize("kind", ["dtype", "shape", "strided", "nonbinary", "all_frozen"])
def test_frozen_selection_is_explicit_exact_uint8_and_preserves_active_occupied_space(kind):
    b = _bundle(mesh=(2, 1, 1), iterations=1)
    mask = b.mask.copy()
    if kind == "dtype":
        mask = mask.astype(bool)
    elif kind == "shape":
        mask = np.zeros((2, 2), dtype=np.uint8)
    elif kind == "strided":
        mask = np.zeros((4, 1), dtype=np.uint8)[::2]
    elif kind == "nonbinary":
        mask[0, 0] = 2
    else:
        mask[:] = 1
    events = []
    with pytest.raises(ValueError, match="frozen|active occupied"):
        _run(b, mask=mask, progress=events.append)
    assert not events


@pytest.mark.parametrize("role", ["ao", "auxiliary", "lattice"])
def test_actual_source_mismatch_rejects_before_progress(role):
    b = _bundle(iterations=1)
    kwargs = {}
    if role == "lattice":
        b.system.lattice = np.eye(3)*8.01
    else:
        kwargs[role] = _normalized_basis([(0, (0, 0, 0), [0.65], [1.0], True),
                                          (0, (0, 0, 0), [1.35], [1.0], True)])
    events = []
    with pytest.raises(ValueError, match="content mismatch|lattice differs"):
        _run(b, progress=events.append, **kwargs)
    assert not events


@pytest.mark.parametrize("field", tuple(_CAP_TO_PLAN))
def test_outer_cap_minus_one_fails_before_any_numerical_callback(field):
    b = _bundle(iterations=1)
    p = _plan(b)
    setattr(b.caps, field, getattr(p, _CAP_TO_PLAN[field])-1)
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(b, progress=events.append)
    assert not events


def test_exact_enclosing_caps_state_bytes_and_replica_inventory():
    b = _bundle(iterations=1)
    p = _plan(b)
    assert p.overlap_hcore_bytes == 32*p.n_kpoints*p.n_basis*p.n_basis
    assert p.frozen_selection_bytes == b.mask.nbytes
    assert p.state_bytes_upper_bound == core._estimate_periodic_restricted_mean_field_resident_bytes(
        list(b.context.mesh), 2, 2)
    for cap, field in _CAP_TO_PLAN.items():
        setattr(b.caps, cap, getattr(p, field))
    r = _run(b)
    assert r.diagnostics.completed_panel_calls == p.one_electron_panel_calls
    b.live.replicas_per_node = 2
    with pytest.raises((ValueError, RuntimeError), match="node"):
        _run(b)


@pytest.mark.parametrize("kind", ["diis", "calls", "scf_work", "other_live"])
def test_unimplemented_acceleration_or_unadmitted_solver_controls_fail_closed(kind):
    b = _bundle(iterations=2)
    p = _plan(b)
    if kind == "diis":
        o = b.options.scf
        o.maximum_diis_history = 1
        b.options.scf = o
    else:
        c = b.caps.scf
        if kind == "calls":
            c.maximum_fock_calls = 1
        elif kind == "scf_work":
            c.maximum_work_units = p.scf.maximum_work_units-1
        else:
            c.other_live_numerical_bytes = 1
        b.caps.scf = c
    events = []
    with pytest.raises((ValueError, RuntimeError), match="DIIS|call ceiling|SCF work|other live"):
        _run(b, progress=events.append)
    assert not events
