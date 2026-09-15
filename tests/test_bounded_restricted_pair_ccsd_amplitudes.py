"""Borrowed ragged CCSD numerical snapshots, not physical/PNO certificates.

Independent dense common-frame and spin-orbital oracles are tiny test data.
Singles spaces deliberately differ from diagonal-double spaces, as required
by Riplinger2013 II.B.4. Mixed-source amplitudes remain in the extended frame
until the completed target residual is projected (II.C, Eq.26).
"""

from __future__ import annotations

from itertools import combinations_with_replacement, product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_bounded_restricted_ccsd_target import _problem, _legacy, _accessor_caps
from tests.test_periodic_correlation_reciprocal_metric import _CanonicalDigest


def _options(tolerance=1e-12):
    options = core._BoundedRestrictedPairCCSDAmplitudesOptions()
    options.coefficient_orthogonality_tolerance = tolerance
    return options


def _inventory():
    inventory = core._BoundedRestrictedPairCCSDAmplitudesInventory()
    inventory.numerical_replicas = 2
    inventory.external_node_bytes = 17
    inventory.other_live_bytes_per_replica = 2**20
    inventory.fixed_backend_margin_bytes_per_replica = 8192
    return inventory


def _caps():
    caps = core._BoundedRestrictedPairCCSDAmplitudesCaps()
    values = dict(maximum_occupied_count=4, maximum_common_virtual_dimension=4, maximum_pair_count=10,
        maximum_singles_rank=4, maximum_pair_rank=4, maximum_table_bytes=4096,
        maximum_borrowed_numerical_bytes=2**20, maximum_per_replica_inventoried_bytes=2**22,
        maximum_node_inventoried_bytes=2**24, maximum_validation_work_units=10**8,
        maximum_singles_work_units_per_query=10**5, maximum_doubles_work_units_per_query=10**6)
    for key, value in values.items():
        setattr(caps, key, value)
    return caps


def _ragged(n=3, singles=(2, 0, 3), pairs=(1, 2, 0, 2, 1, 3), seed=527):
    rng = np.random.default_rng(seed)
    def frame(rank):
        return np.ascontiguousarray(np.linalg.qr(rng.normal(size=(n, n)))[0][:, :rank])
    sc, st = [frame(r) for r in singles], [np.ascontiguousarray(.04*rng.normal(size=r)) for r in singles]
    pc, pt = [frame(r) for r in pairs], []
    for (i, j), rank in zip(combinations_with_replacement(range(len(singles)), 2), pairs):
        t = .035*rng.normal(size=(rank, rank))
        pt.append(np.ascontiguousarray((t+t.T)*.5 if i == j else t))
    return dict(n=n, o=len(singles), sc=sc, st=st, pc=pc, pt=pt)


def _arguments(b, *, options=None, inventory=None, caps=None):
    return (b["n"], b["sc"], b["st"], b["pc"], b["pt"], _options() if options is None else options,
        _inventory() if inventory is None else inventory, _caps() if caps is None else caps)


def _plan(b, **kwargs):
    return core._plan_bounded_restricted_pair_ccsd_amplitudes_diagnostic(*_arguments(b, **kwargs))


def _query(b, *, empty=False, sq=None, dq=None, work=10**12, callback=None, **kwargs):
    if sq is None:
        sq = np.array([] if empty else list(product(range(b["o"]), range(b["n"]))), np.uint64).reshape(-1, 2)
    if dq is None:
        dq = np.array([] if empty else list(product(range(b["o"]), range(b["o"]), range(b["n"]), range(b["n"]))), np.uint64).reshape(-1, 4)
    return core._bounded_restricted_pair_ccsd_amplitudes_query_diagnostic(
        *_arguments(b, **kwargs), sq, dq, work, callback)


def _dense(b):
    o, n = b["o"], b["n"]
    t1 = np.array([c@t for c, t in zip(b["sc"], b["st"])])
    t2 = np.zeros((o, o, n, n))
    for (i, j), c, t in zip(combinations_with_replacement(range(o), 2), b["pc"], b["pt"]):
        t2[i, j] = c@t@c.T
        t2[j, i] = t2[i, j].T
    return t1, t2


def _target(b, problem, i, j, *, caps=None):
    p = _plan(b)
    plan = core._plan_bounded_restricted_ccsd_target_accessor(b["o"], b["n"], p.borrowed_numerical_bytes, 0,
        p.maximum_singles_work_units_per_query, p.maximum_doubles_work_units_per_query, problem["eri"].nbytes, 0)
    o, f = b["o"], problem["fock"]
    return core._bounded_restricted_pair_ccsd_target_diagnostic(*_arguments(b),
        np.ascontiguousarray(f[:o, :o]), np.ascontiguousarray(f[o:, o:]), np.ascontiguousarray(f[:o, o:]),
        problem["eri"], i, j, _accessor_caps(plan) if caps is None else caps)


def _spin_residual(problem):
    from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_residuals
    o, v = problem["o"], problem["v"]
    n = o+v
    fso = np.kron(problem["fock"], np.eye(2))
    offdiagonal = fso.copy()
    np.fill_diagonal(offdiagonal, 0)
    eo, ev = np.diag(fso)[:2*o], np.diag(fso)[2*o:]
    d1 = eo[:, None]-ev[None, :]
    d2 = eo[:, None, None, None]+eo[None, :, None, None]-ev[None, None, :, None]-ev[None, None, None, :]
    one = np.kron(problem["t1"], np.eye(2))
    two = np.zeros((2*o, 2*o, 2*v, 2*v))
    for si, sj in product(range(2), repeat=2):
        two[si::2, sj::2, si::2, sj::2] += problem["t2"]
        two[si::2, sj::2, sj::2, si::2] -= problem["t2"].transpose(0, 1, 3, 2)
    return so_residuals(fso, offdiagonal, _spin_orbital_eri(problem["factors"], n),
        one, two, slice(0, 2*o), slice(2*o, 2*n), d1, d2)


@pytest.mark.parametrize("n,s,p", [(1, (0,), (1,)), (3, (2, 0, 3), (1, 2, 0, 2, 1, 3)),
                                  (4, (4, 1), (0, 3, 2)), (2, (0, 0), (0, 0, 0))])
def test_ragged_scalar_expansions_match_numpy_and_exact_reversal(n, s, p):
    b = _ragged(n, s, p)
    result = _query(b)
    t1, t2 = _dense(b)
    actual1 = result["singles"].reshape(b["o"], n)
    actual2 = result["doubles"].reshape(b["o"], b["o"], n, n)
    np.testing.assert_allclose(actual1, t1, atol=5e-17, rtol=5e-14)
    np.testing.assert_allclose(actual2, t2, atol=5e-17, rtol=5e-14)
    np.testing.assert_array_equal(actual2, actual2.transpose(1, 0, 3, 2))
    assert result["memory"].owned_numerical_bytes == 0
    assert result["provider_transient_numerical_bytes"] == 0
    assert result["provider_retained_numerical_bytes"] == result["memory"].borrowed_numerical_bytes
    assert result["diagnostics"].validated_singles == b["o"]
    assert result["diagnostics"].validated_pairs == b["o"]*(b["o"]+1)//2
    if not any(s+p):
        np.testing.assert_array_equal(actual1, 0)
        np.testing.assert_array_equal(actual2, 0)


def test_separate_singles_and_diagonal_double_frames_are_not_substituted():
    b = dict(n=3, o=1, sc=[np.eye(3)], st=[np.array([.2, -.1, .7])],
        pc=[np.ascontiguousarray(np.eye(3)[:, :1])], pt=[np.array([[.3]])])
    result = _query(b)
    np.testing.assert_array_equal(result["singles"], b["st"][0])
    expected = np.zeros((3, 3))
    expected[0, 0] = .3
    np.testing.assert_array_equal(result["doubles"].reshape(3, 3), expected)


def test_all_zero_ranks_accept_exact_zero_numeric_cap():
    b = _ragged(2, (0, 0), (0, 0, 0))
    caps = _caps()
    caps.maximum_borrowed_numerical_bytes = caps.maximum_singles_rank = caps.maximum_pair_rank = 0
    result = _query(b, caps=caps)
    assert result["memory"].borrowed_numerical_bytes == 0
    np.testing.assert_array_equal(result["singles"], 0)
    np.testing.assert_array_equal(result["doubles"], 0)


def test_independent_within_space_rotations_leave_expanded_amplitudes_unchanged():
    b = _ragged()
    rotated = {"n": b["n"], "o": b["o"], "sc": [], "st": [], "pc": [], "pt": []}
    rng = np.random.default_rng(918)
    for c, t in zip(b["sc"], b["st"]):
        q = np.linalg.qr(rng.normal(size=(len(t), len(t))))[0]
        rotated["sc"].append(np.ascontiguousarray(c@q))
        rotated["st"].append(np.ascontiguousarray(q.T@t))
    for (i, j), c, t in zip(combinations_with_replacement(range(b["o"]), 2), b["pc"], b["pt"]):
        q = np.linalg.qr(rng.normal(size=t.shape))[0]
        value = q.T@t@q
        # Provide an exact-symmetric TEST input, not a native silent repair.
        if i == j:
            value = .5*(value+value.T)
        rotated["pc"].append(np.ascontiguousarray(c@q))
        rotated["pt"].append(np.ascontiguousarray(value))
    first, second = _query(b), _query(rotated)
    np.testing.assert_allclose(first["singles"], second["singles"], atol=8e-17)
    np.testing.assert_allclose(first["doubles"], second["doubles"], atol=8e-17)
    assert first["snapshot_identity_sha256"] != second["snapshot_identity_sha256"]


def test_role_conservative_inventory_upper_bounds_and_snapshot_wire():
    b = _ragged()
    p, inventory = _plan(b), _inventory()
    expected = sum(a.nbytes for key in ("sc", "st", "pc", "pt") for a in b[key])
    assert p.borrowed_numerical_bytes == expected
    assert p.borrowed_table_bytes == p.borrowed_singles_table_bytes+p.borrowed_pair_table_bytes
    assert p.per_replica_inventoried_bytes == expected+p.borrowed_table_bytes+p.fixed_control_storage_bytes+2**20+8192
    assert p.required_node_inventoried_bytes == 17+2*p.per_replica_inventoried_bytes
    upper = core._plan_bounded_restricted_pair_ccsd_amplitudes_upper(b["o"], b["n"], b["n"], b["n"], inventory)
    assert upper.uniform_rank_upper_bound and not p.uniform_rank_upper_bound
    for field in ("borrowed_numerical_bytes", "validation_work_units", "required_node_inventoried_bytes"):
        assert getattr(p, field) <= getattr(upper, field)
    assert p.maximum_singles_work_units_per_query == 32*(max(len(t) for t in b["st"])+1)
    r = max(len(t) for t in b["pt"])
    assert p.maximum_doubles_work_units_per_query == 64*(r*r+r+1)
    result = _query(b)
    assert result["memory"].per_replica_inventoried_bytes-p.per_replica_inventoried_bytes == result["wrapper_extra_live_bytes"]
    h = _CanonicalDigest("vibeqc.bounded-restricted-pair-ccsd-amplitudes.snapshot", 1)
    h.u64(b["o"])
    h.u64(b["n"])
    for frames, values in ((b["sc"], b["st"]), (b["pc"], b["pt"])):
        for c, t in zip(frames, values):
            h.u64(c.shape[1])
            for a in (c, t):
                h.u64(a.size)
                for value in a.ravel():
                    h.binary64(value)
    h.binary64(1e-12)
    h.string("restricted-connected-alpha-beta;separate-singles;unordered-pair-transpose;diagonal-common-upper;role-conservative-borrowed;no-physical-certificate")
    assert result["snapshot_identity_sha256"] == h.finish()
    b["sc"][0] = b["pc"][3]  # Exact same object is still charged in EACH role.
    assert _plan(b).borrowed_numerical_bytes == expected


_CAP_PLAN = {
    "maximum_occupied_count": "n_occupied", "maximum_common_virtual_dimension": "common_virtual_dimension",
    "maximum_pair_count": "pair_count", "maximum_singles_rank": "maximum_singles_rank", "maximum_pair_rank": "maximum_pair_rank",
    "maximum_table_bytes": "borrowed_table_bytes", "maximum_borrowed_numerical_bytes": "borrowed_numerical_bytes",
    "maximum_per_replica_inventoried_bytes": "per_replica_inventoried_bytes",
    "maximum_node_inventoried_bytes": "required_node_inventoried_bytes", "maximum_validation_work_units": "validation_work_units",
    "maximum_singles_work_units_per_query": "maximum_singles_work_units_per_query",
    "maximum_doubles_work_units_per_query": "maximum_doubles_work_units_per_query",
}


@pytest.mark.parametrize("field", _CAP_PLAN)
def test_exact_caps_and_cap_minus_one_before_nan_payload(field):
    b = _ragged()
    p, c = _plan(b), _caps()
    for cap, value in _CAP_PLAN.items():
        setattr(c, cap, getattr(p, value))
    _query(b, empty=True, caps=c)
    setattr(c, field, getattr(c, field)-1)
    b["st"][0][0] = np.nan
    with pytest.raises(ValueError, match="cap"):
        _query(b, empty=True, caps=c)


@pytest.mark.parametrize("role", ["sc", "st", "pc", "pt"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_every_numeric_role_is_scanned_even_without_queries(role, bad):
    b = _ragged()
    b[role][0].flat[0] = bad
    with pytest.raises(OverflowError, match="non-finite"):
        _query(b, empty=True)


def test_orthogonality_diagonal_symmetry_shape_alignment_and_unsupported_dtype_fail():
    b = _ragged()
    b["sc"][0] *= 2
    with pytest.raises(ValueError, match="orthonormal"):
        _query(b, empty=True)
    b = _ragged()
    b["pt"][3][0, 1] += .01
    with pytest.raises(ValueError, match="exactly symmetric"):
        _query(b, empty=True)
    for kind in ("dtype", "alignment", "shape", "table", "converted"):
        b = _ragged()
        if kind == "dtype":
            b["sc"][0] = b["sc"][0].astype(np.float32)
        elif kind == "alignment":
            b["sc"][0] = np.ndarray((3, 2), dtype=np.float64, buffer=bytearray(49), offset=1)
        elif kind == "shape":
            b["st"][0] = np.ones(3)
        elif kind == "table":
            b["pc"].pop()
        else:
            b["st"][0] = [0., 0.]
        with pytest.raises(ValueError, match="dtype|shape|count|actual arrays"):
            _query(b, empty=True)


@pytest.mark.parametrize("tolerance", [0.0, -1., 1., np.inf, np.nan])
def test_scientific_orthogonality_tolerance_is_explicit(tolerance):
    with pytest.raises(ValueError, match="explicit"):
        _query(_ragged(), empty=True, options=_options(tolerance))


def test_callback_snapshot_mutation_and_exception_never_publish_values():
    b = _ragged()
    def mutation():
        b["st"][0][0] += .1
    with pytest.raises(ValueError, match="snapshot changed"):
        _query(b, callback=mutation)
    b = _ragged()
    def resize():
        b["sc"][0].resize((1,), refcheck=False)
    with pytest.raises(ValueError, match="array|dtype"):
        _query(b, callback=resize)
    def fail():
        raise RuntimeError("intentional callback cancellation")
    with pytest.raises(RuntimeError, match="cancellation"):
        _query(_ragged(), callback=fail)


def test_query_work_is_admitted_before_invalid_amplitudes_and_labels_are_checked():
    b = _ragged()
    result = _query(b)
    b["st"][0][0] = np.nan
    with pytest.raises(ValueError, match="query work cap"):
        _query(b, work=result["query_work_units"]-1)
    b = _ragged()
    for sq, dq in ((np.array([[3, 0]], np.uint64), None), (None, np.array([[0, 0, 3, 0]], np.uint64))):
        with pytest.raises(IndexError, match="label"):
            _query(b, sq=sq, dq=dq)


@pytest.mark.parametrize("i,j", [(0, 0), (0, 1), (1, 0), (2, 2)])
def test_ragged_target_common_and_projected_residual_matches_spatial_and_spin_oracles(i, j):
    b = _ragged()
    problem = _problem(b["o"], b["n"], seed=729)
    problem["t1"], problem["t2"] = _dense(b)
    r1, r2 = _legacy(problem)
    spin1, spin2 = _spin_residual(problem)
    result = _target(b, problem, i, j)
    np.testing.assert_allclose(result.target.singles, r1[i], atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(result.target.doubles, r2[i, j], atol=3e-13, rtol=3e-13)
    np.testing.assert_allclose(result.target.singles, spin1[2*i, 0::2], atol=4e-13, rtol=4e-13)
    np.testing.assert_allclose(result.target.doubles, spin2[2*i, 2*j+1, 0::2, 1::2], atol=4e-13, rtol=4e-13)
    slot = list(combinations_with_replacement(range(b["o"]), 2)).index(tuple(sorted((i, j))))
    c, s = b["pc"][slot], b["sc"][i]
    np.testing.assert_allclose(s.T@result.target.singles, s.T@r1[i], atol=3e-13)
    np.testing.assert_allclose(c.T@result.target.doubles@c, c.T@r2[i, j]@c, atol=3e-13)
    assert result.memory.amplitude_retained_numerical_bytes == _plan(b).borrowed_numerical_bytes
    assert result.singles_calls <= result.memory.singles_calls_upper_bound
    assert result.doubles_calls <= result.memory.doubles_calls_upper_bound


def test_mixed_source_outside_target_pno_space_still_changes_target_residual():
    b = dict(n=3, o=2, sc=[np.empty((3, 0)), np.empty((3, 0))], st=[np.empty(0), np.empty(0)],
        pc=[np.empty((3, 0)), np.ascontiguousarray(np.eye(3)[:, :1]), np.ascontiguousarray(np.eye(3)[:, [0, 2]])],
        pt=[np.empty((0, 0)), np.zeros((1, 1)), np.array([[0., .3], [.3, 0.]])])
    problem = _problem(2, 3, seed=333)
    problem["t1"], problem["t2"] = _dense(b)
    _, expected = _legacy(problem)
    result = _target(b, problem, 0, 1)
    c = b["pc"][1]
    np.testing.assert_allclose(c.T@result.target.doubles@c, c.T@expected[0, 1]@c, atol=2e-13)
    b["pt"][2][:] = 0  # The wrong target-only approximation drops this mixed term.
    wrong = _target(b, problem, 0, 1)
    assert abs((c.T@(result.target.doubles-wrong.target.doubles)@c)[0, 0]) > 1e-6


@pytest.mark.parametrize("o,n,s,r", [(0, 1, 0, 0), (1, 0, 0, 0), (1, 2, 3, 1), (2**63, 2, 0, 0)])
def test_count_only_upper_plan_rejects_invalid_or_overflow_dimensions(o, n, s, r):
    with pytest.raises((ValueError, OverflowError)):
        core._plan_bounded_restricted_pair_ccsd_amplitudes_upper(o, n, s, r, _inventory())
