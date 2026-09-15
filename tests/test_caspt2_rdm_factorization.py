"""Phase-oracle tests for the private RDM-contracted CASPT2 kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import build_hamiltonian_mo, casci, caspt2, get_hf_orbital_provider
from vibeqc.solvers._caspt2_rdm import (
    ActiveExternalFactorization,
    _external_signatures_one_body_reachable,
    active_ci_caspt2_corr_signature_iterative,
    build_active_ci_signature_raw_system,
    build_factored_ic_system_dense,
    build_inactive_active_to_virtual_pair_active_ci_raw_system,
    build_inactive_pair_active_virtual_active_ci_raw_system,
    build_inactive_pair_to_virtual_pair_active_ci_raw_system,
    build_inactive_pair_active_ci_raw_system,
    build_inactive_to_active_active_ci_raw_system,
    build_inactive_to_virtual_active_ci_raw_system,
    build_no_inactive_external_active_ci_raw_system,
    build_signature_block_linear_system,
    build_signature_block_linear_system_from_raw,
    contracted_candidates,
    contracted_excitation_labels,
    factored_ic_caspt2_corr_dense,
    factored_ic_caspt2_corr_signature_iterative,
    inactive_active_to_virtual_pair_active_ci_caspt2_corr_signature_iterative,
    inactive_active_to_virtual_pair_active_ci_coupling,
    inactive_active_to_virtual_pair_active_ci_h0,
    inactive_active_to_virtual_pair_active_ci_overlap,
    inactive_pair_to_virtual_pair_active_ci_caspt2_corr_signature_iterative,
    inactive_pair_to_virtual_pair_active_ci_coupling,
    inactive_pair_to_virtual_pair_active_ci_h0,
    inactive_pair_to_virtual_pair_active_ci_overlap,
    inactive_to_virtual_active_ci_caspt2_corr_signature_iterative,
    inactive_to_virtual_active_ci_coupling,
    inactive_to_virtual_active_ci_h0,
    inactive_to_virtual_active_ci_overlap,
    inactive_to_active_active_ci_caspt2_corr_signature_iterative,
    inactive_to_active_active_ci_h0,
    inactive_to_active_rdm_overlap_coupling,
    inactive_pair_active_ci_caspt2_corr_signature_iterative,
    inactive_pair_active_virtual_active_ci_caspt2_corr_signature_iterative,
    inactive_pair_to_active_virtual_active_ci_coupling,
    inactive_pair_to_active_virtual_active_ci_h0,
    inactive_pair_to_active_virtual_active_ci_overlap,
    inactive_pair_to_active_pair_active_ci_coupling,
    inactive_pair_to_active_pair_active_ci_h0,
    inactive_pair_to_active_pair_distinct_overlap,
    inactive_pair_to_active_pair_overlap,
    inactive_pair_to_active_pair_same_hole_overlap,
    no_inactive_active_ci_caspt2_corr_signature_iterative,
    no_inactive_external_active_ci_h0,
    no_inactive_external_rdm_overlap_coupling,
    openmolcas_case_labels,
    orthonormalize_factored_ic_system_signature,
    orthonormal_signature_block_couplings,
    orthonormal_signature_block_components,
    orthonormal_signature_block_matvec,
    orthonormal_signature_block_operator,
    orthonormal_signature_coupling_edges,
    raw_block_system_with_no_inactive_external_active_ci,
    raw_block_system_with_no_inactive_external_active_ci_h0,
    raw_block_system_with_inactive_to_active_active_ci,
    raw_block_system_with_inactive_to_active_active_ci_h0,
    raw_block_system_with_inactive_to_active_rdm_sv,
    raw_block_system_with_inactive_to_virtual_active_ci,
    raw_block_system_with_inactive_to_virtual_h0,
    raw_block_system_with_inactive_to_virtual_overlap,
    raw_block_system_with_inactive_to_virtual_sv,
    raw_block_system_with_inactive_pair_active_virtual_active_ci,
    raw_block_system_with_inactive_pair_active_virtual_h0,
    raw_block_system_with_inactive_pair_active_virtual_overlap,
    raw_block_system_with_inactive_pair_active_virtual_sv,
    raw_block_system_with_inactive_active_to_virtual_pair_active_ci,
    raw_block_system_with_inactive_active_to_virtual_pair_h0,
    raw_block_system_with_inactive_active_to_virtual_pair_overlap,
    raw_block_system_with_inactive_active_to_virtual_pair_sv,
    raw_block_system_with_inactive_pair_to_virtual_pair_active_ci,
    raw_block_system_with_inactive_pair_to_virtual_pair_h0,
    raw_block_system_with_inactive_pair_to_virtual_pair_overlap,
    raw_block_system_with_inactive_pair_to_virtual_pair_sv,
    raw_block_system_with_inactive_pair_distinct_overlap,
    raw_block_system_with_inactive_pair_active_ci_h0,
    raw_block_system_with_inactive_pair_active_ci,
    raw_block_system_with_inactive_pair_overlap,
    raw_block_system_with_inactive_pair_same_hole_overlap,
    raw_block_system_with_inactive_pair_sv,
    raw_block_system_with_no_inactive_external_rdm_sv,
    signature_raw_block_system_from_factored,
    signature_block_operator_components,
    signature_block_operator_matvec,
    solve_factored_ic_system,
    solve_factored_ic_system_class_metric,
    solve_factored_ic_system_independent_signatures,
    solve_factored_ic_system_orthonormal_independent_signatures,
    solve_factored_ic_system_signature_components,
    solve_factored_ic_system_signature_iterative,
    solve_orthonormal_ic_system,
    solve_orthonormal_ic_system_block_components,
    solve_orthonormal_ic_system_block_iterative,
    solve_orthonormal_ic_system_independent_blocks,
    solve_signature_block_linear_system_components,
    solve_signature_block_linear_system_iterative,
    solve_factored_ic_system_signature_metric,
    subset_factored_ic_system,
)
from vibeqc.solvers._mrpt import (
    _build_reference_state,
    _ic_caspt2_corr,
    _ic_caspt2_uses_direct_active_ci,
    _dot,
    _generalized_fock,
    _semicanonical_prep,
    apply_1body,
    apply_2body,
)
from vibeqc.solvers._rdm import make_rdm1

NC = 1
NA = 2
NE = 2
TOL = 5e-14


def test_contracted_candidate_metadata_covers_external_classes():
    candidates = contracted_candidates(n_core=2, n_active=3, n_orb=7)
    labels = contracted_excitation_labels(n_core=2, n_active=3, n_orb=7)
    assert [candidate.label for candidate in candidates] == labels

    classes = {candidate.external_class for candidate in candidates}
    assert classes == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }
    assert {cls: openmolcas_case_labels(cls) for cls in classes} == {
        (0, 0): ("CAS",),
        (1, 0): ("A",),
        (2, 0): ("B+", "B-"),
        (0, 1): ("C",),
        (1, 1): ("D",),
        (2, 1): ("E+", "E-"),
        (0, 2): ("F+", "F-"),
        (1, 2): ("G+", "G-"),
        (2, 2): ("H+", "H-"),
    }

    order_by_class = {}
    for candidate in candidates:
        order_by_class.setdefault(candidate.external_class, set()).add(candidate.order)
        assert candidate.external_class == (
            len(candidate.inactive_holes),
            len(candidate.secondary_particles),
        )
        assert candidate.external_signature == (
            tuple(sorted(candidate.inactive_holes)),
            tuple(sorted(candidate.secondary_particles)),
        )
        assert candidate.openmolcas_cases == openmolcas_case_labels(
            candidate.external_class
        )
        assert len(candidate.label) == candidate.order
    assert order_by_class[(2, 2)] == {2}
    assert order_by_class[(1, 1)] == {1, 2}


def test_external_signature_one_body_reachability_uses_multisets():
    assert _external_signatures_one_body_reachable(((0, 0), ()), ((0,), ()))
    assert _external_signatures_one_body_reachable(((0,), (5,)), ((0,), (6,)))
    assert _external_signatures_one_body_reachable(((0,), (5,)), ((), (5,)))
    assert not _external_signatures_one_body_reachable(
        ((0,), (5,)),
        ((1,), (6,)),
    )
    assert not _external_signatures_one_body_reachable(
        ((0, 0), ()),
        ((1, 1), ()),
    )


def _hamiltonian(mol, basis_name):
    basis = BasisSet(mol, basis_name)
    coeff = get_hf_orbital_provider(mol, basis, method="rhf")
    return build_hamiltonian_mo(mol, basis, coeff)


def _lih_case():
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
    ham = _hamiltonian(mol, "sto-3g")
    norb = ham.norb
    h1 = np.asarray(ham.h1e, dtype=float).copy()
    eri = np.asarray(ham.h2e, dtype=float).transpose(0, 2, 1, 3).copy()

    cas0 = casci(
        h1,
        ham.h2e,
        n_active_elec=NE,
        n_active_orb=NA,
        n_core=NC,
        nuclear_repulsion=0.0,
        ms2=0,
    )
    dm1 = make_rdm1(cas0.ci_coeffs, cas0.determinants, NA)
    fock = _generalized_fock(h1, eri, dm1, NC, NA)

    rot = np.eye(norb)
    for block in (slice(0, NC), slice(NC, NC + NA), slice(NC + NA, norb)):
        if block.stop - block.start > 0:
            _, u_block = eigh(fock[block, block])
            rot[block, block] = u_block
    h1 = rot.T @ h1 @ rot
    eri = np.einsum("pi,qj,rk,sl,pqrs->ijkl", rot, rot, rot, rot, eri, optimize=True)
    fock = rot.T @ fock @ rot
    h2_phys = eri.transpose(0, 2, 1, 3)
    cas = casci(
        h1,
        h2_phys,
        n_active_elec=NE,
        n_active_orb=NA,
        n_core=NC,
        nuclear_repulsion=0.0,
        ms2=0,
    )
    ref = _build_reference_state(cas.ci_coeffs, cas.determinants, NC, norb)
    return norb, ref, cas.ci_coeffs, cas.determinants


def _apply_E_full(state, p, q, norb):
    e = np.zeros((norb, norb))
    e[p, q] = 1.0
    return apply_1body(state, e, norb)


def _sample_labels(norb):
    inactive = list(range(NC))
    active = list(range(NC, NC + NA))
    virtual = list(range(NC + NA, norb))
    i = inactive[0]
    t, u = active[:2]
    a, b = virtual[:2]
    labels = [
        ((a, t),),
        ((t, i),),
        ((a, i),),
        ((b, u), (a, t)),
        ((u, i), (t, i)),
        ((b, u), (a, i)),
        ((a, u), (t, i)),
        ((u, i), (a, i)),
    ]
    return list(dict.fromkeys(labels))


def _full_state(ref, norb, label):
    state = ref
    for p, q in label:
        state = _apply_E_full(state, p, q, norb)
    return state


def _assert_state_close(left, right):
    keys = set(left) | set(right)
    err = max(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
    assert err < TOL


def _scrub_raw_block_sv(block, formula_indices):
    local = [
        idx
        for idx, candidate_idx in enumerate(block.candidate_indices)
        if int(candidate_idx) in formula_indices
    ]
    if not local:
        return block

    local_idx = np.asarray(local, dtype=int)
    overlap = block.overlap.copy()
    coupling = block.coupling.copy()
    overlap[np.ix_(local_idx, local_idx)] = 0.0
    coupling[local_idx] = 0.0
    return type(block)(
        key=block.key,
        candidate_indices=block.candidate_indices,
        overlap=overlap,
        h0=block.h0,
        coupling=coupling,
    )


def _scrub_raw_block_overlap(block, formula_indices):
    local = [
        idx
        for idx, candidate_idx in enumerate(block.candidate_indices)
        if int(candidate_idx) in formula_indices
    ]
    if not local:
        return block

    local_idx = np.asarray(local, dtype=int)
    overlap = block.overlap.copy()
    overlap[np.ix_(local_idx, local_idx)] = 0.0
    return type(block)(
        key=block.key,
        candidate_indices=block.candidate_indices,
        overlap=overlap,
        h0=block.h0,
        coupling=block.coupling,
    )


def _scrub_raw_block_h0(block, formula_indices):
    local = [
        idx
        for idx, candidate_idx in enumerate(block.candidate_indices)
        if int(candidate_idx) in formula_indices
    ]
    if not local:
        return block

    local_idx = np.asarray(local, dtype=int)
    h0 = block.h0.copy()
    h0[np.ix_(local_idx, local_idx)] = 0.0
    return type(block)(
        key=block.key,
        candidate_indices=block.candidate_indices,
        overlap=block.overlap,
        h0=h0,
        coupling=block.coupling,
    )


def _scrub_raw_coupling_h0(coupling, blocks, formula_indices):
    left = blocks[coupling.left_block]
    right = blocks[coupling.right_block]
    left_local = [
        idx
        for idx, candidate_idx in enumerate(left.candidate_indices)
        if int(candidate_idx) in formula_indices
    ]
    right_local = [
        idx
        for idx, candidate_idx in enumerate(right.candidate_indices)
        if int(candidate_idx) in formula_indices
    ]
    if not left_local or not right_local:
        return coupling

    h0 = coupling.h0.copy()
    h0[
        np.ix_(
            np.asarray(left_local, dtype=int),
            np.asarray(right_local, dtype=int),
        )
    ] = 0.0
    return type(coupling)(
        left_block=coupling.left_block,
        right_block=coupling.right_block,
        h0=h0,
    )


def test_factored_contracted_states_match_determinant_oracle():
    norb, ref, ci, dets = _lih_case()
    fac = ActiveExternalFactorization(ci, dets, NC, NA, norb)

    for label in _sample_labels(norb):
        full = _full_state(ref, norb, label)
        factored = fac.apply_E_sequence(fac.ref, label)
        _assert_state_close(full, fac.to_full_determinants(factored))


def test_factored_one_body_operators_match_determinant_oracle():
    norb, ref, ci, dets = _lih_case()
    fac = ActiveExternalFactorization(ci, dets, NC, NA, norb)
    labels = _sample_labels(norb)
    full_states = [_full_state(ref, norb, label) for label in labels]
    factored_states = [fac.apply_E_sequence(fac.ref, label) for label in labels]
    probes = list(range(NC)) + list(range(NC, NC + NA)) + list(range(NC + NA, NC + NA + 2))

    max_err = 0.0
    for p in probes:
        for q in probes:
            e = np.zeros((norb, norb))
            e[p, q] = 1.0
            for bra_full, bra_fac in zip(full_states, factored_states):
                for ket_full, ket_fac in zip(full_states, factored_states):
                    exact = _dot(bra_full, apply_1body(ket_full, e, norb))
                    got = fac.dot(bra_fac, fac.apply_E(ket_fac, p, q))
                    max_err = max(max_err, abs(exact - got))
    assert max_err < TOL


def test_factored_two_body_operator_samples_match_determinant_oracle():
    norb, ref, ci, dets = _lih_case()
    fac = ActiveExternalFactorization(ci, dets, NC, NA, norb)
    labels = _sample_labels(norb)
    full_states = [_full_state(ref, norb, label) for label in labels]
    factored_states = [fac.apply_E_sequence(fac.ref, label) for label in labels]

    i = 0
    t, u = 1, 2
    a, b = 3, 4
    terms = [
        (t, u, t, u),  # active-active
        (a, b, t, u),  # virtual-virtual / active-active
        (t, u, a, b),
        (a, t, b, u),  # virtual-active mixed terms
        (t, a, u, b),
        (a, t, u, b),
        (t, a, b, u),
        (i, i, t, u),  # inactive-active mixed terms
        (t, u, i, i),
        (t, i, u, i),
        (a, i, t, u),
        (t, u, a, i),
        (a, t, u, i),
        (t, i, a, u),
    ]

    max_err = 0.0
    for p, q, r, s in terms:
        eri = np.zeros((norb, norb, norb, norb))
        eri[p, q, r, s] = 1.0
        for bra_full, bra_fac in zip(full_states, factored_states):
            for ket_full, ket_fac in zip(full_states, factored_states):
                exact = _dot(bra_full, apply_2body(ket_full, eri, norb))
                got = fac.dot(bra_fac, fac.apply_chemist_2body_unit(ket_fac, p, q, r, s))
                max_err = max(max_err, abs(exact - got))
    assert max_err < TOL


def test_factored_dense_ic_energy_matches_explicit_oracle():
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            0,
            2,
            2,
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        e_factored, dim_factored = factored_ic_caspt2_corr_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        block = factored_ic_caspt2_corr_signature_iterative(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        e_explicit, dim_explicit = _ic_caspt2_corr(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        assert block.converged
        assert dim_factored == dim_explicit
        assert block.n_basis == dim_explicit
        assert block.residual_norm < 1e-10
        assert abs(e_factored - e_explicit) < TOL
        assert abs(block.energy - e_explicit) < 1e-11


def test_factored_dense_system_exposes_class_sparsity():
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
    ham = _hamiltonian(mol, "sto-3g")
    system = build_factored_ic_system_dense(
        ham.h1e, ham.h2e, n_core=1, n_active=2, n_active_elec=2, ms2=0
    )

    assert system.overlap.shape == system.h0.shape
    assert system.overlap.shape[0] == len(system.candidates)
    for i, left in enumerate(system.candidates):
        hi, pi = left.external_class
        for j, right in enumerate(system.candidates):
            hj, pj = right.external_class
            if left.external_signature != right.external_signature:
                assert abs(system.overlap[i, j]) < TOL
            # H0 is one-body Fock, so one matrix element can alter at most one
            # inactive-hole count and one secondary-particle count.
            if abs(hi - hj) > 1 or abs(pi - pj) > 1:
                assert abs(system.h0[i, j]) < TOL


def test_class_block_metric_solve_matches_global_metric_solve():
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
    ham = _hamiltonian(mol, "sto-3g")
    system = build_factored_ic_system_dense(
        ham.h1e, ham.h2e, n_core=1, n_active=2, n_active_elec=2, ms2=0
    )
    e_global, dim_global = solve_factored_ic_system(system)
    e_block, dim_block = solve_factored_ic_system_class_metric(system)
    e_signature, dim_signature = solve_factored_ic_system_signature_metric(system)
    assert dim_block == dim_global
    assert dim_signature == dim_global
    assert abs(e_block - e_global) < TOL
    assert abs(e_signature - e_global) < TOL


def test_signature_orthonormal_system_matches_signature_metric_solve():
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
    ham = _hamiltonian(mol, "sto-3g")
    system = build_factored_ic_system_dense(
        ham.h1e, ham.h2e, n_core=1, n_active=2, n_active_elec=2, ms2=0
    )
    orth = orthonormalize_factored_ic_system_signature(system)
    e_orth, dim_orth = solve_orthonormal_ic_system(orth)
    e_signature, dim_signature = solve_factored_ic_system_signature_metric(system)

    assert orth.shifted_h0.shape == (dim_signature, dim_signature)
    assert orth.coupling.shape == (dim_signature,)
    assert len(orth.block_keys) == dim_signature
    assert orth.block_ranges
    assert orth.block_ranges[0][1] == 0
    assert orth.block_ranges[-1][2] == dim_signature
    for key, start, stop in orth.block_ranges:
        assert start < stop
        assert all(block_key == key for block_key in orth.block_keys[start:stop])
    for left_range, right_range in zip(orth.block_ranges, orth.block_ranges[1:]):
        assert left_range[2] == right_range[1]
    assert dim_orth == dim_signature
    assert abs(e_orth - e_signature) < TOL


def test_signature_block_iterative_solve_matches_dense_orthonormal_solve():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        system = build_factored_ic_system_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        orth = orthonormalize_factored_ic_system_signature(system)
        e_dense, dim_dense = solve_orthonormal_ic_system(orth)
        iterative = solve_orthonormal_ic_system_block_iterative(orth)
        wrapped = solve_factored_ic_system_signature_iterative(system)

        assert iterative.converged
        assert wrapped.converged
        assert iterative.n_basis == dim_dense
        assert wrapped.n_basis == dim_dense
        assert iterative.n_iter > 0
        assert iterative.residual_norm < 1e-10
        assert wrapped.residual_norm < 1e-10
        assert abs(iterative.energy - e_dense) < 1e-11
        assert abs(wrapped.energy - e_dense) < 1e-11


def test_direct_signature_block_linear_system_matches_orthonormal_oracle():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        system = build_factored_ic_system_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        orth = orthonormalize_factored_ic_system_signature(system)
        block_system = build_signature_block_linear_system(system)
        rng = np.random.default_rng(23)
        vector = rng.normal(size=orth.shifted_h0.shape[0])
        e_dense, dim_dense = solve_orthonormal_ic_system(orth)
        direct = solve_signature_block_linear_system_iterative(block_system)

        assert block_system.block_keys == orth.block_keys
        assert block_system.operator.block_ranges == orth.block_ranges
        assert block_system.operator.shape == orth.shifted_h0.shape
        assert np.linalg.norm(block_system.coupling - orth.coupling) < TOL
        assert np.linalg.norm(
            signature_block_operator_matvec(block_system.operator, vector)
            - orth.shifted_h0 @ vector
        ) < 1e-10
        assert direct.converged
        assert direct.n_basis == dim_dense
        assert direct.residual_norm < 1e-10
        assert abs(direct.energy - e_dense) < 1e-11


def test_signature_raw_block_system_builds_same_linear_system():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        system = build_factored_ic_system_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        raw = signature_raw_block_system_from_factored(system)
        direct = build_signature_block_linear_system(system)
        from_raw = build_signature_block_linear_system_from_raw(raw)
        rng = np.random.default_rng(31)
        vector = rng.normal(size=direct.operator.shape[0])
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_raw = solve_signature_block_linear_system_iterative(from_raw)

        covered = np.concatenate([block.candidate_indices for block in raw.blocks])
        assert sorted(covered.tolist()) == list(range(len(system.candidates)))
        assert raw.e0 == system.e0
        assert [block.key for block in raw.blocks] == [
            block_range[0] for block_range in direct.operator.block_ranges
        ]
        assert from_raw.block_keys == direct.block_keys
        assert from_raw.operator.block_ranges == direct.operator.block_ranges
        assert np.linalg.norm(from_raw.coupling - direct.coupling) < TOL
        assert np.linalg.norm(
            signature_block_operator_matvec(from_raw.operator, vector)
            - signature_block_operator_matvec(direct.operator, vector)
        ) < TOL
        assert e_raw.converged
        assert e_raw.n_basis == e_direct.n_basis
        assert abs(e_raw.energy - e_direct.energy) < TOL


def test_no_inactive_rdm_overlap_coupling_matches_factored_oracle():
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            2,
            2,
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            4,
            4,
        ),
    ]
    for mol, basis_name, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e, ham.h2e, n_core=0, n_act=n_active, n_act_elec=n_elec, ms2=0
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=0,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, overlap, coupling = no_inactive_external_rdm_overlap_coupling(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_active,
            n_elec,
            ms2=0,
        )

        assert indices
        assert len(indices) == len(system.candidates)
        assert {
            system.candidates[idx].external_class for idx in indices
        } == {(0, 1), (0, 2)}
        assert any(system.candidates[idx].active_creations for idx in indices)
        assert np.max(
            np.abs(overlap - system.overlap[np.ix_(indices, indices)])
        ) < 1e-11
        assert np.max(np.abs(coupling - system.coupling[list(indices)])) < 1e-11


def test_no_inactive_rdm_sv_raw_block_system_matches_oracle():
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            2,
            2,
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            4,
            4,
        ),
    ]
    for mol, basis_name, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e, ham.h2e, n_core=0, n_act=n_active, n_act_elec=n_elec, ms2=0
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=0,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        raw = signature_raw_block_system_from_factored(system)
        hybrid_raw = raw_block_system_with_no_inactive_external_rdm_sv(
            raw,
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_active,
            n_elec,
            ms2=0,
        )
        direct = build_signature_block_linear_system_from_raw(raw)
        hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

        assert e_hybrid.converged
        assert e_hybrid.n_basis == e_direct.n_basis
        assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_no_inactive_active_ci_h0_raw_blocks_match_oracle():
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            2,
            2,
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            4,
            4,
        ),
    ]
    for mol, basis_name, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e, ham.h2e, n_core=0, n_act=n_active, n_act_elec=n_elec, ms2=0
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=0,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, h0, e0 = no_inactive_external_active_ci_h0(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_active,
            n_elec,
            ms2=0,
        )

        assert len(indices) == len(system.candidates)
        assert abs(e0 - system.e0) < 1e-11
        assert np.max(np.abs(h0 - system.h0[np.ix_(indices, indices)])) < 1e-11

        raw = signature_raw_block_system_from_factored(system)
        scrubbed_raw = type(raw)(
            blocks=tuple(
                type(block)(
                    key=block.key,
                    candidate_indices=block.candidate_indices,
                    overlap=block.overlap,
                    h0=np.zeros_like(block.h0),
                    coupling=block.coupling,
                )
                for block in raw.blocks
            ),
            couplings=tuple(
                type(coupling)(
                    left_block=coupling.left_block,
                    right_block=coupling.right_block,
                    h0=np.zeros_like(coupling.h0),
                )
                for coupling in raw.couplings
            ),
            e0=raw.e0,
        )
        hybrid_raw = raw_block_system_with_no_inactive_external_active_ci_h0(
            scrubbed_raw,
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_active,
            n_elec,
            ms2=0,
        )
        for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
            assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        for direct_coupling, hybrid_coupling in zip(
            raw.couplings, hybrid_raw.couplings
        ):
            assert (
                np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11
            )
        direct = build_signature_block_linear_system_from_raw(raw)
        hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

        assert e_hybrid.converged
        assert e_hybrid.n_basis == e_direct.n_basis
        assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_no_inactive_active_ci_raw_system_restores_scrubbed_oracle():
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            2,
            2,
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            4,
            4,
        ),
    ]
    for mol, basis_name, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e, ham.h2e, n_core=0, n_act=n_active, n_act_elec=n_elec, ms2=0
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=0,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        raw = signature_raw_block_system_from_factored(system)
        scrubbed_raw = type(raw)(
            blocks=tuple(
                type(block)(
                    key=block.key,
                    candidate_indices=block.candidate_indices,
                    overlap=np.zeros_like(block.overlap),
                    h0=np.zeros_like(block.h0),
                    coupling=np.zeros_like(block.coupling),
                )
                for block in raw.blocks
            ),
            couplings=tuple(
                type(coupling)(
                    left_block=coupling.left_block,
                    right_block=coupling.right_block,
                    h0=np.zeros_like(coupling.h0),
                )
                for coupling in raw.couplings
            ),
            e0=raw.e0,
        )
        hybrid_raw = raw_block_system_with_no_inactive_external_active_ci(
            scrubbed_raw,
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_active,
            n_elec,
            ms2=0,
        )

        for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
            assert np.max(
                np.abs(hybrid_block.overlap - direct_block.overlap)
            ) < 1e-11
            assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
            assert np.max(
                np.abs(hybrid_block.coupling - direct_block.coupling)
            ) < 1e-11
        for direct_coupling, hybrid_coupling in zip(
            raw.couplings, hybrid_raw.couplings
        ):
            assert (
                np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11
            )

        direct = build_signature_block_linear_system_from_raw(raw)
        hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

        assert e_hybrid.converged
        assert e_hybrid.n_basis == e_direct.n_basis
        assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_no_inactive_active_ci_raw_system_builds_without_oracle_blocks():
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            2,
            2,
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            4,
            4,
        ),
    ]
    for mol, basis_name, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e, ham.h2e, n_core=0, n_act=n_active, n_act_elec=n_elec, ms2=0
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=0,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        direct_raw = signature_raw_block_system_from_factored(
            system, coupling_tol=1e-12
        )
        built_raw = build_no_inactive_external_active_ci_raw_system(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_active,
            n_elec,
            ms2=0,
            coupling_tol=1e-12,
        )

        assert len(built_raw.blocks) == len(direct_raw.blocks)
        assert len(built_raw.couplings) == len(direct_raw.couplings)
        assert abs(built_raw.e0 - direct_raw.e0) < 1e-11
        for direct_block, built_block in zip(direct_raw.blocks, built_raw.blocks):
            assert built_block.key == direct_block.key
            assert np.array_equal(
                built_block.candidate_indices, direct_block.candidate_indices
            )
            assert np.max(
                np.abs(built_block.overlap - direct_block.overlap)
            ) < 1e-11
            assert np.max(np.abs(built_block.h0 - direct_block.h0)) < 1e-11
            assert np.max(
                np.abs(built_block.coupling - direct_block.coupling)
            ) < 1e-11
        for direct_coupling, built_coupling in zip(
            direct_raw.couplings, built_raw.couplings
        ):
            assert built_coupling.left_block == direct_coupling.left_block
            assert built_coupling.right_block == direct_coupling.right_block
            assert np.max(
                np.abs(built_coupling.h0 - direct_coupling.h0)
            ) < 1e-11

        direct = build_signature_block_linear_system_from_raw(direct_raw)
        built = build_signature_block_linear_system_from_raw(built_raw)
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_built = solve_signature_block_linear_system_iterative(built)

        assert e_built.converged
        assert e_built.n_basis == e_direct.n_basis
        assert abs(e_built.energy - e_direct.energy) < 1e-10


def test_no_inactive_active_ci_driver_matches_explicit_oracle():
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            2,
            2,
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            4,
            4,
        ),
    ]
    for mol, basis_name, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        dense, dim_dense = factored_ic_caspt2_corr_dense(
            ham.h1e,
            ham.h2e,
            n_core=0,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        direct = no_inactive_active_ci_caspt2_corr_signature_iterative(
            ham.h1e,
            ham.h2e,
            n_active,
            n_elec,
            ms2=0,
        )
        explicit, dim_explicit = _ic_caspt2_corr(
            ham.h1e,
            ham.h2e,
            n_core=0,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )

        assert direct.converged
        assert direct.n_basis == dim_dense == dim_explicit
        assert abs(direct.energy - dense) < 1e-10
        assert abs(direct.energy - explicit) < 1e-10


def test_inactive_to_active_rdm_overlap_coupling_matches_oracle():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, overlap, coupling = inactive_to_active_rdm_overlap_coupling(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )

        assert indices
        assert {
            system.candidates[idx].external_class for idx in indices
        } == {(1, 0)}
        assert all(system.candidates[idx].order == 1 for idx in indices)
        assert np.max(
            np.abs(overlap - system.overlap[np.ix_(indices, indices)])
        ) < 1e-11
        assert np.max(np.abs(coupling - system.coupling[list(indices)])) < 1e-11


def test_inactive_to_active_rdm_sv_raw_block_system_matches_oracle():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, _, _ = inactive_to_active_rdm_overlap_coupling(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )
        formula_indices = set(indices)
        raw = signature_raw_block_system_from_factored(system)
        scrubbed_raw = type(raw)(
            blocks=tuple(
                _scrub_raw_block_sv(block, formula_indices)
                for block in raw.blocks
            ),
            couplings=raw.couplings,
            e0=raw.e0,
        )
        hybrid_raw = raw_block_system_with_inactive_to_active_rdm_sv(
            scrubbed_raw,
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )

        assert indices
        assert {
            system.candidates[idx].external_class for idx in indices
        } == {(1, 0)}
        assert all(system.candidates[idx].order == 1 for idx in indices)
        for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
            assert np.max(
                np.abs(hybrid_block.overlap - direct_block.overlap)
            ) < 1e-11
            assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
            assert np.max(
                np.abs(hybrid_block.coupling - direct_block.coupling)
            ) < 1e-11
        for direct_coupling, hybrid_coupling in zip(
            raw.couplings, hybrid_raw.couplings
        ):
            assert direct_coupling.left_block == hybrid_coupling.left_block
            assert direct_coupling.right_block == hybrid_coupling.right_block
            assert (
                np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11
            )

        direct = build_signature_block_linear_system_from_raw(raw)
        hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

        assert e_hybrid.converged
        assert e_hybrid.n_basis == e_direct.n_basis
        assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_to_active_active_ci_h0_matches_oracle():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, h0, e0 = inactive_to_active_active_ci_h0(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )

        assert indices
        assert {
            system.candidates[idx].external_class for idx in indices
        } == {(1, 0)}
        assert all(system.candidates[idx].order == 1 for idx in indices)
        assert abs(e0 - system.e0) < 1e-11
        assert np.max(np.abs(h0 - system.h0[np.ix_(indices, indices)])) < 1e-11


def test_inactive_to_active_active_ci_h0_raw_blocks_match_oracle():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, _, _ = inactive_to_active_active_ci_h0(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )
        formula_indices = set(indices)
        raw = signature_raw_block_system_from_factored(system)
        scrubbed_raw = type(raw)(
            blocks=tuple(
                _scrub_raw_block_h0(block, formula_indices)
                for block in raw.blocks
            ),
            couplings=tuple(
                _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
                for coupling in raw.couplings
            ),
            e0=raw.e0,
        )
        hybrid_raw = raw_block_system_with_inactive_to_active_active_ci_h0(
            scrubbed_raw,
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )

        assert indices
        assert {
            system.candidates[idx].external_class for idx in indices
        } == {(1, 0)}
        assert all(system.candidates[idx].order == 1 for idx in indices)
        for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
            assert np.max(
                np.abs(hybrid_block.overlap - direct_block.overlap)
            ) < 1e-11
            assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
            assert np.max(
                np.abs(hybrid_block.coupling - direct_block.coupling)
            ) < 1e-11
        for direct_coupling, hybrid_coupling in zip(
            raw.couplings, hybrid_raw.couplings
        ):
            assert direct_coupling.left_block == hybrid_coupling.left_block
            assert direct_coupling.right_block == hybrid_coupling.right_block
            assert (
                np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11
            )

        direct = build_signature_block_linear_system_from_raw(raw)
        hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

        assert e_hybrid.converged
        assert e_hybrid.n_basis == e_direct.n_basis
        assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_to_active_active_ci_raw_system_restores_scrubbed_oracle():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        sv_indices, _, _ = inactive_to_active_rdm_overlap_coupling(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )
        h0_indices, _, _ = inactive_to_active_active_ci_h0(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )
        assert sv_indices == h0_indices
        formula_indices = set(sv_indices)
        raw = signature_raw_block_system_from_factored(system)
        scrubbed_raw = type(raw)(
            blocks=tuple(
                _scrub_raw_block_h0(
                    _scrub_raw_block_sv(block, formula_indices),
                    formula_indices,
                )
                for block in raw.blocks
            ),
            couplings=tuple(
                _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
                for coupling in raw.couplings
            ),
            e0=raw.e0,
        )
        hybrid_raw = raw_block_system_with_inactive_to_active_active_ci(
            scrubbed_raw,
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )

        assert sv_indices
        for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
            assert np.max(
                np.abs(hybrid_block.overlap - direct_block.overlap)
            ) < 1e-11
            assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
            assert np.max(
                np.abs(hybrid_block.coupling - direct_block.coupling)
            ) < 1e-11
        for direct_coupling, hybrid_coupling in zip(
            raw.couplings, hybrid_raw.couplings
        ):
            assert direct_coupling.left_block == hybrid_coupling.left_block
            assert direct_coupling.right_block == hybrid_coupling.right_block
            assert (
                np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11
            )

        direct = build_signature_block_linear_system_from_raw(raw)
        hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
        e_direct = solve_signature_block_linear_system_iterative(direct)
        e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

        assert e_hybrid.converged
        assert e_hybrid.n_basis == e_direct.n_basis
        assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_to_active_active_ci_raw_system_builds_without_oracle_blocks():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, _, _ = inactive_to_active_rdm_overlap_coupling(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )
        built_raw = build_inactive_to_active_active_ci_raw_system(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
            coupling_tol=1e-12,
        )

        groups = {}
        for idx in indices:
            groups.setdefault(system.candidates[idx].external_signature, []).append(idx)
        assert len(built_raw.blocks) == len(groups)
        assert abs(built_raw.e0 - system.e0) < 1e-11

        block_indices = []
        for built_block, (key, group_indices) in zip(
            built_raw.blocks, groups.items()
        ):
            idx = np.asarray(group_indices, dtype=int)
            block_indices.append(idx)
            assert built_block.key == key
            assert np.array_equal(built_block.candidate_indices, idx)
            assert np.max(
                np.abs(built_block.overlap - system.overlap[np.ix_(idx, idx)])
            ) < 1e-11
            assert np.max(
                np.abs(built_block.h0 - system.h0[np.ix_(idx, idx)])
            ) < 1e-11
            assert np.max(
                np.abs(built_block.coupling - system.coupling[idx])
            ) < 1e-11

        expected_couplings = []
        for left_block, left_idx in enumerate(block_indices):
            for right_block in range(left_block + 1, len(block_indices)):
                right_idx = block_indices[right_block]
                matrix = system.h0[np.ix_(left_idx, right_idx)]
                if np.linalg.norm(matrix) > 1e-12:
                    expected_couplings.append((left_block, right_block, matrix))
        assert len(built_raw.couplings) == len(expected_couplings)
        for built_coupling, (left_block, right_block, matrix) in zip(
            built_raw.couplings, expected_couplings
        ):
            assert built_coupling.left_block == left_block
            assert built_coupling.right_block == right_block
            assert np.max(np.abs(built_coupling.h0 - matrix)) < 1e-11

        restricted = subset_factored_ic_system(system, indices)
        e_dense, dim_dense = solve_factored_ic_system(restricted)
        built = build_signature_block_linear_system_from_raw(built_raw)
        e_built = solve_signature_block_linear_system_iterative(built)

        assert e_built.converged
        assert e_built.n_basis == dim_dense
        assert abs(e_built.energy - e_dense) < 1e-10


def test_inactive_to_active_active_ci_driver_matches_restricted_oracle():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, _, _ = inactive_to_active_rdm_overlap_coupling(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )
        restricted = subset_factored_ic_system(system, indices)
        e_dense, dim_dense = solve_factored_ic_system(restricted)
        direct = inactive_to_active_active_ci_caspt2_corr_signature_iterative(
            ham.h1e,
            ham.h2e,
            n_core,
            n_active,
            n_elec,
            ms2=0,
            coupling_tol=1e-12,
        )

        assert direct.converged
        assert direct.n_basis == dim_dense
        assert direct.residual_norm < 1e-10
        assert abs(direct.energy - e_dense) < 1e-10


def test_inactive_pair_distinct_overlap_matches_oracle():
    cases = [
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        prepared = _semicanonical_prep(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_act=n_active,
            n_act_elec=n_elec,
            ms2=0,
        )
        system = build_factored_ic_system_dense(
            ham.h1e,
            ham.h2e,
            n_core=n_core,
            n_active=n_active,
            n_active_elec=n_elec,
            ms2=0,
        )
        indices, overlap = inactive_pair_to_active_pair_distinct_overlap(
            system.candidates,
            prepared["h1"],
            prepared["eri"],
            n_core,
            n_active,
            n_elec,
            ms2=0,
        )

        assert indices
        assert {
            system.candidates[idx].external_class for idx in indices
        } == {(2, 0)}
        assert all(system.candidates[idx].order == 2 for idx in indices)
        assert all(
            len(set(system.candidates[idx].inactive_holes)) == 2
            for idx in indices
        )
        assert np.max(
            np.abs(overlap - system.overlap[np.ix_(indices, indices)])
        ) < 1e-11


def test_inactive_pair_same_hole_overlap_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, overlap = inactive_pair_to_active_pair_same_hole_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    assert {
        system.candidates[idx].external_class for idx in indices
    } == {(2, 0)}
    assert all(system.candidates[idx].order == 2 for idx in indices)
    assert all(
        len(set(system.candidates[idx].inactive_holes)) == 1
        for idx in indices
    )
    assert np.max(
        np.abs(overlap - system.overlap[np.ix_(indices, indices)])
    ) < 1e-11


def test_inactive_pair_overlap_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, overlap = inactive_pair_to_active_pair_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    assert {
        system.candidates[idx].external_class for idx in indices
    } == {(2, 0)}
    assert all(system.candidates[idx].order == 2 for idx in indices)
    assert np.max(
        np.abs(overlap - system.overlap[np.ix_(indices, indices)])
    ) < 1e-11


def test_inactive_pair_active_ci_coupling_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_pair_to_active_pair_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, coupling = inactive_pair_to_active_pair_active_ci_coupling(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert np.max(np.abs(coupling - system.coupling[list(indices)])) < 1e-11


def test_inactive_pair_active_ci_h0_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_pair_to_active_pair_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, h0, e0 = inactive_pair_to_active_pair_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert abs(e0 - system.e0) < 1e-11
    assert np.max(np.abs(h0 - system.h0[np.ix_(indices, indices)])) < 1e-11


def test_inactive_pair_distinct_overlap_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_pair_distinct_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_overlap(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_distinct_overlap(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_same_hole_overlap_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_pair_same_hole_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_overlap(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_same_hole_overlap(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_overlap_raw_blocks_restore_all_simple_i2_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    distinct_indices, _ = inactive_pair_to_active_pair_distinct_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    same_indices, _ = inactive_pair_to_active_pair_same_hole_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(distinct_indices) | set(same_indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_overlap(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_overlap(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert distinct_indices
    assert same_indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_sv_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_pair_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_sv(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_sv(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_active_ci_h0_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _, _ = inactive_pair_to_active_pair_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_active_ci_h0(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_active_ci_raw_system_restores_scrubbed_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_pair_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(
                _scrub_raw_block_sv(block, formula_indices),
                formula_indices,
            )
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_active_ci(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_active_ci_raw_system_builds_without_oracle_blocks():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_pair_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    built_raw = build_inactive_pair_active_ci_raw_system(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    groups = {}
    for idx in indices:
        groups.setdefault(system.candidates[idx].external_signature, []).append(idx)
    assert len(built_raw.blocks) == len(groups)
    assert abs(built_raw.e0 - system.e0) < 1e-11

    block_indices = []
    for built_block, (key, group_indices) in zip(
        built_raw.blocks, groups.items()
    ):
        idx = np.asarray(group_indices, dtype=int)
        block_indices.append(idx)
        assert built_block.key == key
        assert np.array_equal(built_block.candidate_indices, idx)
        assert np.max(
            np.abs(built_block.overlap - system.overlap[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.h0 - system.h0[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.coupling - system.coupling[idx])
        ) < 1e-11

    expected_couplings = []
    for left_block, left_idx in enumerate(block_indices):
        for right_block in range(left_block + 1, len(block_indices)):
            right_idx = block_indices[right_block]
            matrix = system.h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > 1e-12:
                expected_couplings.append((left_block, right_block, matrix))
    assert len(built_raw.couplings) == len(expected_couplings)
    for built_coupling, (left_block, right_block, matrix) in zip(
        built_raw.couplings, expected_couplings
    ):
        assert built_coupling.left_block == left_block
        assert built_coupling.right_block == right_block
        assert np.max(np.abs(built_coupling.h0 - matrix)) < 1e-11

    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    built = build_signature_block_linear_system_from_raw(built_raw)
    e_built = solve_signature_block_linear_system_iterative(built)

    assert e_built.converged
    assert e_built.n_basis == dim_dense
    assert abs(e_built.energy - e_dense) < 1e-10


def test_inactive_pair_active_ci_driver_matches_restricted_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 3
    n_active = 4
    n_elec = 4
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_pair_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    direct = inactive_pair_active_ci_caspt2_corr_signature_iterative(
        ham.h1e,
        ham.h2e,
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    assert direct.converged
    assert direct.n_basis == dim_dense
    assert direct.residual_norm < 1e-10
    assert abs(direct.energy - e_dense) < 1e-10


def test_inactive_pair_to_active_virtual_overlap_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, overlap = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    assert {
        system.candidates[idx].external_class for idx in indices
    } == {(2, 1)}
    assert all(system.candidates[idx].order == 2 for idx in indices)
    assert np.max(
        np.abs(overlap - system.overlap[np.ix_(indices, indices)])
    ) < 1e-11


def test_inactive_pair_active_virtual_overlap_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_overlap(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_active_virtual_overlap(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_active_virtual_coupling_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, coupling = inactive_pair_to_active_virtual_active_ci_coupling(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert np.max(np.abs(coupling - system.coupling[list(indices)])) < 1e-11


def test_inactive_pair_active_virtual_sv_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_sv(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_active_virtual_sv(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_active_virtual_h0_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, h0, e0 = inactive_pair_to_active_virtual_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert abs(e0 - system.e0) < 1e-11
    assert np.max(np.abs(h0 - system.h0[np.ix_(indices, indices)])) < 1e-11


def test_inactive_pair_active_virtual_h0_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _, _ = inactive_pair_to_active_virtual_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_active_virtual_h0(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_active_virtual_raw_system_restores_scrubbed_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(
                _scrub_raw_block_sv(block, formula_indices),
                formula_indices,
            )
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_active_virtual_active_ci(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_active_virtual_raw_system_builds_without_oracle_blocks():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    built_raw = build_inactive_pair_active_virtual_active_ci_raw_system(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    groups = {}
    for idx in indices:
        groups.setdefault(system.candidates[idx].external_signature, []).append(idx)
    assert len(built_raw.blocks) == len(groups)
    assert abs(built_raw.e0 - system.e0) < 1e-11

    block_indices = []
    for built_block, (key, group_indices) in zip(
        built_raw.blocks, groups.items()
    ):
        idx = np.asarray(group_indices, dtype=int)
        block_indices.append(idx)
        assert built_block.key == key
        assert np.array_equal(built_block.candidate_indices, idx)
        assert np.max(
            np.abs(built_block.overlap - system.overlap[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.h0 - system.h0[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.coupling - system.coupling[idx])
        ) < 1e-11

    expected_couplings = []
    for left_block, left_idx in enumerate(block_indices):
        for right_block in range(left_block + 1, len(block_indices)):
            right_idx = block_indices[right_block]
            matrix = system.h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > 1e-12:
                expected_couplings.append((left_block, right_block, matrix))
    assert len(built_raw.couplings) == len(expected_couplings)
    for built_coupling, (left_block, right_block, matrix) in zip(
        built_raw.couplings, expected_couplings
    ):
        assert built_coupling.left_block == left_block
        assert built_coupling.right_block == right_block
        assert np.max(np.abs(built_coupling.h0 - matrix)) < 1e-11

    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    built = build_signature_block_linear_system_from_raw(built_raw)
    e_built = solve_signature_block_linear_system_iterative(built)

    assert e_built.converged
    assert e_built.n_basis == dim_dense
    assert abs(e_built.energy - e_dense) < 1e-10


def test_inactive_pair_active_virtual_driver_matches_restricted_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_active_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    direct = inactive_pair_active_virtual_active_ci_caspt2_corr_signature_iterative(
        ham.h1e,
        ham.h2e,
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    assert direct.converged
    assert direct.n_basis == dim_dense
    assert direct.residual_norm < 1e-10
    assert abs(direct.energy - e_dense) < 1e-10


def test_inactive_active_to_virtual_pair_overlap_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, overlap = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    assert {
        system.candidates[idx].external_class for idx in indices
    } == {(1, 2)}
    assert all(system.candidates[idx].order == 2 for idx in indices)
    assert np.max(
        np.abs(overlap - system.overlap[np.ix_(indices, indices)])
    ) < 1e-11


def test_inactive_active_to_virtual_pair_overlap_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_overlap(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_active_to_virtual_pair_overlap(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_active_to_virtual_pair_coupling_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, coupling = inactive_active_to_virtual_pair_active_ci_coupling(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert np.max(np.abs(coupling - system.coupling[list(indices)])) < 1e-11


def test_inactive_active_to_virtual_pair_sv_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_sv(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_active_to_virtual_pair_sv(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_active_to_virtual_pair_h0_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, h0, e0 = inactive_active_to_virtual_pair_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert abs(e0 - system.e0) < 1e-11
    assert np.max(np.abs(h0 - system.h0[np.ix_(indices, indices)])) < 1e-11


def test_inactive_active_to_virtual_pair_h0_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _, _ = inactive_active_to_virtual_pair_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_active_to_virtual_pair_h0(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_active_to_virtual_pair_raw_system_restores_scrubbed_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(
                _scrub_raw_block_sv(block, formula_indices),
                formula_indices,
            )
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_active_to_virtual_pair_active_ci(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_active_to_virtual_pair_raw_system_builds_without_oracle_blocks():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    built_raw = build_inactive_active_to_virtual_pair_active_ci_raw_system(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    groups = {}
    for idx in indices:
        groups.setdefault(system.candidates[idx].external_signature, []).append(idx)
    assert len(built_raw.blocks) == len(groups)
    assert abs(built_raw.e0 - system.e0) < 1e-11

    block_indices = []
    for built_block, (key, group_indices) in zip(
        built_raw.blocks, groups.items()
    ):
        idx = np.asarray(group_indices, dtype=int)
        block_indices.append(idx)
        assert built_block.key == key
        assert np.array_equal(built_block.candidate_indices, idx)
        assert np.max(
            np.abs(built_block.overlap - system.overlap[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.h0 - system.h0[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.coupling - system.coupling[idx])
        ) < 1e-11

    expected_couplings = []
    for left_block, left_idx in enumerate(block_indices):
        for right_block in range(left_block + 1, len(block_indices)):
            right_idx = block_indices[right_block]
            matrix = system.h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > 1e-12:
                expected_couplings.append((left_block, right_block, matrix))
    assert len(built_raw.couplings) == len(expected_couplings)
    for built_coupling, (left_block, right_block, matrix) in zip(
        built_raw.couplings, expected_couplings
    ):
        assert built_coupling.left_block == left_block
        assert built_coupling.right_block == right_block
        assert np.max(np.abs(built_coupling.h0 - matrix)) < 1e-11

    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    built = build_signature_block_linear_system_from_raw(built_raw)
    e_built = solve_signature_block_linear_system_iterative(built)

    assert e_built.converged
    assert e_built.n_basis == dim_dense
    assert abs(e_built.energy - e_dense) < 1e-10


def test_inactive_active_to_virtual_pair_driver_matches_restricted_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_active_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    direct = inactive_active_to_virtual_pair_active_ci_caspt2_corr_signature_iterative(
        ham.h1e,
        ham.h2e,
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    assert direct.converged
    assert direct.n_basis == dim_dense
    assert direct.residual_norm < 1e-10
    assert abs(direct.energy - e_dense) < 1e-10


def test_inactive_pair_to_virtual_pair_overlap_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, overlap = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    assert {
        system.candidates[idx].external_class for idx in indices
    } == {(2, 2)}
    assert all(system.candidates[idx].order == 2 for idx in indices)
    assert np.max(
        np.abs(overlap - system.overlap[np.ix_(indices, indices)])
    ) < 1e-11


def test_inactive_pair_to_virtual_pair_overlap_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_overlap(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_to_virtual_pair_overlap(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_to_virtual_pair_coupling_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, coupling = inactive_pair_to_virtual_pair_active_ci_coupling(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert np.max(np.abs(coupling - system.coupling[list(indices)])) < 1e-11


def test_inactive_pair_to_virtual_pair_sv_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_sv(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_to_virtual_pair_sv(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_to_virtual_pair_h0_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, h0, e0 = inactive_pair_to_virtual_pair_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert abs(e0 - system.e0) < 1e-11
    assert np.max(np.abs(h0 - system.h0[np.ix_(indices, indices)])) < 1e-11


def test_inactive_pair_to_virtual_pair_h0_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _, _ = inactive_pair_to_virtual_pair_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_to_virtual_pair_h0(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_to_virtual_pair_raw_system_restores_scrubbed_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(
                _scrub_raw_block_sv(block, formula_indices),
                formula_indices,
            )
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_pair_to_virtual_pair_active_ci(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_pair_to_virtual_pair_raw_system_builds_without_oracle_blocks():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    built_raw = build_inactive_pair_to_virtual_pair_active_ci_raw_system(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    groups = {}
    for idx in indices:
        groups.setdefault(system.candidates[idx].external_signature, []).append(idx)
    assert len(built_raw.blocks) == len(groups)
    assert abs(built_raw.e0 - system.e0) < 1e-11

    block_indices = []
    for built_block, (key, group_indices) in zip(
        built_raw.blocks, groups.items()
    ):
        idx = np.asarray(group_indices, dtype=int)
        block_indices.append(idx)
        assert built_block.key == key
        assert np.array_equal(built_block.candidate_indices, idx)
        assert np.max(
            np.abs(built_block.overlap - system.overlap[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.h0 - system.h0[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.coupling - system.coupling[idx])
        ) < 1e-11

    expected_couplings = []
    for left_block, left_idx in enumerate(block_indices):
        for right_block in range(left_block + 1, len(block_indices)):
            right_idx = block_indices[right_block]
            matrix = system.h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > 1e-12:
                expected_couplings.append((left_block, right_block, matrix))
    assert len(built_raw.couplings) == len(expected_couplings)
    for built_coupling, (left_block, right_block, matrix) in zip(
        built_raw.couplings, expected_couplings
    ):
        assert built_coupling.left_block == left_block
        assert built_coupling.right_block == right_block
        assert np.max(np.abs(built_coupling.h0 - matrix)) < 1e-11

    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    built = build_signature_block_linear_system_from_raw(built_raw)
    e_built = solve_signature_block_linear_system_iterative(built)

    assert e_built.converged
    assert e_built.n_basis == dim_dense
    assert abs(e_built.energy - e_dense) < 1e-10


def test_inactive_pair_to_virtual_pair_driver_matches_restricted_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_pair_to_virtual_pair_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    direct = inactive_pair_to_virtual_pair_active_ci_caspt2_corr_signature_iterative(
        ham.h1e,
        ham.h2e,
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    assert direct.converged
    assert direct.n_basis == dim_dense
    assert direct.residual_norm < 1e-10
    assert abs(direct.energy - e_dense) < 1e-10


def test_inactive_to_virtual_overlap_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, overlap = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    assert {
        system.candidates[idx].external_class for idx in indices
    } == {(1, 1)}
    assert all(system.candidates[idx].order in {1, 2} for idx in indices)
    assert np.max(
        np.abs(overlap - system.overlap[np.ix_(indices, indices)])
    ) < 1e-11


def test_inactive_to_virtual_overlap_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_overlap(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_to_virtual_overlap(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_to_virtual_coupling_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, coupling = inactive_to_virtual_active_ci_coupling(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert np.max(np.abs(coupling - system.coupling[list(indices)])) < 1e-11


def test_inactive_to_virtual_sv_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_sv(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=raw.couplings,
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_to_virtual_sv(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_to_virtual_h0_matches_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    overlap_indices, _ = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    indices, h0, e0 = inactive_to_virtual_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices == overlap_indices
    assert abs(e0 - system.e0) < 1e-11
    assert np.max(np.abs(h0 - system.h0[np.ix_(indices, indices)])) < 1e-11


def test_inactive_to_virtual_h0_raw_blocks_match_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _, _ = inactive_to_virtual_active_ci_h0(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(block, formula_indices)
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_to_virtual_h0(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_to_virtual_raw_system_restores_scrubbed_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    formula_indices = set(indices)
    raw = signature_raw_block_system_from_factored(system)
    scrubbed_raw = type(raw)(
        blocks=tuple(
            _scrub_raw_block_h0(
                _scrub_raw_block_sv(block, formula_indices),
                formula_indices,
            )
            for block in raw.blocks
        ),
        couplings=tuple(
            _scrub_raw_coupling_h0(coupling, raw.blocks, formula_indices)
            for coupling in raw.couplings
        ),
        e0=raw.e0,
    )
    hybrid_raw = raw_block_system_with_inactive_to_virtual_active_ci(
        scrubbed_raw,
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )

    assert indices
    for direct_block, hybrid_block in zip(raw.blocks, hybrid_raw.blocks):
        assert np.max(
            np.abs(hybrid_block.overlap - direct_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(hybrid_block.h0 - direct_block.h0)) < 1e-11
        assert np.max(
            np.abs(hybrid_block.coupling - direct_block.coupling)
        ) < 1e-11
    for direct_coupling, hybrid_coupling in zip(
        raw.couplings, hybrid_raw.couplings
    ):
        assert direct_coupling.left_block == hybrid_coupling.left_block
        assert direct_coupling.right_block == hybrid_coupling.right_block
        assert np.max(np.abs(hybrid_coupling.h0 - direct_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(raw)
    hybrid = build_signature_block_linear_system_from_raw(hybrid_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_hybrid = solve_signature_block_linear_system_iterative(hybrid)

    assert e_hybrid.converged
    assert e_hybrid.n_basis == e_direct.n_basis
    assert abs(e_hybrid.energy - e_direct.energy) < 1e-10


def test_inactive_to_virtual_raw_system_builds_without_oracle_blocks():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    built_raw = build_inactive_to_virtual_active_ci_raw_system(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    groups = {}
    for idx in indices:
        groups.setdefault(system.candidates[idx].external_signature, []).append(idx)
    assert len(built_raw.blocks) == len(groups)
    assert abs(built_raw.e0 - system.e0) < 1e-11

    block_indices = []
    for built_block, (key, group_indices) in zip(
        built_raw.blocks, groups.items()
    ):
        idx = np.asarray(group_indices, dtype=int)
        block_indices.append(idx)
        assert built_block.key == key
        assert np.array_equal(built_block.candidate_indices, idx)
        assert np.max(
            np.abs(built_block.overlap - system.overlap[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.h0 - system.h0[np.ix_(idx, idx)])
        ) < 1e-11
        assert np.max(
            np.abs(built_block.coupling - system.coupling[idx])
        ) < 1e-11

    expected_couplings = []
    for left_block, left_idx in enumerate(block_indices):
        for right_block in range(left_block + 1, len(block_indices)):
            right_idx = block_indices[right_block]
            matrix = system.h0[np.ix_(left_idx, right_idx)]
            if np.linalg.norm(matrix) > 1e-12:
                expected_couplings.append((left_block, right_block, matrix))
    assert len(built_raw.couplings) == len(expected_couplings)
    for built_coupling, (left_block, right_block, matrix) in zip(
        built_raw.couplings, expected_couplings
    ):
        assert built_coupling.left_block == left_block
        assert built_coupling.right_block == right_block
        assert np.max(np.abs(built_coupling.h0 - matrix)) < 1e-11

    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    built = build_signature_block_linear_system_from_raw(built_raw)
    e_built = solve_signature_block_linear_system_iterative(built)

    assert e_built.converged
    assert e_built.n_basis == dim_dense
    assert abs(e_built.energy - e_dense) < 1e-10


def test_inactive_to_virtual_driver_matches_restricted_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    indices, _ = inactive_to_virtual_active_ci_overlap(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
    )
    restricted = subset_factored_ic_system(system, indices)
    e_dense, dim_dense = solve_factored_ic_system(restricted)
    direct = inactive_to_virtual_active_ci_caspt2_corr_signature_iterative(
        ham.h1e,
        ham.h2e,
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )

    assert direct.converged
    assert direct.n_basis == dim_dense
    assert direct.residual_norm < 1e-10
    assert abs(direct.energy - e_dense) < 1e-10


def test_active_ci_signature_raw_system_matches_dense_oracle():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    n_core = 2
    n_active = 3
    n_elec = 2
    ham = _hamiltonian(mol, "sto-3g")
    prepared = _semicanonical_prep(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_act=n_active,
        n_act_elec=n_elec,
        ms2=0,
    )
    system = build_factored_ic_system_dense(
        ham.h1e,
        ham.h2e,
        n_core=n_core,
        n_active=n_active,
        n_active_elec=n_elec,
        ms2=0,
    )
    direct_raw = build_active_ci_signature_raw_system(
        system.candidates,
        prepared["h1"],
        prepared["eri"],
        n_core,
        n_active,
        n_elec,
        ms2=0,
        coupling_tol=1e-12,
    )
    dense_raw = signature_raw_block_system_from_factored(
        system, coupling_tol=1e-12
    )

    assert len(direct_raw.blocks) == len(dense_raw.blocks)
    assert len(direct_raw.couplings) == len(dense_raw.couplings)
    assert abs(direct_raw.e0 - dense_raw.e0) < 1e-11
    for direct_block, dense_block in zip(direct_raw.blocks, dense_raw.blocks):
        assert direct_block.key == dense_block.key
        assert np.array_equal(
            direct_block.candidate_indices, dense_block.candidate_indices
        )
        assert np.max(
            np.abs(direct_block.overlap - dense_block.overlap)
        ) < 1e-11
        assert np.max(np.abs(direct_block.h0 - dense_block.h0)) < 1e-11
        assert np.max(
            np.abs(direct_block.coupling - dense_block.coupling)
        ) < 1e-11
    for direct_coupling, dense_coupling in zip(
        direct_raw.couplings, dense_raw.couplings
    ):
        assert direct_coupling.left_block == dense_coupling.left_block
        assert direct_coupling.right_block == dense_coupling.right_block
        assert np.max(np.abs(direct_coupling.h0 - dense_coupling.h0)) < 1e-11

    direct = build_signature_block_linear_system_from_raw(direct_raw)
    dense = build_signature_block_linear_system_from_raw(dense_raw)
    e_direct = solve_signature_block_linear_system_iterative(direct)
    e_dense = solve_signature_block_linear_system_iterative(dense)

    assert e_direct.converged
    assert e_direct.n_basis == e_dense.n_basis
    assert abs(e_direct.energy - e_dense.energy) < 1e-10


def test_signature_block_matvec_matches_dense_shifted_h0():
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
    ham = _hamiltonian(mol, "sto-3g")
    system = build_factored_ic_system_dense(
        ham.h1e, ham.h2e, n_core=1, n_active=2, n_active_elec=2, ms2=0
    )
    orth = orthonormalize_factored_ic_system_signature(system)
    operator = orthonormal_signature_block_operator(orth)
    rng = np.random.default_rng(12)
    vector = rng.normal(size=orth.shifted_h0.shape[0])

    dense = orth.shifted_h0 @ vector
    blocked = orthonormal_signature_block_matvec(orth, vector)
    precomputed = signature_block_operator_matvec(operator, vector)
    assert operator.block_ranges == orth.block_ranges
    assert operator.shape == orth.shifted_h0.shape
    assert len(operator.diagonal_blocks) == len(orth.block_ranges)
    assert np.linalg.norm(blocked - dense) < TOL
    assert np.linalg.norm(precomputed - dense) < TOL

    block_edges = set()
    for coupling in orthonormal_signature_block_couplings(orth):
        left = orth.block_ranges[coupling.left_block][0]
        right = orth.block_ranges[coupling.right_block][0]
        edge = (left, right) if repr(left) <= repr(right) else (right, left)
        block_edges.add(edge)
    assert block_edges == orthonormal_signature_coupling_edges(orth)


def test_signature_block_component_solve_matches_dense_orthonormal_solve():
    cases = [
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec in cases:
        ham = _hamiltonian(mol, basis_name)
        system = build_factored_ic_system_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        orth = orthonormalize_factored_ic_system_signature(system)
        block_system = build_signature_block_linear_system(system)
        components = orthonormal_signature_block_components(orth)
        direct_components = signature_block_operator_components(
            block_system.operator
        )
        e_dense, dim_dense = solve_orthonormal_ic_system(orth)
        e_components, dim_components = solve_orthonormal_ic_system_block_components(
            orth
        )
        e_direct, dim_direct = solve_signature_block_linear_system_components(
            block_system
        )
        e_wrapped, dim_wrapped = solve_factored_ic_system_signature_components(system)

        assert direct_components == components
        flattened = [block for component in components for block in component]
        assert sorted(flattened) == list(range(len(orth.block_ranges)))
        assert len(flattened) == len(set(flattened))
        component_id = {
            block: idx for idx, component in enumerate(components) for block in component
        }
        for coupling in orthonormal_signature_block_couplings(orth):
            assert component_id[coupling.left_block] == component_id[
                coupling.right_block
            ]
        assert dim_components == dim_dense
        assert dim_direct == dim_dense
        assert dim_wrapped == dim_dense
        assert abs(e_components - e_dense) < 1e-11
        assert abs(e_direct - e_dense) < 1e-11
        assert abs(e_wrapped - e_dense) < 1e-11


def test_signature_orthonormal_coupling_edges_are_one_body_reachable():
    mol = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
    ham = _hamiltonian(mol, "sto-3g")
    system = build_factored_ic_system_dense(
        ham.h1e, ham.h2e, n_core=1, n_active=2, n_active_elec=2, ms2=0
    )
    orth = orthonormalize_factored_ic_system_signature(system)
    edges = orthonormal_signature_coupling_edges(orth)

    assert edges
    for left, right in edges:
        h_left, p_left = len(left[0]), len(left[1])
        h_right, p_right = len(right[0]), len(right[1])
        assert abs(h_left - h_right) <= 1
        assert abs(p_left - p_right) <= 1


def test_doubly_external_h0_blocks_are_external_signature_diagonal():
    cases = [
        # Has secondary virtuals, so pins the (0,2) virtual-pair blocks.
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
            (0, 2),
        ),
        # Has multiple inactive orbitals and no secondary virtuals, so pins
        # the (2,0) inactive-hole-pair blocks.
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
            (2, 0),
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec, external_class in cases:
        ham = _hamiltonian(mol, basis_name)
        system = build_factored_ic_system_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        indices = [
            idx
            for idx, candidate in enumerate(system.candidates)
            if candidate.external_class == external_class
        ]
        assert indices
        for i in indices:
            for j in indices:
                left = system.candidates[i].external_signature
                right = system.candidates[j].external_signature
                if left != right:
                    assert abs(system.h0[i, j]) < TOL


def test_doubly_external_independent_signature_solves_match_restricted_dense():
    cases = [
        (
            Molecule([Atom(9, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])]),
            "sto-3g",
            3,
            2,
            4,
            (2, 2),
        ),
        (
            Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])]),
            "sto-3g",
            1,
            2,
            2,
            (0, 2),
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
            (2, 0),
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec, external_class in cases:
        ham = _hamiltonian(mol, basis_name)
        system = build_factored_ic_system_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        indices = [
            idx
            for idx, candidate in enumerate(system.candidates)
            if candidate.external_class == external_class
        ]
        restricted = subset_factored_ic_system(system, indices)
        e_restricted, dim_restricted = solve_factored_ic_system(restricted)
        e_blocks, dim_blocks = solve_factored_ic_system_independent_signatures(
            system, external_class
        )
        orth = orthonormalize_factored_ic_system_signature(system)
        e_orth_blocks, dim_orth_blocks = solve_orthonormal_ic_system_independent_blocks(
            orth,
            key_filter=lambda key: (len(key[0]), len(key[1])) == external_class,
        )
        (
            e_orth_wrapper,
            dim_orth_wrapper,
        ) = solve_factored_ic_system_orthonormal_independent_signatures(
            system, external_class
        )
        assert dim_blocks == dim_restricted
        assert dim_orth_blocks == dim_restricted
        assert dim_orth_wrapper == dim_restricted
        assert abs(e_blocks - e_restricted) < TOL
        assert abs(e_orth_blocks - e_restricted) < TOL
        assert abs(e_orth_wrapper - e_restricted) < TOL


def test_factored_dense_total_matches_openmolcas_references():
    # Recorded OpenMolcas &CASPT2 totals on fixed CASCI(HF) references,
    # Group=C1, Frozen=0, IPEAshift=0.0; see runner_openmolcas.py.
    cases = [
        (
            Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]),
            "6-31g",
            0,
            2,
            2,
            -1.14670322,
        ),
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "sto-3g",
            3,
            4,
            4,
            -74.97302712,
        ),
    ]
    for mol, basis_name, n_core, n_active, n_elec, e_openmolcas in cases:
        ham = _hamiltonian(mol, basis_name)
        cas = casci(
            ham.h1e,
            ham.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_active,
            n_core=n_core,
            nuclear_repulsion=ham.nuclear_repulsion,
        )
        e_corr, _ = factored_ic_caspt2_corr_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        block = factored_ic_caspt2_corr_signature_iterative(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        direct = active_ci_caspt2_corr_signature_iterative(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        assert block.converged
        assert direct.converged
        assert abs(block.energy - e_corr) < 1e-11
        assert abs(direct.energy - e_corr) < 1e-11
        assert direct.n_basis == block.n_basis
        assert abs(cas.e_total + e_corr - e_openmolcas) < 5e-6
        assert abs(cas.e_total + block.energy - e_openmolcas) < 5e-6
        assert abs(cas.e_total + direct.energy - e_openmolcas) < 5e-6


# Heavy multireference validation: builds dense explicit CASPT2 oracles up to
# the internally-contracted dim ~1881 (H2O/6-31G, N2/STO-3G) — minutes per build
# via the dense Python state-dot. The signature-driver contract it checks is
# covered at small scale by the fast oracle-matching tests above; this is the
# large-oracle cross-check, so it runs on the slow/nightly lane.
@pytest.mark.slow
def test_active_ci_signature_driver_matches_larger_explicit_oracles():
    cases = [
        (
            Molecule(
                [
                    Atom(8, [0.0, 0.0, 0.0]),
                    Atom(1, [0.0, 1.43, -0.93]),
                    Atom(1, [0.0, -1.43, -0.93]),
                ]
            ),
            "6-31g",
            3,
            4,
            4,
            1881,
            -76.10408243,
        ),
        (
            Molecule(
                [
                    Atom(7, [0.0, 0.0, 0.0]),
                    Atom(7, [0.0, 0.0, 2.074]),
                ]
            ),
            "sto-3g",
            4,
            6,
            6,
            1020,
            -107.64577483,
        ),
    ]
    for (
        mol,
        basis_name,
        n_core,
        n_active,
        n_elec,
        expected_dim,
        e_openmolcas,
    ) in cases:
        ham = _hamiltonian(mol, basis_name)
        cas = casci(
            ham.h1e,
            ham.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_active,
            n_core=n_core,
            nuclear_repulsion=ham.nuclear_repulsion,
        )
        e_dense, dim_dense = factored_ic_caspt2_corr_dense(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )
        direct = active_ci_caspt2_corr_signature_iterative(
            ham.h1e, ham.h2e, n_core, n_active, n_elec, ms2=0
        )

        assert direct.converged
        assert dim_dense == expected_dim
        assert direct.n_basis == expected_dim
        assert direct.residual_norm < 1e-10
        assert abs(direct.energy - e_dense) < 1e-10
        assert abs(cas.e_total + direct.energy - e_openmolcas) < 5e-6


# cc-pVDZ-scale CASPT2 (O/H₂, 24 basis functions): the largest single SCF +
# CASPT2 in this file. The public-path correctness is covered at STO-3G/6-31G
# scale by the fast tests above; the cc-pVDZ OpenMolcas-parity cross-check runs
# on the slow/nightly lane.
@pytest.mark.slow
def test_caspt2_public_direct_path_matches_ccpvdz_openmolcas_scale():
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    ham = _hamiltonian(mol, "cc-pvdz")
    cas = casci(
        ham.h1e,
        ham.h2e,
        n_active_elec=4,
        n_active_orb=4,
        n_core=3,
        nuclear_repulsion=ham.nuclear_repulsion,
    )
    pt = caspt2(
        cas,
        ham.h1e,
        ham.h2e,
        n_core=3,
        n_virt=ham.norb - 7,
    )

    assert abs(pt.e_total - (-76.22073575)) < 5e-6


def test_imaginary_shift_direct_matches_explicit_small_systems():
    # Forsberg-Malmqvist imaginary shift on the signature-block path: the
    # SPD-solve form E2(σ) = −(V + σ²z)·(Az) with z = (A²+σ²)⁻¹V
    # (solve_signature_block_linear_system_imaginary) must reproduce the
    # explicit engine's eigenbasis form (Forsberg & Malmqvist 1997, Eq 11)
    # to roundoff — both evaluate the same functional of (A, V) on the
    # same retained contracted space.
    lih = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
    h2o = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    for mol, basis, n_core, n_act, n_elec in (
        (lih, "sto-3g", 1, 2, 2),
        (h2o, "sto-3g", 3, 4, 4),
    ):
        ham = _hamiltonian(mol, basis)
        for sigma in (0.1, 0.3):
            e_explicit, _ = _ic_caspt2_corr(
                ham.h1e, ham.h2e, n_core, n_act, n_elec, 0, imaginary=sigma
            )
            direct = active_ci_caspt2_corr_signature_iterative(
                ham.h1e,
                ham.h2e,
                n_core,
                n_act,
                n_elec,
                0,
                tol=1e-12,
                imaginary=sigma,
            )
            assert direct.converged
            assert abs(direct.energy - e_explicit) < 1e-12


def test_imaginary_shift_no_longer_forces_explicit_dispatch():
    # Large shifted jobs now ride the direct path; IPEA stays explicit-only
    # and small jobs stay on the explicit engine regardless of the shift.
    assert _ic_caspt2_uses_direct_active_ci(25, 3, 4, imaginary=0.1)
    assert not _ic_caspt2_uses_direct_active_ci(25, 3, 4, ipea=0.25)
    assert not _ic_caspt2_uses_direct_active_ci(7, 3, 4, imaginary=0.1)


@pytest.mark.slow
def test_caspt2_public_direct_path_imaginary_matches_ccpvdz_openmolcas():
    # Live OpenMolcas reference recorded 2026-06-10 via
    # examples/regression/core/runner_openmolcas.py::run_caspt2(
    #   H2O/cc-pVDZ, CAS(4,4), Frozen=0, Imaginary=0.1, CIonly):
    # CASPT2 total = -76.22073564 (unshifted recorded: -76.22073575).
    # The shifted large-CAS job dispatches to the direct active-CI
    # signature-block solver (dispatch pinned above).
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.43, -0.93]),
            Atom(1, [0.0, -1.43, -0.93]),
        ]
    )
    ham = _hamiltonian(mol, "cc-pvdz")
    cas = casci(
        ham.h1e,
        ham.h2e,
        n_active_elec=4,
        n_active_orb=4,
        n_core=3,
        nuclear_repulsion=ham.nuclear_repulsion,
    )
    pt = caspt2(
        cas,
        ham.h1e,
        ham.h2e,
        n_core=3,
        n_virt=ham.norb - 7,
        imaginary=0.1,
    )
    assert abs(pt.e_total - (-76.22073564)) < 5e-6
