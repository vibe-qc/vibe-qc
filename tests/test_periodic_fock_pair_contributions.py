"""Phase M3b: per-cell-pair J/K contributions.

``build_jk_pair_contributions`` runs the same Γ-only molecular-limit
shell-quartet kernel as ``build_jk_gamma_molecular_limit_explicit`` but
stores each ``(c_g, c_p)`` cell pair's J/K block separately instead of
summing them. The symmetry-reduction layer needs the un-summed blocks so
it can evaluate only orbit representatives and scatter the rest.

``build_jk_gamma_molecular_limit`` computes its cell list internally from
``opts.cutoff_bohr`` — the same list returned by ``direct_lattice_cells``
— so passing that list to ``build_jk_pair_contributions`` gives an exact
like-for-like comparison.

Correctness witnesses:

1. **All pairs sum to the full build.** Summing ``J_contrib`` (resp.
   ``K_contrib``) over the complete ``n_c × n_c`` pair set reproduces
   ``build_jk_gamma_molecular_limit`` exactly.

2. **Subset additivity.** Splitting the pair list into disjoint subsets
   and summing each gives partial J/K whose total equals the full build.

3. **Kernel options carry through** — ``omega`` and the unscreened
   (``schwarz_threshold = 0``) path both reproduce the summed kernel
   under the matching option setting.

The Python orbit-orchestration / Wigner-D reconstruction layer is a
separate milestone (see ``docs/symmetry_status.md``); this file
validates the C++ kernel alone.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _mg_system(a_bohr: float = 5.0, cutoff: float = 8.0):
    """Mg primitive cell — multi-cell at this cutoff, single closed-shell atom."""
    sysp = vq.PeriodicSystem(3, np.eye(3) * a_bohr, [vq.Atom(12, [0, 0, 0])])
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = cutoff
    mol = sysp.unit_cell_molecule()
    P = np.asarray(vq.run_rhf(mol, basis).density)
    return sysp, basis, opts, P


def _all_pairs(n_c: int) -> list[tuple[int, int]]:
    return [(g, p) for g in range(n_c) for p in range(n_c)]


# ---------------------------------------------------------------------------
# Witness 1 — all pairs sum to the full build
# ---------------------------------------------------------------------------

def test_all_pairs_sum_matches_full_kernel():
    """Summing every (c_g, c_p) contribution reproduces the summed kernel."""
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    n_c = len(cells)
    assert n_c > 1, "test system must span multiple cells to be meaningful"

    contribs = vq.build_jk_pair_contributions(
        basis, sysp, cells, _all_pairs(n_c), opts, P
    )
    assert len(contribs) == n_c * n_c

    nbf = int(basis.nbasis)
    J_sum = np.zeros((nbf, nbf))
    K_sum = np.zeros((nbf, nbf))
    for c in contribs:
        J_sum += np.asarray(c.J_contrib)
        K_sum += np.asarray(c.K_contrib)

    jk_full = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P)
    assert np.allclose(J_sum, np.asarray(jk_full.J), atol=1e-12)
    assert np.allclose(K_sum, np.asarray(jk_full.K), atol=1e-12)


def test_pair_indices_roundtrip():
    """Each output entry reports the (c_g, c_p) of its input pair, in order."""
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    n_c = len(cells)
    pairs = _all_pairs(n_c)

    contribs = vq.build_jk_pair_contributions(basis, sysp, cells, pairs, opts, P)
    for (g, p), c in zip(pairs, contribs):
        assert c.c_g == g
        assert c.c_p == p
        assert np.asarray(c.J_contrib).shape == (int(basis.nbasis),) * 2
        assert np.asarray(c.K_contrib).shape == (int(basis.nbasis),) * 2


# ---------------------------------------------------------------------------
# Witness 2 — subset additivity
# ---------------------------------------------------------------------------

def test_subset_partition_sums_to_full():
    """Disjoint pair subsets give partial J/K that sum to the full build."""
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    n_c = len(cells)
    all_pairs = _all_pairs(n_c)

    subset_a = all_pairs[0::2]
    subset_b = all_pairs[1::2]

    nbf = int(basis.nbasis)

    def _sum(pairs):
        J = np.zeros((nbf, nbf))
        K = np.zeros((nbf, nbf))
        if not pairs:
            return J, K
        for c in vq.build_jk_pair_contributions(
            basis, sysp, cells, pairs, opts, P
        ):
            J += np.asarray(c.J_contrib)
            K += np.asarray(c.K_contrib)
        return J, K

    Ja, Ka = _sum(subset_a)
    Jb, Kb = _sum(subset_b)

    jk_full = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P)
    assert np.allclose(Ja + Jb, np.asarray(jk_full.J), atol=1e-12)
    assert np.allclose(Ka + Kb, np.asarray(jk_full.K), atol=1e-12)

    # A subset must give strictly partial (non-full) results.
    assert not np.allclose(Ja, np.asarray(jk_full.J), atol=1e-9)


def test_single_pair_contribution_is_self_consistent():
    """A one-pair call equals that pair's slice of the all-pairs call."""
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    n_c = len(cells)

    full = vq.build_jk_pair_contributions(
        basis, sysp, cells, _all_pairs(n_c), opts, P
    )
    target = (0, n_c - 1)
    one = vq.build_jk_pair_contributions(basis, sysp, cells, [target], opts, P)
    assert len(one) == 1

    flat_idx = target[0] * n_c + target[1]
    assert np.allclose(
        np.asarray(one[0].J_contrib), np.asarray(full[flat_idx].J_contrib),
        atol=1e-13,
    )
    assert np.allclose(
        np.asarray(one[0].K_contrib), np.asarray(full[flat_idx].K_contrib),
        atol=1e-13,
    )


# ---------------------------------------------------------------------------
# Witness 3 — kernel options carry through
# ---------------------------------------------------------------------------

def test_omega_screened_path_matches_full_kernel():
    """With omega > 0 the per-pair sum still reproduces the summed kernel."""
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    n_c = len(cells)
    omega = 0.4

    contribs = vq.build_jk_pair_contributions(
        basis, sysp, cells, _all_pairs(n_c), opts, P, omega
    )
    nbf = int(basis.nbasis)
    J_sum = sum((np.asarray(c.J_contrib) for c in contribs), np.zeros((nbf, nbf)))
    K_sum = sum((np.asarray(c.K_contrib) for c in contribs), np.zeros((nbf, nbf)))

    jk_full = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, omega)
    assert np.allclose(J_sum, np.asarray(jk_full.J), atol=1e-12)
    assert np.allclose(K_sum, np.asarray(jk_full.K), atol=1e-12)


def test_unscreened_path_matches_unscreened_kernel():
    """schwarz_threshold = 0 (unscreened): the pair sum still reproduces the
    summed kernel run under the *same* unscreened setting.

    Note: screened and unscreened results are not expected to agree with
    each other — cell-level Schwarz also drops K terms whose ket-pair cell
    falls outside the truncation list. The witness here is that each option
    setting is reproduced faithfully, like-for-like."""
    sysp, basis, opts, _P = _mg_system()
    opts.schwarz_threshold = 0.0  # unscreened
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    n_c = len(cells)
    mol = sysp.unit_cell_molecule()
    P = np.asarray(vq.run_rhf(mol, basis).density)
    nbf = int(basis.nbasis)

    J = np.zeros((nbf, nbf))
    K = np.zeros((nbf, nbf))
    for c in vq.build_jk_pair_contributions(
        basis, sysp, cells, _all_pairs(n_c), opts, P
    ):
        J += np.asarray(c.J_contrib)
        K += np.asarray(c.K_contrib)

    jk_full = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P)
    assert np.allclose(J, np.asarray(jk_full.J), atol=1e-12)
    assert np.allclose(K, np.asarray(jk_full.K), atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_pair_list_returns_empty():
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    contribs = vq.build_jk_pair_contributions(basis, sysp, cells, [], opts, P)
    assert list(contribs) == []


def test_out_of_range_pair_index_raises():
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    n_c = len(cells)
    with pytest.raises(Exception):
        vq.build_jk_pair_contributions(
            basis, sysp, cells, [(0, n_c)], opts, P
        )
    with pytest.raises(Exception):
        vq.build_jk_pair_contributions(
            basis, sysp, cells, [(-1, 0)], opts, P
        )


def test_density_shape_mismatch_raises():
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    bad_P = np.zeros((P.shape[0] + 1, P.shape[1] + 1))
    with pytest.raises(Exception):
        vq.build_jk_pair_contributions(
            basis, sysp, cells, [(0, 0)], opts, bad_P
        )


def test_negative_omega_raises():
    sysp, basis, opts, P = _mg_system()
    cells = vq.direct_lattice_cells(sysp, opts.cutoff_bohr)
    with pytest.raises(Exception):
        vq.build_jk_pair_contributions(
            basis, sysp, cells, [(0, 0)], opts, P, -1.0
        )
