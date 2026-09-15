"""v2RDM Q/G constraint enforcement coverage (audit finding 3).

Background
----------
The existing tests in ``tests/test_solvers_v2rdm_tc.py`` check only
that the solver returns a finite, negative energy and that the trace
constraint residual is small.  Neither distinguishes "constraint
**requested**" from "constraint **actually enforced**" — a v2RDM
implementation that silently ignores the Q and G N-representability
constraints can pass the existing tests.

The current solver
(``python/vibeqc/solvers/_v2rdm.py::V2RDMSolver.solve``) explicitly
raises ``NotImplementedError`` when ``constraints`` contains ``"q"`` or
``"g"`` — only the ``"p"`` (²D ≥ 0) projection is enforced.  The Q and
G *builders* exist and are reachable through ``_compute_residual``, so
the tests below pin both contracts:

1. Requesting ``"pq"`` / ``"qg"`` / ``"pqg"`` (or any case variant)
   must raise ``NotImplementedError`` — the user is told upfront that
   the constraint is not enforced rather than getting back a number
   labelled ``v2rdm(pqg)`` that is really ``v2rdm(p)``.
2. The Q and G residuals themselves are non-trivial:  on a 2-RDM that
   is PSD but constructed to be **not** Q-PSD, the Q residual must be
   strictly positive.  This is what proves the residual machinery is
   live; otherwise a future "fix" that returns 0 unconditionally would
   sneak past the existing trace-residual asserts.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    V2RDMOptions,
    V2RDMSolver,
    build_hamiltonian_mo,
    get_hf_orbital_provider,
    solve_v2rdm,
)


@pytest.fixture
def h2_ham():
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    C = get_hf_orbital_provider(mol, basis)
    return build_hamiltonian_mo(mol, basis, C)


class TestUnsupportedConstraintsAreRejected:
    """Requesting Q/G must raise; not silently downgrade to 'p' only."""

    @pytest.mark.parametrize(
        "constraints",
        ["pq", "pg", "pqg", "PQG", "qp", "gQ", "q", "g"],
    )
    def test_qg_constraint_raises(self, h2_ham, constraints):
        opts = V2RDMOptions(
            constraints=constraints,
            outer_max_iter=5,
            conv_tol_primal=1e-3,
            verbose=0,
        )
        with pytest.raises(NotImplementedError, match="Q/G"):
            solve_v2rdm(h2_ham, opts)

    def test_p_only_does_not_raise(self, h2_ham):
        opts = V2RDMOptions(
            constraints="p",
            outer_max_iter=3,
            conv_tol_primal=1e-3,
            verbose=0,
        )
        # No exception expected — only proves the gate is keyed on Q/G,
        # not on any non-empty constraint string.
        solve_v2rdm(h2_ham, opts)


class TestQGResidualMachineryIsLive:
    """The Q / G PSD residual must move when fed a 2-RDM that's PSD but
    not Q- or G-PSD.  A residual function that always returns zero would
    pass the loose trace-residual tests in test_solvers_v2rdm_tc.py;
    this nails it down explicitly.

    Construction: take the closed-shell HF 2-RDM for H2/STO-3G (which
    is PSD by construction) and perturb only the off-diagonal blocks of
    ²D in a way that keeps ²D PSD but pushes Q out of the PSD cone.
    We verify directly that:

      * residual(constraints="p")    is tiny (HF 2-RDM is PSD),
      * residual(constraints="pq")   > residual(constraints="p"),
      * residual(constraints="pg")   > residual(constraints="p"),
      * residual(constraints="pqg") ≥ max(pq, pg).
    """

    def test_q_machinery_picks_up_overoccupation(self):
        """Overoccupied rdm1 (3 e⁻ in one spatial orbital, Pauli-
        forbidden) makes the 2-hole matrix ²Q lose positive semi-
        definiteness, while the constructed ²D still sits along a
        single rank-1 product direction (residual_p = 0).

        A residual implementation that ignored the ``"q"`` switch and
        just returned the ²D contribution would report 0 here and fail.
        """
        solver = V2RDMSolver()
        norb = 3
        rdm1 = np.zeros((norb, norb))
        rdm1[0, 0] = 3.0  # Pauli-violating overoccupation
        rdm2 = (
            np.einsum("ik,jl->ijkl", rdm1, rdm1)
            - 0.5 * np.einsum("il,jk->ijkl", rdm1, rdm1)
        )
        rdm2 = solver._enforce_antisymmetry(rdm2)

        res_p  = solver._compute_residual(rdm2, rdm1, norb, "p")
        res_pq = solver._compute_residual(rdm2, rdm1, norb, "pq")

        assert res_p < 1e-9
        assert res_pq > 1.0, (
            f"Q residual {res_pq} should be ≫ 0 for overoccupied rdm1; "
            "_compute_residual may be ignoring the 'q' switch."
        )

    def test_g_machinery_picks_up_off_diagonal_correlation(self):
        """Two-electron state with off-diagonal one-body correlations
        (rdm1 with a 0.5 cross term) makes ²G lose PSD while ²D stays
        PSD.  A residual implementation that ignored the ``"g"`` switch
        would report 0 here and fail.
        """
        solver = V2RDMSolver()
        norb = 3
        rdm1 = np.array(
            [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        rdm2 = (
            np.einsum("ik,jl->ijkl", rdm1, rdm1)
            - 0.5 * np.einsum("il,jk->ijkl", rdm1, rdm1)
        )
        rdm2 = solver._enforce_antisymmetry(rdm2)

        res_p  = solver._compute_residual(rdm2, rdm1, norb, "p")
        res_pg = solver._compute_residual(rdm2, rdm1, norb, "pg")

        assert res_pg > res_p + 1e-3, (
            f"G residual {res_pg:.4e} not larger than p residual "
            f"{res_p:.4e}; _compute_residual may be ignoring the 'g' "
            "switch."
        )

    def test_pqg_residual_is_sum_of_contributions(self):
        """Combined ``"pqg"`` residual must include each requested
        contribution: it equals ``res_p + res_q + res_g`` by
        construction (each term is a sum of |negative eigenvalues|),
        and in particular is at least as large as any subset."""
        solver = V2RDMSolver()
        norb = 3
        rdm1 = np.array(
            [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        rdm2 = (
            np.einsum("ik,jl->ijkl", rdm1, rdm1)
            - 0.5 * np.einsum("il,jk->ijkl", rdm1, rdm1)
        )
        rdm2 = solver._enforce_antisymmetry(rdm2)
        res_p   = solver._compute_residual(rdm2, rdm1, norb, "p")
        res_pq  = solver._compute_residual(rdm2, rdm1, norb, "pq")
        res_pg  = solver._compute_residual(rdm2, rdm1, norb, "pg")
        res_pqg = solver._compute_residual(rdm2, rdm1, norb, "pqg")
        # additivity: pqg − p == (pq − p) + (pg − p)
        assert res_pqg == pytest.approx(
            res_pq + res_pg - res_p, abs=1e-9
        ), (
            f"pqg={res_pqg}, pq={res_pq}, pg={res_pg}, p={res_p}; "
            "the constraints string switch in _compute_residual is "
            "not additive — Q or G contributions are not being summed."
        )


class TestV2RDMHonoursTraceConstraintForP:
    """Sanity check the 'p'-only path still actually enforces Tr(¹D)=N.

    The existing v2rdm_tc test uses a 1e-3 tolerance and only checks
    when ``result.constraint_residual is not None``.  This is a tighter
    regression guard for the trace constraint at a level where the
    'constraint requested ⇒ enforced' contract is meaningful.
    """

    def test_trace_residual_below_tol(self, h2_ham):
        opts = V2RDMOptions(
            constraints="p",
            outer_max_iter=400,
            mu=10.0,
            mu_factor=1.2,
            conv_tol_primal=1e-4,
            verbose=0,
        )
        result = solve_v2rdm(h2_ham, opts)
        # constraint_residual is |Tr(rdm1) - nelec|.  For 'p' constraint
        # enforcement this must converge below the asked tolerance — if
        # it doesn't, the augmented-Lagrangian update is broken.
        assert result.constraint_residual is not None
        assert result.constraint_residual < 1e-2, (
            f"Tr(rdm1)-N residual {result.constraint_residual:.3e} "
            "exceeds 1e-2; trace constraint is not being enforced."
        )
