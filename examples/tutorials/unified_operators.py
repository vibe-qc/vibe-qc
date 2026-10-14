"""Shared operator tutorial and reproducible, bounded eigensolver benchmark.

Run after the editable native build:
    python examples/tutorials/unified_operators.py --size 80

No result files are created. Operator counts include residual verification.
Times are local observations, not cross-machine performance claims.
"""
from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from vibeqc.output.eigensolver import eigensolver_diagnostics
from vibeqc.solvers.eigensolver import SolverOptions
from vibeqc.solvers.operators import (
    FockOperator, HermitianOperator, OperatorMetadata, solve_operator,
    subspace_overlap,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=80)
    args = parser.parse_args()
    if args.size < 20:
        parser.error("--size must be at least 20 for this partial-root example")
    rng = np.random.default_rng(472)
    n = args.size
    perturbation = rng.normal(scale=.002, size=(n, n))
    matrix = np.diag(np.linspace(-2., 3., n)) + perturbation + perturbation.T
    opts = SolverOptions(n_roots=3, tol=1e-9, n_guess=8)
    op = HermitianOperator(
        matvec=lambda v: matrix @ v,
        matmat=lambda v: matrix @ v,
        n=n, diagonal=matrix.diagonal(),
        metadata=OperatorMetadata(kind="response", basis_id="example-response"),
    )
    reference = np.linalg.eigvalsh(matrix)[:3]
    previous = None
    for label in ("cold", "warm"):
        start = perf_counter()
        result = solve_operator(op, opts, solver="davidson", previous=previous)
        seconds = perf_counter() - start
        error = max(abs(result.eigenvalues-reference))
        assert error < 1e-8
        print(f"{label}: seconds={seconds:.6f}, iterations={result.n_iter}, "
              f"vector_applications={result.n_matvec}, eigenvalue_error={error:.3e}")
        previous = result
    print("Residual convergence is not a completeness certificate:", result.selection_verified)

    # Complex block callbacks use the same native iteration as explicit input.
    # The block-call count measures Python crossings; vector applications sum
    # the block widths and include independent residual diagnostics.
    complex_matrix = matrix + 1j*(perturbation - perturbation.T)
    blocks = []
    def complex_block(v):
        blocks.append(v.shape[1])
        return complex_matrix @ v
    complex_op = HermitianOperator(matmat=complex_block, n=n, dtype=complex,
        diagonal=complex_matrix.diagonal().real,
        metadata=OperatorMetadata(kind="response", basis_id="example-complex-response"))
    complex_result = solve_operator(complex_op, opts, solver="hermitian_davidson")
    complex_error = max(abs(complex_result.eigenvalues-np.linalg.eigvalsh(complex_matrix)[:3]))
    assert complex_error < 1e-8
    assert complex_result.n_matvec == sum(blocks)
    print(f"complex callback: iterations={complex_result.n_iter}, "
          f"block_calls={len(blocks)}, vector_applications={sum(blocks)}, "
          f"eigenvalue_error={complex_error:.3e}")

    # A frozen separable AO model: A = diag(h) + B B^H. The action and
    # reduced diagonal are evaluated from factors, without constructing A.
    # This illustrates a provider boundary, not an integral or J/K engine.
    metric_diagonal = np.linspace(.7, 1.3, n)
    x = np.diag(1 / np.sqrt(metric_diagonal))
    ao_h = np.linspace(-2., 3., n) * metric_diagonal
    factors = rng.normal(scale=.01, size=(n, 4)) + 1j*rng.normal(scale=.01, size=(n, 4))
    ao_h.setflags(write=False)
    factors.setflags(write=False)
    def frozen_ao_block(v):
        return ao_h[:, None]*v + factors @ (factors.conj().T @ v)
    reduced_diagonal = (ao_h + np.sum(abs(factors)**2, axis=1)) / metric_diagonal
    ao_action = FockOperator(
        matmat=frozen_ao_block, n=n, dtype=complex, X=x,
        overlap=np.diag(metric_diagonal), reduced_diagonal=reduced_diagonal,
        metadata=OperatorMetadata(kind="ks", basis_id="example-factorized-AO", generation=1),
    )
    ao_result = solve_operator(ao_action, opts, solver="hermitian_davidson")
    # Explicit matrices are used only for this small independent reference.
    reduced_factors = x.conj().T @ factors
    ao_reference = np.linalg.eigvalsh(np.diag(ao_h / metric_diagonal)
                                    + reduced_factors @ reduced_factors.conj().T)[:3]
    ao_error = max(abs(ao_result.eigenvalues-ao_reference))
    assert ao_error < 1e-8
    assert max(ao_result.residuals) < 2e-9
    print(f"factorized AO callback: vector_applications={ao_result.n_matvec}, "
          f"eigenvalue_error={ao_error:.3e}, "
          f"AO_residual={max(ao_result.residuals):.3e}, "
          f"projected_residual={max(ao_result.projected_residuals):.3e}")

    # Complex AO Fock/KS with a nontrivial positive definite overlap.
    f = np.array([[-.7, .12j, .03], [-.12j, -.1, .08j], [.03, -.08j, .8]])
    s = np.array([[1., .05, 0.], [.05, 1.2, .03], [0., .03, .9]])
    ks = FockOperator(matrix=f, overlap=s, metadata=OperatorMetadata(
        kind="ks", spin="alpha", kpoint=(.25, 0., 0.), basis_id="example-AO"))
    result = solve_operator(ks, SolverOptions(n_roots=2, tol=1e-11))
    np.testing.assert_allclose(result.eigenvectors.conj().T @ s @ result.eigenvectors, np.eye(2), atol=1e-12)
    print("Complex AO diagnostics:", eigensolver_diagnostics(result))
    _, principal_cosines = subspace_overlap(result.eigenvectors, result.eigenvectors, overlap=s)
    print("Same-subspace principal cosines:", principal_cosines)

    # Interior selection enumerates the complete dense spectrum. No inverse
    # is needed, so a root exactly at the target is well defined.
    interior = solve_operator(HermitianOperator(matrix=matrix),
        SolverOptions(n_roots=2, sigma=.4248), require_selection=True)
    exact = np.linalg.eigvalsh(matrix)
    expected = np.sort(exact[np.argsort(abs(exact-.4248))[:2]])
    np.testing.assert_allclose(interior.eigenvalues, expected, atol=1e-12)
    print("Verified nearest roots:", interior.eigenvalues)

    # The kind keeps the mathematics honest: a CI Hamiltonian is Hermitian
    # but is not a one-particle Fock matrix.
    ci = HermitianOperator(matrix=np.array([[-1., .04], [.04, -.5]]),
        metadata=OperatorMetadata(kind="ci", basis_id="two-determinants"))
    print("CI roots:", solve_operator(ci).eigenvalues)


if __name__ == "__main__":
    main()
