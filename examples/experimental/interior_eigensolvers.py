"""Compare experimental JD/GPLHR interior roots with dense diagonalization.

Run with the checkout's Python after rebuilding the native extension:
    python examples/experimental/interior_eigensolvers.py
"""

import numpy as np

from vibeqc.solvers.eigensolver import (
    EigenProblem,
    InteriorEigenProblem,
    SolverOptions,
    solve_eigenproblem,
)


def main():
    sigma = 0.4248
    for n in (60, 120, 300):
        for coupling in (0.01, 0.05):
            rng = np.random.default_rng(42)
            upper = np.triu(rng.normal(scale=coupling, size=(n, n)), 1)
            matrix = np.diag(np.linspace(-2.0, 3.0, n)) + upper + upper.T
            spectrum = np.linalg.eigvalsh(matrix)
            reference = spectrum[np.argmin(np.abs(spectrum - sigma))]
            problem = InteriorEigenProblem(EigenProblem(matrix=matrix), sigma)
            options = SolverOptions(n_roots=1, tol=1e-8, max_iter=500)
            print(f"n={n}, coupling={coupling}, nearest root={reference:.10f}")
            for solver in ("jd", "gplhr"):
                try:
                    result = solve_eigenproblem(problem, options, solver=solver)
                except RuntimeError as exc:
                    print(f"  {solver}: {exc}")
                    continue
                residual = np.linalg.norm(
                    matrix @ result.eigenvectors
                    - result.eigenvectors * result.eigenvalues,
                    axis=0,
                ).max()
                print(
                    f"  {solver}: value={result.eigenvalues[0]:.10f}, "
                    f"converged={result.converged}, iterations={result.n_iter}, "
                    f"residual={residual:.3e}"
                )


if __name__ == "__main__":
    main()
