"""Internal bridges from method-owned Fock policies to operator solves."""
from dataclasses import replace

import numpy as np

from .eigensolver import SolverOptions
from .operators import FockOperator, OperatorMetadata, _snapshot, solve_operator


def solve_dense_fock(fock, *, X=None, overlap=None, kind="fock",
                     spin="restricted", kpoint=(0., 0., 0.), generation=0,
                     basis_id=None):
    """Stateless dense Fock solve in an explicitly identified coordinate frame.

    Pass both X and S for an AO Fock, retaining the method's rank policy.
    Omit both for a Fock already projected/extrapolated in an orthonormal
    frame, including a semicanonical MO partition. That result contains
    reduced coefficients and residuals; its caller lifts the coefficients
    through the corresponding basis. No AO operator is reconstructed
    from a projected DIIS matrix, and no all-k metric copies are retained.
    Occupations and k-point weights are always the caller's responsibility.
    """
    if (X is None) != (overlap is None):
        raise ValueError("AO Fock solves require both X and overlap")
    orthogonal = X is None
    metadata = OperatorMetadata(
        kind=kind, spin=spin, kpoint=kpoint, generation=generation,
        basis_id=basis_id if basis_id is not None else (
            "orthogonal-scf" if orthogonal else "ao-scf"),
    )
    result = solve_operator(
        FockOperator(matrix=fock, X=X, overlap=overlap, metadata=metadata),
        solver="dense",
    )
    if orthogonal:
        result.residual_frame = "orthogonal"
    return result


class FockDiagonalizer:
    """One SCF channel with a fixed AO frame and its own warm subspace.

    The method supplies its existing X and S, chooses the size threshold,
    and retains occupation/acceleration policy. Native option objects are
    read once and never mutated. Use separate instances for spins/k-points.
    ``n_required`` is the method's required orbital count, which can include
    the full retained space when occupations depend on virtual candidates.
    Real-only consumers opt into ``real_only``: their transpose-based density
    construction cannot accept arbitrary complex orbital phases.
    """

    def __init__(self, X, overlap, scf_options, *, kind="fock",
                 spin="restricted", kpoint=(0., 0., 0.), n_required=0,
                 real_only=False, frame=None):
        self.frame = frame
        if frame is not None and (X is not frame.X or overlap is not frame.overlap):
            raise ValueError("Shared-frame SCF must use that frame's X and overlap")
        self.X = _snapshot(X) if frame is None else frame.X
        self.overlap = _snapshot(overlap) if frame is None else frame.overlap
        self.real_only = bool(real_only)
        if self.real_only and (np.iscomplexobj(self.X) or np.iscomplexobj(self.overlap)):
            raise NotImplementedError("Real-only SCF requires real X and overlap")
        self.metadata = OperatorMetadata(
            kind=kind, spin=spin, kpoint=kpoint,
            basis_id="fixed-scf-ao" if frame is None else frame.basis_id,
            frame_id=None if frame is None else frame.identity)
        self.iterative = bool(getattr(scf_options, "use_davidson", False)
                              and self.X.shape[0] >= getattr(scf_options, "davidson_min_dim", 100))
        self.options = SolverOptions()
        if self.iterative:
            native = getattr(scf_options, "davidson", None)
            if native is not None:
                if native.max_refine != 5:
                    raise NotImplementedError("Davidson max_refine is not implemented by the native kernel")
                self.options = SolverOptions(
                    n_roots=native.n_eig, n_guess=native.n_guess,
                    max_subspace=native.max_subspace, tol=native.conv_tol,
                    max_iter=native.max_iter, preshift=native.preshift,
                    verbosity=native.verbosity,
                )
                guess = np.asarray(native.guess_vectors_cplx)
                if self.real_only and guess.size:
                    raise NotImplementedError("Real-only SCF does not support complex Davidson guesses")
                if not guess.size:
                    guess = np.asarray(native.guess_vectors)
                if guess.size:
                    # Native guesses are reduced coordinates; public guesses are AO.
                    self.options.guess_vectors = self.X @ guess
        self.options.n_roots = self.options.n_roots or self.X.shape[1]
        if self.options.n_roots > self.X.shape[1]:
            raise ValueError(
                f"Requested Fock roots ({self.options.n_roots}) exceed "
                f"the retained orbital space ({self.X.shape[1]})"
            )
        if self.options.n_roots < n_required:
            raise ValueError(
                f"Requested Fock roots ({self.options.n_roots}) do not cover "
                f"the required orbital space ({n_required})"
            )
        self.last_result = None

    def __call__(self, fock):
        operator = fock if isinstance(fock, FockOperator) else None
        if operator is not None:
            if (self.frame is None or operator.frame is None
                    or self.frame.identity != operator.frame.identity
                    or (operator.metadata.kind, operator.metadata.spin, operator.metadata.kpoint)
                    != (self.metadata.kind, self.metadata.spin, self.metadata.kpoint)):
                raise ValueError("Fock build belongs to a different SCF channel/frame")
        complex_ = ((operator.dtype.kind == "c" if operator is not None else np.iscomplexobj(fock))
                    or np.iscomplexobj(self.X)
                    or np.iscomplexobj(self.options.guess_vectors))
        if self.real_only and complex_:
            raise NotImplementedError("Real-only SCF does not support complex Fock matrices")
        solver = ("hermitian_davidson" if complex_ else "davidson") if self.iterative else "dense"
        previous = self.last_result if self.iterative else None
        options = replace(self.options, guess_vectors=None) if previous is not None else self.options
        if operator is None:
            operator = FockOperator(matrix=fock, metadata=self.metadata, **(
                dict(X=self.X, overlap=self.overlap) if self.frame is None else dict(frame=self.frame)))
        result = solve_operator(
            operator,
            options, solver=solver, previous=previous,
        )
        self.last_result = result
        self.metadata = replace(self.metadata, generation=self.metadata.generation + 1)
        # Occupation/state-following code may rotate its coefficients in place.
        return result.eigenvectors.copy(), result.eigenvalues.copy()
