"""Non-mean-field solvers and eigensolvers for vibe-qc.

This package provides wavefunction-based electronic-structure methods
that do **not** conceptually rely on a Hartree-Fock reference state.
All solvers share a common interface:

    result = solver.solve(hamiltonian, options) -> SolverResult

where ``hamiltonian`` is a :class:`Hamiltonian` carrying one- and
two-electron integrals in an orthonormal spatial-orbital basis.

The submodule :mod:`vibeqc.solvers.eigensolver` provides a separate
unified eigensolver framework for SCF diagonalisation, with problem
descriptors, a solver registry, and multiple backends (dense, Davidson,
Lanczos).

Solvers included
----------------
* :class:`SelectedCISolver` - CIPSI-style selected CI with iterative
  determinant selection, variational diagonalization, and optional
  PT2 correction.
* :class:`DMRGSolver` - two-site DMRG with MPS wavefunction and
  bond-dimension control.
* :class:`V2RDMSolver` - exact small-active-space N-representable 1/2-RDMs
  from the determinant Hamiltonian; Q/G SDP feedback is not yet implemented.
* :func:`build_transcorrelated_hamiltonian` - similarity-transformed
  Hamiltonian (Jastrow/transcorrelated) for use with CI/DMRG backends.
* :func:`casci` - Complete Active Space CI.
* :func:`casscf` - Complete Active Space SCF (orbital-optimized CASCI),
  with optional state-averaging.

Eigensolvers (see :mod:`vibeqc.solvers.eigensolver`)
---------------------------------------------------
* :class:`eigensolver.DenseSolver` - full dense diagonalisation
* :class:`eigensolver.DavidsonSolver` - blocked Davidson (C++ backend)
* :class:`eigensolver.HermitianDavidsonSolver` - complex Hermitian Davidson
* :class:`eigensolver.LanczosSolver` - Lanczos tridiagonalisation (Python)
* :func:`eigensolver.solve_eigenproblem` - unified entry point
* :func:`eigensolver.solve_scf_step` - solve a Fock step from SCFStep

Quick start
-----------
>>> from vibeqc import Atom, Molecule, BasisSet
>>> from vibeqc.solvers import (
...     Hamiltonian,
...     build_hamiltonian_mo,
...     get_hf_orbital_provider,
...     solve_selected_ci,
...     SelectedCIOptions,
... )
>>> mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
>>> basis = BasisSet(mol, "sto-3g")
>>> C = get_hf_orbital_provider(mol, basis)
>>> H = build_hamiltonian_mo(mol, basis, C)
>>> opts = SelectedCIOptions(target_size=100, verbose=1)
>>> result = solve_selected_ci(H, opts)
>>> print(f"E = {result.energy:.8f} Ha")

Eigensolver quick start
-----------------------
>>> from vibeqc.solvers.eigensolver import (
...     EigenProblem, SolverOptions, solve_eigenproblem,
... )
>>> import numpy as np
>>> F = np.array([[2.0, 0.1], [0.1, 1.0]])
>>> problem = EigenProblem(matrix=F, n=2)
>>> options = SolverOptions(n_roots=2, which="SA", tol=1e-8)
>>> result = solve_eigenproblem(problem, options, solver="dense")
>>> print(result.eigenvalues)
"""

from __future__ import annotations

# CASCI -- Complete Active Space CI
from ._casci import (
    CASCIOptions,
    CASCIResult,
    casci,
)

# CASSCF -- Complete Active Space SCF (orbital-optimized CASCI)
from ._casscf import (
    CASSCFOptions,
    CASSCFResult,
    casscf,
)

# CISD -- fixed-space configuration interaction with singles and doubles
from ._cisd import (
    CISDOptions,
    CISDResult,
    cisd,
)

# CCSDT -- full iterative singles, doubles, and triples
from ._ccsdt import (
    CCSDTIteration,
    CCSDTOptions,
    CCSDTResult,
    ccsdt,
)

# CC3 -- iterative singles/doubles with perturbative-order triples
from ._cc3 import (
    CC3Iteration,
    CC3Options,
    CC3Result,
    cc3,
)

# Common interfaces
from ._common import (
    Hamiltonian,
    SolverOptions,
    SolverResult,
)

# Determinant representations
from ._determinant import (
    Det,
    SpinDet,
    determinant_string,
    excitation_rank,
    generate_cisd_determinants,
    generate_closed_shell_determinants,
    generate_determinants,
    generate_doubles,
    generate_singles,
    is_connected,
    reference_determinant,
)

# DMRG
from ._dmrg import (
    DMRGOptions,
    DMRGSolver,
    solve_dmrg,
)

# Hamiltonian construction
from ._hamiltonian import (
    build_hamiltonian_ao,
    build_hamiltonian_mo,
    canonical_orthogonalize,
    get_hf_orbital_provider,
    transform_hamiltonian,
)

# Multi-reference configuration interaction
from ._mrci import (
    MRCIResult,
    mrci,
)

# Multi-reference perturbation theory
from ._mrpt import (
    CASPT2Options,
    NEVPT2Options,
    NEVPT2Result,
    caspt2,
    nevpt2,
)

# Multi-state CASPT2 (MS / XMS effective Hamiltonian)
from ._ms_caspt2 import (
    MSCASPT2Result,
    ms_caspt2,
)

# Reduced density matrices
from ._rdm import (
    energy_from_rdms,
    make_rdm1,
    make_rdm12,
    make_rdm12_sa,
    make_rdm123,
    make_rdm1234,
)

# Selected CI
from ._selected_ci import (
    SelectedCIOptions,
    SelectedCIPT2Options,
    SelectedCISolver,
    selected_casci,
    selected_ci_pt2,
    solve_selected_ci,
)

# Slater-Condon rules
from ._slater_condon import (
    build_hamiltonian_matrix,
    build_hamiltonian_matrix_unrestricted,
    diagonal_matrix_element,
    diagonal_matrix_element_unrestricted,
    double_excitation_matrix_element,
    hamiltonian_dot,
    hamiltonian_matrix_element,
    hamiltonian_matrix_element_unrestricted,
    pair_excitation_matrix_element,
    single_excitation_matrix_element,
)

# Transcorrelated
from ._transcorrelated import (
    TCDiagnostics,
    TranscorrelatedOptions,
    build_transcorrelated_hamiltonian,
)

# v2RDM
from ._v2rdm import (
    V2RDMOptions,
    V2RDMSolver,
    solve_v2rdm,
)

# Eigensolver framework
from .eigensolver import (
    DavidsonSolver,
    DenseSolver,
    EigenProblem,
    GeneralizedEigenProblem,
    HermitianDavidsonSolver,
    InteriorEigenProblem,
    LanczosSolver,
    # Note: SolverOptions and SolverResult from eigensolver are distinct from
    # _common.SolverOptions / _common.SolverResult (non-mean-field solvers).
    # Import the eigensolver variants directly:
    #   from vibeqc.solvers.eigensolver import SolverOptions, SolverResult
    SCFStep,
    SolverStrategy,
    available_solvers,
    get_solver,
    solve_eigenproblem,
    solve_scf_step,
)

__all__ = [
    # Common
    "Hamiltonian",
    "SolverOptions",
    "SolverResult",
    # Hamiltonian
    "build_hamiltonian_ao",
    "build_hamiltonian_mo",
    "canonical_orthogonalize",
    "get_hf_orbital_provider",
    "transform_hamiltonian",
    # Determinants
    "Det",
    "SpinDet",
    "determinant_string",
    "excitation_rank",
    "generate_cisd_determinants",
    "generate_closed_shell_determinants",
    "generate_determinants",
    "generate_doubles",
    "generate_singles",
    "is_connected",
    "reference_determinant",
    # Slater-Condon
    "build_hamiltonian_matrix",
    "build_hamiltonian_matrix_unrestricted",
    "diagonal_matrix_element",
    "diagonal_matrix_element_unrestricted",
    "double_excitation_matrix_element",
    "hamiltonian_dot",
    "hamiltonian_matrix_element",
    "hamiltonian_matrix_element_unrestricted",
    "pair_excitation_matrix_element",
    "single_excitation_matrix_element",
    # Selected CI
    "SelectedCIOptions",
    "SelectedCISolver",
    "selected_casci",
    "SelectedCIPT2Options",
    "selected_ci_pt2",
    "solve_selected_ci",
    # DMRG
    "DMRGOptions",
    "DMRGSolver",
    "solve_dmrg",
    # v2RDM
    "V2RDMOptions",
    "V2RDMSolver",
    "solve_v2rdm",
    # Transcorrelated
    "TranscorrelatedOptions",
    "TCDiagnostics",
    "build_transcorrelated_hamiltonian",
    # Multireference PT2
    "CASPT2Options",
    "MSCASPT2Result",
    "ms_caspt2",
    # CASCI / CASSCF
    "CASCIOptions",
    "CASCIResult",
    "casci",
    "CASSCFOptions",
    "CASSCFResult",
    "casscf",
    # CISD
    "CISDOptions",
    "CISDResult",
    "cisd",
    # CC3
    "CC3Iteration",
    "CC3Options",
    "CC3Result",
    "cc3",
    # CCSDT
    "CCSDTIteration",
    "CCSDTOptions",
    "CCSDTResult",
    "ccsdt",
    # Reduced density matrices
    "energy_from_rdms",
    "make_rdm1",
    "make_rdm12",
    "make_rdm12_sa",
    "make_rdm123",
    "make_rdm1234",
    # MRCI
    "MRCIResult",
    "mrci",
    # Eigensolver framework
    "DavidsonSolver",
    "DenseSolver",
    "EigenProblem",
    "GeneralizedEigenProblem",
    "HermitianDavidsonSolver",
    "InteriorEigenProblem",
    "LanczosSolver",
    "SCFStep",
    "SolverStrategy",
    "available_solvers",
    "get_solver",
    "solve_eigenproblem",
    "solve_scf_step",
]
