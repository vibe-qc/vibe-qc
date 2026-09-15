"""Post-SCF molecular properties: atomic charges, bond orders, dipole.

The standard "sanity-check" output you expect from every QC program,
built on top of a converged SCF result.

Public API
----------

.. autofunction:: mulliken_charges
.. autofunction:: loewdin_charges
.. autofunction:: mayer_bond_orders
.. autofunction:: dipole_moment
.. autofunction:: center_of_mass
.. autofunction:: natural_orbitals
.. autofunction:: idempotency_deviation

All functions accept the ``result`` object returned by
:func:`vibeqc.run_rhf`, :func:`vibeqc.run_uhf`, :func:`vibeqc.run_rks`,
or :func:`vibeqc.run_uks`; the matching :class:`Molecule` and
:class:`BasisSet` used to run the SCF; and for dipole moments an
optional origin.

Implementation notes
--------------------

Mulliken and Löwdin population analyses are AO-basis-dependent in well-
known ways -- Mulliken in particular is very sensitive to diffuse
functions. The output is useful for trend-watching across a series
(e.g. charge transfer along a reaction coordinate) but atomic charges
beyond the leading digit should never be taken too seriously. Mayer
bond orders are rotation-invariant and less basis-sensitive.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    GridOptions,
    Molecule,
    build_grid,
    compute_dipole,
    compute_overlap,
    evaluate_ao,
    sad_density,
)
from .spin_channels import is_open_shell_result, spin_densities

if TYPE_CHECKING:  # pragma: no cover -- import cycle avoidance
    from ._vibeqc_core import RHFResult  # noqa: F401


__all__ = [
    "DipoleMoment",
    "HirshfeldResult",
    "NaturalOrbitals",
    "mulliken_charges",
    "loewdin_charges",
    "hirshfeld_charges",
    "mayer_bond_orders",
    "dipole_moment",
    "center_of_mass",
    "natural_orbitals",
    "idempotency_deviation",
]


# Standard atomic masses (u), index 0 unused.  Dalton approximate values --
# good to ~1e-3 u -- plenty for center-of-mass computation. We ship a
# subset sufficient for H-Kr; for heavier elements callers pass an
# explicit origin.
_ATOMIC_MASSES: tuple[float, ...] = (
    0.0,
    1.008,   4.003,                                              # H  He
    6.94,    9.012,  10.81,  12.011, 14.007, 15.999, 18.998, 20.180,  # Li-Ne
    22.990, 24.305, 26.982, 28.085, 30.974, 32.06,  35.45,  39.948,   # Na-Ar
    39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845,   # K-Fe
    58.933, 58.693, 63.546, 65.38,  69.723, 72.630, 74.922, 78.971,   # Co-Se
    79.904, 83.798,                                                    # Br Kr
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_if_hermitian(P: np.ndarray, *, what: str = "density matrix") -> np.ndarray:
    """Return the real part of a complex-but-Hermitian density matrix.

    Periodic SCF results carry a *complex-typed* density that is Hermitian;
    Bloch phases can leave non-trivial imaginary off-diagonal entries even
    though all real one-electron observables remain real. The molecular-
    property code forms
    **real observables** -- ``tr(P.O)`` (dipole, Mayer bond order), per-atom
    ``(P.S)`` population sums, the density on a real grid (Hirshfeld) -- for
    which the imaginary part of a Hermitian ``P`` contracted with a real
    operator cancels exactly. Returning the real part keeps those
    observables real and stops a periodic run from emitting a
    ``ComplexWarning`` on every property (the silent ``float(...)`` /
    ``float64`` casts that warning came from also discarded the imaginary
    part -- this does it once, intentionally, after a Hermiticity check).

    A *non-negligible* Hermiticity residual means a genuinely non-Hermitian
    density (a bug), surfaced with an actionable warning rather than
    discarded silently. Real-typed input is returned unchanged (molecular
    RHF/RKS/UHF/UKS path -- zero behaviour change). A per-k ``(nk, n, n)``
    stack (a periodic ``List[np.ndarray]`` density) is checked block by
    block and returned with its shape unchanged.
    """
    P = np.asarray(P)
    if not np.iscomplexobj(P):
        return P
    if P.size:
        max_abs = float(np.abs(P).max())
        max_im = float(np.abs(P.imag).max())
        max_re = float(np.abs(P.real).max())
        # Adjoint over the AO axes only; ``.T`` would also reverse the k axis.
        herm_resid = float(np.abs(P - P.conj().swapaxes(-1, -2)).max())
        if herm_resid > 1e-8 * max(max_abs, 1.0):
            warnings.warn(
                f"{what} is non-Hermitian "
                f"(max|P-P^H|={herm_resid:.2e}, max|Im|={max_im:.2e}, "
                f"max|Re|={max_re:.2e}); molecular-property values are "
                "derived from its real part and may be unreliable. This "
                "indicates a density-matrix assembly bug rather than a "
                "well-converged periodic SCF.",
                RuntimeWarning,
                stacklevel=3,
            )
    return np.ascontiguousarray(P.real)


def _total_density(result) -> np.ndarray:
    """Return the total (closed-shell + open-shell) density matrix.

    RHF/RKS store the already-combined density on ``.density``; UHF/UKS
    expose ``density_alpha`` and ``density_beta`` separately. Complex
    periodic densities are reduced to their (Hermitian) real part via
    :func:`_real_if_hermitian` so downstream observables stay real.
    """
    alpha, beta = spin_densities(result)
    if alpha is not None:
        P = np.asarray(alpha) + np.asarray(beta)
    else:
        P = np.asarray(result.density)
    return _real_if_hermitian(P)


def _shell_nao(shell) -> int:
    """Number of AOs a single shell contributes.

    ``2l+1`` for a pure (spherical-harmonic) shell, ``(l+1)(l+2)/2`` for a
    Cartesian one. ``pure`` defaults to True for duck-typed shell objects
    that predate the attribute."""
    angular = int(shell.l)
    if bool(getattr(shell, "pure", True)):
        return 2 * angular + 1
    return (angular + 1) * (angular + 2) // 2


def _shell_to_atom(basis: BasisSet) -> np.ndarray:
    """1-D int array, length ``nbasis``, mapping each AO to the 0-based
    atom index it lives on. Computed from ``basis.shells()`` (which is
    public C++ API exposed for basis-set I/O).

    This is the canonical AO-to-atom map. :mod:`vibeqc.bands`,
    :mod:`vibeqc.bond_analysis`, :mod:`vibeqc.nbo`,
    :mod:`vibeqc.output.formats.population`,
    :mod:`vibeqc.periodic.chi.localization` and
    :mod:`vibeqc.periodic.chi.properties` delegate here rather than
    re-deriving it; a Cartesian basis makes an independently derived
    ``2l+1`` map silently short, which misaligns every per-atom sum
    built on it. Only ``basis.shells()`` is touched, so a duck-typed
    basis exposing just that method works.

    ``tests/test_properties.py::_ao_to_atom_helpers`` lists every
    delegate and asserts they agree; add new ones there."""
    per_ao: list[int] = []
    for shell in basis.shells():
        per_ao.extend([int(shell.atom_index)] * _shell_nao(shell))
    return np.asarray(per_ao, dtype=np.int64)


def _per_atom_sum(ao_values: np.ndarray, ao_to_atom: np.ndarray,
                  n_atoms: int) -> np.ndarray:
    """Sum an nbasis-length array into per-atom totals."""
    out = np.zeros(n_atoms, dtype=np.float64)
    for a, v in zip(ao_to_atom, ao_values):
        out[a] += v
    return out


def _symmetric_matrix_power(
    matrix: np.ndarray,
    power: float,
    *,
    what: str,
    min_eigenvalue: float = 1.0e-10,
) -> np.ndarray:
    """Return ``matrix**power`` for a real symmetric positive matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    min_eval = float(np.min(eigvals))
    if min_eval < min_eigenvalue:
        raise ValueError(
            f"{what}: overlap matrix is near-singular "
            f"(min eigenvalue {min_eval:.2e})"
        )
    return (eigvecs * (eigvals ** power).reshape(1, -1)) @ eigvecs.T


def center_of_mass(molecule: Molecule) -> np.ndarray:
    """Center of mass (bohr). Atomic masses from a built-in table up to
    Z = 36; callers with heavier elements should specify the origin to
    dipole_moment directly."""
    atoms = list(molecule.atoms)
    if not atoms:
        return np.zeros(3)
    total_mass = 0.0
    com = np.zeros(3)
    for atom in atoms:
        z = int(atom.Z)
        if z < len(_ATOMIC_MASSES):
            m = _ATOMIC_MASSES[z]
        else:
            # Crude fall-back: 2 u per nucleon ≈ A, and A ≈ 2 Z on average
            # for light elements -- good enough not to throw. Heavy-element
            # users should pass an explicit origin.
            m = 2.0 * z
        pos = np.array([atom.xyz[0], atom.xyz[1], atom.xyz[2]])
        com += m * pos
        total_mass += m
    if total_mass == 0.0:
        return np.zeros(3)
    return com / total_mass


# ---------------------------------------------------------------------------
# Mulliken population analysis
# ---------------------------------------------------------------------------

def _resolve_nuclear_charges(
    molecule: Molecule,
    nuclear_charges: Optional[Sequence[float]],
    *,
    what: str,
) -> np.ndarray:
    """Per-atom nuclear charges for the electrostatic properties below.

    ``None`` means bare ``Z`` -- correct for an all-electron SCF. On an ECP
    calculation the density integrates to the *valence* count only, so the
    caller must pass the effective charges ``Z - n_core`` the valence-only
    Hamiltonian uses (:func:`vibeqc.ecp_metadata.effective_nuclear_charges`,
    fed from the SCF options that ran; GitLab #642). The runner, the FD
    Hessian, the ASE calculator and the population writers do so; a direct
    caller on an ECP system must too, or every quantity below carries a
    spurious ``n_core`` per ECP atom.
    """
    if nuclear_charges is None:
        return np.array([float(atom.Z) for atom in molecule.atoms], dtype=np.float64)
    q = np.asarray(nuclear_charges, dtype=np.float64).reshape(-1)
    n_atoms = len(molecule.atoms)
    if q.shape != (n_atoms,):
        raise ValueError(
            f"{what}: nuclear_charges must have one entry per atom "
            f"({n_atoms}), got shape {q.shape}"
        )
    return q


def mulliken_charges(
    result,
    basis: BasisSet,
    molecule: Molecule,
    *,
    nuclear_charges: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Mulliken atomic partial charges q_A = Z_A - S_{mu in A} (P.S)_mumu.

    Returns a 1-D array of length ``n_atoms``. Charges sum to the total
    molecular charge (``molecule.charge``) to machine precision. The
    overall partition is AO-basis-dependent; Mulliken is sensitive to
    diffuse functions and should be used for trend analysis rather than
    for quantitative charge assignment.

    ``nuclear_charges`` -- per-atom ``Z_A`` to subtract the population
    from; ``None`` is bare ``Z``. On an ECP calculation pass the effective
    charges ``Z - n_core`` (see :func:`_resolve_nuclear_charges`, #642).
    """
    P = _total_density(result)
    S = np.asarray(compute_overlap(basis))
    PS_diag = np.einsum("ij,ji->i", P, S)   # diag(P . S)
    ao_to_atom = _shell_to_atom(basis)
    n_atoms = len(molecule.atoms)
    electron_pop = _per_atom_sum(PS_diag, ao_to_atom, n_atoms)
    Z = _resolve_nuclear_charges(molecule, nuclear_charges, what="mulliken_charges")
    return Z - electron_pop


# ---------------------------------------------------------------------------
# Löwdin population analysis
# ---------------------------------------------------------------------------

def loewdin_charges(
    result,
    basis: BasisSet,
    molecule: Molecule,
    *,
    nuclear_charges: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Löwdin (symmetric-orthogonalization) atomic partial charges.

    The spin-summed AO density is transformed to the global symmetric-
    orthogonalized basis before its diagonal is partitioned by atom:

        q_A = Z_A - sum_{mu in A} (S^{1/2} P S^{1/2})_mu,mu

    A block-diagonal, per-atom basis transformation cannot define this
    population: the per-atom trace of the correspondingly transformed ``P S``
    product is invariant and therefore reduces exactly to Mulliken charges.

    ``nuclear_charges`` as in :func:`mulliken_charges` (ECP: ``Z - n_core``,
    #642).
    """
    P = _total_density(result)
    S = np.asarray(compute_overlap(basis))
    S_half = _symmetric_matrix_power(
        S,
        0.5,
        what="loewdin_charges",
    )
    P_lowdin = S_half @ P @ S_half
    ao_to_atom = _shell_to_atom(basis)
    n_atoms = len(molecule.atoms)
    electron_pop = _per_atom_sum(np.diag(P_lowdin), ao_to_atom, n_atoms)
    Z = _resolve_nuclear_charges(molecule, nuclear_charges, what="loewdin_charges")
    return Z - electron_pop


# ---------------------------------------------------------------------------
# Hirshfeld population analysis
# ---------------------------------------------------------------------------

@dataclass
class HirshfeldResult:
    """Output of :func:`hirshfeld_charges`.

    Attributes
    ----------
    charges : np.ndarray
        ``(n_atoms,)`` array of Hirshfeld atomic partial charges in
        electrons. Sums to ``molecule.charge`` to grid precision.
        Sign convention matches :func:`mulliken_charges` /
        :func:`loewdin_charges` (positive = electron-deficient).
    electron_population : np.ndarray
        ``(n_atoms,)`` array of ∫ w_A(r) r(r) dV -- the Hirshfeld-
        partitioned electron count on each atom. ``Z_A - this`` is
        ``charges[A]``.
    promolecule_norm : float
        ∫ r_pro dV evaluated on the grid; should ≈ n_electrons.
        Diagnostic: when this deviates by > 1e-3 from the integer
        electron count, the integration grid is too coarse or the
        SAD promolecule didn't converge for some atom (very rare).
    molecule_norm : float
        ∫ r_mol dV evaluated on the same grid; should also
        ≈ n_electrons. Comparing the two norms tells you whether
        grid error is in the molecular density or the promolecule.
    n_grid_points : int
        Total Becke-Lebedev-Treutler grid point count used. Scales
        with ``GridOptions.n_radial x angular order x n_atoms``.
    """

    charges: np.ndarray
    electron_population: np.ndarray
    promolecule_norm: float
    molecule_norm: float
    n_grid_points: int

    def __repr__(self) -> str:
        return (
            f"HirshfeldResult(charges=<{len(self.charges)} atoms>, "
            f"Sq={self.charges.sum():+.6f}, "
            f"n_e(mol)={self.molecule_norm:.4f}, "
            f"n_e(pro)={self.promolecule_norm:.4f}, "
            f"n_grid={self.n_grid_points})"
        )


def _factor_density(P: np.ndarray,
                    rel_tol: float = 1.0e-14) -> tuple[np.ndarray, np.ndarray]:
    """Factor a symmetric density matrix into natural-orbital modes.

    Returns ``(W, s)`` such that for AO values ``chi`` of shape
    ``(n_points, n_bf)``::

        rho = ((chi @ W) ** 2) @ s

    reproduces ``einsum("gm,gm->g", chi @ P, chi)`` to round-off, where
    ``W = v_k sqrt(|n_k|)`` and ``s = sign(n_k)`` over the eigenpairs
    ``(n_k, v_k)`` of ``P``.

    Only modes with ``|n_k| <= rel_tol * max|n|`` are dropped -- the
    numerical-zero tail of an idempotent density, whose eigenvalues sit at
    the diagonalisation round-off floor (~1e-16 relative). A density with
    no such tail keeps every mode, so this never trades accuracy for speed:
    it exploits low rank when it is there and costs the same when it is
    not. ``sign`` is carried explicitly because correlated relaxed
    densities can carry slightly negative natural occupations.
    """
    occupations, vectors = np.linalg.eigh(np.asarray(P, dtype=np.float64))
    scale = float(np.abs(occupations).max()) if occupations.size else 0.0
    if scale == 0.0:                       # all-zero block (ghost atom)
        return (np.zeros((P.shape[0], 0), dtype=np.float64),
                np.zeros(0, dtype=np.float64))
    keep = np.abs(occupations) > rel_tol * scale
    occ = occupations[keep]
    return (vectors[:, keep] * np.sqrt(np.abs(occ)), np.sign(occ))


def hirshfeld_charges(
    result,
    basis: BasisSet,
    molecule: Molecule,
    *,
    grid_options: Optional[GridOptions] = None,
    rho_floor: float = 1.0e-30,
    max_block_elems: int = 50_000_000,
    nuclear_charges: Optional[Sequence[float]] = None,
) -> HirshfeldResult:
    """Classical Hirshfeld atomic partial charges from a converged SCF.

    Hirshfeld, F. L. *Theor. Chim. Acta* **44**, 129 (1977).

        q_A = Z_A - ∫ w_A(r) r(r) d^3r,
        w_A(r) = r_A^free(r) / S_B r_B^free(r).

    Unlike :func:`mulliken_charges` and :func:`loewdin_charges`,
    Hirshfeld is a *real-space* partition rather than a basis-space
    partition -- far less sensitive to diffuse functions, and the
    standard input charge for charge-dependent dispersion methods
    like the D4 refinement scheduled for vibe-qc v0.10.0 D2b.

    The promolecular reference {r_A^free} is obtained for free from
    :func:`vibeqc.sad_density` -- it returns the SAD initial-guess
    density matrix, which is block-diagonal by atom (each atomic
    SCF runs in vacuum, contributes to its own AO range, off-
    diagonal blocks are zero). Restricting the AO sum to atom A's
    basis-function range gives r_A^free evaluated at the molecular
    geometry. No per-atom SCF, no ionic-fragment branching.

    Parameters
    ----------
    result
        SCF result from ``vibeqc.run_rhf`` / ``run_rks`` /
        ``run_uhf`` / ``run_uks``. The total density matrix is
        extracted via :func:`_total_density` (handles both the
        closed-shell ``.density`` and the open-shell
        ``.density_alpha`` + ``.density_beta`` schemas).
    basis
        The same :class:`BasisSet` used to run the SCF.
    molecule
        The same :class:`Molecule` used to run the SCF.
    grid_options
        Optional :class:`GridOptions` for the Becke-Lebedev-
        Treutler integration grid. Defaults to ``GridOptions()``
        (vibe-qc's DFT-default level). Hirshfeld weights are
        smooth so the default grid is plenty for sub-millielectron
        charge accuracy; tighten only for explicit
        grid-convergence studies.
    rho_floor
        Promolecule density floor to keep w_A = r_A / r_pro
        well-defined in vacuum regions far from any atom. Default
        ``1e-30`` is well below any grid point that contributes
        meaningfully to the integral.
    max_block_elems
        Memory cap for the grid sweep. The AO matrix chi_mu(r_g) is
        ``(n_block, n_bf)``; the grid is processed in blocks sized
        so ``n_block x n_bf`` never exceeds this many elements
        (~``8 x max_block_elems`` bytes of float64). Default
        ``5x10⁷`` ≈ 400 MB per block. Small molecules fit in a
        single block (no behaviour change); large systems are
        swept block-by-block so the function never materialises a
        multi-gigabyte AO matrix. The result is independent of
        the block size to floating-point round-off.

    Returns
    -------
    HirshfeldResult
        Rich dataclass; ``HirshfeldResult.charges`` is the
        ``(n_atoms,)`` array if you want bare-array semantics
        matching :func:`mulliken_charges`. The other fields carry
        per-atom integrated electron populations and grid-
        normalisation sanity numbers.

    Notes
    -----
    Numerical-quality knobs:

    * ``∫ r_mol - n_electrons`` should be < 5x10⁻⁵ on the default
      grid for typical first-row systems.
    * ``S q_A - molecule.charge`` should be < 1x10⁻⁶ e (exact
      identity from the Hirshfeld weight normalisation; only grid
      error introduces residual).

    Classical Hirshfeld charges in vibe-qc come out ~0.04-0.06 e
    *more negative* on heavy atoms than ORCA-reported values
    because the promolecule is constructed in the molecular basis
    (SAD-derived) rather than from tabulated Slater-type atomic
    densities. Documentable trade-off -- switch sources via a
    future ``promolecule="slater"`` kwarg if byte-equal ORCA
    parity matters.

    The iterative Hirshfeld variant (Bultinck et al., *J. Chem.
    Phys.* **126**, 144111 (2007)) -- which is less promolecule-
    sensitive -- is a clean follow-on; same API, different inner
    loop. Tracked as a v0.10.0 D2b-i candidate.
    """
    grid = build_grid(molecule, grid_options if grid_options is not None
                      else GridOptions())
    points = np.asarray(grid.points, dtype=np.float64)
    weights = np.asarray(grid.weights, dtype=np.float64)
    n_grid = points.shape[0]

    n_bf = basis.nbasis
    n_atoms = len(molecule.atoms)
    P_mol = _total_density(result)
    # Promolecular density matrix is the SAD guess (block-diagonal
    # by atom -- see docstring above).
    P_pro = np.asarray(sad_density(molecule, basis), dtype=np.float64)
    ao_to_atom = _shell_to_atom(basis)
    # Precompute per-atom AO masks + the diagonal P_pro blocks once.
    atom_masks = [(ao_to_atom == A) for A in range(n_atoms)]
    atom_P_blocks = [
        (P_pro[np.ix_(m, m)] if m.any() else None) for m in atom_masks
    ]

    # Rank-reduce every density matrix ONCE, before the grid sweep.
    #
    # A density matrix is symmetric, so P = sum_k n_k v_k v_k^T with
    # (n_k, v_k) its eigenpairs, and the grid density collapses from a
    # double AO sum to a single sum over natural orbitals:
    #
    #     r(r_g) = sum_muν chi_mu P_muν chi_ν
    #            = sum_k n_k ( sum_mu chi_mu(r_g) v_k,mu )^2 .
    #
    # Folding sqrt(|n_k|) into the vectors (``W``) and carrying the sign
    # separately (``s``) turns the per-block (n_block, n_bf) x (n_bf, n_bf)
    # product into (n_block, n_bf) x (n_bf, rank). A converged SCF density
    # has rank = n_occ, so the dominant matmul shrinks by n_bf / n_occ --
    # 6x for C20H42/def2-SVP, 10.8x at def2-TZVP, and the saving grows with
    # basis size because n_occ does not. This is an exact rewrite of the
    # same contraction, not an approximation: only eigenvalues at the
    # round-off floor are dropped, so a full-rank density simply keeps
    # every mode and costs what it did before.
    W_mol, sign_mol = _factor_density(P_mol)
    atom_factors = [
        (_factor_density(P_AA) if P_AA is not None else None)
        for P_AA in atom_P_blocks
    ]

    # Process the grid in blocks so the (n_block, n_bf) AO matrix
    # never exceeds ``max_block_elems`` float64 entries. Small
    # molecules collapse to a single block (identical to the old
    # one-shot path); large systems are swept without ever holding
    # a multi-GB chi matrix in memory.
    block = max(1, int(max_block_elems // max(n_bf, 1)))

    electron_pop = np.zeros(n_atoms, dtype=np.float64)
    promolecule_norm = 0.0
    molecule_norm = 0.0

    for start in range(0, n_grid, block):
        stop = min(start + block, n_grid)
        pts = points[start:stop]
        w = weights[start:stop]

        # chi_mu(r_g) for this block: (n_block, n_bf).
        chi = np.asarray(evaluate_ao(basis, pts), dtype=np.float64)

        # r_mol(r_g) = sum_k n_k (chi.v_k)^2 over the natural orbitals.
        t_mol = chi @ W_mol
        rho_mol = (t_mol * t_mol) @ sign_mol

        # r_A^free(r_g) per atom -- AO sum restricted to A's range
        # thanks to P_pro's block-diagonal structure, then the same
        # rank reduction within that range.
        rho_atoms = np.zeros((n_atoms, stop - start), dtype=np.float64)
        for A in range(n_atoms):
            factored = atom_factors[A]
            if factored is None:           # ghost atom -- row stays 0
                continue
            W_A, sign_A = factored
            t_A = chi[:, atom_masks[A]] @ W_A
            rho_atoms[A] = (t_A * t_A) @ sign_A

        rho_pro = rho_atoms.sum(axis=0)

        # Hirshfeld weights with a r_pro floor for far-out points
        # that would otherwise be ~0/~0.
        safe_pro = np.maximum(rho_pro, rho_floor)
        weights_atoms = rho_atoms / safe_pro

        electron_pop += weights_atoms @ (w * rho_mol)
        promolecule_norm += float((w * rho_pro).sum())
        molecule_norm += float((w * rho_mol).sum())

    # ECP systems: the promolecule weights are the all-electron free-atom
    # densities (shape only), but the charge must be referenced to the
    # valence-only nuclear charge the density integrates against (#642).
    Z = _resolve_nuclear_charges(molecule, nuclear_charges, what="hirshfeld_charges")
    charges = Z - electron_pop

    return HirshfeldResult(
        charges=charges,
        electron_population=electron_pop,
        promolecule_norm=promolecule_norm,
        molecule_norm=molecule_norm,
        n_grid_points=n_grid,
    )


# ---------------------------------------------------------------------------
# Mayer bond orders
# ---------------------------------------------------------------------------

def mayer_bond_orders(result, basis: BasisSet,
                      molecule: Molecule) -> np.ndarray:
    """Mayer bond-order matrix, shape (n_atoms, n_atoms).

    In terms of the spin-summed density ``P`` and spin density ``R``:

        B_AB = sum_{mu in A, nu in B} [
            (P S)_mu,nu (P S)_nu,mu + (R S)_mu,nu (R S)_nu,mu
        ]

    The spin term is zero for RHF/RKS. For UHF/UKS the same definition is
    evaluated as ``2 * [(P_alpha S)^2 + (P_beta S)^2]``. This is distinct
    from a Wiberg index, which squares the density transformed to a Löwdin
    orthonormal basis.

    Off-diagonal entries are atom-pair bond orders; diagonal entries are zero.
    """
    S = np.asarray(compute_overlap(basis))
    ao_to_atom = _shell_to_atom(basis)
    n_atoms = len(molecule.atoms)

    alpha, beta = spin_densities(result)
    if alpha is not None:
        # Real part for complex (periodic, Hermitian) densities -- the Mayer
        # bond order is a real observable; see _real_if_hermitian.
        Pa = _real_if_hermitian(alpha, what="alpha density")
        Pb = _real_if_hermitian(beta, what="beta density")
        PS_a = Pa @ S
        PS_b = Pb @ S
        # Element-wise Mayer: M_muν = 2.[(PS_a)_muν (PS_a)_νmu
        #                              + (PS_b)_muν (PS_b)_νmu]
        M = 2.0 * (PS_a * PS_a.T + PS_b * PS_b.T)
    else:
        P = _real_if_hermitian(result.density)
        PS = P @ S
        M = PS * PS.T   # broadcasting element-wise; equivalent to
                        # M_muν = (PS)_muν . (PS)_νmu since the matrix is
                        # real.

    bond_orders = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for mu in range(M.shape[0]):
        a = ao_to_atom[mu]
        for nu in range(M.shape[1]):
            b = ao_to_atom[nu]
            if a != b:
                bond_orders[a, b] += M[mu, nu]
    # Symmetrize numerically -- analytical B_AB = B_BA for real AOs.
    bond_orders = 0.5 * (bond_orders + bond_orders.T)
    return bond_orders


def prominent_bonds(
    bond_orders: np.ndarray,
    molecule: Molecule,
    *,
    threshold: float = 0.10,
) -> list[tuple[int, int, float]]:
    """Return ``[(i, j, B_ij)]`` pairs with ``i < j`` and
    ``B_ij >= threshold`` -- convenience for formatting a compact
    bond-order table in the log output."""
    n = bond_orders.shape[0]
    out: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if bond_orders[i, j] >= threshold:
                out.append((i, j, float(bond_orders[i, j])))
    # Sort by descending bond order so the covalent bonds float to the top.
    # Symmetry-equivalent bonds (e.g. the two O-H bonds of C2v water) are
    # degenerate up to floating-point noise, so a raw float sort leaks that
    # noise into the row order. Quantize well below the displayed 4-decimal
    # precision and break ties by atom indices for a machine-stable order.
    out.sort(key=lambda t: (-round(t[2], 6), t[0], t[1]))
    return out


# ---------------------------------------------------------------------------
# Dipole moment
# ---------------------------------------------------------------------------

_BOHR_TO_DEBYE = 2.541746473     # 1 e.bohr = 2.541746473 Debye


@dataclass
class DipoleMoment:
    """Dipole moment components in atomic units (e.bohr), plus Debye."""
    x: float
    y: float
    z: float
    origin: tuple[float, float, float]

    @property
    def total(self) -> float:
        r"""\|mu\| in atomic units (e\*bohr)."""
        return float(np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2))

    @property
    def total_debye(self) -> float:
        return self.total * _BOHR_TO_DEBYE

    def components_debye(self) -> tuple[float, float, float]:
        return (self.x * _BOHR_TO_DEBYE,
                self.y * _BOHR_TO_DEBYE,
                self.z * _BOHR_TO_DEBYE)


def dipole_moment(
    result,
    basis: BasisSet,
    molecule: Molecule,
    *,
    origin: Optional[Sequence[float]] = None,
    nuclear_charges: Optional[Sequence[float]] = None,
) -> DipoleMoment:
    """Electric dipole moment of a converged SCF calculation.

    ``origin`` (bohr) defaults to the molecular center of mass, which
    makes the dipole origin-independent for neutral systems (the
    convention every standard QC code uses). For charged systems the
    dipole depends on origin; pass an explicit vector if you need a
    particular reference.

    ``nuclear_charges`` -- per-atom charges for the nuclear term
    ``sum_A Z_A (R_A - O)``; ``None`` is bare ``Z``. On an ECP calculation
    the density integrates to the valence count only, so pass the
    effective charges ``Z - n_core``
    (:func:`vibeqc.ecp_metadata.effective_nuclear_charges` on the SCF
    options that ran; Dolg & Cao 2011, Sec. 5, Eqs. 54-55). With bare
    ``Z`` a neutral ECP molecule's dipole is origin-dependent by exactly
    ``-n_core`` au per bohr of origin shift (GitLab #642).
    """
    if origin is None:
        origin_vec = center_of_mass(molecule)
    else:
        origin_vec = np.asarray(origin, dtype=np.float64)
        if origin_vec.shape != (3,):
            raise ValueError("dipole_moment: origin must be a 3-vector (bohr)")

    dip = compute_dipole(basis, [float(x) for x in origin_vec])
    Mx = np.asarray(dip.x)
    My = np.asarray(dip.y)
    Mz = np.asarray(dip.z)
    P = _total_density(result)

    # Electronic contribution (electrons are negative): -tr(P . M_c).
    mu_e_x = -np.einsum("ij,ji->", P, Mx)
    mu_e_y = -np.einsum("ij,ji->", P, My)
    mu_e_z = -np.einsum("ij,ji->", P, Mz)

    # Nuclear contribution (with origin shift): S_A Z_A (R_A - O), with
    # Z_A the charge the valence-only Hamiltonian sees (Z - n_core on ECP
    # atoms; #642).
    Z = _resolve_nuclear_charges(molecule, nuclear_charges, what="dipole_moment")
    mu_n_x = 0.0
    mu_n_y = 0.0
    mu_n_z = 0.0
    for z, atom in zip(Z, molecule.atoms):
        mu_n_x += z * (atom.xyz[0] - origin_vec[0])
        mu_n_y += z * (atom.xyz[1] - origin_vec[1])
        mu_n_z += z * (atom.xyz[2] - origin_vec[2])

    return DipoleMoment(
        x=float(mu_e_x + mu_n_x),
        y=float(mu_e_y + mu_n_y),
        z=float(mu_e_z + mu_n_z),
        origin=(float(origin_vec[0]),
                float(origin_vec[1]),
                float(origin_vec[2])),
    )


# ---------------------------------------------------------------------------
# Natural orbitals
# ---------------------------------------------------------------------------

@dataclass
class NaturalOrbitals:
    """Natural orbitals + occupations, sorted by descending occupation.

    Attributes
    ----------
    occupations
        ``(n_bf,)`` real array. For ``kind="rhf"`` and
        ``kind="uhf-total"`` occupations are in ``[0, 2]``; for
        ``kind="uhf-alpha"`` / ``kind="uhf-beta"`` they are in
        ``[0, 1]``; for ``kind="uhf-spin"`` they are in ``[-1, 1]``.
    coefficients
        ``(n_bf, n_bf)`` real matrix. Each column is one NO expressed in
        the AO basis, S-normalized: ``C^T S C = I``. Columns are
        ordered by descending occupation so ``coefficients[:, :n_occ]``
        is the natural-occupation analogue of "occupied MOs".
    kind
        One of ``"rhf"``, ``"uhf-total"``, ``"uhf-alpha"``,
        ``"uhf-beta"``, ``"uhf-spin"`` -- describes which density matrix
        was diagonalized so the user knows how to interpret occupations.
    """

    occupations: np.ndarray
    coefficients: np.ndarray
    kind: str

    @property
    def n_orbitals(self) -> int:
        return int(self.occupations.size)

    @property
    def n_electrons(self) -> float:
        """Sum of occupations -- equals the underlying electron count
        (or ``N_a - N_b`` for ``kind="uhf-spin"``) up to FP noise."""
        return float(self.occupations.sum())


def _diagonalise_density(D: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``D S c_i = n_i c_i`` via the Löwdin route. Returns
    ``(occupations, coefficients)`` sorted by *descending* occupation
    with the AO-basis NOs S-normalized (``C^T S C = I``).

    Numerically: form ``D̃ = S^{1/2} D S^{1/2}`` (real symmetric),
    diagonalize, then transform eigenvectors back via
    ``C = S^{-1/2} U``. This avoids the non-symmetric eigenproblem
    ``D . S`` (which has real eigenvalues but complex-arithmetic
    eigenvectors)."""
    s_eig, U_S = np.linalg.eigh(S)
    if np.min(s_eig) < 1e-10:
        raise ValueError(
            f"natural_orbitals: overlap is near-singular "
            f"(min eigenvalue {np.min(s_eig):.2e})"
        )
    sqrt_s = np.sqrt(s_eig)
    inv_sqrt_s = 1.0 / sqrt_s
    S_half = U_S @ np.diag(sqrt_s) @ U_S.T
    S_inv_half = U_S @ np.diag(inv_sqrt_s) @ U_S.T
    D_tilde = S_half @ D @ S_half
    # Force-symmetrize -- D̃ is exactly symmetric in exact arithmetic.
    D_tilde = 0.5 * (D_tilde + D_tilde.T)
    n, U = np.linalg.eigh(D_tilde)
    # eigh returns ascending; flip to descending occupation.
    order = np.argsort(-n)
    n = n[order]
    U = U[:, order]
    C = S_inv_half @ U
    return n, C


def natural_orbitals(result, basis: BasisSet, *,
                     kind: str = "auto") -> NaturalOrbitals:
    """Diagonalize the SCF one-particle density matrix to get natural
    orbitals + occupations.

    For a single-determinant RHF/RKS the occupations come out exactly
    integer (2.0 for occupied, 0.0 for virtual) and the NOs span the
    same occupied/virtual subspaces as the canonical MOs (different
    rotations within those subspaces only). For UHF/UKS the *total*
    natural orbitals (default) carry fractional occupations whose
    deviation from {0, 2} measures spin contamination /
    multireference character; the *spin* natural orbitals
    (``kind="uhf-spin"``) carry the unpaired-electron distribution
    whose largest-eigenvalue magnitudes localise the open-shell
    character.

    Parameters
    ----------
    result
        Output of :func:`vibeqc.run_rhf`, :func:`vibeqc.run_rks`,
        :func:`vibeqc.run_uhf`, or :func:`vibeqc.run_uks`.
    basis
        The same :class:`BasisSet` used to run the SCF (the AO overlap
        is recomputed from it).
    kind
        ``"auto"`` (default) chooses ``"rhf"`` for closed-shell results
        and ``"uhf-total"`` for open-shell. Other accepted values:

        * ``"rhf"`` -- for an RHF/RKS result, diagonalize ``D`` directly
          (occupations 0..2).
        * ``"uhf-total"`` -- for a UHF/UKS result, diagonalize
          ``D_a + D_b`` (occupations 0..2, sum = N_e).
        * ``"uhf-alpha"`` / ``"uhf-beta"`` -- diagonalize ``D_a`` or
          ``D_b`` alone (occupations 0..1, sum = N_a or N_b).
        * ``"uhf-spin"`` -- diagonalize the spin density ``D_a - D_b``
          (occupations -1..1, sum = N_a - N_b). Eigenvectors with the
          largest |occupation| pick out where the unpaired spins live.
    """
    is_open_shell = is_open_shell_result(result)
    if kind == "auto":
        kind = "uhf-total" if is_open_shell else "rhf"

    if kind == "rhf":
        if is_open_shell:
            raise ValueError(
                "natural_orbitals: kind='rhf' requires a closed-shell "
                "(RHF/RKS) result; got an open-shell result. Use "
                "kind='uhf-total' instead."
            )
        D = np.asarray(result.density)
    elif kind == "uhf-total":
        if not is_open_shell:
            raise ValueError(
                "natural_orbitals: kind='uhf-total' requires an "
                "open-shell (UHF/UKS) result."
            )
        D = (np.asarray(result.density_alpha)
             + np.asarray(result.density_beta))
    elif kind == "uhf-alpha":
        if not is_open_shell:
            raise ValueError(
                "natural_orbitals: kind='uhf-alpha' requires an "
                "open-shell result.")
        D = np.asarray(result.density_alpha)
    elif kind == "uhf-beta":
        if not is_open_shell:
            raise ValueError(
                "natural_orbitals: kind='uhf-beta' requires an "
                "open-shell result.")
        D = np.asarray(result.density_beta)
    elif kind == "uhf-spin":
        if not is_open_shell:
            raise ValueError(
                "natural_orbitals: kind='uhf-spin' requires an "
                "open-shell result.")
        D = (np.asarray(result.density_alpha)
             - np.asarray(result.density_beta))
    else:
        raise ValueError(
            f"natural_orbitals: unknown kind={kind!r}. Valid: 'auto', "
            f"'rhf', 'uhf-total', 'uhf-alpha', 'uhf-beta', 'uhf-spin'."
        )

    S = np.asarray(compute_overlap(basis))
    occupations, coefficients = _diagonalise_density(D, S)
    return NaturalOrbitals(
        occupations=occupations,
        coefficients=coefficients,
        kind=kind,
    )


def idempotency_deviation(no: NaturalOrbitals) -> float:
    """Scalar measure of how far the density matrix is from a single
    Slater determinant. Larger values flag multireference character or
    spin contamination.

    For ``kind="rhf"`` / ``kind="uhf-total"`` (occupations in [0, 2]):

        Δ  =  S_i  n_i (2 - n_i)  /  2

    A pure single-determinant Hartree-Fock state gives ``Δ = 0`` (every
    NO is exactly 0 or 2). For UHF the value is the standard
    "non-idempotency" diagnostic -- small for well-behaved closed-shell
    systems and growing with spin contamination.

    For ``kind="uhf-alpha"``, ``"uhf-beta"`` (occupations in [0, 1]):

        Δ  =  S_i  n_i (1 - n_i)

    For ``kind="uhf-spin"`` (occupations in [-1, 1]):

        Δ  =  S_i  (1 - n_i^2) / 2     (Yamaguchi-style estimate of the
                                       number of unpaired electrons; the
                                       proper "N_unpaired" via the
                                       Head-Gordon definition uses
                                       2.(D_a D_b S) eigenvalues, which
                                       this function does not compute.)
    """
    n = np.asarray(no.occupations, dtype=float)
    if no.kind in ("rhf", "uhf-total"):
        return 0.5 * float(np.sum(n * (2.0 - n)))
    if no.kind in ("uhf-alpha", "uhf-beta"):
        return float(np.sum(n * (1.0 - n)))
    if no.kind == "uhf-spin":
        return 0.5 * float(np.sum(1.0 - n * n))
    raise ValueError(f"idempotency_deviation: unknown NO kind {no.kind!r}")
