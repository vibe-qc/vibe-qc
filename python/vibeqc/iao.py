"""Intrinsic atomic orbitals (IAOs) and IAO partial charges.

IAOs are a minimal set of atom-centred orbitals, polarised by the molecular
environment, that exactly span the occupied space of an SCF wave function.
They give a basis-set-stable definition of "the electrons on atom A", which
Mulliken populations famously do not, and they are the population basis the
intrinsic bond orbitals (:func:`vibeqc.localise.ibo_localise`) are built on.

Reference
---------
Knizia, *J. Chem. Theory Comput.* **9**, 4834 (2013),
doi:10.1021/ct400687b -- "Intrinsic Atomic Orbitals: An Unbiased Bridge
between Quantum Theory and Chemical Concepts".

The construction is Appendix C of that paper; the partial charges are its
eq 3.  Both are transcribed at the call sites below.

The reference basis
-------------------
The IAO construction needs a minimal free-atom basis B2.  vibe-qc uses
**Huzinaga MINI** (``"mini"``), which is one of the variants Knizia himself
validates (Table 1 footnote c).  It is *not* the ``"ano-rcc-mb"`` basis that
the MINAO initial guess uses -- that one is a fine SCF starting guess but a
poor IAO reference, and it shifts CH4 partial charges by ~20 %:

    B2 = "mini"         q(C) = -0.473   <- matches Knizia Table 1 (-0.49)
    B2 = "ano-rcc-mb"   q(C) = -0.389
    B2 = "sto-3g"       q(C) = -0.132

(measured on the same CH4/def2-SVP RHF wave function).  See
``handovers/HANDOVER_IBO.md`` § 4a.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    Molecule,
    compute_dipole,
    compute_overlap,
    compute_overlap_two_basis,
)
from .properties import _shell_to_atom

__all__ = [
    "IAOReference",
    "IBOAnalysis",
    "IAO_REFERENCE_BASIS",
    "IAO_REFERENCE_MAX_Z",
    "LOCALIZATION_METHODS",
    "analyse_ibo",
    "analyse_localization",
    "build_iaos",
    "iao_atom_populations",
    "iao_charges",
    "iao_populations",
    "iao_reference",
    "iao_unsupported_reason",
]

#: Minimal free-atom basis used as B2 in the IAO construction.
IAO_REFERENCE_BASIS = "mini"

#: Highest atomic number ``IAO_REFERENCE_BASIS`` covers.  Verified against the
#: bundled ``mini.g94``: Z=86 (Rn) builds, Z=87 (Fr) raises "no shells loaded".
IAO_REFERENCE_MAX_Z = 86

# Relative eigenvalue floor for the metric inversions below.  Knizia's
# Appendix C warns that explicit inverse overlap matrices are numerically
# fragile for large or diffuse bases and prescribes a Cholesky or spectral
# decomposition; we use a thresholded spectral (pseudo-)inverse, matching
# `_hermitian_power` in the periodic AICCM2026DEV-B localizer.
_METRIC_THRESHOLD = 1e-10


@dataclass(frozen=True)
class IAOReference:
    """The minimal-basis side of an IAO construction.

    Attributes
    ----------
    basis
        The minimal free-atom ``BasisSet`` (B2).
    overlap
        ``S2``, the overlap within B2.
    cross_overlap
        ``S12 = <mu in B1 | sigma in B2>``.
    atom_indices
        AO index -> 0-based atom index, in B2's own AO order.
    name
        The basis-set name actually used.
    """

    basis: BasisSet
    overlap: np.ndarray
    cross_overlap: np.ndarray
    atom_indices: np.ndarray
    name: str


def _hermitian_power(matrix: np.ndarray, power: float) -> np.ndarray:
    """``matrix ** power`` for a Hermitian matrix, dropping null directions.

    Symmetrised before the eigendecomposition so that accumulated asymmetry
    in an assembled metric cannot produce complex eigenvalues.
    """
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    largest = float(values[-1])
    if largest <= 0.0:
        raise np.linalg.LinAlgError(
            "IAO construction hit a metric with no positive eigenvalue"
        )
    keep = values > _METRIC_THRESHOLD * largest
    if not np.any(keep):
        raise np.linalg.LinAlgError(
            "IAO construction hit a numerically singular metric"
        )
    return (vectors[:, keep] * values[keep] ** power) @ vectors[:, keep].conj().T


def _orth(coefficients: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Knizia Appendix C:  ``orth(C) = C [C^T S1 C]^(-1/2)``."""
    metric = coefficients.conj().T @ overlap @ coefficients
    return coefficients @ _hermitian_power(metric, -0.5)


def iao_unsupported_reason(molecule: Molecule, uses_ecp: bool = False) -> str | None:
    """Return why IAOs cannot be built for ``molecule``, or ``None`` if they can.

    Two hard blockers, both verified rather than assumed:

    * **ECPs.** The reference basis is all-electron, so pairing it with a
      valence-only target basis is meaningless -- for Au, ``lanl2dz`` carries
      22 functions for 19 explicit electrons while the all-electron minimal
      reference carries 43 for all 79.  There is no consistent partition.
    * **Element coverage.** ``IAO_REFERENCE_BASIS`` stops at
      ``IAO_REFERENCE_MAX_Z``.
    """
    if uses_ecp:
        return (
            "IAO analysis needs an all-electron reference basis, but this job "
            "uses ECPs; the valence-only target basis and the all-electron "
            f"'{IAO_REFERENCE_BASIS}' reference do not describe the same "
            "electrons"
        )
    charges = [int(atom.Z) for atom in molecule.atoms]
    beyond = sorted({z for z in charges if z > IAO_REFERENCE_MAX_Z})
    if beyond:
        return (
            f"IAO reference basis '{IAO_REFERENCE_BASIS}' covers Z<="
            f"{IAO_REFERENCE_MAX_Z}; this system contains Z={beyond}"
        )
    return None


def iao_reference(
    molecule: Molecule,
    target_basis: BasisSet,
    *,
    name: str = IAO_REFERENCE_BASIS,
) -> IAOReference:
    """Build the minimal-basis side (B2, S2, S12, atom map) of an IAO job.

    ``S12`` comes from :func:`compute_overlap_two_basis`, the molecular
    cross-basis overlap.
    """
    reference_basis = BasisSet(molecule, name)
    for shell in reference_basis.shells():
        if not bool(getattr(shell, "pure", True)):
            raise ValueError(
                "IAO construction requires spherical (pure) reference shells; "
                f"'{name}' produced a Cartesian shell"
            )
    return IAOReference(
        basis=reference_basis,
        overlap=np.asarray(compute_overlap(reference_basis)),
        cross_overlap=np.asarray(
            compute_overlap_two_basis(target_basis, reference_basis)
        ),
        atom_indices=_shell_to_atom(reference_basis),
        name=name,
    )


def build_iaos(
    occupied: np.ndarray,
    overlap: np.ndarray,
    reference_overlap: np.ndarray,
    cross_overlap: np.ndarray,
) -> np.ndarray:
    """Construct IAO coefficients in the main (B1) basis.

    Knizia Appendix C, matrix form.  With ``C`` the occupied MO coefficients,
    ``S1``/``S2`` the overlaps within B1/B2 and ``S12`` the cross-basis
    overlap:

        P12 = S1^-1 S12
        C~  = orth(S1^-1 S12 S2^-1 S21 C)                        (eq 1)
        A   = C C^T S1 C~ C~^T S1 P12
              + (1 - C C^T S1)(1 - C~ C~^T S1) P12               (eq 2)

    and ``A`` is then multiplied from the right by ``[A^T S1 A]^(-1/2)`` to
    give a symmetrically orthonormalised set.

    Parameters
    ----------
    occupied
        ``C``, shape ``(n_ao, n_occ)`` -- occupied MO coefficients in B1.
    overlap
        ``S1``, shape ``(n_ao, n_ao)``.
    reference_overlap
        ``S2``, shape ``(n_ref, n_ref)``.
    cross_overlap
        ``S12``, shape ``(n_ao, n_ref)``.

    Returns
    -------
    ndarray
        IAO coefficients in B1, shape ``(n_ao, n_ref)``, orthonormal in ``S1``.
    """
    p12 = _hermitian_power(overlap, -1.0) @ cross_overlap
    depolarized = _orth(
        p12
        @ _hermitian_power(reference_overlap, -1.0)
        @ cross_overlap.conj().T
        @ occupied,
        overlap,
    )

    projector_occ = occupied @ occupied.conj().T @ overlap          # O
    projector_dep = depolarized @ depolarized.conj().T @ overlap    # O~
    identity = np.eye(overlap.shape[0])

    raw = (
        projector_occ @ projector_dep
        + (identity - projector_occ) @ (identity - projector_dep)
    ) @ p12
    return raw @ _hermitian_power(raw.conj().T @ overlap @ raw, -0.5)


def iao_populations(
    occupied: np.ndarray,
    iaos: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    """Per-IAO electron populations of a closed-shell occupied set.

    ``2 * sum_i |<rho|i>|^2`` for each IAO ``rho``; the inner product is
    ``<rho|i> = (A^T S1 C)_{rho i}``.
    """
    amplitudes = iaos.conj().T @ overlap @ occupied
    return 2.0 * np.einsum("ri,ri->r", amplitudes, amplitudes.conj()).real


def iao_charges(
    occupied: np.ndarray,
    iaos: np.ndarray,
    overlap: np.ndarray,
    nuclear_charges: np.ndarray,
    atom_indices: np.ndarray,
) -> np.ndarray:
    """IAO partial charges.

    Knizia eq 3:  ``q_A = Z_A - sum_{rho in A} <rho|gamma|rho>`` with the
    closed-shell density ``gamma = 2 sum_i |i><i|``.

    Published targets (Knizia Table 1, Hartree-Fock): CH4/def2-SVP gives
    C -0.49, H +0.12; CH4/def2-TZVPP gives C -0.52, H +0.13.  With the
    Huzinaga-MINI reference this implementation gives -0.473/+0.118 and
    -0.500/+0.125 -- inside the 0.02-0.03 e spread Knizia reports between
    reference-basis choices (his footnote c).
    """
    populations = iao_populations(occupied, iaos, overlap)
    charges = np.asarray(nuclear_charges, dtype=np.float64).copy()
    np.subtract.at(charges, np.asarray(atom_indices, dtype=np.int64), populations)
    return charges


def iao_atom_populations(
    orbitals_in_iao_basis: np.ndarray,
    atom_indices: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Per-orbital, per-atom IAO populations ``n_A(i)``.

    ``orbitals_in_iao_basis`` is the occupied set expressed in the orthonormal
    IAO basis (``A^T S1 C``), shape ``(n_ref, n_occ)``.  Returns shape
    ``(n_occ, n_atoms)``; each row sums to 1 because the IAOs span the
    occupied space exactly.
    """
    coefficients = np.asarray(orbitals_in_iao_basis)
    out = np.zeros((coefficients.shape[1], n_atoms), dtype=np.float64)
    for atom in range(n_atoms):
        rows = atom_indices == atom
        if not np.any(rows):
            continue
        block = coefficients[rows]
        out[:, atom] = np.einsum("ri,ri->i", block, block.conj()).real
    return out


def _dipole_tensor(basis: BasisSet) -> np.ndarray:
    """Cartesian dipole integrals stacked as ``(n_ao, n_ao, 3)``."""
    integrals = compute_dipole(basis)
    return np.stack(
        [
            np.asarray(integrals.x),
            np.asarray(integrals.y),
            np.asarray(integrals.z),
        ],
        axis=-1,
    )


@dataclass(frozen=True)
class IBOAnalysis:
    """Result of a closed-shell IAO + IBO analysis.

    Attributes
    ----------
    coefficients
        Localised occupied orbitals in the AO basis, shape ``(n_ao, n_occ)``.
    charges
        IAO partial charges, one per atom.
    atom_populations
        ``n_A(i)``, shape ``(n_occ, n_atoms)``; each row sums to 1.
    centroids
        Orbital centroids ``<i|r|i>`` in bohr, shape ``(n_occ, 3)``.  These
        replace the (meaningless) orbital energy as the spatial handle for a
        localised orbital.
    n_centres
        Per-orbital count of atoms carrying more than ``centre_threshold`` of
        the orbital -- 1 for a core or lone pair, 2 for an ordinary bond,
        3+ for a delocalised or multi-centre bond.
    objective_initial, objective_final
        IBO functional before / after.  ``final >= initial`` always.
    n_sweeps
        Jacobi sweeps performed.
    reference_basis
        Name of the minimal basis used as B2.
    power
        Localisation power actually used.
    """

    coefficients: np.ndarray
    charges: np.ndarray
    atom_populations: np.ndarray
    centroids: np.ndarray
    n_centres: np.ndarray
    objective_initial: float
    objective_final: float
    n_sweeps: int
    reference_basis: str
    power: int
    method: str = "ibo"


#: Localization criteria that :func:`analyse_localization` can run, in the
#: order they are emitted.  All three are *described* with the same IAO
#: yardstick so their pictures are directly comparable; only the criterion
#: being maximised differs.
LOCALIZATION_METHODS = ("ibo", "boys", "pipek-mezey")


def analyse_localization(
    molecule: Molecule,
    basis: BasisSet,
    occupied: np.ndarray,
    *,
    method: str = "ibo",
    dipoles: np.ndarray | None = None,
    reference_basis: str = IAO_REFERENCE_BASIS,
    power: int = 4,
    centre_threshold: float = 0.10,
    max_iter: int = 200,
    conv_tol: float = 1e-10,
) -> IBOAnalysis:
    """Localize a converged closed-shell occupied set and describe the result.

    Parameters
    ----------
    molecule, basis
        The system and its main (B1) basis.
    occupied
        Occupied MO coefficients, shape ``(n_ao, n_occ)``.
    method
        One of :data:`LOCALIZATION_METHODS`.  ``"ibo"`` maximises IAO
        populations to the fourth power (Knizia eq 4); ``"boys"`` maximises
        the orbital dipole spread; ``"pipek-mezey"`` maximises *Mulliken*
        populations squared.
    dipoles
        Optional ``(n_ao, n_ao, 3)`` Cartesian dipole integrals.  Required by
        ``"boys"``, and used for the reported centroids in every case.
        Computed on demand when omitted.
    centre_threshold
        Atomic population above which an atom counts toward ``n_centres``.

    Notes
    -----
    Whichever criterion runs, the *descriptors* -- atomic populations, centre
    counts, IAO charges -- are computed in the IAO basis. Boys and
    Pipek-Mezey orbitals are therefore reported on the same yardstick as IBOs
    rather than each in its own units, which is what makes a side-by-side
    comparison mean anything. It also means the reported charges are IAO
    charges in all three cases; they are a property of the occupied space,
    which is invariant under the localizing rotation, so they are identical
    across methods by construction.

    Caller is responsible for the ECP / element-coverage gate --
    see :func:`iao_unsupported_reason`.
    """
    from .localise import (
        foster_boys_localise,
        ibo_localise,
        ibo_objective,
        pipek_mezey_localise,
    )
    from .properties import _shell_to_atom

    if method not in LOCALIZATION_METHODS:
        raise ValueError(
            f"unknown localization method {method!r}; "
            f"expected one of {LOCALIZATION_METHODS}"
        )

    overlap = np.asarray(compute_overlap(basis))
    reference = iao_reference(molecule, basis, name=reference_basis)

    iaos = build_iaos(
        occupied, overlap, reference.overlap, reference.cross_overlap
    )

    # Occupied set in the orthonormal IAO basis.  Lossless: the IAOs span the
    # occupied space exactly, which is the defining property of the
    # construction (Knizia Appendix C).
    in_iao_basis = iaos.conj().T @ overlap @ occupied

    n_atoms = len(molecule.atoms)
    objective_initial = ibo_objective(in_iao_basis, reference.atom_indices, power)

    if dipoles is None and method == "boys":
        dipoles = _dipole_tensor(basis)

    if method == "ibo":
        localised_iao, sweeps = ibo_localise(
            in_iao_basis,
            reference.atom_indices,
            power=power,
            max_iter=max_iter,
            conv_tol=conv_tol,
        )
        coefficients = iaos @ localised_iao
    else:
        if method == "boys":
            coefficients = foster_boys_localise(
                occupied, dipoles, max_iter=max_iter
            )
        else:
            atom_of_ao = _shell_to_atom(basis)
            atom_basis_map = np.zeros((overlap.shape[0], n_atoms))
            atom_basis_map[np.arange(overlap.shape[0]), atom_of_ao] = 1.0
            coefficients = pipek_mezey_localise(
                occupied, overlap, atom_basis_map, max_iter=max_iter
            )
        # Re-express on the IAO yardstick so the descriptors below are
        # comparable with the IBO ones.
        localised_iao = iaos.conj().T @ overlap @ coefficients
        sweeps = 0

    objective_final = ibo_objective(localised_iao, reference.atom_indices, power)
    populations = iao_atom_populations(
        localised_iao, reference.atom_indices, n_atoms
    )

    if dipoles is None:
        dipoles = _dipole_tensor(basis)
    centroids = np.stack(
        [
            np.einsum("mi,mn,ni->i", coefficients, dipoles[:, :, c], coefficients)
            for c in range(3)
        ],
        axis=-1,
    )

    return IBOAnalysis(
        coefficients=coefficients,
        charges=iao_charges(
            occupied,
            iaos,
            overlap,
            np.array([atom.Z for atom in molecule.atoms], dtype=np.float64),
            reference.atom_indices,
        ),
        atom_populations=populations,
        centroids=centroids,
        n_centres=(populations > centre_threshold).sum(axis=1).astype(np.int64),
        objective_initial=objective_initial,
        objective_final=objective_final,
        n_sweeps=sweeps,
        reference_basis=reference.name,
        power=power,
        method=method,
    )


def analyse_ibo(
    molecule: Molecule,
    basis: BasisSet,
    occupied: np.ndarray,
    **kwargs,
) -> IBOAnalysis:
    """IBO-specific wrapper around :func:`analyse_localization`."""
    kwargs.setdefault("method", "ibo")
    return analyse_localization(molecule, basis, occupied, **kwargs)
