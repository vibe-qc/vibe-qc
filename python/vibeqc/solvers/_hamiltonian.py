"""Hamiltonian construction from AO integrals.

Provides a clean interface for building :class:`Hamiltonian` objects
from vibe-qc's native C++ integral bindings, with optional orbital
transformation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._vibeqc_core import (
    BasisSet,
    Molecule,
    compute_eri,
)
from ._common import Hamiltonian, _chemist_to_physicist


def build_hamiltonian_ao(
    molecule: Molecule,
    basis: BasisSet,
    *,
    description: str = "",
    reference: object = None,
) -> Hamiltonian:
    """Build a :class:`Hamiltonian` in the **AO basis**.

    The returned ``h1e`` and ``h2e`` are in the original (non-orthogonal)
    atomic-orbital basis.  Most solvers require an orthonormal basis;
    transform with :func:`transform_hamiltonian_to_mo` after orthogonalising.

    Parameters
    ----------
    molecule : Molecule
        Molecular geometry, charge, multiplicity.
    basis : BasisSet
        AO basis set.
    description : str
        Label for logging.

    Returns
    -------
    Hamiltonian
        With ``h1e``, ``h2e``, ``nuclear_repulsion``, ``nelec``, ``ms2``,
        and ``norb`` populated.
    """
    from ..ecp_metadata import one_electron_hamiltonian

    # T + V_ne(Z_eff) + V_ECP, the Z_eff repulsion and the valence count when
    # ``reference`` carries an ECP on either route; the bare-Z all-electron
    # Hamiltonian otherwise. The solver then diagonalises the same operator
    # the mean-field reference did instead of silently dropping the ECP.
    h1e_ao, e_nuc, nelec, _z_eff = one_electron_hamiltonian(
        molecule, basis, reference
    )
    h1e_ao = np.asarray(h1e_ao, order="C")

    # Chemist's notation (muν|ls) from libint -> physicist's g_{muνls}
    eri_chem_ao = np.asarray(compute_eri(basis), order="C")
    h2e_ao = _chemist_to_physicist(eri_chem_ao)  # g_{muνls} = (mul|νs)

    norb = h1e_ao.shape[0]
    multiplicity = molecule.multiplicity
    ms2 = multiplicity - 1  # 2*S_z

    return Hamiltonian(
        h1e=h1e_ao,
        h2e=h2e_ao,
        nuclear_repulsion=float(e_nuc),
        norb=norb,
        nelec=int(nelec),
        ms2=ms2,
        description=description or f"{int(nelec)}e, {norb} orbitals",
    )


def build_hamiltonian_mo(
    molecule: Molecule,
    basis: BasisSet,
    mo_coeffs: np.ndarray,
    *,
    description: str = "",
    reference: object = None,
) -> Hamiltonian:
    """Build a :class:`Hamiltonian` in the **MO basis**.

    Transforms the one- and two-electron integrals from the AO basis
    into the molecular-orbital basis given by ``mo_coeffs``.

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet
    mo_coeffs : (n_ao, n_mo) ndarray
        AO -> MO coefficient matrix (columns are MOs).
    description : str

    Returns
    -------
    Hamiltonian
        With integrals in the MO basis. ``norb`` = n_mo.
    """
    ham_ao = build_hamiltonian_ao(
        molecule, basis, description=description, reference=reference
    )
    return transform_hamiltonian(ham_ao, mo_coeffs)


def transform_hamiltonian(
    ham: Hamiltonian,
    C: np.ndarray,
) -> Hamiltonian:
    """Transform a Hamiltonian into a new orthonormal basis given by ``C``.

    New one-electron matrix:  h̃_{pq} = S_{muν} C_{mup} h_{muν} C_{νq}
    New two-electron tensor:  g̃_{pqrs} = S_{muνls} C_{mup} C_{νq} g_{muνls} C_{lr} C_{ss}

    Parameters
    ----------
    ham : Hamiltonian
        Source Hamiltonian.
    C : (n_ao, n_mo) ndarray
        Transformation matrix.  Columns are the new basis vectors in the
        old (AO) basis.
    """
    h1e_new = np.einsum("up,uv,vq->pq", C, ham.h1e, C, optimize=True)
    n_mo = C.shape[1]
    h2e_new = np.einsum("ap,bq,cr,ds,abcd->pqrs", C, C, C, C, ham.h2e, optimize=True)

    return Hamiltonian(
        h1e=h1e_new,
        h2e=h2e_new,
        nuclear_repulsion=ham.nuclear_repulsion,
        norb=n_mo,
        nelec=ham.nelec,
        ms2=ham.ms2,
        description=ham.description,
    )


@dataclass(frozen=True)
class OrbitalFrameReport:
    """What :func:`canonicalize_degenerate_orbitals` did to an orbital set.

    ``clusters`` are the half-open column ranges whose clustering values
    (orbital energies or occupations) were within ``tol`` of their
    neighbour; ``unresolved`` are the sub-blocks the three second moments
    could not split (left as given, signs fixed). ``origin`` is the nuclear
    centre of charge in bohr about which the moments were taken.
    """

    tol: float
    origin: tuple[float, float, float]
    clusters: tuple[tuple[int, int], ...]
    unresolved: tuple[tuple[int, int], ...]

    @property
    def description(self) -> str:
        n = len(self.clusters)
        text = (
            "canonical frame: degenerate clusters aligned by successive "
            "projected second moments about the nuclear centre of charge "
            f"({n} cluster{'s' if n != 1 else ''}, tol {self.tol:g})"
        )
        if self.unresolved:
            text += f"; {len(self.unresolved)} sub-block(s) left as given"
        return text


def _resolve_block(
    block: np.ndarray, operator: np.ndarray, moment_tol: float
) -> tuple[np.ndarray, list[list[int]]]:
    """Re-span one degenerate block by an operator's eigenvectors, where it can.

    Returns the new columns (ordered by ascending eigenvalue of
    ``block^T operator block``) and the groups of local column indices the
    operator left degenerate (to ``moment_tol``). A group of one is
    resolved: its column is the eigenvector. A group of several is not
    resolved by this operator, and its frame must then not be the arbitrary
    one ``eigh`` returns inside a degenerate eigenspace (that frame changes
    from call to call, so a second pass would move the orbitals): its
    columns are the projections of the *input* columns onto the group's
    eigenspace, orthonormalised in input order, which is a deterministic
    function of the input and leaves an already-canonical block untouched.
    A block the operator does not split at all is returned unchanged.
    """
    w, U = np.linalg.eigh(block.T @ operator @ block)
    k = len(w)
    groups: list[list[int]] = []
    start = 0
    for t in range(1, k + 1):
        if t == k or w[t] - w[t - 1] > moment_tol:
            groups.append(list(range(start, t)))
            start = t
    if len(groups) == 1:
        return block, [list(range(k))]
    columns: list[np.ndarray] = []
    out_groups: list[list[int]] = []
    col = 0
    for group in groups:
        V = U[:, group]
        if len(group) == 1:
            columns.append(block @ V)
        else:
            P = V @ V.T
            basis: list[np.ndarray] = []
            for j in range(k):
                q = P[:, j].copy()
                for b in basis:
                    q -= b * float(b @ q)
                norm = float(np.linalg.norm(q))
                if norm > 0.5:
                    basis.append(q / norm)
                if len(basis) == len(group):
                    break
            Q = np.column_stack(basis) if len(basis) == len(group) else V
            columns.append(block @ Q)
        out_groups.append(list(range(col, col + len(group))))
        col += len(group)
    return np.hstack(columns), out_groups


def canonicalize_degenerate_orbitals(
    molecule: Molecule,
    basis: BasisSet,
    mo_coeffs: np.ndarray,
    values: np.ndarray,
    *,
    tol: float = 1e-8,
    moment_tol: float = 1e-6,
) -> tuple[np.ndarray, OrbitalFrameReport]:
    """Fix the frame inside every degenerate cluster of an orbital set (#322).

    ``values`` (one per column of ``mo_coeffs``: orbital energies, or
    occupations for natural orbitals) define the clusters: consecutive
    columns whose value is within ``tol`` of the previous column's form
    one. Inside a cluster the eigensolver's frame is rounding noise --
    N2/cc-pVDZ's 1pi_u and 1pi_g pairs come out mixed differently on
    different hosts and even at two bond lengths on one host -- and every
    quantity that is not invariant under rotations within the cluster
    inherits that noise: the seniority-zero (DOCI) energy swept 16 mHa
    across the recorded frames, a truncated selected-CI rung, a DMRG at
    finite bond dimension and an active window ending inside the cluster
    behave the same way.

    Each cluster is re-spanned by the eigenvectors of the projected second
    moments ``X2 = B^T D_x S^-1 D_x B`` (then ``Y2`` and ``Z2`` inside the
    sub-blocks the previous operator left degenerate, to ``moment_tol``),
    with ``D_k`` the dipole integrals about the nuclear centre of charge
    and ``S`` the AO overlap. ``x^2`` is totally symmetric, so wherever the
    point group allows an axis-aligned choice (pi_x/pi_y of a linear
    molecule along z, t2g, ...) this is it; elsewhere the frame is still a
    function of the deck alone. Columns outside clusters are returned
    unchanged; inside a cluster they are ordered by ascending moment
    eigenvalue and signed so their largest coefficient is positive. The
    rotation stays inside eigenspaces of the operator that produced
    ``values``, so a diagonal Fock matrix stays diagonal and every
    rotation-invariant energy (SCF, MP2, CC, CASCI, FCI) is unchanged.

    Returns the new coefficient matrix and an :class:`OrbitalFrameReport`.
    """
    from .._vibeqc_core import compute_dipole, compute_overlap

    C = np.array(mo_coeffs, dtype=float, order="C")
    v = np.asarray(values, dtype=float).reshape(-1)
    n = C.shape[1]
    if v.shape[0] != n:
        raise ValueError(
            "canonicalize_degenerate_orbitals: need one clustering value per "
            f"orbital column; got {v.shape[0]} values for {n} columns"
        )
    charges = np.array([float(atom.Z) for atom in molecule.atoms])
    positions = np.array([list(atom.xyz) for atom in molecule.atoms], dtype=float)
    if charges.size and charges.sum() > 0:
        centre = (charges[:, None] * positions).sum(axis=0) / charges.sum()
    else:
        centre = np.zeros(3)
    origin = (float(centre[0]), float(centre[1]), float(centre[2]))
    overlap = np.asarray(compute_overlap(basis))
    dipole = compute_dipole(basis, list(origin))
    # Projected second moments in the AO basis: D S^-1 D, one per axis.
    moments = [
        np.asarray(D) @ np.linalg.solve(overlap, np.asarray(D))
        for D in (dipole.x, dipole.y, dipole.z)
    ]

    clusters: list[tuple[int, int]] = []
    unresolved: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(v[j] - v[j - 1]) <= tol:
            j += 1
        if j - i > 1:
            clusters.append((i, j))
            blocks = [list(range(i, j))]
            for A in moments:
                split: list[list[int]] = []
                for blk in blocks:
                    if len(blk) == 1:
                        split.append(blk)
                        continue
                    new_block, groups = _resolve_block(C[:, blk], A, moment_tol)
                    C[:, blk] = new_block
                    split.extend([blk[g] for g in group] for group in groups)
                blocks = split
            for blk in blocks:
                if len(blk) > 1:
                    unresolved.append((blk[0], blk[-1] + 1))
            for k in range(i, j):
                m = int(np.argmax(np.abs(C[:, k])))
                if C[m, k] < 0.0:
                    C[:, k] *= -1.0
        i = j
    report = OrbitalFrameReport(
        tol=float(tol),
        origin=origin,
        clusters=tuple(clusters),
        unresolved=tuple(unresolved),
    )
    return np.ascontiguousarray(C), report


def get_hf_orbital_provider(
    molecule: Molecule,
    basis: BasisSet,
    *,
    method: str = "rhf",
    _with_result: bool = False,
    scf_options: object = None,
    canonical_frame: bool = True,
    degeneracy_tol: float = 1e-8,
) -> np.ndarray | tuple[np.ndarray, object]:
    """Run vibe-qc's RHF and return the MO coefficient matrix.

    This is a **convenience** function -- the solver layer does not
    require HF orbitals.  Any source of orthonormal orbitals works.

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet
    method : str
        ``"rhf"`` (default), ``"uhf"``, ``"uno"`` or ``"rohf"``.
    canonical_frame : bool
        ``True`` (default) passes the orbitals through
        :func:`canonicalize_degenerate_orbitals`, keyed on the orbital
        energies (``"uno"``: the natural occupations), so that the frame
        inside every degenerate cluster is a function of the deck rather
        than of the host's eigensolver (#322). ``False`` returns the SCF's
        own eigenvectors: a frame-dependent result computed on them (a
        seniority-zero energy, a truncated selected-CI rung) is then
        reproducible only on the host that produced it.
    degeneracy_tol : float
        Cluster tolerance handed to the canonicaliser (1e-8 Ha, or 1e-8 in
        occupation for ``"uno"``).
    scf_options : RHFOptions | UHFOptions, optional
        The caller's SCF options, used as the base instead of a fresh
        object. Callers that accept ``rhf_options=`` / ``uhf_options=``
        must thread them here: an ECP supplied only through those options
        (rather than through a basis sidecar) reaches the reference this
        way and nowhere else, and a dropped one is silent -- the run
        completes on a bare-Z Hamiltonian (#740). Convergence controls
        below are applied on top unless the caller set them.

    Returns
    -------
    mo_coeffs : (n_ao, n_ao) ndarray
        Columns are canonical HF molecular orbitals, in the canonical
        frame unless ``canonical_frame=False``.
    """
    # The package-level wrappers (not the raw bindings) attach a bundled
    # basis' ECP sidecar, so the orbitals -- and the result handed back with
    # ``_with_result`` -- describe the ECP Hamiltonian when the basis has one.
    from .. import run_rhf, run_uhf
    from .._vibeqc_core import RHFOptions, UHFOptions

    def finish(coeffs: np.ndarray, result: object, values: object):
        orbitals = np.asarray(coeffs, order="C")
        if canonical_frame:
            orbitals, _frame = canonicalize_degenerate_orbitals(
                molecule, basis, orbitals, np.asarray(values), tol=degeneracy_tol
            )
        if _with_result:
            return orbitals, result
        return orbitals

    def _tighten(opts: object, default_cls: type) -> object:
        """Caller's options when given, else a fresh tight-convergence set.

        A caller-supplied object keeps everything it carries -- an ECP
        route above all -- and only gains the tight thresholds this
        provider relies on, and only where the caller left the default.
        """
        if opts is None:
            opts = default_cls()
            opts.max_iter = 200
            opts.conv_tol_energy = 1e-12
            opts.conv_tol_grad = 1e-10
            return opts
        if not isinstance(opts, default_cls):
            raise TypeError(
                "get_hf_orbital_provider: scf_options must be a "
                f"{default_cls.__name__} for method={method!r}, got "
                f"{type(opts).__name__}"
            )
        fresh = default_cls()
        if opts.max_iter == fresh.max_iter:
            opts.max_iter = 200
        if opts.conv_tol_energy == fresh.conv_tol_energy:
            opts.conv_tol_energy = 1e-12
        if opts.conv_tol_grad == fresh.conv_tol_grad:
            opts.conv_tol_grad = 1e-10
        return opts

    if method == "rhf":
        result = run_rhf(molecule, basis, _tighten(scf_options, RHFOptions))
        return finish(result.mo_coeffs, result, result.mo_energies)
    elif method == "uhf":
        result = run_uhf(molecule, basis, _tighten(scf_options, UHFOptions))
        return finish(result.mo_coeffs_alpha, result, result.mo_energies_alpha)
    elif method == "uno":
        # UHF natural orbitals (UNO-CAS) -- P. Pulay & T. P. Hamilton,
        # J. Chem. Phys. 88, 4926 (1988), doi:10.1063/1.454704: diagonalize
        # the total UHF density in the symmetrically-orthogonalized AO
        # basis,
        #   n, U = eigh(S^{1/2} (Pa + Pb) S^{1/2}),   C = S^{-1/2} U,
        # ordered by descending occupation n.  Occupations cluster near 2
        # (doubly occupied), 1 (open shell) and 0 (virtual), so a single
        # spin-restricted orbital set carries the open-shell character and
        # the descending order matches the CAS family's lowest-core /
        # active-window convention.  This is the recommended starting
        # reference for open-shell CASCI / CASSCF / MR-PT2 (the plain
        # "uhf" choice uses the a orbitals only, leaving the b space -- and
        # hence the doubly-occupied core -- only approximately represented).
        from .._vibeqc_core import compute_overlap

        # Same threading as the plain "uhf" branch: an open-shell CAS
        # reference resolves to "uno", so a manual ECP on uhf_options must
        # reach the UHF that builds the natural orbitals (#740).
        result = run_uhf(molecule, basis, _tighten(scf_options, UHFOptions))
        dm_total = np.asarray(result.density_alpha) + np.asarray(
            result.density_beta
        )
        s_ao = np.asarray(compute_overlap(basis))
        s_val, s_vec = np.linalg.eigh(s_ao)
        if float(np.min(s_val)) < 1e-10:
            raise ValueError(
                "AO overlap is near-singular (min eigenvalue "
                f"{np.min(s_val):.2e}); UNO construction needs a "
                "non-degenerate basis."
            )
        s_half = (s_vec * np.sqrt(s_val)) @ s_vec.T
        s_inv_half = (s_vec / np.sqrt(s_val)) @ s_vec.T
        occ, u_no = np.linalg.eigh(s_half @ dm_total @ s_half)
        order = np.argsort(occ)[::-1]  # descending occupation
        return finish(
            np.ascontiguousarray(s_inv_half @ u_no[:, order]), result, occ[order]
        )
    elif method == "rohf":
        # Restricted open-shell HF reference (Roothaan, Rev. Mod. Phys. 32,
        # 179 (1960)).  ROHF gives ONE spin-restricted orbital set already
        # ordered closed (doubly occ) -> open (singly occ) -> virtual, so
        # the doubly-occupied core and the active window are well-defined
        # AND the determinant is spin-pure (<S^2> = S(S+1) exactly) -- a
        # cleaner CAS starting reference than the spin-contaminated UHF
        # alpha set, and an alternative to UNO that needs no natural-orbital
        # diagonalisation.
        from ..rohf import ROHFOptions, run_rohf

        opts = ROHFOptions()
        opts.max_iter = 200
        opts.conv_tol_energy = 1e-12
        opts.conv_tol_grad = 1e-9
        result = run_rohf(molecule, basis, opts)
        return finish(result.mo_coeffs, result, result.mo_energies)
    else:
        raise ValueError(f"Unknown HF method: {method!r}")


def canonical_orthogonalize(
    S: np.ndarray,
    threshold: float = 1e-8,
) -> np.ndarray:
    """Return the canonical (Löwdin-symmetric) orthogonalisation matrix X.

    X = U s^{-1/2} U^T  where S = U s U^T is the overlap eigen-decomposition.
    This produces orthonormal orbitals closest to the original AO basis.

    Parameters
    ----------
    S : (n, n) ndarray
        Overlap matrix.
    threshold : float
        Eigenvalue threshold for linear-dependence removal.

    Returns
    -------
    X : (n, n_active) ndarray
        Orthogonalisation matrix: C_orth = X^T . C_ao.
    """
    evals, evecs = np.linalg.eigh(S)
    mask = evals > threshold
    s_inv_sqrt = np.diag(1.0 / np.sqrt(evals[mask]))
    return evecs[:, mask] @ s_inv_sqrt
