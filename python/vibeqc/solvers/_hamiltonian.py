"""Hamiltonian construction from AO integrals.

Provides a clean interface for building :class:`Hamiltonian` objects
from vibe-qc's native C++ integral bindings, with optional orbital
transformation.
"""

from __future__ import annotations

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


def get_hf_orbital_provider(
    molecule: Molecule,
    basis: BasisSet,
    *,
    method: str = "rhf",
    _with_result: bool = False,
    scf_options: object = None,
) -> np.ndarray | tuple[np.ndarray, object]:
    """Run vibe-qc's RHF and return the MO coefficient matrix.

    This is a **convenience** function -- the solver layer does not
    require HF orbitals.  Any source of orthonormal orbitals works.

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet
    method : str
        ``"rhf"`` (default) or ``"uhf"``.
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
        Columns are canonical HF molecular orbitals.
    """
    # The package-level wrappers (not the raw bindings) attach a bundled
    # basis' ECP sidecar, so the orbitals -- and the result handed back with
    # ``_with_result`` -- describe the ECP Hamiltonian when the basis has one.
    from .. import run_rhf, run_uhf
    from .._vibeqc_core import RHFOptions, UHFOptions

    def finish(coeffs: np.ndarray, result: object):
        orbitals = np.asarray(coeffs, order="C")
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
        return finish(result.mo_coeffs, result)
    elif method == "uhf":
        result = run_uhf(molecule, basis, _tighten(scf_options, UHFOptions))
        return finish(result.mo_coeffs_alpha, result)
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
        return finish(np.ascontiguousarray(s_inv_half @ u_no[:, order]), result)
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
        return finish(result.mo_coeffs, result)
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
