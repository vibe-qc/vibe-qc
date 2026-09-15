"""Embedded one-electron Hamiltonian: inject external point charges
into the molecular SCF Hamiltonian via ``compute_nuclear_with_charges``.

Milestone 1 enabler for embedded-cluster CAS: the QM region sits in a
field of classical point charges (the embedding / Madelung potential).
Zero new core code — reuses the existing pybind11 binding
``compute_nuclear_with_charges`` (also used by the CPCM solvation
driver) and ``run_rhf_scf_with_jk``.

Productionised from ``studies/embedded-cluster-cas/spike_external_field.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import vibeqc as vq
from vibeqc._vibeqc_core import (
    ECPCenter,
    compute_ecp_matrix,
    compute_kinetic,
    compute_nuclear,
    compute_nuclear_with_charges,
    compute_overlap,
    make_direct_jk_builder,
)
from vibeqc.solvers._hamiltonian import build_hamiltonian_ao, transform_hamiltonian


def _attr(obj: object, *names: str) -> object:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"none of {names} on {type(obj).__name__}")


def _energy(res: object) -> float:
    """RHFResult -> .energy ; CASCI/CASSCF/NEVPT2Result -> .e_total."""
    return float(_attr(res, "e_total", "energy"))


def qm_ext_repulsion(mol: vq.Molecule, ext_pos: list, ext_q: list) -> float:
    r"""Σ_{QM nucleus A, external charge B} Z_A q_B / |R_A − R_B|."""
    e = 0.0
    for a in mol.atoms:
        Ra = np.asarray(a.xyz, dtype=float)
        for q, Rb in zip(ext_q, ext_pos):
            e += float(a.Z) * float(q) / np.linalg.norm(Ra - np.asarray(Rb, float))
    return e


def embedded_pieces(
    mol: vq.Molecule,
    basis: vq.BasisSet,
    ext_pos: list,
    ext_q: list,
    *,
    ecp_centers: list | None = None,
    ecp_library: str = "",
    ecp_share_dir: str = "",
) -> tuple[np.ndarray, np.ndarray, float, object]:
    """Build the embedded one-electron pieces.

    Returns ``(S, Hcore, E_nuc, jk)`` where *Hcore* = T + V_nuc(QM) +
    V_ext + V_aimp (if *ecp_centers* provided) and *E_nuc* = E_nuc(QM-QM)
    + Σ_{QM,ext} Z_A q_B / R_AB.

    Parameters
    ----------
    mol, basis, ext_pos, ext_q :
        The QM molecule + external point-charge array.
    ecp_centers : list[ECPCenter], optional
        AIMP or other ECP centres whose potential is added to Hcore.
    ecp_library : str, optional
        ECP library name (e.g. ``"aimp_ecp"``).
    ecp_share_dir : str, optional
        Directory containing the ECP XML library.
    """
    S = np.asarray(compute_overlap(basis))
    T = np.asarray(compute_kinetic(basis))
    V_nuc = np.asarray(compute_nuclear(basis, mol))
    if ext_q:
        V_ext = np.asarray(
            compute_nuclear_with_charges(
                basis,
                [list(map(float, p)) for p in ext_pos],
                list(map(float, ext_q)),
            )
        )
    else:
        V_ext = np.zeros_like(T)
    Hcore = T + V_nuc + V_ext
    # AIMP / ECP contribution (MR7).
    if ecp_centers:
        V_aimp = np.asarray(
            compute_ecp_matrix(basis, list(ecp_centers), ecp_library, ecp_share_dir)
        )
        Hcore = Hcore + V_aimp
    E_nuc = mol.nuclear_repulsion() + qm_ext_repulsion(mol, ext_pos, ext_q)
    jk = make_direct_jk_builder(basis)
    return S, Hcore, float(E_nuc), jk


def embedded_mo_hamiltonian(
    mol: vq.Molecule,
    basis: vq.BasisSet,
    C: np.ndarray,
    Hcore: np.ndarray,
    E_nuc: float,
) -> object:
    """MO Hamiltonian whose h1e carries V_ext (*Hcore* already includes it).

    Bypasses ``build_hamiltonian_ao`` (which hardcodes
    ``compute_nuclear``, QM nuclei only) by replacing its h1e with the
    already-constructed *Hcore* via ``dataclasses.replace``.
    """
    ham_ao = build_hamiltonian_ao(mol, basis)
    ham_ao = dataclasses.replace(
        ham_ao, h1e=np.asarray(Hcore), nuclear_repulsion=float(E_nuc)
    )
    return transform_hamiltonian(ham_ao, np.asarray(C))
