"""Energy Decomposition Analysis (EDA) for two molecular fragments.

Partitions the interaction energy of two closed-shell fragments, computed at
the Hartree-Fock level in the *dimer* basis (ghost atoms, so the result is
counterpoise consistent), into physically labelled terms. The partition
follows the path

    E_A + E_B  ->  promolecule (P_A + P_B, orbitals of A and B as they are)
               ->  frozen state (one determinant of the mutually
                   orthogonalised occupied orbitals of A and B)
               ->  relaxed dimer (the converged SCF)

and reads, for the LMO-EDA scheme (Su & Li, J. Chem. Phys. 131, 014102,
2009):

    dE_int = dE_elstat + dE_exch + dE_rep + dE_pol + dE_disp

- ``elstat``: classical Coulomb interaction of the two frozen charge
  distributions, nuclei included (the same term as Kitaura-Morokuma ES).
- ``exch``: the inter-fragment exchange energy of the promolecule state.
- ``rep``: the further Pauli repulsion the frozen state adds, that is the
  cost of orthogonalising the occupied orbitals of A and B.
- ``pol``: everything the converged dimer gains beyond the frozen state,
  orbital polarisation and charge transfer together.
- ``disp``: the difference of the additive dispersion corrections, if any.

:func:`eda_morokuma` reports the Kitaura-Morokuma (Int. J. Quantum Chem. 10,
325, 1976) partition of the same states. Its exchange-repulsion term EX is the
sum ``exch + rep``. Its polarisation, charge-transfer and coupling terms need
restricted SCF variations that are not implemented, so they are reported as
one lumped term.

Scope
-----
Both functions handle closed-shell Hartree-Fock. The terms are functionals of
the fragment densities and need the AO integrals, so the caller passes the
basis, the molecule and the atom indices of each fragment beside the SCF
results. Su and Li define the terms through localised occupied orbitals; every
term above is invariant to unitary rotations within the occupied space of a
fragment, so no localisation is carried out. Kohn-Sham, open-shell,
fractionally occupied and ECP inputs are refused (see ``consistency_tol``).

Public API
----------

.. autofunction:: eda_lmo
.. autofunction:: eda_morokuma
.. autoclass:: EDAResult

Theory references
-----------------

- Su, P.; Li, H., J. Chem. Phys. 131, 014102 (2009). DOI: 10.1063/1.3159673
  (LMO-EDA: localised molecular orbital energy decomposition analysis)
- Morokuma, K., J. Chem. Phys. 55, 1236 (1971). DOI: 10.1063/1.1676210
  (Morokuma EDA, the original energy decomposition scheme)
- Kitaura, K.; Morokuma, K., Int. J. Quantum Chem. 10, 325 (1976).
  DOI: 10.1002/qua.560100211 (the ES/EX/PL/CT/MIX partition within the
  Hartree-Fock approximation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "EDAResult",
    "eda_lmo",
    "eda_morokuma",
    "fragment_density_matrix",
]

_KCAL_PER_HARTREE = 627.5094740631

# A closed-shell density satisfies P S P = 2 P, so the eigenvalues of
# S^(1/2) P S^(1/2) are 0 or 2. Converged SCF densities sit far inside this.
_IDEMPOTENCY_TOL = 1e-5


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EDAResult:
    """Energy decomposition analysis for a two-fragment system.

    All energies in Hartree. The components sum to the total interaction
    energy::

        E_int = E_total - E_frag1 - E_frag2
              = E_elstat + E_exch + E_rep + E_pol + E_disp

    Attributes
    ----------
    e_total : float
        Total energy of the supersystem.
    e_frag1 : float
        Energy of isolated fragment 1 in the supersystem basis (ghost atoms).
    e_frag2 : float
        Energy of isolated fragment 2 in the supersystem basis (ghost atoms).
    e_int : float
        Total interaction energy: E_total - E_frag1 - E_frag2.
    e_elstat : float
        Electrostatic (Coulomb) interaction between the frozen fragment
        charge distributions, including the nuclei.
    e_exch : float
        For ``method="lmo"``, the inter-fragment exchange energy of the
        promolecule state. For ``method="morokuma"``, the full
        Kitaura-Morokuma exchange repulsion EX, which is the LMO-EDA
        ``e_exch + e_rep``.
    e_rep : float
        Pauli repulsion beyond the exchange term (orthogonalisation of the
        occupied orbitals of the two fragments). Zero for
        ``method="morokuma"``, whose EX already contains it.
    e_pol : float
        Orbital relaxation of the frozen state into the converged dimer:
        polarisation and charge transfer together (for
        ``method="morokuma"``: PL + CT + MIX).
    e_disp : float
        Difference of the additive dispersion corrections (zero for plain
        Hartree-Fock).
    method : str
        EDA variant used (``"lmo"`` or ``"morokuma"``).
    extra : dict
        Method-specific numbers: ``e_frozen`` (energy of the antisymmetrised
        frozen determinant, electronic part), ``e_promolecule`` (energy of the
        ``P_A + P_B`` state), ``consistency`` (residuals of the energy check,
        Hartree) and, for Morokuma, ``morokuma_components``.
    """

    e_total: float = 0.0
    e_frag1: float = 0.0
    e_frag2: float = 0.0
    e_int: float = 0.0
    e_elstat: float = 0.0
    e_exch: float = 0.0
    e_rep: float = 0.0
    e_pol: float = 0.0
    e_disp: float = 0.0
    method: str = "lmo"
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary table."""
        rows = [("Electrostatic", self.e_elstat)]
        if self.method == "morokuma":
            rows += [
                ("Exchange (Pauli) EX", self.e_exch),
                ("Polarisation + CT", self.e_pol),
            ]
        else:
            rows += [
                ("Exchange", self.e_exch),
                ("Repulsion (Pauli)", self.e_rep),
                ("Polarisation (+CT)", self.e_pol),
            ]
        if self.e_disp != 0.0:
            rows.append(("Dispersion", self.e_disp))

        def line(label: str, value: float) -> str:
            return (
                f"  {label:<23} = {value:>14.8f} E_h"
                f" {value * _KCAL_PER_HARTREE:>10.2f} kcal/mol"
            )

        lines = [
            f"Energy Decomposition Analysis ({self.method.upper()}-EDA)",
            "-" * 64,
            f"  E_total (supersystem)   = {self.e_total:>14.8f} E_h",
            f"  E_fragment_1 (isolated) = {self.e_frag1:>14.8f} E_h",
            f"  E_fragment_2 (isolated) = {self.e_frag2:>14.8f} E_h",
            "-" * 64,
            line("dE_int (total)", self.e_int),
            "",
            *(line(label, value) for label, value in rows),
            "-" * 64,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fragment density matrix utilities
# ---------------------------------------------------------------------------


def fragment_density_matrix(
    mo_coeffs: np.ndarray,
    overlap: np.ndarray,
    n_occ: int,
    fragment_indices: Sequence[int],
    ao_to_atom: Optional[list[int]] = None,
) -> np.ndarray:
    """Extract the density matrix for a fragment defined by atom indices.

    This zeroes-out AO contributions that don't belong to the fragment
    atoms, then builds the fragment density from the MO coefficients.
    This is a *projection* -- the fragment density is not idempotent, so it
    is **not** a valid ``density_frag1`` / ``density_frag2`` for
    :func:`eda_lmo` (which refuses it). Use the density of a fragment SCF
    run in the dimer basis with ghost atoms.

    Parameters
    ----------
    mo_coeffs : np.ndarray, shape ``(n_ao, n_orb)``
        MO coefficients of the supersystem.
    overlap : np.ndarray, shape ``(n_ao, n_ao)``
        AO overlap matrix.
    n_occ : int
        Number of occupied orbitals.
    fragment_indices : sequence of int
        Atom indices belonging to the fragment.
    ao_to_atom : list of int, optional
        AO-to-atom mapping. If None, all AOs are used (no projection).

    Returns
    -------
    np.ndarray, shape ``(n_ao, n_ao)``
        Fragment density matrix (real part).
    """
    C = np.asarray(mo_coeffs)
    S = np.asarray(overlap)
    n_ao = C.shape[0]

    if ao_to_atom is not None:
        frag_set = set(fragment_indices)
        mask = np.array([ao_to_atom[mu] in frag_set for mu in range(n_ao)], dtype=bool)
        C_frag = C.copy()
        C_frag[~mask, :] = 0.0
    else:
        C_frag = C

    C_occ = C_frag[:, :n_occ]
    P_frag = C_occ @ C_occ.T
    return P_frag.real if np.iscomplexobj(P_frag) else P_frag


# ---------------------------------------------------------------------------
# Shared partition (closed-shell Hartree-Fock)
# ---------------------------------------------------------------------------


def _tr(a: np.ndarray, b: np.ndarray) -> float:
    """Tr[a b] for square matrices, without forming the product."""
    return float(np.sum(a * b.T))


def _fragment_atoms(molecule, indices: Sequence[int], label: str) -> list[int]:
    n_atoms = len(molecule.atoms)
    idx = [int(i) for i in indices]
    if not idx:
        raise ValueError(f"{label} is empty; each fragment needs at least one atom")
    if len(set(idx)) != len(idx):
        raise ValueError(f"{label} lists an atom twice: {idx}")
    if any(i < 0 or i >= n_atoms for i in idx):
        raise ValueError(f"{label} has an atom index outside 0..{n_atoms - 1}: {idx}")
    return idx


def _positions_and_charges(molecule, idx: Sequence[int]):
    atoms = list(molecule.atoms)
    pos = [[float(x) for x in atoms[i].xyz] for i in idx]
    chg = [float(atoms[i].Z) for i in idx]
    return pos, chg


def _coulomb_repulsion(pos_a, chg_a, pos_b=None, chg_b=None) -> float:
    """Sum of Z_i Z_j / r_ij: within one set, or between two sets."""
    if pos_b is None:
        pairs = [(i, j) for i in range(len(chg_a)) for j in range(i + 1, len(chg_a))]
        other_pos, other_chg = pos_a, chg_a
    else:
        pairs = [(i, j) for i in range(len(chg_a)) for j in range(len(chg_b))]
        other_pos, other_chg = pos_b, chg_b
    total = 0.0
    for i, j in pairs:
        r = np.linalg.norm(np.asarray(pos_a[i]) - np.asarray(other_pos[j]))
        total += chg_a[i] * other_chg[j] / r
    return float(total)


def _occupied_from_density(P: np.ndarray, s_half, s_minus_half, label: str) -> np.ndarray:
    """S-orthonormal occupied orbitals C (P = 2 C C^T) of a closed-shell density."""
    X = s_half @ P @ s_half
    w, V = np.linalg.eigh(0.5 * (X + X.T))
    occ = w > 1.0
    dev = max(
        float(np.max(np.abs(w[occ] - 2.0))) if occ.any() else 0.0,
        float(np.max(np.abs(w[~occ]))) if (~occ).any() else 0.0,
    )
    if dev > _IDEMPOTENCY_TOL:
        raise ValueError(
            f"{label} is not a closed-shell density (P S P = 2 P fails by "
            f"{dev:.1e}); the energy decomposition needs the converged RHF "
            "density of the fragment, not a projected, smeared, "
            "fractionally occupied or open-shell density"
        )
    return s_minus_half @ V[:, occ]


def _partition(
    e_total,
    e_frag1,
    e_frag2,
    density_total,
    density_frag1,
    density_frag2,
    overlap,
    basis,
    molecule,
    frag1_atoms,
    frag2_atoms,
    e_disp,
    e_disp_frag1,
    e_disp_frag2,
    consistency_tol,
) -> dict:
    """Energies of the promolecule and frozen states and the resulting terms."""
    from ._vibeqc_core import (
        compute_kinetic,
        compute_nuclear_with_charges,
        compute_overlap,
        make_direct_jk_builder,
    )

    P1 = np.asarray(density_frag1, dtype=float)
    P2 = np.asarray(density_frag2, dtype=float)
    P12 = np.asarray(density_total, dtype=float)
    S = np.asarray(overlap, dtype=float)
    n_bf = int(basis.nbasis)
    for name, arr in (
        ("density_frag1", P1),
        ("density_frag2", P2),
        ("density_total", P12),
        ("overlap", S),
    ):
        if arr.shape != (n_bf, n_bf):
            raise ValueError(
                f"{name} has shape {arr.shape} but the basis has {n_bf} functions; "
                "all three SCF runs must use the dimer basis (ghost atoms for the "
                "fragments)"
            )
    if not np.allclose(S, np.asarray(compute_overlap(basis)), atol=1e-8):
        raise ValueError("overlap does not match the overlap matrix of the given basis")

    idx1 = _fragment_atoms(molecule, frag1_atoms, "frag1_atoms")
    idx2 = _fragment_atoms(molecule, frag2_atoms, "frag2_atoms")
    if set(idx1) & set(idx2):
        raise ValueError("frag1_atoms and frag2_atoms overlap")
    if len(idx1) + len(idx2) != len(molecule.atoms):
        raise ValueError(
            "frag1_atoms and frag2_atoms must together cover every atom of the molecule"
        )

    # Electronic parts: whatever dispersion correction the caller folded into
    # the energies is subtracted here and reported as its own term.
    E_ab = float(e_total) - float(e_disp)
    E_a = float(e_frag1) - float(e_disp_frag1)
    E_b = float(e_frag2) - float(e_disp_frag2)

    pos1, chg1 = _positions_and_charges(molecule, idx1)
    pos2, chg2 = _positions_and_charges(molecule, idx2)
    T = np.asarray(compute_kinetic(basis))
    V1 = np.asarray(compute_nuclear_with_charges(basis, pos1, chg1))
    V2 = np.asarray(compute_nuclear_with_charges(basis, pos2, chg2))
    e_nn_1 = _coulomb_repulsion(pos1, chg1)
    e_nn_2 = _coulomb_repulsion(pos2, chg2)
    e_nn_12 = _coulomb_repulsion(pos1, chg1, pos2, chg2)

    jk = make_direct_jk_builder(basis, 1e-12)

    def hf_energy(P, h, e_nn):
        """RHF energy expression of an arbitrary closed-shell density."""
        G = jk.build_J(P) - 0.5 * jk.build_K(P)
        return _tr(P, h) + 0.5 * _tr(P, G) + e_nn

    if consistency_tol is not None:
        # The terms below are functionals of the densities. If the supplied
        # energies are not the Hartree-Fock energies of the supplied densities
        # (Kohn-Sham, ECPs, wrong ghost set-up), the partition would be wrong
        # without warning, so refuse.
        residual = {
            "frag1": hf_energy(P1, T + V1, e_nn_1) - E_a,
            "frag2": hf_energy(P2, T + V2, e_nn_2) - E_b,
            "total": hf_energy(P12, T + V1 + V2, e_nn_1 + e_nn_2 + e_nn_12) - E_ab,
        }
        worst = max(abs(v) for v in residual.values())
        if worst > consistency_tol:
            raise ValueError(
                "the supplied energies are not the closed-shell Hartree-Fock "
                "energies of the supplied densities (residuals in Ha: "
                + ", ".join(f"{k} {v:+.2e}" for k, v in residual.items())
                + "). eda_lmo and eda_morokuma decompose Hartree-Fock energies "
                "of fragments computed in the dimer basis with ghost atoms; "
                "Kohn-Sham, ECP or mismatched inputs are not supported"
            )
    else:
        residual = {}

    # Electrostatics: frozen charge distributions interacting classically.
    J2 = jk.build_J(P2)
    e_elstat = _tr(P1, V2) + _tr(P2, V1) + _tr(P1, J2) + e_nn_12

    # Exchange of the promolecule state: the inter-fragment part of
    # -1/4 Tr[P K(P)] for P = P1 + P2.
    e_exch = -0.5 * _tr(P1, jk.build_K(P2))

    # Frozen state: one determinant of the mutually orthogonalised occupied
    # orbitals, P = 2 C (C^T S C)^-1 C^T.
    w, U = np.linalg.eigh(S)
    if w.min() < 1e-10:
        raise ValueError("the overlap matrix is numerically singular")
    s_half = (U * np.sqrt(w)) @ U.T
    s_minus_half = (U / np.sqrt(w)) @ U.T
    C = np.hstack(
        [
            _occupied_from_density(P1, s_half, s_minus_half, "density_frag1"),
            _occupied_from_density(P2, s_half, s_minus_half, "density_frag2"),
        ]
    )
    M = C.T @ S @ C
    if np.linalg.eigvalsh(M).min() < 1e-8:
        raise ValueError(
            "the occupied orbitals of the two fragments are linearly dependent; "
            "the frozen state does not exist"
        )
    P_frozen = 2.0 * C @ np.linalg.solve(M, C.T)
    e_frozen = hf_energy(P_frozen, T + V1 + V2, e_nn_1 + e_nn_2 + e_nn_12)

    e_pauli = e_frozen - E_a - E_b - e_elstat  # Kitaura-Morokuma EX = exch + rep
    return {
        "e_elstat": e_elstat,
        "e_exch": e_exch,
        "e_rep": e_pauli - e_exch,
        "e_pauli": e_pauli,
        "e_pol": E_ab - e_frozen,
        "e_disp": float(e_disp) - float(e_disp_frag1) - float(e_disp_frag2),
        "extra": {
            "e_frozen": e_frozen,
            "e_promolecule": E_a + E_b + e_elstat + e_exch,
            "consistency": residual,
        },
    }


# ---------------------------------------------------------------------------
# LMO-EDA (Su & Li 2009)
# ---------------------------------------------------------------------------


def eda_lmo(
    e_total: float,
    e_frag1: float,
    e_frag2: float,
    fock_total: Optional[np.ndarray],
    fock_frag1: Optional[np.ndarray],
    fock_frag2: Optional[np.ndarray],
    density_total: np.ndarray,
    density_frag1: np.ndarray,
    density_frag2: np.ndarray,
    overlap: np.ndarray,
    *,
    basis,
    molecule,
    frag1_atoms: Sequence[int],
    frag2_atoms: Sequence[int],
    e_disp: float = 0.0,
    e_disp_frag1: float = 0.0,
    e_disp_frag2: float = 0.0,
    consistency_tol: Optional[float] = 1e-6,
) -> EDAResult:
    r"""LMO-EDA: localised molecular orbital energy decomposition analysis.

    Decomposes the Hartree-Fock interaction energy of two closed-shell
    fragments (Su & Li, J. Chem. Phys. 131, 014102, 2009) into

    - **Electrostatic**: the classical Coulomb interaction of the frozen
      fragment charge distributions, nuclei included,

      .. math::

         \Delta E_\mathrm{elstat} = \mathrm{Tr}[\mathbf P_1 \mathbf V_2]
           + \mathrm{Tr}[\mathbf P_2 \mathbf V_1]
           + \mathrm{Tr}[\mathbf P_1 \mathbf J(\mathbf P_2)] + E_{nn}^{12}

    - **Exchange**: the inter-fragment exchange energy of the promolecule
      state (the fragment orbitals as they are, not yet orthogonal),

      .. math::

         \Delta E_\mathrm{exch} = -\tfrac12\,\mathrm{Tr}[\mathbf P_1 \mathbf K(\mathbf P_2)]

    - **Repulsion**: the Pauli repulsion beyond exchange, the energy change
      when the occupied orbitals of the two fragments are made mutually
      orthogonal and the energy of the resulting determinant is evaluated
      with the full Hamiltonian,
      :math:`E_\mathrm{frozen} - E_1 - E_2 - \Delta E_\mathrm{elstat} - \Delta E_\mathrm{exch}`.

    - **Polarisation**: the relaxation from the frozen determinant to the
      converged dimer, :math:`E_{12} - E_\mathrm{frozen}`. As in LMO-EDA it
      contains the charge transfer.

    - **Dispersion**: the difference of the additive dispersion corrections.

    The five terms sum to ``e_total - e_frag1 - e_frag2`` exactly. Each is
    invariant to rotations of the occupied orbitals within a fragment, so the
    localisation that gives the scheme its name is not carried out.

    Parameters
    ----------
    e_total : float
        Total energy of the supersystem, including any additive dispersion
        correction.
    e_frag1, e_frag2 : float
        Energies of the isolated fragments in the supersystem basis (ghost
        atoms), including any additive dispersion correction.
    fock_total, fock_frag1, fock_frag2 : np.ndarray or None
        Accepted for signature compatibility and not used: the terms need the
        integrals, which the Fock matrices cannot supply.
    density_total, density_frag1, density_frag2 : np.ndarray
        Converged closed-shell densities of the supersystem and of the
        fragments, all in the supersystem basis, as ``RHFResult.density``.
    overlap : np.ndarray
        AO overlap matrix of the supersystem basis.
    basis : BasisSet
        The supersystem basis.
    molecule : Molecule
        The supersystem with its real nuclear charges, even if a fragment
        was computed with ghost atoms.
    frag1_atoms, frag2_atoms : sequence of int
        Indices into ``molecule.atoms`` of the two fragments. Together they
        must cover every atom.
    e_disp, e_disp_frag1, e_disp_frag2 : float
        The additive dispersion corrections contained in ``e_total``,
        ``e_frag1`` and ``e_frag2``. They are removed before the electronic
        partition and reported as ``e_disp``.
    consistency_tol : float or None
        The Hartree-Fock energy expression is evaluated for the supplied
        densities and compared with the supplied energies. A residual above
        this many Hartree raises ``ValueError``; ``None`` skips the check.

    Returns
    -------
    EDAResult

    Raises
    ------
    ValueError
        If the inputs are not closed-shell Hartree-Fock quantities of the
        dimer basis (Kohn-Sham, ECP, open-shell, smeared or projected
        densities, mismatched energies), or the fragment atoms do not
        partition the molecule.

    Examples
    --------
    The fragments are computed in the dimer basis with ghost atoms (a
    ``Molecule`` accepts ``Z = 0`` atoms)::

        eda = eda_lmo(
            r_ab.energy, r_a.energy, r_b.energy, None, None, None,
            r_ab.density, r_a.density, r_b.density, overlap,
            basis=basis, molecule=dimer,
            frag1_atoms=[0, 1, 2], frag2_atoms=[3, 4, 5],
        )
        print(eda.summary())
"""
    part = _partition(
        e_total, e_frag1, e_frag2,
        density_total, density_frag1, density_frag2, overlap,
        basis, molecule, frag1_atoms, frag2_atoms,
        e_disp, e_disp_frag1, e_disp_frag2, consistency_tol,
    )  # fmt: skip
    return EDAResult(
        e_total=e_total,
        e_frag1=e_frag1,
        e_frag2=e_frag2,
        e_int=e_total - e_frag1 - e_frag2,
        e_elstat=part["e_elstat"],
        e_exch=part["e_exch"],
        e_rep=part["e_rep"],
        e_pol=part["e_pol"],
        e_disp=part["e_disp"],
        method="lmo",
        extra=part["extra"],
    )


# ---------------------------------------------------------------------------
# Morokuma EDA (1971, 1976)
# ---------------------------------------------------------------------------


def eda_morokuma(
    e_total: float,
    e_frag1: float,
    e_frag2: float,
    fock_total: Optional[np.ndarray],
    fock_frag1: Optional[np.ndarray],
    fock_frag2: Optional[np.ndarray],
    density_total: np.ndarray,
    density_frag1: np.ndarray,
    density_frag2: np.ndarray,
    overlap: np.ndarray,
    *,
    basis,
    molecule,
    frag1_atoms: Sequence[int],
    frag2_atoms: Sequence[int],
    e_disp: float = 0.0,
    e_disp_frag1: float = 0.0,
    e_disp_frag2: float = 0.0,
    consistency_tol: Optional[float] = 1e-6,
) -> EDAResult:
    r"""Kitaura-Morokuma EDA of the same fragment states.

    Same inputs and states as :func:`eda_lmo`, partitioned as in
    Morokuma (J. Chem. Phys. 55, 1236, 1971) and Kitaura and Morokuma
    (Int. J. Quantum Chem. 10, 325, 1976):

    - **ES** (``e_elstat``): as in :func:`eda_lmo`.
    - **EX** (``e_exch``): the exchange repulsion, the energy of the
      antisymmetrised frozen determinant relative to the isolated fragments
      and the electrostatic term, :math:`E_\mathrm{frozen} - E_1 - E_2 - ES`.
      It equals the sum ``e_exch + e_rep`` of :func:`eda_lmo`, so ``e_rep``
      of the result is zero.
    - **PL + CT + MIX** (``e_pol``): everything the converged dimer gains
      beyond the frozen state. Kitaura and Morokuma split this by restricted
      SCF variations (polarisation within each fragment, charge transfer in
      each direction, and a coupling remainder); those variations are not
      implemented here, so the terms are reported as one lumped number, also
      under ``extra["morokuma_components"]["PL+CT+MIX"]``.

    Parameters, return value and exceptions are those of :func:`eda_lmo`.
    """
    part = _partition(
        e_total, e_frag1, e_frag2,
        density_total, density_frag1, density_frag2, overlap,
        basis, molecule, frag1_atoms, frag2_atoms,
        e_disp, e_disp_frag1, e_disp_frag2, consistency_tol,
    )  # fmt: skip
    extra = dict(part["extra"])
    extra["morokuma_components"] = {
        "ES": part["e_elstat"],
        "EX": part["e_pauli"],
        "PL+CT+MIX": part["e_pol"],
    }
    return EDAResult(
        e_total=e_total,
        e_frag1=e_frag1,
        e_frag2=e_frag2,
        e_int=e_total - e_frag1 - e_frag2,
        e_elstat=part["e_elstat"],
        e_exch=part["e_pauli"],
        e_rep=0.0,
        e_pol=part["e_pol"],
        e_disp=part["e_disp"],
        method="morokuma",
        extra=extra,
    )

